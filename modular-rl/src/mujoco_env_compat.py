"""mujoco_py-API-compatible MujocoEnv, backed by the modern `mujoco` bindings.

gym==0.13.1's bundled gym.envs.mujoco.mujoco_env.MujocoEnv hard-requires
mujoco_py, which (a) has no arm64 wheels and (b) depends on an old
activation-key/license-server flow that no longer exists (MuJoCo dropped it
when it went free in 2021), so it cannot be installed or used today.

This module reimplements the same self.model / self.data / self.sim.* /
do_simulation / set_state / state_vector / dt / viewer_setup surface using
the modern `mujoco` package instead, so every environments/*.py ModularEnv
subclass keeps working unchanged -- only its
`from gym.envs.mujoco import mujoco_env` import line needs to become
`import mujoco_env_compat as mujoco_env`.
"""
import os

# gym==0.13.1's gym/__init__.py does `import distutils.version`, which was
# removed from the stdlib in Python 3.12. Importing setuptools first installs
# its distutils shim into sys.modules before gym needs it.
import setuptools  # noqa: F401
import numpy as np
import mujoco
import gym
from gym import spaces
from gym.utils import seeding

from environments.terrain import inject_terrain, get_obstacle_geom_ids, recycle_obstacles

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    low = np.full(observation.shape, -float("inf"))
    high = np.full(observation.shape, float("inf"))
    return spaces.Box(low, high, dtype=observation.dtype)


class _MjModelWrapper:
    """Adds mujoco_py-style name/id helpers on top of mujoco.MjModel."""

    def __init__(self, model):
        self._model = model
        self.body_names = tuple(model.body(i).name for i in range(model.nbody))
        self.camera_names = tuple(model.camera(i).name for i in range(model.ncam))

    def body_name2id(self, name):
        return self._model.body(name).id

    def camera_name2id(self, name):
        return self._model.camera(name).id

    def __getattr__(self, name):
        return getattr(self._model, name)


class _MjDataWrapper:
    """Adds mujoco_py-style get_body_xpos/xquat/xvelp/xvelr helpers.

    xvelp/xvelr are computed via the body Jacobian dotted with qvel, matching
    mujoco_py's own get_body_xvelp/get_body_xvelr implementation exactly (both
    ultimately call the same underlying MuJoCo mj_jacBody routine) -- this is
    a different (and, for this repo's obs convention, correct) quantity than
    mujoco.MjData.cvel, which is a center-of-mass-frame spatial velocity.
    """

    def __init__(self, model, data):
        self._model = model
        self._data = data

    def get_body_xpos(self, name):
        return self._data.xpos[self._model.body(name).id].copy()

    def get_body_xquat(self, name):
        return self._data.xquat[self._model.body(name).id].copy()

    def get_body_xvelp(self, name):
        bid = self._model.body(name).id
        jacp = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, None, bid)
        return jacp @ self._data.qvel

    def get_body_xvelr(self, name):
        bid = self._model.body(name).id
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, None, jacr, bid)
        return jacr @ self._data.qvel

    def __getattr__(self, name):
        return getattr(self._data, name)


class _Sim:
    """Minimal stand-in for mujoco_py.MjSim."""

    def __init__(self, model):
        self._raw_model = model
        self._raw_data = mujoco.MjData(model)
        self.model = _MjModelWrapper(model)
        self.data = _MjDataWrapper(model, self._raw_data)

    def step(self):
        mujoco.mj_step(self._raw_model, self._raw_data)

    def forward(self):
        mujoco.mj_forward(self._raw_model, self._raw_data)

    def reset(self):
        mujoco.mj_resetData(self._raw_model, self._raw_data)


class _Camera:
    def __init__(self):
        self.trackbodyid = -1
        self.distance = 0.0
        self.lookat = np.zeros(3)
        self.elevation = -20.0
        self.azimuth = 90.0


class _Viewer:
    """Best-effort stand-in for mujoco_py's MjViewer / MjRenderContextOffscreen.

    Only exercised by render()/visualize.py, not by the train/step/reset path
    that main.py uses -- untested against a real display in this session.
    """

    def __init__(self, sim, mode):
        self.sim = sim
        self.mode = mode
        self.cam = _Camera()
        self._renderer = None
        self._passive_handle = None
        self._last_frame = None

    def _mjv_camera(self):
        cam = mujoco.MjvCamera()
        if self.cam.trackbodyid >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.cam.trackbodyid
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = self.cam.lookat
        cam.distance = self.cam.distance
        cam.elevation = self.cam.elevation
        cam.azimuth = self.cam.azimuth
        return cam

    def render(self, width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_id=None):
        if self.mode == "human":
            import mujoco.viewer

            if self._passive_handle is None:
                self._passive_handle = mujoco.viewer.launch_passive(
                    self.sim._raw_model, self.sim._raw_data
                )
            self._passive_handle.sync()
        else:
            if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
                self._renderer = mujoco.Renderer(self.sim._raw_model, height, width)
            self._renderer.update_scene(self.sim._raw_data, camera=self._mjv_camera())
            self._last_frame = self._renderer.render()

    def read_pixels(self, width, height, depth=False):
        if depth:
            return None, self._last_frame
        return self._last_frame


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments -- same public surface as
    gym==0.13.1's gym.envs.mujoco.mujoco_env.MujocoEnv, backed by `mujoco`."""

    def __init__(self, model_path, frame_skip, rgb_rendering_tracking=True,
                 terrain="flat", seed=None, direction_deg=0.0, terrain_kwargs=None):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip

        # terrain (default "flat") wraps the agent's own xml in a temp copy
        # with terrain geoms added; the agent's kinematic tree itself is
        # never touched -- see environments/terrain.py
        terrain_xml_path, hfield_heights = inject_terrain(
            fullpath, terrain=terrain, seed=seed, direction_deg=direction_deg,
            **(terrain_kwargs or {}))
        self.sim = _Sim(mujoco.MjModel.from_xml_path(terrain_xml_path))
        self.model = self.sim.model
        self.data = self.sim.data
        self.viewer = None
        self.rgb_rendering_tracking = rgb_rendering_tracking
        self._viewers = {}

        if hfield_heights is not None:
            self.model.hfield_data[:] = hfield_heights.ravel()
        # obstacle recycling needs its own rng and heading (do_simulation uses
        # these on every step) -- set before the auto-inference step() call
        # below, which already exercises do_simulation once.
        self._terrain_direction_deg = direction_deg
        self._terrain_rng = np.random.default_rng(seed)
        self._obstacle_geom_ids = (
            get_obstacle_geom_ids(self.model) if terrain == "obstacles" else [])

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel). Implement in each subclass."""
        raise NotImplementedError

    def viewer_setup(self):
        """Called when the viewer is initialized; override to set camera position etc."""
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
        if self._obstacle_geom_ids:
            # qpos[:2] is the agent's (x, y): only correct for a free-joint
            # (3D) root -- terrain='obstacles' is only meant for those.
            agent_xy = self.sim.data.qpos[:2].copy()
            recycle_obstacles(self.model, agent_xy, self._terrain_direction_deg,
                              self._terrain_rng, self._obstacle_geom_ids)

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == "rgb_array":
            camera_id = None
            camera_name = "track"
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            self.viewer = _Viewer(self.sim, mode)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
