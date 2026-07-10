"""Vendored from openai/baselines (baselines/common/vec_env/{__init__,subproc_vec_env}.py),
baselines==0.1.5, commit-pinned via PyPI release.

Vendored instead of depended-on because baselines' setup.py hard-requires
tensorflow>=1.4.0 for the whole package (needed by its RL algorithms, not by
this file), and no tensorflow wheel exists for current Python -- so `pip
install baselines` cannot succeed at all. SubprocVecEnv/VecEnv themselves have
no tensorflow dependency; only `cloudpickle` (used so multiprocessing can
serialize the env-constructor closures instead of failing on plain pickle).

API matches the old-gym 4-tuple convention (step -> obs, reward, done, info)
that every environments/*.py wrapper in this repo already uses.
"""
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
import warnings

import numpy as np


class VecEnv(ABC):
    """An abstract asynchronous, vectorized environment."""

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """Reset all the environments and return an array of observations."""

    @abstractmethod
    def step_async(self, actions):
        """Tell all the environments to start taking a step with the given actions."""

    @abstractmethod
    def step_wait(self):
        """Wait for the step taken with step_async(). Returns (obs, rews, dones, infos)."""

    @abstractmethod
    def close(self):
        """Clean up the environments' resources."""

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        warnings.warn("Render not defined for %s" % self)


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (multiprocessing's default pickle
    can't serialize local closures, which is exactly what env_fns are here)."""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset()
            remote.send(ob)
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """envs: list of gym environment constructors to run in subprocesses."""
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
