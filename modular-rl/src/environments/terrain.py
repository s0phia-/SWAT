"""Terrain variants for MuJoCo ant-style agents: flat, variable (rough),
obstacles, or incline (constant-grade uphill slope).

Terrain is added as loose <geom>/<asset> elements written to a *new* temporary
copy of the agent's XML - the original file on disk is never touched. Since
these are bare geoms (not wrapped in their own <body>), they stay invisible to
utils.graph_utils.getGraphStructure (which only walks the single <body> nested
under <worldbody>) and to the per-limb observation extractors (which only read
named torso/limb bodies) - so this works unchanged for any agent XML,
including the whole unimals_100 dataset, without editing any of those files,
and without the policy perceiving the terrain.

Both terrain types are sized for long, roughly-straight-line running tasks
(see `direction_deg`) rather than small pens the agent quickly runs out of:
  - 'obstacles' uses a small, fixed pool of geoms recycled from behind the
    agent to ahead of it (see `recycle_obstacles`), so the field is
    effectively unbounded regardless of episode length.
  - 'variable' uses one large (but finite) tileable heightfield - MuJoCo has
    no native streaming/scrolling heightfield, so "unbounded" here means
    "comfortably larger than any realistic episode's travel distance", not
    literally infinite.
"""
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from os import path
from typing import List, Optional, Tuple

import numpy as np

TERRAIN_TYPES = ("flat", "variable", "obstacles", "incline")
OBSTACLE_PREFIX = "obstacle_"


def _direction_vectors(direction_deg: float) -> Tuple[np.ndarray, np.ndarray]:
  """Returns (forward, leftward) unit vectors for a running direction."""
  rad = math.radians(direction_deg)
  fwd = np.array([math.cos(rad), math.sin(rad)])
  perp = np.array([-fwd[1], fwd[0]])
  return fwd, perp


def _resolve_includes(tree, xml_path):
  """Rewrites every <include file="..."> to an absolute path, resolved
  relative to xml_path's own directory (MuJoCo's own resolution rule for a
  relative include path).

  ElementTree doesn't understand MuJoCo's <include> directive -- it's a
  MuJoCo-compiler-only mechanism, not standard XML entity inclusion, so
  ET.parse copies it verbatim as an unexpanded element with its `file`
  attribute untouched. Left relative, that attribute would still be resolved
  against wherever the injected copy ends up living (the system tempdir),
  not the original xml's directory, breaking the reference. Rewriting to an
  absolute path here means the copy can be written anywhere.
  """
  base_dir = path.dirname(path.abspath(xml_path))
  for include in tree.getroot().iter("include"):
    file_attr = include.get("file")
    if file_attr and not path.isabs(file_attr):
      include.set("file", path.abspath(path.join(base_dir, file_attr)))


def inject_terrain(
    xml_path: str,
    terrain: str = "flat",
    seed: Optional[int] = None,
    direction_deg: float = 0.0,
    num_obstacles: int = 100,
    obstacle_size_range: Tuple[float, float] = (0.2, 0.6),
    obstacle_forward_range: Tuple[float, float] = (5.0, 100.0),
    obstacle_lateral_range: float = 12.0,
    obstacle_floor_extent: float = 200.0,
    hfield_radius: float = 150.0,
    hfield_max_height: float = 0.5,
    hfield_resolution: int = 400,
    hfield_smoothing: int = 4,
    incline_deg: float = 10.0,
    incline_floor_extent: float = 200.0,
) -> Tuple[str, Optional[np.ndarray]]:
  """Returns (xml_path, hfield_heights) with the requested terrain added.

  For 'flat', returns `(xml_path, None)` unchanged. For 'variable' or
  'obstacles', parses `xml_path`, adds terrain geoms to a copy of its
  <worldbody>, and writes the result to a new temp file, whose path is
  returned instead.

  `direction_deg` is the agent's intended running direction (degrees, 0 =
  +x); obstacles are laid out ahead of the origin along this heading rather
  than scattered symmetrically, since a running task only ever needs terrain
  in front of the agent.

  `hfield_heights` is only non-None for terrain='variable': MuJoCo's XML can
  declare an <hfield>'s resolution but not its elevation data, so the caller
  must assign it to `model.hfield_data` after the MjModel is loaded from the
  returned xml path.

  `incline_deg` (terrain='incline' only) is the slope's grade in degrees; the
  floor plane is tilted about the origin so that +x is uphill, matching the
  forward-progress reward convention used throughout this repo. Rotating
  about the origin (rather than the slope's own surface) keeps z=0 at x=0, so
  an agent spawned at its usual flat-terrain height rests correctly on the
  slope with no special initial-position handling.

  The agent's own kinematic tree is untouched, so code that treats `xml_path`
  as the source of morphology (e.g. utils.graph_utils.getGraphStructure)
  should keep using the original path, not the one returned here.
  """
  assert terrain in TERRAIN_TYPES, (
      f"terrain must be one of {TERRAIN_TYPES}, got {terrain!r}")
  if terrain == "flat":
    return xml_path, None

  rng = np.random.default_rng(seed)
  tree = ET.parse(xml_path)
  _resolve_includes(tree, xml_path)
  worldbody = tree.getroot().find("worldbody")

  heights = None
  if terrain == "obstacles":
    _add_obstacles(worldbody, rng, direction_deg, num_obstacles,
                    obstacle_size_range, obstacle_forward_range,
                    obstacle_lateral_range, obstacle_floor_extent)
  elif terrain == "variable":
    heights = _add_heightfield(tree, worldbody, rng, hfield_radius,
                               hfield_max_height, hfield_resolution,
                               hfield_smoothing)
  elif terrain == "incline":
    _add_incline(worldbody, incline_deg, incline_floor_extent)

  tmp_path = path.join(
      tempfile.gettempdir(), f"ant_terrain_{terrain}_{time.time()}.xml")
  tree.write(tmp_path)
  return tmp_path, heights


def _add_obstacles(worldbody, rng, direction_deg, num_obstacles, size_range,
                   forward_range, lateral_range, floor_extent):
  """Seeds a fixed pool of boxes ahead of the origin, along direction_deg.

  This pool is meant to be recycled at runtime via `recycle_obstacles` - the
  initial layout just needs to fill the agent's starting view.
  """
  # The agent's own floor plane is usually sized for its native (short-range)
  # task; MuJoCo planes collide infinitely regardless of `size`, but the
  # *visual* extent is finite, so a long run would appear to run off the edge
  # of a textured floor into plain gray. Stretch it to match.
  floor = worldbody.find("geom[@name='floor']")
  if floor is not None:
    sx, sy, sz = (float(v) for v in floor.get("size").split())
    floor.set("size", f"{max(sx, floor_extent)} {max(sy, floor_extent)} {sz}")

  fwd, perp = _direction_vectors(direction_deg)
  min_size, max_size = size_range
  for i in range(num_obstacles):
    forward = rng.uniform(*forward_range)
    lateral = rng.uniform(-lateral_range, lateral_range)
    x, y = forward * fwd + lateral * perp
    hx, hy, hz = rng.uniform(min_size, max_size, size=3)
    ET.SubElement(
        worldbody, "geom",
        name=f"{OBSTACLE_PREFIX}{i}",
        pos=f"{x} {y} {hz}",
        size=f"{hx} {hy} {hz}",
        type="box",
        contype="1",
        conaffinity="1",
        rgba="0.85 0.6 0.3 1")


def _add_incline(worldbody, incline_deg, floor_extent):
  """Tilts the floor plane into a constant-grade uphill slope along +x.

  A plane geom's local +x axis, expressed in world coordinates, gains a
  positive z-component (i.e. becomes uphill as +x increases) when rotated by
  a *negative* angle about the Y axis -- empirically verified against
  MuJoCo's actual euler convention, not just derived on paper.
  """
  floor = worldbody.find("geom[@name='floor']")
  if floor is not None:
    sx, sy, sz = (float(v) for v in floor.get("size").split())
    floor.set("size", f"{max(sx, floor_extent)} {max(sy, floor_extent)} {sz}")
    floor.set("euler", f"0 {-incline_deg} 0")


def get_obstacle_geom_ids(model) -> List[int]:
  """IDs of the recyclable obstacle geoms, for use with `recycle_obstacles`."""
  return [i for i in range(model.ngeom)
          if model.geom(i).name.startswith(OBSTACLE_PREFIX)]


def recycle_obstacles(model, agent_xy: np.ndarray, direction_deg: float,
                      rng, obstacle_geom_ids: List[int],
                      forward_range: Tuple[float, float] = (40.0, 100.0),
                      lateral_range: float = 12.0,
                      behind_margin: float = 5.0) -> None:
  """Teleports any obstacle that has fallen behind the agent to a new spot
  ahead of it, so a small fixed geom pool gives an effectively unbounded
  obstacle field regardless of episode length.

  Mutates `model.geom_pos` in place; call once per env step. `agent_xy` is
  the agent's current (x, y) world position.
  """
  fwd, perp = _direction_vectors(direction_deg)
  agent_xy = np.asarray(agent_xy)
  for geom_id in obstacle_geom_ids:
    rel = model.geom_pos[geom_id, :2] - agent_xy
    forward_dist = rel @ fwd
    if forward_dist < -behind_margin:
      forward = rng.uniform(*forward_range)
      lateral = rng.uniform(-lateral_range, lateral_range)
      new_xy = agent_xy + forward * fwd + lateral * perp
      model.geom_pos[geom_id, 0] = new_xy[0]
      model.geom_pos[geom_id, 1] = new_xy[1]


def _generate_heights(resolution: int, smoothing: int, rng) -> np.ndarray:
  """Smoothed random heightfield in [0, 1], via repeated box-blur of noise."""
  heights = rng.uniform(0., 1., size=(resolution, resolution))
  for _ in range(smoothing):
    # 'wrap' (not 'edge') padding: edge-clamped padding biases repeated
    # blurring toward a spurious large-scale gradient/hill across the grid,
    # and 'wrap' additionally makes the result seamlessly tileable.
    padded = np.pad(heights, 1, mode="wrap")
    heights = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.
  heights -= heights.min()
  peak = heights.max()
  if peak > 0:
    heights /= peak
  return heights


def _add_heightfield(tree, worldbody, rng, radius, max_height, resolution,
                     smoothing) -> np.ndarray:
  """Replaces the flat floor plane with a large randomized rough-terrain
  hfield, sized to outlast any realistic episode's travel distance.

  Returns the [0, 1]-normalized height grid; the caller must assign it to
  `model.hfield_data` once the MjModel has been compiled from this XML.
  """
  floor = worldbody.find("geom[@name='floor']")
  if floor is not None:
    worldbody.remove(floor)

  root = tree.getroot()
  asset = root.find("asset")
  if asset is None:
    asset = ET.SubElement(root, "asset")
  # Define our own checker texture/material (rather than assuming the agent's
  # own XML happens to define one, e.g. "MatPlane") so elevation is visible
  # under any agent XML, including the unimals_100 set.
  ET.SubElement(
      asset, "texture",
      name="terrain_checker_tex",
      type="2d", builtin="checker",
      rgb1="0 0 0", rgb2="0.8 0.8 0.8", width="100", height="100")
  ET.SubElement(
      asset, "material",
      name="terrain_checker_mat",
      texture="terrain_checker_tex", texrepeat=f"{resolution} {resolution}",
      reflectance="0.2")
  ET.SubElement(
      asset, "hfield",
      name="terrain",
      nrow=str(resolution),
      ncol=str(resolution),
      size=f"{radius} {radius} {max_height} 0.1")
  ET.SubElement(
      worldbody, "geom",
      name="terrain",
      type="hfield",
      hfield="terrain",
      pos="0 0 0",
      contype="1",
      conaffinity="1",
      material="terrain_checker_mat")
  return _generate_heights(resolution, smoothing, rng)
