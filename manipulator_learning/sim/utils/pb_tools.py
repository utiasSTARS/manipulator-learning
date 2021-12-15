from __future__ import absolute_import
from __future__ import division
import functools
import inspect
import pybullet
import numpy as np
from manipulator_learning.sim.utils.general import TransformMat


def load_table(pb_client, urdf_root, mark_on_table=False, short=False):
  if mark_on_table:
    table = pb_client.loadURDF(urdf_root + "/table_with_mark/table.urdf",
                               [1.000000, -0.200000, 0.000000],
                               [0.000000, 0.000000, 0.707107, 0.707107])
  elif short:
    table = pb_client.loadURDF(urdf_root + "/table/table_short.urdf",
                               [1.000000, -0.200000, 0.000000],
                               [0.000000, 0.000000, 0.707107, 0.707107])
  else:
    table = pb_client.loadURDF(urdf_root + "/table/table.urdf",
                               [1.000000, -0.200000, 0.000000],
                               [0.000000, 0.000000, 0.707107, 0.707107])
  return table


def add_pb_frame_marker(pb_client, body_id, link_id=None, line_length=.1, line_width=3):
  if link_id is not None:
    pb_client.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0],
                               parentObjectUniqueId=body_id,
                               parentLinkIndex=link_id, lineWidth=line_width)
    pb_client.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0],
                               parentObjectUniqueId=body_id,
                               parentLinkIndex=link_id, lineWidth=line_width)
    pb_client.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1],
                               parentObjectUniqueId=body_id,
                               parentLinkIndex=link_id, lineWidth=line_width)
  else:
    pb_client.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0],
                               parentObjectUniqueId=body_id,
                               lineWidth=line_width)
    pb_client.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0],
                               parentObjectUniqueId=body_id,
                               lineWidth=line_width)
    pb_client.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1],
                               parentObjectUniqueId=body_id,
                               lineWidth=line_width)


def add_pb_frame_marker_by_pose(pb_client, pos, orient, ids=(), line_length=.1, line_width=3):
  # transform along each axis to get x, y, and z positions
  # Note: orient should be an xyzw quaternion
  pose_T = TransformMat(pb_pose=(pos, orient))
  x_axis_T = np.eye(4)
  x_axis_T[:3, 3] = [line_length, 0, 0]
  x_axis_pos = (pose_T.pose_mat @ x_axis_T)[:3, 3]
  y_axis_T = np.eye(4)
  y_axis_T[:3, 3] = [0, line_length, 0]
  y_axis_pos = (pose_T.pose_mat @ y_axis_T)[:3, 3]
  z_axis_T = np.eye(4)
  z_axis_T[:3, 3] = [0, 0, line_length]
  z_axis_pos = (pose_T.pose_mat @ z_axis_T)[:3, 3]
  if not ids:
    x_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], x_axis_pos, [1, 0, 0],
                                      lineWidth=line_width)
    y_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], y_axis_pos, [0, 1, 0],
                                      lineWidth=line_width)
    z_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], z_axis_pos, [0, 0, 1],
                                      lineWidth=line_width)
  else:
    x_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], x_axis_pos, [1, 0, 0],
                                      replaceItemUniqueId=ids[0], lineWidth=line_width)
    y_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], y_axis_pos, [0, 1, 0],
                                      replaceItemUniqueId=ids[1], lineWidth=line_width)
    z_id = pb_client.addUserDebugLine([pos[0], pos[1], pos[2]], z_axis_pos, [0, 0, 1],
                                      replaceItemUniqueId=ids[2], lineWidth=line_width)

  return x_id, y_id, z_id


class BulletClient(object):
  """A wrapper for pybullet to manage different clients.
  Borrowed from pybullet:
  https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/bullet_client.py
  """

  def __init__(self, connection_mode=None):
    """Creates a Bullet client and connects to a simulation.
    Args:
      connection_mode:
        `None` connects to an existing simulation or, if fails, creates a
          new headless simulation,
        `pybullet.GUI` creates a new simulation with a GUI,
        `pybullet.DIRECT` creates a headless simulation,
        `pybullet.SHARED_MEMORY` connects to an existing simulation.
    """
    self._shapes = {}

    if connection_mode is None:
      self._client = pybullet.connect(pybullet.SHARED_MEMORY)
      if self._client >= 0:
        return
      else:
        connection_mode = pybullet.DIRECT
    self._client = pybullet.connect(connection_mode)

  def __del__(self):
    """Clean up connection if not already done."""
    try:
      pybullet.disconnect(physicsClientId=self._client)
    except pybullet.error:
      pass

  def __getattr__(self, name):
    """Inject the client id into Bullet functions."""
    attribute = getattr(pybullet, name)
    if inspect.isbuiltin(attribute):
      if name not in [
          "invertTransform", "multiplyTransforms", "getMatrixFromQuaternion",
          "getEulerFromQuaternion", "computeViewMatrixFromYawPitchRoll",
          "computeProjectionMatrixFOV", "getQuaternionFromEuler",
      ]:  # A temporary hack for now.
        attribute = functools.partial(attribute, physicsClientId=self._client)
    return attribute