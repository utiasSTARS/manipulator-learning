# Various simple utility functions
import transforms3d as tf3d
import numpy as np


class TransformMat:
    def __init__(self, mat=None, pb_pose=None):
        """
        Create a 4x4 transformation matrix object from a pybullet pose or a 4x4 matrix.
        :param mat: A 4x4 matrix as a numpy array or a list.
        :param pb_pose: Should be a tuple with first element being 3-tuple for position, second element being
        4-tuple of xyzw quaternion.
        """
        if pb_pose is not None:
            self.pose_mat = np.eye(4)
            self.pose_mat[:3, :3] = tf3d.quaternions.quat2mat(
                convert_quat_pb_to_tf(pb_pose[1]))
            self.pose_mat[:3, 3] = pb_pose[0]

        if mat is not None:
            self.pose_mat = mat

    def __call__(self):
        return self.pose_mat

    def __str__(self):
        return np.array2string(self.pose_mat)

    def to_pb(self, single_tuple=False):
        """ Return the equivalent pybullet pose of this T matrix. """
        pos = tuple(self.pose_mat[:3, 3])
        rot = tuple(convert_quat_tf_to_pb(tf3d.quaternions.mat2quat(self.pose_mat[:3, :3])))
        if single_tuple:
            return (*pos, *rot)
        else:
            return pos, rot


def q_convert(q, instr='xyzw', outstr='wxyz'):
  inorder = {instr[0]: 0, instr[1]: 1, instr[2]: 2, instr[3]: 3}
  return np.array([q[inorder[outstr[i]]] for i in range(4)])


def convert_quat_tf_to_pb(quat):
    ''' Convert from transformations.py quat (w,x,y,z) to pybullet quat (x,y,z,w) '''
    return [quat[1], quat[2], quat[3], quat[0]]


def convert_quat_pb_to_tf(quat):
    ''' Convert from pybullet quat (x,y,z,w) to transformation.py quat (w,x,y,z) '''
    return [quat[3], quat[0], quat[1], quat[2]]


def trans_quat_to_mat(trans, quat):
    ''' Convert from trans and pybullet quat (xyzw) to 4x4 transformation matrix '''
    mat = np.eye(4)
    mat[:3, :3] = tf3d.quaternions.quat2mat(convert_quat_pb_to_tf(quat))
    mat[:3, 3] = trans
    return mat


def invert_transform(T_in):
    """return inverse transform of T_in, assuming T_in is affine 4x4 transformation"""
    T_out = np.eye(4)
    C_out_inv = T_in[:3, :3].T # exploiting that C^T = C^-1 for rotations
    T_out[:3, :3] = C_out_inv
    T_out[:3, 3] = -C_out_inv.dot(T_in[:3, 3])
    return T_out


def change_pose_ref_frame(pose_world, ref_pose_world, ref_pose_inverted=False):
    """ Transform pose_world to be in the frame of ref_pose_world, assuming both are [3-dof, 4-dof] pos + xyzw quats."""
    T_w_to_rel = trans_quat_to_mat(ref_pose_world[0], ref_pose_world[1])
    if ref_pose_inverted:
        T_rel_to_w = T_w_to_rel
    else:
        T_rel_to_w = invert_transform(T_w_to_rel)
    T_world_obj = trans_quat_to_mat(*pose_world)
    T_rel_to_obj = T_rel_to_w.dot(T_world_obj)
    pose = TransformMat(T_rel_to_obj).to_pb()
    return pose


def change_vel_ref_frame(vel_world, ref_pose_world):
    """ Transform vel_world to be in the frame of ref_pose_world, assuming vel_world is [3-dof, 3-dof] and
    ref_pose_world is 7-dof pos + xyzw quat. """
    T_w_to_rel = trans_quat_to_mat(ref_pose_world[:3], ref_pose_world[3:])
    vel_rel = np.concatenate([T_w_to_rel[:3, :3].dot(vel_world[0]), T_w_to_rel[:3, :3].dot(vel_world[1])])
    return vel_rel


def convert_pose_to_3_pts(pose, dist=0.1, axes='xy'):
    """
    Convert a pose from xyz position and xyzw quaternion to set of 3 pts, the idea being that,
    for a neural network, 3 (non-colinear) points can encode both position and orientation.
    :param pose: Pose as list or tuple (pos, xyzw quat).
    :param dist: Distance from original pose to other two points.
    :param axes: Axes of pose orientation to transform other 2 points along. Must be a two letter
        combination of x, y, and z, in any order.
    :return: A 3x3 np array of each point, with each row containing a single point. The first point
        is the original point from the 'pose' argument.
    """
    assert len(axes) == 2, "axes must be xy, yx, xz, zx, yz, or zy"
    points = []
    points.append(pose[0])
    quat_wxyz = convert_quat_pb_to_tf(pose[1])
    pose_T = np.eye(4)
    pose_T[:3, :3] = tf3d.quaternions.quat2mat(quat_wxyz)
    pose_T[:3, 3] = pose[0]
    axes = axes.replace('x', '0')
    axes = axes.replace('y', '1')
    axes = axes.replace('z', '2')
    r2 = np.zeros(3,)  # for transforming to 2nd pt
    r3 = np.zeros(3,)  # for transforming to 3rd pt
    r2[int(axes[0])] = dist
    r3[int(axes[1])] = dist
    T2 = np.eye(4)
    T3 = np.eye(4)
    T2[:3, 3] = r2
    T3[:3, 3] = r3
    pose_2_T = np.dot(pose_T, T2)
    pose_3_T = np.dot(pose_T, T3)
    points.append(pose_2_T[:3, 3])
    points.append(pose_3_T[:3, 3])

    return np.array(points)
