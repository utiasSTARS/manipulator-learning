# Attempting to view images from various cameras -- hardcoded values based on vr_kuka_setup
#
# Most values can be found by taking them from the actual GUI (shown at the bottom)

import pybullet as p
import numpy as np

import transforms3d as tf3d
from manipulator_learning.sim.utils.general import convert_quat_pb_to_tf


class EyeInHandCam():
    ''' Eye in hand camera in pybullet. In other words, a camera that maintains a constant transform from an
    existing frame. '''

    def __init__(self, pb_client, frame_rel_trans, frame_rel_rot, forward_axis, up_axis, renderer, render_shadows,
                 light_direction=None, focus_dist=2., aspect=1.333, fov=70, width=320, height=240,
                 far=5, near=.01):
        """
        Instantiate an EyeInHandCam.

        :param list frame_rel_trans: 3 float list, relative translation of camera eye from desired frame
        :param list frame_rel_rot: 4 float list, relative rotation (quaternion) of camera eye from desired frame (given in xyzw)
        :param list forward_axis: 3 int list, chooses the "forward" axis for the camera (given in frame of camera)
        :param list up_axis: 3 int list, chooses up axis for camera (given in frame of camera)
        :param string renderer: string containing 'opengl' to render with gpu or anything else for cpu
        :param bool render_shadows: render shadows or not
        :param list light_direction: 3 int/float list, light direction vector.
        :param float focus_dist: Focus distance
        :param float aspect: Aspect ratio
        :param float fov: Field of view
        :param int width: Width of cam image (pixels)
        :param int height: Height of cam image (pixels)
        """
        self._pb_client = pb_client
        self.fov = fov
        self.aspect = aspect
        self.width = width
        self.height = height
        self.forward_axis = forward_axis
        self.focus_dist = focus_dist
        self.frame_rel_tf = np.eye(4)
        self.frame_rel_tf[:3, :3] = tf3d.quaternions.quat2mat(convert_quat_pb_to_tf(frame_rel_rot))  # frame rel rot is 4 float list for quat
        self.frame_rel_tf[:3, 3] = frame_rel_trans  # frame_rel_trans is 3 float list for xyz
        self.far = far
        self.near = near
        self._latest_view_mat = None

        target_tf = np.eye(4)
        target_tf[:3, 3] = np.dot(np.array(forward_axis), focus_dist)
        self.frame_rel_target_tf = np.dot(self.frame_rel_tf, target_tf)

        self.cam_up_vec = np.array(up_axis)

        if renderer == 'opengl':
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            self.renderer = p.ER_TINY_RENDERER
        if render_shadows:
            self.render_shadows = 1
        else:
            self.render_shadows = 0

        self.light_direction = light_direction

    def get_img(self, frame_pose, width=None, height=None, depth_type='original', segment_mask=False):
        """ Get the camera image. frame_pose should be a 4x4 transform relative to the world, and
         is the frame that we want the image relative to.

         In most cases, this will be the wrist frame or the tool frame of the robot.

         depth_type options are 'original', 'fixed', 'true'. See note below on difference between
          'original' and 'fixed'. 'true' corresponds to actual values in m. """

        depth_options = ['original', 'fixed', 'true']

        if width is None:
            width = self.width
        if height is None:
            height = self.height
        aspect = width / height

        cam_pose = np.dot(frame_pose, self.frame_rel_tf)
        cam_target_pose = np.dot(frame_pose, self.frame_rel_target_tf)

        # The intuition for calculating this up_axis is that we are taking the particular
        # basis vector from the final cam pose corresponding to the desired up axis, which should be
        # a vector that points in the user-defined "up_axis" relative to the current camera pose
        up_axis = np.dot(cam_pose[:3, :3], self.cam_up_vec)

        # use these values for making a new cam for papers, etc
        # print(cam_pose[:3, 3], cam_target_pose[:3, 3])

        view_mat = self._pb_client.computeViewMatrix(cameraEyePosition=cam_pose[:3,3],
                                                     cameraTargetPosition=cam_target_pose[:3,3],
                                                     cameraUpVector=up_axis)
        self._latest_view_mat = view_mat
        project_mat = self._pb_client.computeProjectionMatrixFOV(fov=self.fov, aspect=aspect,
                                                                 nearVal=self.near, farVal=self.far)

        # start_render = time.time()
        if segment_mask:
            _, _, img, depth, segment = self._pb_client.getCameraImage(width=width, height=height, viewMatrix=view_mat,
                                                                       projectionMatrix=project_mat,
                                                                       renderer=self.renderer,
                                                                       shadow=self.render_shadows,
                                                                       lightDirection=self.light_direction,
                                                                       flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                                                                       )
        else:
            _, _, img, depth, _ = self._pb_client.getCameraImage(width=width, height=height, viewMatrix=view_mat,
                                                                 projectionMatrix=project_mat,
                                                                 renderer=self.renderer,
                                                                 shadow=self.render_shadows,
                                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                                 lightDirection=self.light_direction
                                                                 )

        # fix depth values to be linear between minimum and maximum value, from 0 to 1
        # true depth from depth = far * near / (far - (far - near) * depthImg) -- see pybullet docs
        # to then get values between 0 and 1, divide by far
        if depth_type == 'fixed':
            depth = self.near / (self.far - (self.far - self.near) * depth)
        elif depth_type == 'original':
            # this depth_type is deprecated, and should not be used
            # it was added in error, and now creates an ambiguity where any depth below .02m gives a mirrored value to
            # depths beyond .02, e.g. if depth out of pybullet corresponds to 3cm, this depth will be the same value if
            # depth out of pybullet corresponds to 1cm. everything beyond 4cm is for sure okay.
            # to be even clearer, any raw value of depth out of pybullet lower than .501 is definitely broken
            depth = self.near / (self.far - (self.far - self.near) * depth) / depth
        elif depth_type == 'true':
            depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        else:
            raise NotImplementedError("depth_type %s is not implemented, options are %s" % (depth_type, depth_options))

        # print("Cam img render: %f" % (time.time() - start_render))
        if segment_mask:
            return img, depth, segment
        else:
            return img, depth

    def get_latest_view_mat(self):
        """ Get a 4x4 numpy T view matrix. Should be called directly after calling get_img, and before moving the
        frame the camera is attached to again. """
        return self._latest_view_mat


class WorkspaceCam:
    def __init__(self, pb_client, width, height, aspect, eye, target, renderer,
                 render_shadows, light_direction=None, up_axis=(0, 0, 1)):
        self._pb_client = pb_client
        self.width = width
        self.height = height
        self.aspect = aspect
        self.eye = eye  # store so that other classes can see it
        self.target = target  # store so that other classes can see it
        self.view_mat = self._pb_client.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target,
                                                     cameraUpVector=up_axis)
        self.project_mat = self._pb_client.computeProjectionMatrixFOV(fov=70, aspect=self.aspect, nearVal=.1, farVal=5)
        if renderer == 'opengl':
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            self.renderer = p.ER_TINY_RENDERER
        if render_shadows:
            self.render_shadows = 1
        else:
            self.render_shadows = 0
        self.light_direction = light_direction

    @property
    def aspect(self):
        return self.__aspect

    @aspect.setter
    def aspect(self, ratio):
        self.project_mat = self.project_mat = self._pb_client.computeProjectionMatrixFOV(fov=70, aspect=ratio,
                                                                                         nearVal=.1, farVal=5)
        self.__aspect = ratio

    def get_img(self):
        _, _, img, depth, _ = self._pb_client.getCameraImage(width=self.width, height=self.height,
                                                             viewMatrix=self.view_mat,
                                                             projectionMatrix=self.project_mat,
                                                             renderer=self.renderer,
                                                             shadow=self.render_shadows,
                                                             flags=p.ER_NO_SEGMENTATION_MASK,
                                                             lightDirection=self.light_direction)

        return img, depth


# use the code below for testing what the view and projection matrices do to an image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cid = p.connect(p.SHARED_MEMORY)
    print("Physics server: ", cid)

    # workspace camera
    # view_mat = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.5, -.2, 0], distance=2.0,
    #                                yaw=-88, pitch=-53, roll=0, upAxisIndex=2)
    view_mat = p.computeViewMatrix(cameraEyePosition=[0, 0, 2.5], cameraTargetPosition=[1.5, -.2, 0], cameraUpVector=[0,1,1])
    project_mat = p.computeProjectionMatrixFOV(fov=70, aspect=1.333, nearVal=.1, farVal=100)



    width, height, img, depth_img, seg_img = p.getCameraImage(width=320, height=240, viewMatrix=view_mat, projectionMatrix=project_mat)
    plt.imshow(img)
    plt.show()
