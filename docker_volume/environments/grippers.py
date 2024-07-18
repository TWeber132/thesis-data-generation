import os
import time
import numpy as np
import pybullet as p

from environments.pybullet_utils import JointInfo


ROBOTIQ140_URDF = "robot/robotiq140.urdf"


class Robotiq140():
    def __init__(self, assets_root, env, robot_uid, flange_id):
        # Load gripper
        _, _, _, _, ee_pos, ee_rot = p.getLinkState(
            robot_uid, flange_id, computeForwardKinematics=True)
        self.assets_root = assets_root
        self.env = env
        self.gripper_urdf_path = ROBOTIQ140_URDF
        self.uid = p.loadURDF(os.path.join(self.assets_root, self.gripper_urdf_path), ee_pos, ee_rot,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        # Attach it
        p.createConstraint(parentBodyUniqueId=robot_uid,
                           parentLinkIndex=flange_id,
                           childBodyUniqueId=self.uid,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=(0, 0, 0),
                           parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0))

        self.closed = False
        self.leading_joint_range = [0.0, 0.7]
        self.homej = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        n_urdf_joints = p.getNumJoints(self.uid)
        _urdf_joints_info = [p.getJointInfo(
            self.uid, i) for i in range(n_urdf_joints)]
        self.joints_info = [JointInfo(j[0], j[1].decode("utf-8"), j[10], j[11])
                            for j in _urdf_joints_info if j[2] == p.JOINT_REVOLUTE]

        self.n_joints = len(self.joints_info)
        left_pad_joint_name = 'left_inner_finger_pad_joint'
        self.left_pad_joint_id = [
            joint[0] for joint in _urdf_joints_info if joint[1].decode("utf-8") == left_pad_joint_name][0]
        right_pad_joint_name = 'right_inner_finger_pad_joint'
        self.right_pad_joint_id = [
            joint[0] for joint in _urdf_joints_info if joint[1].decode("utf-8") == right_pad_joint_name][0]
        left_pad_link_name = 'left_inner_finger_pad'
        self.left_pad_link_id = []
        self.__post_load__()

    def __post_load__(self):
        # To control the gripper
        leading_joint_name = 'left_outer_knuckle_joint'

        following_joint_names_and_multiplier = {'right_outer_knuckle_joint': 1,
                                                'left_inner_knuckle_joint': 1,
                                                'right_inner_knuckle_joint': 1,
                                                'left_inner_finger_joint': -1,
                                                'right_inner_finger_joint': -1}

        self.__setup_following_joints__(
            leading_joint_name, following_joint_names_and_multiplier)

    def __setup_following_joints__(self, leading_joint_name, following_joint_names_and_multiplier):
        self.leading_joint_id = [
            joint.id for joint in self.joints_info if joint.name == leading_joint_name][0]

        self.following_joint_ids_and_multiplier = {
            joint.id: following_joint_names_and_multiplier[joint.name] for joint in self.joints_info if joint.name in following_joint_names_and_multiplier}

        for joint_id, multiplier in self.following_joint_ids_and_multiplier.items():
            c = p.createConstraint(parentBodyUniqueId=self.uid,
                                   parentLinkIndex=self.leading_joint_id,
                                   childBodyUniqueId=self.uid,
                                   childLinkIndex=joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            # Note: the mysterious `erp` is of EXTREME importance
            p.changeConstraint(c, gearRatio=multiplier, maxForce=100, erp=1)

        # Enable joint force-torque sensor in leading joint
        p.enableJointForceTorqueSensor(bodyUniqueId=self.uid,
                                       jointIndex=self.left_pad_joint_id,
                                       enableSensor=1)
        p.enableJointForceTorqueSensor(bodyUniqueId=self.uid,
                                       jointIndex=self.right_pad_joint_id,
                                       enableSensor=1)

    def move_m(self):
        # Moves leading joint until the measured y-moment exceeds a threshold
        targ_m = 0.05
        timeout = 2.0
        t0 = time.time()
        while (time.time() - t0) < timeout:
            # Control the mimic gripper joint(s)
            curr_ft_left = p.getJointState(
                bodyUniqueId=self.uid, jointIndex=self.left_pad_joint_id)[2]
            curr_ft_right = p.getJointState(
                bodyUniqueId=self.uid, jointIndex=self.right_pad_joint_id)[2]
            _, _, _, curr_mx_left, curr_my_left, curr_mz_left = curr_ft_left
            sum_curr_m_left = abs(curr_mx_left) + \
                abs(curr_my_left) + abs(curr_mz_left)
            _, _, _, curr_mx_right, curr_my_right, curr_mz_right = curr_ft_right
            sum_curr_m_right = abs(curr_mx_right) + \
                abs(curr_my_right) + abs(curr_mz_right)

            if (sum_curr_m_left > targ_m) or (sum_curr_m_right > targ_m):
                return False
            p.setJointMotorControl2(bodyUniqueId=self.uid,
                                    jointIndex=self.leading_joint_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=2.0,
                                    force=1.0)
            self.env.step_simulation()
        print(
            f'Warning: gripper movet exceeded {timeout} second timeout. Skipping.')
        return True

    def move_j(self, targ_j: float = 0.0):
        # Moves leading joint to target joint position
        timeout = 2.0
        t0 = time.time()
        while (time.time() - t0) < timeout:
            targ_j = np.clip(targ_j, *self.leading_joint_range)
            # Control the mimic gripper joint(s)
            curr_j = p.getJointState(
                bodyUniqueId=self.uid, jointIndex=self.leading_joint_id)[0]
            diff_j = targ_j - curr_j
            if abs(diff_j) < 1e-3:
                return False
            p.setJointMotorControl2(bodyUniqueId=self.uid,
                                    jointIndex=self.leading_joint_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targ_j,
                                    force=5,
                                    maxVelocity=10)
            self.env.step_simulation()
        print(
            f'Warning: gripper movej exceeded {timeout} second timeout. Skipping.')
        return True

    def close(self):
        self.closed = True
        return self.move_j(targ_j=self.leading_joint_range[1])

    def open(self):
        self.closed = False
        return self.move_j(targ_j=self.leading_joint_range[0])

    def grasp(self):
        self.closed = True
        return self.move_m()

    @property
    def something_grasped(self):
        "Check whether an object was grasped"
        contact_points_right = p.getContactPoints(
            bodyA=self.uid, linkIndexA=self.right_pad_joint_id)
        contact_points_left = p.getContactPoints(
            bodyA=self.uid, linkIndexA=self.left_pad_joint_id)

        if contact_points_right or contact_points_left:
            points = contact_points_left + contact_points_right
            for point in points:
                obj_uid = point[2]
                # Gripper grasped something but not itself
                if obj_uid != self.uid:
                    return True
        # Gripper grasped nothing or itself
        return False

    def obj_grasped(self, obj_uid):
        contact_points_right = p.getContactPoints(
            bodyA=self.uid, linkIndexA=self.right_pad_joint_id, bodyB=obj_uid, linkIndexB=-1)
        contact_points_left = p.getContactPoints(
            bodyA=self.uid, linkIndexA=self.left_pad_joint_id, bodyB=obj_uid, linkIndexB=-1)

        if contact_points_right or contact_points_left:
            return True
        return False

    def reset(self):
        for i in range(self.n_joints):
            p.resetJointState(self.uid, self.joints_info[i].id, self.homej[i])
