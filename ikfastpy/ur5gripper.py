#########################
#  UR5 Gripper class
########################

# Still need gripper control

import time
import os
import numpy as np
import numpy.linalg as la
import math
import ikfastpy
from robot_util import *


class Robot(object):
    def __init__(self, MjSim, MjViewer=None, Render = False):
        armActuatorNames = ['shoulder_pan_T', 'shoulder_lift_T', 'forearm_T', 'wrist_1_T', 'wrist_2_T', 'wrist_3_T']
        gripperActuatorNames = ['gripper']
        armJointNames = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        gripperJointNames = ['l_f', 'r_f']
        cameraNames = ['c1']

        self.sim = MjSim
        self.viewer = MjViewer
        self.RENDER = Render
        self.sim_init_state = MjSim.get_state()  # get sim initial state, later used for reset sim

        # extracting qpos id for joints
        self.armJointQposID = []
        for name in armJointNames:
            self.armJointQposID.append(MjSim.model._joint_name2id[name])             
        self.armJointAngles = np.array([0., 0., 0., 0., 0., 0.])                                     # in radians
        self.ur5_kin = ikfastpy.PyKinematics()                                       # ur5 kinematics class
        self.numArmJoints = self.ur5_kin.getDOF()                                     
        self.ee_pose = np.asarray(self.ur5_kin.forward(self.armJointAngles)).reshape(3,4)  # 3x4 rigid transformation matrix

        # extracting qpos id for gripper
        self.gripperJointQposID = []
        for name in gripperJointNames:
            self.gripperJointQposID.append(MjSim.model._joint_name2id[name]) 
        self.gripperJointAngles = np.array([0., 0.])

        # extracting ctrl id for arm joints
        self.armCtrlID=[]
        for name in armActuatorNames:
            self.armCtrlID.append(MjSim.model._actuator_name2id[name])

        # extracting ctrl id for gripper joints
        self.gripperCtrlID = MjSim.model._actuator_name2id[gripperActuatorNames[0]]


        self.currentToolPosition = None
        self.currentToolQuaternion = None



    def getArmJointAngles(self):
        return self.sim.data.qpos[self.armJointQposID]

    def getGripperJointAngles(self):
        return self.sim.data.qpos[self.gripperJointQposID]

    def setJointAngles(self, armJointAngles, gripperJointAngles):
        self.sim.data.qpos[self.armJointQposID] = armJointAngles
        self.sim.data.qpos[self.gripperJointQposID] = gripperJointAngles


    def getClosestJointConfig(self, joint_configs, currentJointAngles, n_solutions):
        # find which solution is closest to current state in joint space
        for joint_config in joint_configs:
            print(joint_config)
        # for each joint_config in joint_configs, compare the euclidian distance between it and current joint angles
        jointConfigs_Remainder = np.mod(joint_configs + 2*np.pi, 2*np.pi)
        currentJoint_Remainder = np.mod(currentJointAngles + 2*np.pi, 2*np.pi)
        print('Remainder of joint configs\n')
        for joint_config in jointConfigs_Remainder:
            print(joint_config)
        idx = np.linalg.norm(np.subtract(jointConfigs_Remainder, currentJoint_Remainder), axis='1').argmin()
        return idx

    def getElbowUpJointConfig(self, joint_configs, n_solutions):
        # find the elbow up configuration for ur5
        # the second joint limit between (-3.1416, 0), elbow up should be the one closest to -1.57
        secondJointSolutions = joint_configs[:, 1]
        idx = np.absolute(secondJointSolutions + 1.57).argmin()
        return idx

    def IKsolver(self, ee_pose):
        # ee_pose is 3x4 ndarray
        joint_configs = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())    # python wrapper for ikfast solver
        n_solutions = int(len(joint_configs)/self.numArmJoints)
        print("%d solutions found:"%(n_solutions))

        joint_configs = np.asarray(joint_configs).reshape(n_solutions,self.numArmJoints)
        currentArmJointAngles = self.getArmJointAngles()
        # idx = self.getClosestJointConfig(joint_configs, currentArmJointAngles, n_solutions)   # find the closest joint_config among all solutions
        idx = self.getElbowUpJointConfig(joint_configs, n_solutions)   # find the elbow up configuration
        print('Closest solution is number {}\n'.format(idx))

        goal_position = joint_configs[idx]
        return goal_position

    def convert2EEpose(self, toolPosition, toolQuaternion):
        # convert tool position and quaternion to 3x4 ndarray ee_pose
        R = quaternion2Rotation(toolQuaternion)
        ee_pose = np.zeros((3,4))
        ee_pose[:,0:3] = R
        ee_pose[0,3] = toolPosition[0]
        ee_pose[1,3] = toolPosition[1]
        ee_pose[2,3] = toolPosition[2]
        return ee_pose

    def convertFromEEpose(self, ee_pose):
        # convert ee_pose into toolPosition and toolQuaternion
        R = ee_pose[:, 0:3]
        toolQuaternion = rotation2Quaternion(R)
        toolPosition = np.zeros(3)
        toolPosition[0] = ee_pose[0,3]
        toolPosition[1] = ee_pose[1,3]
        toolPosition[2] = ee_pose[2,3]
        return toolPosition, toolQuaternion


    def move_to(self, toolPosition, toolQuaternion, numStep = 200):
        # orientation format is quaternion qw,qx,qy,qz
        ee_pose = self.convert2EEpose(toolPosition, toolQuaternion)
        goalJointConfig = self.IKsolver(ee_pose)        
        currentArmJointAngles = self.getArmJointAngles()
        currentGripperJointAngles = self.getGripperJointAngles()
        step = np.subtract(goalJointConfig, currentArmJointAngles) / numStep

        for i in range(numStep):
            self.setJointAngles(np.add(currentArmJointAngles, step), currentGripperJointAngles)
            currentArmJointAngles = self.getArmJointAngles()
            self.simulation_step()
        
        for i in range(20):                # just in case the robot failed to follow the trajectory, force it to reach taraget pose 
            self.setJointAngles(goalJointConfig, currentGripperJointAngles)
            self.simulation_step()
        # get EE pose after moving gripper
        ee_pose = self.get_EE_pos()
        self.currentToolPosition, self.currentToolQuaternion = self.convertFromEEpose(ee_pose)

        # step sim after setting up the joint angles is probably not optimal since joints will move a little after one step  
        # but 5e-3 should be able to tolerate this error
        if la.norm(self.currentToolPosition - toolPosition) < 5e-3:
            print('Moving toolPose succedd!\n')
            return True
        else:
            print(toolPosition, self.currentToolPosition)
            print('Moving failed to reach goal state\n')
            return False

    def move_while_grasping(self, toolPosition, toolQuaternion, numStep = 1000, torque = 0.1):
        # orientation format is quaternion qw,qx,qy,qz
        ee_pose = self.convert2EEpose(toolPosition, toolQuaternion)
        goalJointConfig = self.IKsolver(ee_pose)        
        currentArmJointAngles = self.getArmJointAngles()
        step = np.subtract(goalJointConfig, currentArmJointAngles) / numStep
        # Move EE while grasping
        for i in range(numStep):
            self.sim.data.qpos[self.armJointQposID] = np.add(currentArmJointAngles, step)
            currentArmJointAngles = self.getArmJointAngles()
            self.sim.data.ctrl[self.gripperCtrlID] = torque
            self.simulation_step()
        # get EE pose after moving gripper
        ee_pose = self.get_EE_pos()
        self.currentToolPosition, self.currentToolQuaternion = self.convertFromEEpose(ee_pose)

        # step sim after setting up the joint angles is probably not optimal since joints will move a little after one step  
        # but 5e-3 should be able to tolerate this error
        if la.norm(self.currentToolPosition - toolPosition) < 5e-3:
            print('Moving toolPose succedd!\n')
            return True
        else:
            print(toolPosition, self.currentToolPosition)
            print('Moving failed to reach goal state\n')
            return False        

    def get_EE_pos(self):
        currentArmJointAngles = self.getArmJointAngles()
        return np.array(self.ur5_kin.forward(currentArmJointAngles)).reshape(3,4)

    def rest_simulation(self):
        self.sim.set_state(self.sim_init_state)
        return True

    def close_gripper(self, torque = 1):
        currentArmJointAngles = self.getArmJointAngles()
        while True:
            currentGripperJointAngles = self.getGripperJointAngles()
            self.sim.data.qpos[self.armJointQposID] = currentArmJointAngles
            self.sim.data.ctrl[self.gripperCtrlID] = torque
            self.simulation_step()

            # check if gripper is fully closed
            if la.norm(currentGripperJointAngles - self.getGripperJointAngles()) < 1e-7:
                print('Gripper successfully closed.\n')
                return True


    def open_gripper(self, torque = -1):
        currentArmJointAngles = self.getArmJointAngles()
        torque = -abs(torque)
        while True:
            currentGripperJointAngles = self.getGripperJointAngles()
            self.sim.data.qpos[self.armJointQposID] = currentArmJointAngles
            self.sim.data.ctrl[self.gripperCtrlID] = torque
            self.simulation_step()

            # check if gripper is fully opened
            if la.norm(self.getGripperJointAngles()) < 5e-2:
                for i in range(10):
                    self.setJointAngles(currentArmJointAngles, np.array([0,0]))
                    self.simulation_step()
                print('Gripper successfully opened.\n')
                break



    def robot_freeze(self, numStep = 1000):
        # freeze the robot for one step, helpful during debug
        currentArmJointAngles = self.getArmJointAngles()
        currentGripperJointAngles = self.getGripperJointAngles()
        for i in range(numStep):
            self.setJointAngles(currentArmJointAngles, currentGripperJointAngles)
            self.simulation_step()

    def simulation_step(self):
        self.sim.step()
        if self.RENDER:
            self.viewer.render()


