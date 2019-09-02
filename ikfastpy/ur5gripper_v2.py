#########################
#  UR5 Gripper class
########################

# Since it is required to specify control parameters (data.ctrl and data.qfrc_applied) for all joints at every step, 
# it is hard to implement features like "close gripper" (without taking care of the parameters for arm).
# So basically I implemented 6 different control modes here: 
# 1) move arm while keeping gripper at home position,                                 
#       self.move_to()
# 2) move arm while closing gripper with user-specified force,                        
#       self.move_while_grasping()
# 3) freeze arm while closing gripper with user-specified force,                      
#       self.close_gripper()
# 4) freeze arm while opening gripper to home position,                               
#       self.open_gripper()         
# 5) freeze both arm and gripper for user-defined time steps,                         
#       self.robot_freeze()
# 6) freeze arm and gripper while gripper is loaded with user-specified force, for user-defined time steps,            
#       self.robot_freeze_while_grasping()

import time
import os
import numpy as np
import numpy.linalg as la
import math
import ikfastpy
from robot_util import *
from timeScaling import *
from drawnow import *
import datetime as dt

q1 = []
v1 = []
s1 = []
#pre-load dummy data
for i in range(0,1):
    q1.append(0)
    v1.append(0)
    s1.append(0)
plt.ion()
cnt=0

def plotQ1():
    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.ylabel('q1 Values')
    plt.plot(np.array(range(len(q1))), q1, 'r-', np.array(range(len(v1))), v1, 'b-')
    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.ylabel('s1 Values')
    plt.plot(np.array(range(len(s1))), s1, 'r-')


class Robot(object):
    def __init__(self, MjSim, MjViewer=None, Render = False):
        # When initializing, extract qpos id and ctrl id 
        # qpos id for check joint values and velocities
        # ctrl id for assigning commands to position/velocity actuators

        armPositionActuatorNames = ['shoulder_pan', 'shoulder_lift', 'forearm', 'wrist_1', 'wrist_2', 'wrist_3']              # arm position actuators
        armVelocityActuatorNames = ['shoulder_pan_v', 'shoulder_lift_v', 'forearm_v', 'wrist_1_v', 'wrist_2_v', 'wrist_3_v']  # arm velocity actuators
        gripperPositionActuatorNames = ['gripper']                                                                             # gripper position actuators
        gripperVelocityActuatorNames = ['gripper_v']                                                                           # gripper velocity actuators
        armJointNames = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        gripperJointNames = ['l_f', 'r_f']
        cameraNames = ['c1']
        gripperMainTouchNames = ['lf_main', 'rf_main']  # main touch sensors are used to check whether or not gripper is fully closed

        self.sim = MjSim
        self.viewer = MjViewer
        self.RENDER = Render                     # bool to determine whether or not visualize simulation
        self.sim_init_state = MjSim.get_state()  # get sim initial state, later used for reset sim

        # extracting qpos id for joints
        self.armJointQposID = []
        for name in armJointNames:
            self.armJointQposID.append(MjSim.model._joint_name2id[name])             
        self.armJointAngles = np.array([0., 0., 0., 0., 0., 0.])                                     # in radians

        # extracting qpos id for gripper
        self.gripperJointQposID = []
        for name in gripperJointNames:
            self.gripperJointQposID.append(MjSim.model._joint_name2id[name]) 
        self.gripperJointAngles = np.array([0., 0.])

        # extracting ctrl id for arm position actuators
        self.armPositionCtrlID=[]
        for name in armPositionActuatorNames:
            self.armPositionCtrlID.append(MjSim.model._actuator_name2id[name])

        # extracting ctrl id for arm velocity actuators
        self.armVelocityCtrlID=[]
        for name in armVelocityActuatorNames:
            self.armVelocityCtrlID.append(MjSim.model._actuator_name2id[name])

        # extracting ctrl id for gripper actuator
        self.gripperPositionCtrlID = MjSim.model._actuator_name2id[gripperPositionActuatorNames[0]]
        self.gripperVelocityCtrlID = MjSim.model._actuator_name2id[gripperVelocityActuatorNames[0]]

        # extracting gripper main touch sensor id
        self.gripperMainTouchID=[]
        for name in gripperMainTouchNames:
            self.gripperMainTouchID.append(MjSim.model._sensor_name2id[name])

        self.ur5_kin = ikfastpy.PyKinematics()                                       # ur5 kinematics class
        self.numArmJoints = self.ur5_kin.getDOF()                                     
        self.ee_pose = np.asarray(self.ur5_kin.forward(self.armJointAngles)).reshape(3,4)  # 3x4 rigid transformation matrix

        self.currentToolPosition = None
        self.currentToolQuaternion = None

        self.timeStep = MjSim.model.opt.timestep   # simulation timestep


    def getArmJointAngles(self):
        return self.sim.data.qpos[self.armJointQposID]

    def getArmJointVelocities(self):
        return self.sim.data.qvel[self.armJointQposID]

    def getGripperJointAngles(self):
        return self.sim.data.qpos[self.gripperJointQposID]

    def getGripperJointVelocities(self):
        return self.sim.data.qvel[self.gripperJointQposID]

    def getClosestJointConfig(self, joint_configs, currentJointAngles, n_solutions):
        # find which solution is closest to current state in joint space
        # for joint_config in joint_configs:
        #     print(joint_config)
        # for each joint_config in joint_configs, compare the euclidian distance between it and current joint angles
        jointConfigs_Remainder = np.mod(joint_configs + 2*np.pi, 2*np.pi)
        currentJoint_Remainder = np.mod(currentJointAngles + 2*np.pi, 2*np.pi)
        # print('Remainder of joint configs\n')
        # for joint_config in jointConfigs_Remainder:
        #     print(joint_config)
        idx = np.linalg.norm(np.subtract(jointConfigs_Remainder, currentJoint_Remainder), axis='1').argmin()
        return idx

    def getElbowUpJointConfig(self, joint_configs, n_solutions):
        # find the elbow up configuration for ur5
        # the second joint limit between (-3.1416, 0), elbow up should be the one closest to -1.57
        secondJointValues = joint_configs[:, 1]
        idx = np.absolute(secondJointValues + 1.57).argmin()
        return idx

    def IKsolver(self, ee_pose):
        print('IK solver calculating solutions for target end effector pose...\n')
        startTime = dt.datetime.now()      
        # ee_pose is 3x4 ndarray
        joint_configs = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())    # python wrapper for ikfast solver
        n_solutions = int(len(joint_configs)/self.numArmJoints)
        print("%d solutions found:"%(n_solutions))

        joint_configs = np.asarray(joint_configs).reshape(n_solutions,self.numArmJoints)
        currentArmJointAngles = self.getArmJointAngles()
        idx = self.getElbowUpJointConfig(joint_configs, n_solutions)   # find the elbow up configuration
        print('Optimal solution is number {}'.format(idx))

        endTime = dt.datetime.now()
        timeDuration = (endTime.microsecond - startTime.microsecond) * 1e-3 + (endTime.second - startTime.second)*1e3
        print('Time it took for IK solver is: {}ms\n'.format(timeDuration))
        print('===========================================================================')

        goal_joint_config = joint_configs[idx]
        return goal_joint_config

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

    def get_EE_pos(self):
        currentArmJointAngles = self.getArmJointAngles()
        return np.array(self.ur5_kin.forward(currentArmJointAngles)).reshape(3,4)

    def reset_simulation(self):
        self.sim.set_state(self.sim_init_state)
        return True

    def move_to(self, toolPosition, toolQuaternion):
        # orientation format is quaternion qw,qx,qy,qz
        print('===========================================================================')
        ee_pose = self.convert2EEpose(toolPosition, toolQuaternion)
        goalJointConfig = self.IKsolver(ee_pose) 
        currentArmJointAngles = self.getArmJointAngles()
        ret, traj = p2pTrajectory(currentArmJointAngles, goalJointConfig, resolution=3e-3)
        if not ret:  # if ret is False
            print('Goal is very close to start point, no need to move end effector.\n')
        else:
            print('Robot recieved goal joint config: {}\nCurrent joint config is: {}\nStarting to move end effector'
                        .format(goalJointConfig, currentArmJointAngles))
            # compute velocity and position actuator command from the sampled trajectory
            velocityCommand = velocityGeneration(traj, self.timeStep)
            positionCommand = traj[1:]
            print('Max speed in velocityCommand is: {}'.format(np.amax(np.array(velocityCommand))))
            nStep = len(velocityCommand)
            print('Total number of trajectory step is {}'.format(nStep))
            for t in range(nStep):
                # gravity compensation for arm joints
                self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
                # assign velocity and position actuator commands
                self.sim.data.ctrl[self.armPositionCtrlID] = positionCommand[t]
                self.sim.data.ctrl[self.armVelocityCtrlID] = velocityCommand[t]
                # assign gripper actuator commands to remain in home config, i.e. position = velocity = 0
                self.sim.data.ctrl[self.gripperVelocityCtrlID] = np.zeros(
                                self.sim.data.ctrl[self.gripperVelocityCtrlID].shape)
                self.sim.data.ctrl[self.gripperPositionCtrlID] = np.zeros(
                                self.sim.data.ctrl[self.gripperPositionCtrlID].shape)
                self.simulation_step()

            # get EE pose after moving gripper
            ee_pose = self.get_EE_pos()
            self.currentToolPosition, self.currentToolQuaternion = self.convertFromEEpose(ee_pose)
            # check error
            err = la.norm(self.currentToolPosition - toolPosition)
            if err < 5e-3:
                print('Moving toolPose succedd!\nNow end effector at position: {}\n\n'
                            .format(self.currentToolPosition))
                return True
            else:
                print(toolPosition, self.currentToolPosition)
                print('Failed to reach goal state\nError is {}, Now end effector at position: {}'
                            .format(err, self.currentToolPosition))
                return False

    def move_while_grasping(self, toolPosition, toolQuaternion, force = 10):
        # The only difference from self.move_to() is gripper actuator commands
        print('===========================================================================')

        # orientation format is quaternion qw,qx,qy,qz
        ee_pose = self.convert2EEpose(toolPosition, toolQuaternion)
        goalJointConfig = self.IKsolver(ee_pose) 
        currentArmJointAngles = self.getArmJointAngles()
        # The resolution in p2pTrajectory controls how fast the joints will move, it shouldn't be too fast,
        # otherwise the grasped object might slip out of gripper
        ret, traj = p2pTrajectory(currentArmJointAngles, goalJointConfig, resolution=1e-3)
        if not ret:  # if ret is False
            print('Goal is very close to start point, no need to move end effector.\n')
        else:
            print('Recieved goal joint config: {}\nCurrent joint config is: {}\nStarting to move end effector'
                        .format(goalJointConfig, currentArmJointAngles))
            # compute velocity and position actuator command from the sampled trajectory
            velocityCommand = velocityGeneration(traj, self.timeStep)
            positionCommand = traj[1:]
            print('Max speed in velocityCommand is: {}'.format(np.amax(np.array(velocityCommand))))
            nStep = len(velocityCommand)
            print('Total number of trajectory step is {}\n'.format(nStep))

            # compute the accelaration
            



            for t in range(nStep):
                # gravity compensation for arm joints
                self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
                # assign velocity and position actuator commands
                self.sim.data.ctrl[self.armPositionCtrlID] = positionCommand[t]
                self.sim.data.ctrl[self.armVelocityCtrlID] = velocityCommand[t]
                print(velocityCommand[t])
                # assign gripper grasping force
                self.sim.data.qfrc_applied[self.gripperJointQposID] = \
                        self.sim.data.qfrc_bias[self.gripperJointQposID] + force
                self.simulation_step()
                # # drawnow plot gripper joint values
                # q1.append(self.sim.data.qpos[self.gripperJointQposID][0]*10)
                # s1.append(self.sim.data.sensordata[0])
                # v1.append(self.sim.data.qvel[self.gripperJointQposID][0])
                # drawnow(plotQ1)

            # get EE pose after moving gripper
            ee_pose = self.get_EE_pos()
            self.currentToolPosition, self.currentToolQuaternion = self.convertFromEEpose(ee_pose)
            # check error
            err = la.norm(self.currentToolPosition - toolPosition)
            if err < 5e-3:
                print('Moving toolPose succedd!\nNow end effector at position: {}\n\n'
                            .format(self.currentToolPosition))
                return True
            else:
                print(toolPosition, self.currentToolPosition)
                print('Failed to reach goal state\nError is {}, Now end effector at position: {}'
                            .format(err, self.currentToolPosition))
                return False

    def close_gripper(self, force = 10, maxTime = 5):
        # maintain arm joint configurations while moving gripper joints
        print('===========================================================================')
        # initialization 
        initialArmJointAngles = self.getArmJointAngles()
        initialGripperJointAngles = self.getGripperJointAngles()
        initialTime = self.sim.data.time                  # starting time of gripper operation
        timeHist = []
        gripperJointValueHist = []
        gripperJointVelocityHist = []
        gripperMainTouchHist = [[], []]        

        while True:
            # gravity compensation for arm joints
            self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
            # assign velocity and position actuator commands
            self.sim.data.ctrl[self.armPositionCtrlID] = initialArmJointAngles
            self.sim.data.ctrl[self.armVelocityCtrlID] = np.zeros(initialArmJointAngles.shape)
            # assign gripper velocity actuator command, assign gripper grasping force
            self.sim.data.qfrc_applied[self.gripperJointQposID] = \
                    self.sim.data.qfrc_bias[self.gripperJointQposID] + force
            self.simulation_step()
            # keep track of duration of this operation
            timeDuration = self.sim.data.time - initialTime
            # keep track of gripper joint value, velocity and main touch sensor reading
            timeHist.append(timeDuration)
            gripperJointValueHist.append(self.sim.data.qpos[self.gripperJointQposID][0])
            gripperJointVelocityHist.append(self.sim.data.qvel[self.gripperJointQposID][0])
            gripperMainTouchHist[0].append(self.sim.data.sensordata[self.gripperMainTouchID][0])
            gripperMainTouchHist[1].append(self.sim.data.sensordata[self.gripperMainTouchID][1])
            if len(gripperJointValueHist) > 10:  # only keep values of the past 20 steps
                gripperJointValueHist.pop(0)
                gripperJointVelocityHist.pop(0)
                gripperMainTouchHist[0].pop(0)
                gripperMainTouchHist[1].pop(0)
                timeHist.pop(0)
                Y = np.array([gripperJointValueHist, gripperJointVelocityHist, gripperMainTouchHist[0]])
                X = np.array(timeHist)
                slope = fitSlope(Y.transpose(), X)

                # check if gripper is fully closed
                # if gripper joint value, velocity and main touch sensor reading has stablized, that means gripper successfully closed
                if np.abs(slope)[0] < 1e-1 and np.abs(slope)[1] < 1e-1 and np.abs(slope)[2] < 1e-1 \
                        and gripperMainTouchHist[0][-1] + gripperMainTouchHist[1][-1] > force :
                    print('Gripper successfully closed.\n Time duration for closing gripper is: {}\n'.format(timeDuration))
                    return True
                if timeDuration > maxTime:
                    print('[WARNING] Maximum time {}seconds reached while closing gripper.\n'.format(maxTime))
                    return False                   
            # # drawnow plot gripper joint values
            # q1.append(self.sim.data.qpos[self.gripperJointQposID][0]*10)
            # s1.append(self.sim.data.sensordata[0])
            # v1.append(self.sim.data.qvel[self.gripperJointQposID][0])
            # drawnow(plotQ1)

    def open_gripper(self, force = 1, maxTime = 5):
        print('===========================================================================')
        # initialization 
        initialArmJointAngles = self.getArmJointAngles()
        initialGripperJointAngles = self.getGripperJointAngles()
        initialTime = self.sim.data.time                  # starting time of gripper operation
        timeHist = []
        gripperJointValueHist = []
        gripperJointVelocityHist = []
        gripperMainTouchHist = [[], []]
        # maintain arm joint configurations while moving gripper joints
        initialArmJointAngles = self.getArmJointAngles()
        initialGripperJointAngles = self.getGripperJointAngles()
        initialTime = self.sim.data.time                  # starting time of gripper operation
        leftFingerJointAngle = [initialGripperJointAngles[0]]
        goalGripperJointAngle = [0]
        while True:
            # gravity compensation for arm joints
            self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
            # assign velocity and position actuator commands
            self.sim.data.ctrl[self.armPositionCtrlID] = initialArmJointAngles
            self.sim.data.ctrl[self.armVelocityCtrlID] = np.zeros(initialArmJointAngles.shape)
            # assign gripper velocity actuator command, assign gripper grasping force
            self.sim.data.qfrc_applied[self.gripperJointQposID] = \
                    self.sim.data.qfrc_bias[self.gripperJointQposID] - force
            self.simulation_step()
            # keep track of duration of this operation
            timeDuration = self.sim.data.time - initialTime
            # keep track of gripper joint value, velocity and main touch sensor reading
            timeHist.append(timeDuration)
            gripperJointValueHist.append(self.sim.data.qpos[self.gripperJointQposID][0])
            gripperJointVelocityHist.append(self.sim.data.qvel[self.gripperJointQposID][0])
            gripperMainTouchHist[0].append(self.sim.data.sensordata[self.gripperMainTouchID][0])
            gripperMainTouchHist[1].append(self.sim.data.sensordata[self.gripperMainTouchID][1])
            if len(gripperJointValueHist) > 10:  # only keep values of the past 20 steps
                gripperJointValueHist.pop(0)
                gripperJointVelocityHist.pop(0)
                gripperMainTouchHist[0].pop(0)
                gripperMainTouchHist[1].pop(0)
                timeHist.pop(0)
                Y = np.array([gripperJointValueHist, gripperJointVelocityHist, gripperMainTouchHist[0]])
                X = np.array(timeHist)
                slope = fitSlope(Y.transpose(), X)

                # check if gripper is fully opened
                # if gripper joint value, velocity and main touch sensor reading has stablized, that means gripper successfully opened
                if np.abs(slope)[0] < 1e-1 and np.abs(slope)[1] < 1e-1 and np.abs(slope)[2] < 1e-1 \
                        and gripperMainTouchHist[0][-1] + gripperMainTouchHist[1][-1] < 0.01 * force :
                    print('Gripper successfully opended.\n Time duration for opening gripper is: {}\n'.format(timeDuration))
                    return True
                if timeDuration > maxTime:
                    print('[WARNING] Maximum time {}seconds reached while opening gripper.\n'.format(maxTime))
                    return False

    def robot_freeze(self, numStep = 1000):
        print('===========================================================================')
        # freeze the robot for numStep steps, helpful during debug
        currentArmJointAngles = self.getArmJointAngles()
        currentGripperJointAngles = self.getGripperJointAngles()
        print('Robot in freeze mode, arm is in {}, gripper in home position\n'.format(currentArmJointAngles))
        for t in range(numStep):
            # gravity compensation for arm joints
            self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
            self.sim.data.qfrc_applied[self.gripperJointQposID] = self.sim.data.qfrc_bias[self.gripperJointQposID]
            # assign velocity and position actuator commands for arm
            self.sim.data.ctrl[self.armPositionCtrlID] = currentArmJointAngles
            self.sim.data.ctrl[self.armVelocityCtrlID] = np.zeros(currentArmJointAngles.shape)
            # assign velocity and position actuator commands for gripper
            self.sim.data.ctrl[self.gripperPositionCtrlID] = currentGripperJointAngles[0]
            self.sim.data.ctrl[self.gripperVelocityCtrlID] = 0
            self.simulation_step()

    def simulation_step(self):
        self.sim.step()
        if self.RENDER:
            self.viewer.render()

    def pr2eepose(self, P, R):
        ee_pose = np.zeros((3,4))
        ee_pose[:,0:3] = R
        ee_pose[0,3] = P[0]
        ee_pose[1,3] = P[1]
        ee_pose[2,3] = P[2]
        return  ee_pose

    def eepose2pr(self, ee_pose):
        R = ee_pose[:, 0:3]
        P = np.zeros(3)
        P[0] = ee_pose[0,3]
        P[1] = ee_pose[1,3]
        P[2] = ee_pose[2,3]
        return P, R

    def getArmJointCommand(self, P, R, P_target, R_target):
        ee_pose_target = self.pr2eepose(P_target, R_target)
        target_joint_configs = self.ur5_kin.inverse(ee_pose_target.reshape(-1).tolist())    # python wrapper for ikfast solver
        n_solutions = int(len(target_joint_configs)/self.numArmJoints)
        # print("%d solutions found:"%(n_solutions))
        target_joint_configs = np.asarray(target_joint_configs).reshape(n_solutions,self.numArmJoints)
        idx = self.getClosestJointConfig(target_joint_configs, self.armJointAngles, n_solutions)   # find the closest joint_config among all solutions
        positionCommand = target_joint_configs[idx]
        velocityCommand = (target_joint_configs[idx] - self.armJointAngles) / self.timeStep
        return positionCommand, velocityCommand

    def move_in_cartesian(self, cartesianTrajectory_P, cartesianTrajectory_R, gripperClose = True, gripperCloseForce = 10, gripperOpenForce = 1, maxTime = 120):
        # cartesianTrajectory is a list of sampled points and rotational matrix in cartesian space
        # if gripperClose is Ture, the gripper will grasp with the user-defined gripperForce in gripper joint
        #
        # For each timestep in cartesianTrajectory, run the IK solver to find the next joint space point, use
        # the next point to compute velocity command for arm joint actuators. If the velocity is larger than the 
        # limit, resample between these two points.

        v_max_joint = 0.8
        print('===========================================================================')
        print('Robot recieved cartesian Trajectory.\nStarting P and R is: {}\n{}'.format(cartesianTrajectory_P[0], cartesianTrajectory_R[0]))
        print('Ending P and R is: {}\n{}'.format(cartesianTrajectory_P[-1], cartesianTrajectory_R[-1]))
        print('Number of timeStep in cartesian space: {}\n\nStarting planning and moving...\n'.format(len(cartesianTrajectory_P)))

        # initialization
        initialTime = self.sim.data.time                  # starting time of gripper operation
        timeHist = []
        P_goal = cartesianTrajectory_P[-1]

        while len(cartesianTrajectory_P) > 0:
            timeDuration = self.sim.data.time - initialTime 
            if timeDuration > maxTime:
                print('[ERROR] Planning failed, time exceeded maxTime {}s \n'.format(maxTime))
                break
            self.armJointAngles = self.getArmJointAngles()         # current arm joint angles
            ee_pose = np.asarray(self.ur5_kin.forward(self.armJointAngles)).reshape(3,4)    # current end effector pose
            P, R = self.eepose2pr(ee_pose)
            P_target = cartesianTrajectory_P[0]
            R_target = cartesianTrajectory_R[0]
            positionCommand, velocityCommand = self.getArmJointCommand(P, R, P_target, R_target)
            print(velocityCommand)
            if np.amax(velocityCommand) > v_max_joint: 
                # if the largest velocity command is above limit, re-sample a point in the middle
                # R_new = R * exp(log(R_t * R_target) * 0.5)
                P_new = P + (P_target - P) / 2
                R_new = np.dot(R, MatrixExp3(MatrixLog3(np.dot(R.transpose(), R_target)) * 0.5))
                cartesianTrajectory_P.insert(0, P_new)
                cartesianTrajectory_R.insert(0, R_new)
                continue
            else:
                # otherwise, use control command to move arm and remove the first element of trajectory list
                cartesianTrajectory_P.pop(0)
                cartesianTrajectory_R.pop(0)
                # gravity compensation for arm joints
                self.sim.data.qfrc_applied[self.armJointQposID] = self.sim.data.qfrc_bias[self.armJointQposID]
                # assign velocity and position actuator commands
                self.sim.data.ctrl[self.armPositionCtrlID] = positionCommand
                self.sim.data.ctrl[self.armVelocityCtrlID] = velocityCommand
                if gripperClose:
                    self.sim.data.qfrc_applied[self.gripperJointQposID] = \
                                    self.sim.data.qfrc_bias[self.gripperJointQposID] + gripperCloseForce
                else:
                    self.sim.data.qfrc_applied[self.gripperJointQposID] = \
                                    self.sim.data.qfrc_bias[self.gripperJointQposID] - gripperOpenForce
                # drawnow plot gripper joint values
                q1.append(self.sim.data.qpos[self.gripperJointQposID][0]*10)
                s1.append(self.sim.data.sensordata[0])
                v1.append(self.sim.data.qvel[self.gripperJointQposID][0])
                drawnow(plotQ1)

                self.simulation_step()

        # get EE pose after moving gripper
        ee_pose = self.get_EE_pos()
        self.currentToolPosition, self.currentToolQuaternion = self.convertFromEEpose(ee_pose)
        # check error
        err = la.norm(self.currentToolPosition - P_goal)
        if err < 5e-3:
            print('Moving toolPose succedd!\nNow end effector at position: {}\n\n'
                        .format(self.currentToolPosition))
            return True
        else:
            print(P_goal, self.currentToolPosition)
            print('Failed to reach goal state\nError is {}, Now end effector at position: {}'
                        .format(err, self.currentToolPosition))
            return False


            




