import numpy as np
import ikfastpy
import time
from mujoco_py import MjViewer, load_model_from_path, MjSim
from ur5gripper_v2 import Robot
from robot_util import *

model = load_model_from_path('../ur5/ur5gripperTablePNP.xml')
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()

robot = Robot(sim, viewer, Render=True)

while True:
    sim.set_state(sim_state)
    robot.move_to(np.array([0.402, 0.2264, 0.245]), np.array([-4.29355962e-04, 7.21356638e-01, -6.92255947e-01,2.06427328e-02]))
    robot.close_gripper()
    robot.move_while_grasping(np.array([0.402, 0.2264, 0.395]), np.array([-4.29355962e-04, 7.21356638e-01, -6.92255947e-01,2.06427328e-02]))
    robot.open_gripper()
    robot.robot_freeze(numStep=2000)


    # ee_pose_1 = robot.get_EE_pos()
    # P1, R1 = robot.eepose2pr(ee_pose_1)
    # q2 = np.array([-4.29355962e-04, 7.21356638e-01, -6.92255947e-01,2.06427328e-02])
    # R2 = quaternion2Rotation(q2)
    # P2 = np.array([0.402, 0.2264, 0.345])
    # ret, cartesianTrajectory_P, cartesianTrajectory_R = sampleTrajectoryCartesian(P1, R1, P2, R2)
    # robot.move_in_cartesian(cartesianTrajectory_P, cartesianTrajectory_R, gripperCloseForce = 2)
    # robot.open_gripper()
    # robot.robot_freeze(numStep=2000)
