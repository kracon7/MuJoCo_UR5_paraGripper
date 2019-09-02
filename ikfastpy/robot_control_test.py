import numpy as np
import ikfastpy
import time
from mujoco_py import MjViewer, load_model_from_path, MjSim
from ur5gripper import Robot

model = load_model_from_path('../ur5/ur5gripper.xml')
sim = MjSim(model)
# viewer = MjViewer(sim)
sim_state = sim.get_state()

robot = Robot(sim, viewer, Render=True)

while True:
    sim.set_state(sim_state)
    # robot.move_to(np.array([0.402, 0.2264, 0.245]), np.array([-4.29355962e-04, 7.21356638e-01, -6.92255947e-01,2.06427328e-02]), numStep = 2000)
    # robot.close_gripper(torque=0.5)
    # robot.move_while_grasping(np.array([0.402, 0.2264, 0.345]), np.array([-4.29355962e-04, 7.21356638e-01, -6.92255947e-01,2.06427328e-02]), numStep=4000, torque=1)
    # robot.open_gripper(torque=-0.1)
    # robot.robot_freeze(numStep=3000)
    # bias = -sim.data.qfrc_bias[0:7]
    for i in range(5000):
        sim.data.qfrc_applied[:] = sim.data.qfrc_bias[:]
        sim.step()
        viewer.render()
