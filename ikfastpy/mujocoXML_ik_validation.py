import numpy as np
import ikfastpy
from mujoco_py import MjViewer, load_model_from_path, MjSim

model = load_model_from_path('../ur5/ur5gripper.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

joint_angles_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
joint_angles_2 = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.0]
joint_angles_3 = [3.1, -1.6, -1.6, 1.6, -1.6, 0.0]


while True:
    # set initial position
    for i in range(1000):
        sim.data.qpos[0:6] = np.array(joint_angles_1)
        sim.step()
        viewer.render()
    # move end effector to goal pose
    for i in range(1000):
        sim.data.qpos[0:6] = np.array(joint_angles_2)
        sim.step()
        viewer.render()

    for i in range(1000):
        sim.data.qpos[0:6] = np.array(joint_angles_3)
        sim.step()
        viewer.render()