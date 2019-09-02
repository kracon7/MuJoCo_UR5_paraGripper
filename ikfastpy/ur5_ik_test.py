import numpy as np
import ikfastpy
from mujoco_py import MjViewer, load_model_from_path, MjSim

model = load_model_from_path('../ur5/ur5gripper.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

joint_angles = [-3.1,-1.6,1.6,-1.6,-1.6,0.] # in radians

# Goal end effector pose
ee_pose = np.array([[0.04071115, -0.99870914, 0.03037599, 0.5020009],
                    [-0.99874455, -0.04156303, -0.02796067, 0.06648243],
                    [0.0291871, -0.02919955, -0.99914742, 0.35451169]])

print("\nTesting inverse kinematics:\n")
joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

# find which solution is closest to current state in joint space
idx = np.linalg.norm(np.subtract(joint_configs, joint_angles), axis='1').argmin()
print('Closest solution is number {}\n'.format(idx))
goal_position = joint_configs[idx]

sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)
    # set initial position
    for i in range(100):
        sim.data.qpos[0:6] = np.array(joint_angles)
        sim.step()
        viewer.render()
    # move end effector to goal pose
    for i in range(1000):
        sim.data.qpos[0:6] = np.array(joint_angles + i/1000 * (goal_position-joint_angles))
        sim.step()
        viewer.render()
    # stay in goal pose
    for i in range(1000):
        sim.data.qpos[0:6] = np.array(goal_position)
        sim.step()
        viewer.render()


