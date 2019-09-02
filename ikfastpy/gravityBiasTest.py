import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
import matplotlib.pyplot as plt
from drawnow import *
import atexit
from timeScaling import *

model = load_model_from_path('../ur5/gravity_test.xml')
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()


q1_values = []
q2_values = []
j2_values = []
#pre-load dummy data
for i in range(0,10):
    q1_values.append(0)
    q2_values.append(0)
    j2_values.append(0)
plt.ion()
cnt=0

def plotJointAngles():
    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.ylabel('J1 Values')
    plt.plot(np.array(range(len(q1_values))), q1_values, 'r-', np.array(range(len(q2_values))), q2_values, 'b')

    # plt.subplot(2, 1, 2)
    # plt.grid(True)
    # plt.ylabel('J2 Values')
    # plt.plot(j2_values, 'b-')

while True:
    sim.set_state(sim_state)
    # for i in range(19000):
        # if i < 10000:
        #     # sim.data.qfrc_applied[1] = sim.data.qfrc_bias[1]
        #     # sim.data.qfrc_applied[0] =sim.data.qfrc_bias[0] + model.body_mass[-1] * 9.8 * 5
        #     sim.data.qfrc_applied[:] = sim.data.qfrc_bias[:]
        #     sim.step()
        #     # q1_values.append(np.mod(sim.data.qpos[0]/np.pi, 1))
        #     # q2_values.append(np.mod(sim.data.qpos[1]/np.pi ,1))
        #     # drawnow(plotJointAngles)
        #     viewer.render()
        # else:
        #     sim.data.qfrc_applied[:] = sim.data.qfrc_bias[:]
        #     sim.step()
        #     q1_values.append(np.mod(sim.data.qpos[0]/np.pi, 1))
        #     q2_values.append(np.mod(sim.data.qpos[1]/np.pi, 1))
        #     drawnow(plotJointAngles)
        #     # viewer.render()
    for i in range(5):
        currentJointAngles = sim.data.qpos[:]
        goalJointAngles = currentJointAngles + np.array([1.57,1.57])
        ret, traj = p2pTrajectory(currentJointAngles, goalJointAngles, resolution=5e-4)
        timeStep = sim.model.opt.timestep
        velocityCommand = velocityGeneration(traj, timeStep)
        positionCommand = traj[1:]
        nStep = len(velocityCommand)
        print('trajectory number {}\nnumber of step is {}\n\n\n'.format(i, nStep))
        for t in range(nStep):
            sim.data.qfrc_applied[:] = sim.data.qfrc_bias[:]
            sim.data.ctrl[0:2] = positionCommand[t]
            sim.data.ctrl[2:] = velocityCommand[t]
            sim.step()
            viewer.render()



