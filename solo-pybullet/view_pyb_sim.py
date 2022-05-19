import ocp_Giotto as optimalControlProblem
import ocp_parameters_conf as conf
import time
import numpy as np
import pybullet as p  # PyBullet simulator

# Functions to initialize the simulation and retrieve joints positions/velocities
from initialization_simulation import configure_simulation, getPosVelJoints
import matplotlib.pyplot as plt
import os
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
solver = 'ipopt'
dt = conf.dt
timestep_per_phase = 10
horizon = 20

realTimeSimulation = True
enableGUI = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices = configure_simulation(dt, enableGUI)


nq = solo.nq 
nv = solo.nv
q0 = solo.q0[:nq]
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0

fs0 = [np.ones(3)*0, np.ones(3) *0]
FR_foot0 = np.array([0.1946, -0.16891, 0.0191028])

# Create target for free foot
A = np.array([0, 0.05, 0.05])
offset = np.array([0.15, 0, 0.05])
freq = np.array([0, 1, 1])
phase = np.array([0,0,np.pi/2])

#     10       18        26        34
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,1,1,1 ] ] * timestep_per_phase



x_mpc = []
x_mpc.append(x0)
residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = []

data = np.load("/tmp/sol_mpc.npy",allow_pickle=True).item()
u_mpc = data['u_mpc']

""" u_mpc = np.zeros((10000, 12))
u_mpc[:, 4] = 0.001
 """

q = np.zeros((nq, len(u_mpc)))
v = np.zeros((nv, len(u_mpc)))

for i in range(len(u_mpc)-1):
    print('Iteration ', str(i), ' / ', str(horizon))
    if realTimeSimulation:
        t0 = time.time()

    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=u_mpc[i])

    # Compute one step of simulation
    p.stepSimulation()

    # Sleep to get a real time simulation
    if realTimeSimulation:
        t_sleep = dt - (time.time() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)
    
    # Get position and velocity of all joints in PyBullet (free flying base + motors)
    q_tmp, v_tmp = getPosVelJoints(robotId, revoluteJointIndices)
    q[:, i], v[:, i] = q_tmp[:,0], v_tmp[:, 0]
    x_p = np.concatenate([q[:, i], v[:, i]])

#p.disconnect()


