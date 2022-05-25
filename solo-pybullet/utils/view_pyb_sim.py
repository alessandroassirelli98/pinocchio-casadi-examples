#####################
#  LOADING MODULES ##
#####################

import time
import numpy as np
import pybullet as p  # PyBullet simulator
import matplotlib.pyplot as plt
from PyBulletSimulator import PyBulletSimulator
from loader import Solo12

plt.style.use("seaborn")

# Functions to initialize the simulation and retrieve joints positions/velocities
####################
#  INITIALIZATION ##
####################

p_dt = 0.001  # time step of the simulation
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
solo = Solo12()

nq = solo.nq 
nv = solo.nv
q0 = solo.q0
v0 = np.zeros(12)

device = PyBulletSimulator()
device.Init(q0[7:], 0, True, True, p_dt)

data = np.load("/tmp/sol_ocp.npy",allow_pickle=True).item()
u = data['u_mpc']
time.sleep(2)

###############
#  MAIN LOOP ##
###############
tau_log = []
for i in range (len(u)):
    device.joints.set_torques(u[i])
    device.send_command_and_wait_end_of_cycle()
    tau_log.append(device.jointTorques)
tau_log = np.array(tau_log)


# Shut down the PyBullet client
for i in range(12):
    plt.subplot(6,2, i+1)
    plt.plot(tau_log[:, i])


