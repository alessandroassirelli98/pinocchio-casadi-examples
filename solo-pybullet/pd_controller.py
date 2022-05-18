# coding: utf8

#####################
#  LOADING MODULES ##
#####################

import time
import numpy as np
import pybullet as p  # PyBullet simulator

# Functions to initialize the simulation and retrieve joints positions/velocities
from initialization_simulation import configure_simulation, getPosVelJoints

####################
#  INITIALIZATION ##
####################

dt = 0.015  # time step of the simulation
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation = True
enableGUI = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices = configure_simulation(dt, enableGUI)

nq = solo.nq 
nv = solo.nv
q0 = solo.q0[:nq]
v0 = np.zeros(nv)


###############
#  MAIN LOOP ##
###############
n_simu = 10000
q = np.zeros((nq, n_simu))
v = np.zeros((nv, n_simu))
for i in range(50):  # run the simulation during dt * i_max seconds (simulation time)

    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.time()
    # Get position and velocity of all joints in PyBullet (free flying base + motors)
    q_tmp, v_tmp = getPosVelJoints(robotId, revoluteJointIndices)
    q[:, i], v[:, i] = q_tmp[:,0], v_tmp[:, 0]

    # Call controller to get torques for all joints
    #jointTorques = c_walking_IK(q, qdot, dt, solo, i * dt)


    jointTorques = 5*(solo.q0- q[:, i])[7:] + 0.03 * (solo.v0- v[:, i])[6:]
    # Set control torques for all joints in PyBullet
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    p.stepSimulation()

    # Sleep to get a real time simulation
    if realTimeSimulation:
        t_sleep = dt - (time.time() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)

# Shut down the PyBullet client
p.disconnect()
