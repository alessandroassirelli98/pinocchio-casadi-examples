import ocp_Giotto as optimalControlProblem
import ocp_parameters_conf as conf
import time
import numpy as np
from pinocchio.visualize import GepettoVisualizer
import pybullet as p  # PyBullet simulator
from PyBulletSimulator import PyBulletSimulator
from loader import Solo12

# Functions to initialize the simulation and retrieve joints positions/velocities
from initialization_simulation import configure_simulation, getPosVelJoints
import matplotlib.pyplot as plt
import os
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
solver = 'ipopt'
dt = conf.dt
timestep_per_phase = 40
solo = Solo12()

nq = solo.nq 
nv = solo.nv
q0 = solo.q0
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0

fs0 = [np.ones(3)*0, np.ones(3) *0]
FR_foot0 = np.array([0.1946, -0.16891, 0.0191028])


device = PyBulletSimulator()
device.Init(q0[7:], 0, True, True, conf.dt)


# Create target for free foot
A = np.array([0, 0.05, 0.05])
offset = np.array([0.15, 0, 0.05])
freq = np.array([0, 1, 1])
phase = np.array([0,0,np.pi/2])

#     10       18        26        34
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,1,1,1 ] ] * timestep_per_phase

warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = []

start_time = time.time()


# Target is disabled in OCP formulation now
# I want it to be still with 4 contacts
target = []
for t in range(timestep_per_phase): target += [FR_foot0 + offset +A*np.sin(2*np.pi*freq * (t)*dt + phase)]
target = np.array(target)


ocp = optimalControlProblem.OCP(robot=solo, gait=gait, x0=x0, x_ref=x0.copy(),\
                                    u_ref = u0, target = target, solver=solver)


allContactIds = ocp.allContactIds
contactNames = ocp.contactNames
feet_log = {i:[] for i in allContactIds}
feet_vel_log = {i:[] for i in allContactIds}
                                    

ocp.solve(guess=warmstart)
print('OCP time: ', ocp.iterationTime)

dx, x, a, u, f = ocp.get_results()  

q_des = x[:, 7: nq]
v_des = x[:, nq + 6: ]
q_sol = x[:, :nq] 

ocp_feet_log = ocp.get_feet_position(x)
ocp_feet_vel_log = ocp.get_feet_velocities(x)
ocp_predictions.append(ocp.get_base_log(x))
ocp_times.append(ocp.iterationTime)
[residuals_log[key].append(ocp.opti.stats()['iterations'][key]) for key in residuals_log]
for foot in allContactIds: feet_log[foot] += [ocp_feet_log[foot][:, :]]
for foot in allContactIds: feet_vel_log[foot] += [ocp_feet_vel_log[foot][:, :]]

for foot in allContactIds: feet_log[foot] = np.array(feet_log[foot])
for foot in allContactIds: feet_vel_log[foot] = np.array(feet_vel_log[foot])

""" np.save(open('/tmp/sol_mpc.npy', "wb"),
        {
            "x": x,
            "u": u,
        })
 """

## Simulate in PyBullet
for i in range(u.shape[0]):
    device.joints.set_torques(np.array([0.0, 0.022, 0.5] * 2 + [0.0, -0.022, -0.5] * 2))
    device.parse_sensor_data()
    device.Print()
    device.send_command_and_wait_end_of_cycle()

### Show in Gepetto gui
try:
    viz = GepettoVisualizer(solo.model,solo.robot.collision_model, solo.robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

viz.play(q_sol.T, dt)






### --------------------------------------------------- ###

t_scale = np.linspace(0, (ocp.T)*dt, ocp.T+1)



plt.figure(figsize=(12, 6), dpi = 90)
for i in range(1):
    plt.subplot(1, 2, i+1)
    plt.title('Residuals' + str(i))
    plt.semilogy(residuals_log['inf_du'][i])
    plt.semilogy(residuals_log['inf_pr'][i])
    plt.legend(['dual', 'primal'])

plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in ocp.terminalModel.freeIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot position on ' + legend[i])
        plt.plot(feet_log[foot][0, :, i])
        plt.legend(['FR_foot', 'Ref'])
plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in ocp.terminalModel.freeIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot velocity on ' + legend[i])
        plt.plot(feet_vel_log[foot][0, :, i])
plt.draw()

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint velocity of ' + contactNames[i])
    [plt.plot(x[:, nq + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
    plt.ylabel('Velocity [Deg/s]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint torques of ' + contactNames[i])
    [plt.plot(u[:, (3*i+jj)]) for jj in range(3) ]
    plt.ylabel('Torque [N/m]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()


plt.plot()