import ocp_Giotto as optimalControlProblem
import ocp_parameters_conf as conf
import time
import numpy as np
from pinocchio.visualize import GepettoVisualizer
import pybullet as p  # PyBullet simulator
from PyBulletSimulator import PyBulletSimulator
from loader import Solo12

# Functions to initialize the simulation and retrieve joints positions/velocities
import matplotlib.pyplot as plt
import os
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
solver = 'ipopt'
dt = 0.015
timestep_per_phase = 50
p_dt = 0.001
t_init = 1
step_init = int(t_init/p_dt)
solo = Solo12()

nq = solo.nq 
nv = solo.nv
q0 = solo.q0
qj0 = q0[7:]
v0 = np.zeros(nv)
vj0 = v0[6:]
x0 = np.concatenate([q0, v0])
u0 = np.array([-0.02613785, -0.2584939 ,  0.51698191,  0.02860206, -0.25719823,
        0.51440234, -0.02613105,  0.25848988, -0.51698564,  0.02860886,
        0.25720224, -0.51439861])

fs0 = [np.ones(3)*0, np.ones(3) *0]
FR_foot0 = np.array([0.1946, -0.16891, 0.0191028])


device = PyBulletSimulator()
device.Init(q0[7:], 0, True, True, p_dt)

def put_on_ground(steps):
    for i in range(steps):
        device.parse_sensor_data()
        q = device.joints.positions
        v = device.joints.velocities

        jointTorques = 5*(q0[7:]- q) + 0.5 * (np.zeros(12)- v)
        device.joints.set_torques(jointTorques)

        device.send_command_and_wait_end_of_cycle()


# Create target for free foot
A = np.array([0, 0.05, 0.05])
offset = np.array([0.15, 0, 0.05])
freq = np.array([0, 1, 1])
phase = np.array([0,0,np.pi/2])

#     10       18        26        34
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,0,1,1 ] ] * timestep_per_phase

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
                                    u_ref = u0, target = target)


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


u_log = []
# Constatnt torque between timesteps
for i in range (u.shape[0]):
    for _ in range(int(dt/p_dt)):
        device.joints.set_position_gains(3)
        device.joints.set_velocity_gains(0.1)
        device.joints.set_desired_positions(q_des[i, :])
        device.joints.set_desired_velocities(v_des[i, :])
        device.joints.set_torques(u[i, :])
        device.send_command_and_wait_end_of_cycle()
        u_log.append(device.joints.measured_torques)

u_log = np.array(u_log)
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

legend = ['Hip_opt', 'Shoulder_opt', 'Knee_opt', 'Hip_meas', 'Shoulder_meas', 'Knee_meas']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint torques of ' + contactNames[i])
    [plt.plot(u[:, (3*i+jj)], linestyle = '--') for jj in range(3) ]
    [plt.plot(u_log[:, (3*i+jj)], linestyle = '--') for jj in range(3) ]
    plt.ylabel('Torque [N/m]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()

plt.plot()