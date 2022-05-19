import ocp_Giotto as optimalControlProblem
import ocp_parameters_conf as conf
import time
import numpy as np
from pinocchio.visualize import GepettoVisualizer
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
horizon = 50

realTimeSimulation = True
enableGUI = True  # enable PyBullet GUI or not
USE_BULLET = False
if not USE_BULLET: enableGUI = False
robotId, solo, revoluteJointIndices = configure_simulation(dt, enableGUI)

if not USE_BULLET:
    try:
        viz = GepettoVisualizer(solo.model,solo.collision_model, solo.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        gv = viz.viewer.gui
        viz.display(solo.q0)
    except:
        print("No viewer"  )
    

nq = solo.nq 
nv = solo.nv
q0 = solo.q0[:nq]
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
x_p = np.array([ 3.81852412e-21, -5.06186889e-22,  2.32792750e-01,  1.57686364e-19,
        2.89014799e-19, -4.94965397e-20,  1.00000000e+00,  1.00000000e-01,
        8.00000000e-01, -1.60000000e+00, -1.00000000e-01,  8.00000000e-01,
       -1.60000000e+00,  1.00000000e-01, -8.00000000e-01,  1.60000000e+00,
       -1.00000000e-01, -8.00000000e-01,  1.60000000e+00,  2.54568275e-19,
       -3.37457926e-20, -1.47150000e-01,  2.10248485e-17,  3.85353065e-17,
       -6.59953862e-18,  0.00000000e+00,  0.00000000e+00,  1.00979361e-16,
        0.00000000e+00, -1.71537341e-16,  2.01958723e-16, -9.22225583e-17,
        1.71537341e-16, -3.02938084e-16,  9.25916364e-17,  0.00000000e+00,
       -1.00979361e-16])
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
    + [ [ 1,0,1,1 ] ] * timestep_per_phase

warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

x_mpc = []
u_mpc = []
x_mpc.append(x_p)
residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = []

q = np.zeros((nq, horizon))
v = np.zeros((nv, horizon))

start_time = time.time()
for i in range(horizon):
    print('Iteration ', str(i), ' / ', str(horizon))
    target = []
    for t in range(timestep_per_phase): target += [FR_foot0 + offset +A*np.sin(2*np.pi*freq * (t+i)*dt + phase)]
    target = np.array(target)

    if i != 0:
        gait = np.roll(gait, -1, axis=0)

    if realTimeSimulation:
        t0 = time.time()

    ocp = optimalControlProblem.OCP(robot=solo, gait=gait, x0=x_mpc[-1], x_ref=x0.copy(),\
                                        u_ref = u0, target = target, solver=solver)

    if i == 0:
        allContactIds = ocp.allContactIds
        contactNames = ocp.contactNames
        feet_log = {i:[] for i in allContactIds}
        feet_vel_log = {i:[] for i in allContactIds}
                                        
    
    ocp.solve(guess=warmstart)
    print('OCP time: ', ocp.iterationTime)
    
    dx, x, a, u, f = ocp.get_results()  
    warmstart['xs'] = x[1:]
    warmstart['acs'] = a[1:]
    warmstart['us'] = u[1:]
    warmstart['fs'] = f[1:]

    if USE_BULLET:

        p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=u[0])
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

        x = np.concatenate([q[:, i], v[:, i]])
        x_mpc.append(x)

    else:
        x_mpc.append(x[1])    

    u_mpc.append(u[0])

    ocp_feet_log = ocp.get_feet_position(x)
    ocp_feet_vel_log = ocp.get_feet_velocities(x)
    ocp_predictions.append(ocp.get_base_log(x))
    ocp_times.append(ocp.iterationTime)
    [residuals_log[key].append(ocp.opti.stats()['iterations'][key]) for key in residuals_log]
    for foot in allContactIds: feet_log[foot] += [ocp_feet_log[foot][0, :]]
    for foot in allContactIds: feet_vel_log[foot] += [ocp_feet_vel_log[foot][0, :]]

p.disconnect()

np.save(open('/tmp/sol_mpc.npy', "wb"),
        {
            "x_mpc": x_mpc,
            "u_mpc": u_mpc,

        })

for foot in allContactIds: feet_log[foot] = np.array(feet_log[foot])
for foot in allContactIds: feet_vel_log[foot] = np.array(feet_vel_log[foot])
print( 'Total MPC time: ', time.time() - start_time, '\n\n')

x_mpc = np.array(x_mpc)
u_mpc = np.array(u_mpc)
q_sol = x_mpc[:, :nq] # GET THE SOLUTION OF THE SINGLE OCP I'M solving
base_log_mpc = ocp.get_base_log(x_mpc)



### --------------------------------------------------- ###

t_scale = np.linspace(0, (ocp.T)*dt, ocp.T+1)
t_scale_mpc = np.linspace(0, (horizon)*dt, horizon+1)


plt.figure(figsize=(12, 6), dpi = 90)
for i in range(horizon):
    plt.subplot(int((horizon + 1)/2), 2, i+1)
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
        plt.plot(feet_log[foot][ :, i])
        plt.legend(['FR_foot', 'Ref'])
plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in ocp.terminalModel.freeIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot velocity on ' + legend[i])
        plt.plot(feet_vel_log[foot][:, i])
plt.draw()

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint velocity of ' + contactNames[i])
    [plt.plot(x_mpc[:, nq + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
    plt.ylabel('Velocity [Deg/s]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint torques of ' + contactNames[i])
    [plt.plot(u_mpc[:, (3*i+jj)]) for jj in range(3) ]
    plt.ylabel('Torque [N/m]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()


plt.plot()