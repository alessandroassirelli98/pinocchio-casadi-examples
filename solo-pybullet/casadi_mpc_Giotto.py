import ocp_Giotto as optimalControlProblem
import ocp_parameters_conf as conf
import numpy as np
from pinocchio.visualize import GepettoVisualizer
import example_robot_data as robex
import matplotlib.pyplot as plt
import os
import time
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
solver = 'ipopt'
dt = conf.dt
timestep_per_phase = 66
horizon = 1
robot = robex.load('solo12')
nq, nv = robot.model.nq, robot.model.nv
x0 = np.concatenate([robot.q0, np.zeros(nv)])
u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0

fs0 = [np.ones(3)*0, np.ones(3) *0]
FR_foot0 = np.array([0.1946, -0.16891, 0.0191028])

# Create target for free foot
target = []
A = np.array([0, 0.05, 0.05])
offset = np.array([0.15, 0, 0.1])
freq = np.array([0, 1, 1])
phase = np.array([0,0,np.pi/2])
for t in range(timestep_per_phase): target += [FR_foot0 + offset + \
                                            A*np.sin(2*np.pi*freq * t*dt + phase)]
target = np.array(target)

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

#     10       18        26        34
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,0,1,1 ] ] * timestep_per_phase



warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

x_mpc = []
x_mpc.append(x0)
residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = []

guess = np.load("/tmp/sol.npy",allow_pickle=True).item()
warmstart['dxs'] = guess['dxs'][1:]
warmstart['xs'] = guess['xs'][1:]
warmstart['acs'] = guess['acs'][1:]
warmstart['us'] = guess['us'][1:]
warmstart['fs'] = guess['fs'][1:]


start_time = time.time()
for i in range(horizon):
    print('Iteration ', str(i), ' / ', str(horizon))

    if i != 0:
        gait = np.roll(gait, -1, axis=0)

    ocp = optimalControlProblem.OCP(robot=robot, gait=gait, x0=x_mpc[-1], x_ref=x0,\
                                        u_ref = u0, target = target, solver=solver)
                                        
    if i == 0:
        allContactIds = ocp.allContactIds
        contactNames = ocp.contactNames
        feet_log = {i:[] for i in allContactIds}
        feet_vel_log = {i:[] for i in allContactIds}
    
    ocp.solve(guess=warmstart)
    print('OCP time: ', ocp.iterationTime)
    
    dx, x, a, u, f = ocp.get_results()  

    ocp_feet_log = ocp.get_feet_position(x)
    ocp_feet_vel_log = ocp.get_feet_velocities(x)
    ocp_predictions.append(ocp.get_base_log(x))
    ocp_times.append(ocp.iterationTime)
    [residuals_log[key].append(ocp.opti.stats()['iterations'][key]) for key in residuals_log]
    for foot in allContactIds: feet_log[foot] += [ocp_feet_log[foot][:, :]]
    for foot in allContactIds: feet_vel_log[foot] += [ocp_feet_vel_log[foot][:, :]]

    x_mpc.append(x[1])
    
    warmstart['dxs'] = dx[1:]
    warmstart['xs'] = x[1:]
    warmstart['acs'] = a[1:]
    warmstart['us'] = u[1:]
    warmstart['fs'] = f[1:]
    


for foot in allContactIds: feet_log[foot] = np.array(feet_log[foot])
for foot in allContactIds: feet_vel_log[foot] = np.array(feet_vel_log[foot])
print( 'Total MPC time: ', time.time() - start_time, '\n\n')

x_mpc = np.array(x_mpc)
q_sol = x[:, :nq] # GET THE SOLUTION OF THE SINGLE OCP I'M solving
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
        plt.plot(t_scale, feet_log[foot][0, :, i])
        plt.plot(t_scale, target[:, i], linestyle = '--')
        plt.legend(['FR_foot', 'Ref'])
plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in ocp.terminalModel.freeIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot position on ' + legend[i])
        plt.plot(t_scale, feet_vel_log[foot][0, :, i])
plt.draw()

""" legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 18), dpi = 90)
for p in range(3):
    plt.subplot(3,1, p+1)
    plt.title('Base on ' + legend[p])
    for i in range(horizon):
        t = np.linspace(i*dt, (ocp.T+ i)*dt, ocp.T+1)
        y = ocp_predictions[i][:,p]
        for j in range(len(y) - 1):
            plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([0.8 - j/len(y), 0]))
    plt.plot(t_scale_mpc, base_log_mpc[:, p], linewidth=0.5, color = 'tomato', marker='o')
plt.draw() """

plt.show()

np.save(open('/tmp/sol.npy', "wb"),
        {
            "dxs": dx,
            "xs": x,
            "us": u,
            "acs": a,
            "fs": f
        })