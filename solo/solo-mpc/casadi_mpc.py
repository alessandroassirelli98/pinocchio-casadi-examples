import ocp
import conf
import numpy as np
from pinocchio.visualize import GepettoVisualizer
import example_robot_data as robex
import matplotlib.pyplot as plt
import os
import time
path = os.getcwd()

dt = conf.dt

v_lin_target = np.array([1, 0, 0])
v_ang_target = np.array([0, 0, 0])

robot = robex.load('solo12')
nq, nv = robot.model.nq, robot.model.nv
x0 = np.concatenate([robot.q0, np.zeros(nv)])
u0 =  np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0
fs0 = [np.ones(3)*0, np.ones(3) *0]
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
    + [ [ 1,0,0,1 ] ] * conf.timestep_per_phase  \
    + [ [ 0,1,1,0 ] ] * conf.timestep_per_phase

ocp = ocp.OCP(robot, gait)
allContactIds = ocp.allContactIds
contactNames = ocp.contactNames

warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

x_mpc = []
x_mpc.append(x0)
feet_log = {i:[] for i in allContactIds}
residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = []

start_time = time.time()
for i in range(conf.horizon):
    print('Iteration ', str(i), ' / ', str(conf.horizon))

    gait = np.roll(gait, -1, axis=0)
    ocp.solve(gait=gait, x0=x_mpc[-1], x_ref=x0, u_ref = u0, v_lin_target=v_lin_target, \
                    v_ang_target=v_ang_target, guess=warmstart)
    print('OCP time: ', ocp.iterationTime)
    
    ocp_times.append(ocp.iterationTime)
    
    x, a, u, f, _ = ocp.get_results()  
    ocp_feet_log = ocp.get_feet_position(x)
    ocp_predictions.append(ocp.get_base_log(x))
    [residuals_log[key].append(ocp.opti.stats()['iterations'][key]) for key in residuals_log]

    x_mpc.append(x[1])
    
    """ warmstart['xs'] = np.append(x[1:], x[-1].reshape(1, -1), axis = 0)
    warmstart['acs'] = np.append(a[1:], a[-1].reshape(1, -1), axis = 0)
    warmstart['us'] = np.append(u[1:], u[-1].reshape(1, -1), axis = 0)
    f.append(f[-1])
    warmstart['fs'] = f[1:] """

    
    warmstart['xs'] = x[1:]
    warmstart['acs'] = a[1:]
    warmstart['us'] = u[1:]
    warmstart['fs'] = f[1:]
    for foot in allContactIds: feet_log[foot] += [ocp_feet_log[foot][0, :]]

for foot in allContactIds: feet_log[foot] = np.array(feet_log[foot])  
print( 'Total MPC time: ', time.time() - start_time, '\n\n')

x_mpc = np.array(x_mpc)
q_sol = x_mpc[:, :nq]
base_log_mpc = ocp.get_base_log(x_mpc)



### --------------------------------------------------- ###

t_scale = np.linspace(0, (ocp.T)*dt, ocp.T+1)
t_scale_mpc = np.linspace(0, (conf.horizon)*dt, conf.horizon+1)


plt.figure(figsize=(12, 6), dpi = 90)
for i in range(conf.horizon):
    plt.subplot(int((conf.horizon + 1)/2), 2, i+1)
    plt.title('Residuals' + str(i))
    plt.semilogy(residuals_log['inf_du'][i])
    plt.semilogy(residuals_log['inf_pr'][i])
    plt.legend(['dual', 'primal'])


plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in allContactIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot position on ' + legend[i])
        plt.plot(feet_log[foot][:, i])
        plt.legend(contactNames)
plt.draw()

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 18), dpi = 90)
for p in range(3):
    plt.subplot(3,1, p+1)
    plt.title('Base on ' + legend[p])
    for i in range(conf.horizon):
        t = np.linspace(i*dt, (ocp.T+ i)*dt, ocp.T+1)
        y = ocp_predictions[i][:,p]
        for j in range(len(y) - 1):
            plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([0.8 - j/len(y), 0]))
    plt.plot(t_scale_mpc, base_log_mpc[:, p], linewidth=0.5, color = 'tomato', marker='o')
plt.draw()

plt.show()