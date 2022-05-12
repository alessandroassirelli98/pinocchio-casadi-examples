'''
OCP with constrained dynamics

Simple example of walk  

It takes around 22 iter to converge without warmstart
If warmstarted with the solution obtained by the ocp without warmstart then it takes 900 iter
'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from pinocchio.visualize import GepettoVisualizer
from time import time
import os
import ocp as optimalControlProblem
import ocp_parameters_conf as conf

import proxnlp
from proxnlp.manifolds import MultibodyPhaseSpace, VectorSpace
from proxnlp.utils import CasadiFunction, plot_pd_errs


plt.style.use('seaborn')
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
dt = conf.dt
timestep_per_phase = 12
v_lin_target = np.array([3, 0, 0])
v_ang_target = np.array([0, 0, 0])

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('solo12')

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )
viz.display(robot.q0)

model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
#x0[3:7] = np.array([0,0,1,0])
# quasi static for x0, used for warm-start and regularization
u0 =  np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0
a0 = np.zeros(robot.nv)
fs0 = [np.ones(3)*0, np.ones(3) *0]

#     10       18        26        34
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,0,0,1 ] ] * timestep_per_phase  \
    + [ [ 0,1,1,0 ] ] * timestep_per_phase

gait = gait*2



ocp = optimalControlProblem.OCP(robot=robot ,gait=gait, x0=x0, x_ref=x0, u_ref=u0, \
    v_lin_target=v_lin_target, v_ang_target=v_ang_target, solver= 'ipopt' )

""" allContactIds = ocp.allContactIds
contactNames = ocp.contactNames """

warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

""" guess = np.load(os.getcwd() + "/sol.npy",allow_pickle=True).item()
warmstart['xs'] = guess['xs']
warmstart['acs'] = guess['acs']
warmstart['us'] = guess['us']
warmstart['fs'] = guess['fs'] """

""" feet_log = {i:[] for i in allContactIds}
residuals_log = {'inf_pr': [], 'inf_du': []}
ocp_times = []
ocp_predictions = [] """

ocp.solve(guess=warmstart)
x, a, u, f, _ = ocp.get_results()  
q_sol = x[:, : robot.nq]

viz.play(q_sol.T, dt)

np.save(open(os.getcwd() + '/sol.npy', "wb"),
        {
            "xs": x,
            "us": u,
            "acs": a,
            "fs": f
        })

### -------------------------------------------------------------------------------------- ###
### CHECK CONTRAINTS
