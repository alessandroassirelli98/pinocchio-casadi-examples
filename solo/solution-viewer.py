
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from pinocchio.visualize import GepettoVisualizer
from time import time
import os
plt.style.use('seaborn')
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
DT = 0.015
walking_steps = 15
mu = 1
kx = 1
ky = 1
k = np.array([kx, ky])
step_height = 0.05
v_lin_target = np.array([1, 0, 0])
v_ang_target = np.array([0, 0, 0])

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('solo12')
# The pinocchio model is what we are really interested by.
model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

try:
    data = np.load(path + '/sol.npy',allow_pickle=True).item()
    xs_sol = data['xs']
    print("Got data")
except:
    print("No data")


q_sol = xs_sol[:,: robot.nq]