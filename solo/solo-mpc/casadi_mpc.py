import ocp
import conf
import numpy as np
from pinocchio.visualize import GepettoVisualizer
import example_robot_data as robex

dt = conf.dt

v_lin_target = np.array([1, 0, 0])
v_ang_target = np.array([0, 0, 0])

robot = robex.load('solo12')
nq, nv = robot.model.nq, robot.model.nv

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )


# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
gait = [] \
    + [ [ 1,0,0,1 ] ] * conf.walking_steps  \
    + [ [ 0,1,1,0 ] ] * conf.walking_steps

ocp = ocp.OCP(robot, gait)
x0 = ocp.x0
xs, us = ocp.solve(gait=gait, x_ref=x0, v_lin_target=v_lin_target, v_ang_target=v_ang_target)

q_sol = xs[:,: nq]