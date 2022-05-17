import numpy as np
import example_robot_data as robex
import pinocchio.casadi as cpin

# OCP parameters
dt = 0.015
mu = 1
kx = 30
ky = 30
k = np.array([kx, ky])
foot_tracking_cost = 1e2
force_reg_weight = 1e-2
control_weight = 1e1
base_reg_cost = 1e1
joints_reg_cost = 1e1
joints_vel_reg_cost = 0
sw_feet_reg_cost = 1e1


 

