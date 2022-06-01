import numpy as np
import example_robot_data as robex
import pinocchio.casadi as cpin

# OCP parameters
mu = 1

foot_tracking_cost = 1e3
force_reg_weight = 1e-2
control_weight = 1e1
base_reg_cost = 1e1
base_translation_weight = 1e-1
joints_reg_cost = np.array([1e0, 1e0, 1e0, 1e-1, 1e-1, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
joints_vel_reg_cost = np.array([1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1])
terminal_cost = 1e5


 

