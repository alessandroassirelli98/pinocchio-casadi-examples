import numpy as np
import example_robot_data as robex
import pinocchio.casadi as cpin

# OCP parameters
mu = 0.8
kx = 30
ky = 30
k = np.array([kx, ky])
foot_tracking_cost = 1e3
force_reg_weight = 1e-2
stiff_contact_weight = 1e5*0
control_weight = 1e1
base_reg_cost = 1e1
base_translation_weight = 1e3
joints_reg_cost = 1e1
joints_vel_reg_cost = 1e2
terminal_cost = 1e4


 

