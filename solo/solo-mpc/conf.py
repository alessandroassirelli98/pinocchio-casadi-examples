import numpy as np
import example_robot_data as robex
import pinocchio.casadi as cpin

# OCP parameters
dt = 0.015
timestep_per_phase = 5
horizon = 10

mu = 1
kx = 10
ky = 10
k = np.array([kx, ky])
lin_vel_weight = np.array([10, 10, 10])
ang_vel_weight = np.array([10, 10, 10])
force_reg_weight = 1e-1
control_weight = 1e1
base_reg_cost = 1e1
joints_reg_cost = 1e2


 

