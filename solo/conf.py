import numpy as np
import example_robot_data as robex
import pinocchio.casadi as cpin

# OCP parameters
dt = 0.015
timestep_per_phase = 8

mu = 1
kx = 30
ky = 30
k = np.array([kx, ky])
v_lin_target = np.array([1, 0, 0])
v_ang_target = np.array([0, 0, 0])

lin_vel_weight = np.array([10, 10, 10])
ang_vel_weight = np.array([10, 10, 10])



force_reg_weight = 1e-2
control_weight = 1e1
base_reg_cost = 1e1
joints_reg_cost = 1e2
sw_feet_reg_cost = 1e1


u0 =  np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314]) 

