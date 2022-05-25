from utils.PyBulletSimulator import PyBulletSimulator
from utils.plot_utils import plot
import numpy as np
from Controller import Controller, Results

horizon = 30
dt_ocp = 0.015
dt_sim = 0.001
r = int(dt_ocp/dt_sim)

def Init_simulation(q_init):
    device = PyBulletSimulator()
    device.Init(q_init, 0, True, True, dt_sim)
    return device

def tuple_to_array(tup):
    a = np.array([element for tupl in tup for element in tupl])
    return a

def interpolate_traj():
    measures = read_state()
    qj_des_i = np.linspace(measures['qj_m'], ctrl.results.qj_des, r)
    vj_des_i = np.linspace(measures['vj_m'], ctrl.results.vj_des, r)

    return qj_des_i, vj_des_i

def read_state():
    device.parse_sensor_data()
    qj_m = device.joints.positions
    vj_m = device.joints.velocities
    bp_m = tuple_to_array(device.baseState)
    bv_m = tuple_to_array(device.baseVel)
    x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])

    return {'qj_m': qj_m, 'vj_m': vj_m, 'x_m': x_m}


def send_torques():
    q, v = interpolate_traj()
    for t in range(r):
        device.joints.set_desired_positions(q[t])
        device.joints.set_desired_velocities(v[t])
        device.joints.set_position_gains(4)
        device.joints.set_velocity_gains(0.1)
        device.joints.set_torques(ctrl.results.tau_ff)
        device.send_command_and_wait_end_of_cycle()

        local_res.tau.append(device.jointTorques)

def control_loop(ctrl):
    for t in range(horizon):      
        measures = read_state()
        local_res.tau.append(device.jointTorques)

        ctrl.compute_step(measures['x_m'], ctrl.x0, ctrl.u0)

        send_torques()       


if __name__ ==  '__main__':
    ctrl = Controller(50, dt_ocp)
    local_res = Results()
    device = Init_simulation(ctrl.qj0)

    ctrl.create_target()
    control_loop(ctrl)
    plot(ctrl.results)
    
    
    np.save(open('/tmp/sol_mpc.npy', "wb"),
        {
            "u_mpc": local_res.tau

        })


