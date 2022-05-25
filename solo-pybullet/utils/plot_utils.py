import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

def plot(ctrl, ocp_results, local_results, dt_simu):
    r = int(ctrl.dt/dt_simu)

    t15 = np.linspace(0, (ctrl.ocp.T)*ctrl.dt, ctrl.ocp.T+1)
    t1 = np.arange(0, (ctrl.ocp.T)*ctrl.dt, dt_simu)[:ctrl.ocp.T*r]

### ------------------------------------------------------------------------- ###
    # FORCES IN WORLD FRAME 
    forces = ocp_results.ocp_storage['fw'][0]
    legend = ['F_x', 'F_y', 'F_z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i, foot in enumerate(forces):
        plt.subplot(2,2,i+1)
        plt.title('OCP Forces on ' + str(i))
        [plt.plot(forces[foot][:, jj]) for jj in range(3) ]
        plt.ylabel('Force [N]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()


### ------------------------------------------------------------------------- ###
    # JOINT VELOCITIES
    x = ocp_results.ocp_storage['xs'][0]
    legend = ['Hip', 'Shoulder', 'Knee']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title('Joint velocity of ' + str(i))
        [plt.plot(x[:, 19 + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
        plt.ylabel('Velocity [Deg/s]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()


### ------------------------------------------------------------------------- ###
    # BASE POSITION

    base_log_ocp = ctrl.ocp.get_base_log(ocp_results.ocp_storage['xs'][0])
    base_log_m = ctrl.ocp.get_base_log(local_results.x_m)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Base position on ' + legend[i])
            plt.plot(t15, base_log_ocp[:, i])
            plt.plot(t1, base_log_m[:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()


### ------------------------------------------------------------------------- ###
    # FEET POSITIONS 
    feet_log_ocp = ctrl.ocp.get_feet_position(ocp_results.ocp_storage['xs'][0])
    feet_log_m = ctrl.ocp.get_feet_position(local_results.x_m)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Foot position on ' + legend[i])
            plt.plot(t15, feet_log_ocp[foot][:, i])
            plt.plot(t1, feet_log_m[foot][:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()









    plt.show()
