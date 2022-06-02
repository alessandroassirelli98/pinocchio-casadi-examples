import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

def plot_ocp(ctrl, ocp_results, local_results, dt_simu):
    r = int(ctrl.dt/dt_simu)

    t15 = np.linspace(0, (ctrl.ocp.T)*ctrl.dt, ctrl.ocp.T+1)
    t1 = np.linspace(0, (ctrl.ocp.T)*ctrl.dt, ctrl.ocp.T*r + 1)

### ------------------------------------------------------------------------- ###
    # FORCES IN WORLD FRAME 
    forces = ocp_results.ocp_storage['fw'][0]
    legend = ['F_x', 'F_y', 'F_z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i, foot in enumerate(forces):
        plt.subplot(2,2,i+1)
        plt.title('OCP Forces on ' + str(i))
        [plt.plot(t15[:-1], forces[foot][:, jj]) for jj in range(3) ]
        plt.ylabel('Force [N]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()

### ------------------------------------------------------------------------- ###
    # JOINT VELOCITIES
    x = ocp_results.ocp_storage['xs'][1]
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
    # JOINT TORQUES
    u = ocp_results.ocp_storage['us'][0]
    legend = ['Hip', 'Shoulder', 'Knee']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title('Joint torques of ' + str(i))
        [plt.plot(u[:, (3*i+jj)]) for jj in range(3) ]
        plt.ylabel('Velocity [Deg/s]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()

### ------------------------------------------------------------------------- ###
    # BASE POSITION

    base_log_ocp = ctrl.ocp.get_base_log(ocp_results.ocp_storage['xs'][1])
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
    feet_log_ocp = ctrl.ocp.get_feet_position(ocp_results.ocp_storage['xs'][1])
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


def plot_mpc(ctrl, ocp_results, local_results, dt_simu):

    horizon = len(ocp_results.ocp_storage['xs'])-1
    r = int(ctrl.dt/dt_simu)
    t15 = np.linspace(0, horizon*ctrl.dt, horizon+1)
    t1 = np.linspace(0, (horizon)*ctrl.dt, (horizon)*r+1)
    t_mpc = np.linspace(0, (horizon)*ctrl.dt, horizon+1)

    x_mpc = []
    [x_mpc.append(ctrl.results.ocp_storage['xs'][i][1, :]) for i in range(horizon+1)]
    x_mpc = np.array(x_mpc)

    feet_log_mpc = ctrl.ocp.get_feet_position(x_mpc)
    feet_log_m = ctrl.ocp.get_feet_position(local_results.x_m)
    all_ocp_feet_log = [ctrl.ocp.get_feet_position(x)[18] for x in ctrl.results.ocp_storage['xs']]
    all_ocp_feet_log = np.array(all_ocp_feet_log)

    u_mpc = []
    [u_mpc.append(ocp_results.ocp_storage['us'][i][1, :]) for i in range(horizon)]
    u_mpc = np.array(u_mpc)
    all_u_log = np.array(ocp_results.ocp_storage['us'])
    u_m = np.array(local_results.tau)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Foot position on ' + legend[i])
            plt.plot(t15, feet_log_mpc[foot][:, i])
            plt.plot(t1, feet_log_m[foot][:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 18), dpi = 90)
    for p in range(3):
        plt.subplot(3,1, p+1)
        plt.title('Free foot on ' + legend[p])
        for i in range(horizon):
            t = np.linspace(i*ctrl.dt, (ctrl.ocp.T+ i)*ctrl.dt, ctrl.ocp.T+1)
            y = all_ocp_feet_log[i+1][:,p]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))
        plt.plot(t_mpc, feet_log_mpc[18][:, p], linewidth=0.8, color = 'tomato', marker='o')
        plt.plot(t1, feet_log_m[18][:, p], linewidth=2, color = 'lightgreen')
    plt.draw()


    plt.figure(figsize=(12, 36), dpi = 90)
    for p in range(12):
        plt.subplot(12,1, p+1)
        plt.title('u ' + str(p))
        for i in range(horizon-1):
            t = np.linspace(i*ctrl.dt, (ctrl.ocp.T+ i)*ctrl.dt, ctrl.ocp.T+1)
            y = all_u_log[i][:,p]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))

            plt.plot(t_mpc[:-1], u_mpc[:, p], linewidth=0.8, color = 'tomato', marker='o')
            plt.plot(t1, u_m[:, p], linewidth=2, color = 'lightgreen')
    plt.draw()


    plt.show()