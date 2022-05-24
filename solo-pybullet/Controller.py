from utils.loader import Solo12
import numpy as np
import ocp_Giotto as optimalControlProblem

class Results:
    def __init__(self):
        # Optimal state and control to be followed by LLC
        self.qj_des = []
        self.vj_des =  []
        self.tau_ff =  []
        self.tau = []
        self.x_m = []
        self.ocp_storage = {'xs': [], 'us': [], 'fw': [], 'qj_des': [], 'vj_des': []}


class Controller:
    def __init__(self, n_nodes, dt):
        self.dt = dt
        self.n_nodes = n_nodes

        self.solo = Solo12()
        self.results = Results()

        self.nq = self.solo.nq 
        nv = self.solo.nv
        self.q0 = self.solo.q0
        self.qj0 = self.q0[7:self.nq]
        v0 = np.zeros(nv)
        self.x0 = np.concatenate([self.q0, v0])
        self.u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])

        self.warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

        self.gait = [] \
            + [ [ 1,0,1,1 ] ] * self.n_nodes

    def create_target(self, t_):
        # Here we create the target of the control
        # FR Foot moving in a circular way

        FR_foot0 = np.array([0.1946, -0.16891, 0.0191028])
        A = np.array([0, 0, 0])
        offset = np.array([0., 0.05, 0.])
        freq = np.array([0, 0, 0])
        phase = np.array([0,0,0])

        target = []
        for t in range(self.n_nodes): target += [FR_foot0 + offset +A*np.sin(2*np.pi*freq * (t+t_)* self.dt + phase)]
        self.target = np.array(target)

    def compute_step(self, x0, x_ref, u_ref):
        self.ocp = optimalControlProblem.OCP(robot=self.solo, gait=self.gait, x0=x0, x_ref=x_ref,\
                                    u_ref = u_ref, target = self.target, dt=self.dt)

        self.ocp.solve(guess=self.warmstart)
        _, x, a, u, f, fw = self.ocp.get_results()  
        self.warmstart['xs'] = x[1:]
        self.warmstart['acs'] = a[1:]
        self.warmstart['us'] = u[1:]
        self.warmstart['fs'] = f[1:]

        self.results.ocp_storage['fw'] += [fw]
        self.results.ocp_storage['xs'] += [x]
        self.results.ocp_storage['us'] += [u]
        self.results.ocp_storage['qj_des'] += [x[:, 7: self.nq]]
        self.results.ocp_storage['vj_des'] += [x[:, self.nq + 6: ]]

        self.results.qj_des = x[:, 7: self.nq][1]
        self.results.vj_des = x[:, self.nq + 6: ][1]
        self.results.tau_ff = u[0]

            






    


