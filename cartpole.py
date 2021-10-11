import casadi
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

### CARTPOLE HYPERPARAMS
T = 50
x0 = np.array([0., 2.0, 0., 0.])
#costWeightsRunning = np.array([1., 1., 0.1, 0.001, 0.001, 1.])  # sin, 1-cos, x, xdot, thdot, f
costWeightsRunning = np.array([1., 1., 0.1, 0.001, 0.001, 1.])  # sin, 1-cos, x, xdot, thdot, f
costWeightsTerminal = np.array([100, 100, 1, 0.1, 0.01, 0.0001])

### MODEL
class CasadiActionModelCartPole:
    m1 = 1.
    m2 = .1
    l = .5
    g = 9.81
    dt = 5e-2

    def __init__(self,weights):
        self.weights = weights.copy()
        
    def calc(self,x, u):
        
        y     = x[0]
        th    = x[1]
        ydot  = x[2]
        thdot = x[3]
        f     = u[0] if u is not None else 0
        s, c  = casadi.sin(th), casadi.cos(th)
        
        ### Dynamics
        m = self.m1 + self.m2
        mu = self.m1 + self.m2 * s**2
        yddot = (f + self.m2 * c * s * self.g - self.m2 * self.l * s * thdot**2) / mu
        thddot = (c * f / self.l + m * self.g * s / self.l - self.m2 * c * s * thdot**2) / mu
        a = casadi.vertcat(thddot, yddot)
        x_next = casadi.vertcat(x[:2] + x[-2:] * self.dt, x[-2:] + a * self.dt)

        ### Cost
        residual = casadi.vertcat(s, 1 - c, y, ydot, thdot, f)
        wresidual = residual * self.weights
        rcost = wresidual.T @ residual
        
        return x_next,rcost

### PROBLEM
opti = casadi.Opti()
X = opti.variable(4, T + 1)   # state variable
U = opti.variable(1, T)       # control variable

runningModels = [ CasadiActionModelCartPole(costWeightsRunning) for t in range(T) ]
terminalModel = CasadiActionModelCartPole(costWeightsTerminal)

totalcost = 0
opti.subject_to(X[:, 0] == x0)
for t in range(T):
    xnext,rcost = runningModels[t].calc(X[:, t], U[:, t])
    opti.subject_to(X[:, t + 1] == xnext )
    totalcost += rcost
#totalcost += terminalModel.calc(X[:,-1],None)[1]
#opti.subject_to(X[:, -1] == np.zeros(4))
opti.subject_to(X[1, -1] == 0)
opti.subject_to(X[3, -1] == 0)

### SOLVE
opti.minimize(totalcost)
opti.solver('ipopt')
sol = opti.solve()

x_sol = opti.value(X)
u_sol = opti.value(U)

plt.figure( 'x')
[plt.plot(x) for x in x_sol]
plt.legend(['x', '$\\theta$', '$\dot{x}$', '$\dot{\\theta}$'])
plt.show()

plt.figure( 'u')
[plt.plot(u_sol)]
plt.legend(['u'])
plt.show()
