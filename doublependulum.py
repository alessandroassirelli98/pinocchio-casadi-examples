'''
Solve a pinocchio-based double-pendulum problem, formulated as multiple shooting with RK4 integration.
min_xs,us    sum_0^T-1  l(x,u)
       s.t.     x_0 = x0
                x_t+1 = f(x_t,u_t)  for all t=0..T-1
                c(x_T) = 0

where xs = [ x_0 ... x_T ] and us = [ u_0 ... u_T-1 ]
      l(x,u) = l(u) = u**2  is the running cost.
      f  is the integrated dynamics, writen as xnext = x + RK4(ABA(q,v,tau) with q=x[:NQ], v=x[NQ:] and TAU=U
      c is a terminal (hard) constraints asking the pendulum to be standing up with 0 velocity.

The model is stored in a so-called action-model, mostly defining the [xnext,cost] = calc(x,u) concatenating
l and f functions.

As a results, it plots the state and control trajectories and display the movement in gepetto viewer.
'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
from utils.double_pendulum import create_double_pendulum_model
import matplotlib.pyplot as plt; plt.ion()
from pinocchio.visualize import GepettoVisualizer

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
T = 100
x0 = np.array([-np.pi,0., 0., 0.])
costWeightsRunning = np.array([])  # sin, 1-cos, y, ydot, thdot, f
costWeightsTerminal = np.array([])
WARMSTART = "doublependulum_unbounded_solution.npy"

### LOAD AND DISPLAY PENDULUM
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot,viz = create_double_pendulum_model()
# The pinocchio model is what we are really interested by.
model = robot.model

### ACTION MODEL
# The action model stores the computation of the dynamic -f- and cost -l- functions, both return by
# calc as self.calc(x,u) -> [xnext=f(x,u),cost=l(x,u)]
# The dynamics is obtained by RK4 integration of the pinocchio ABA function.
# The cost is a sole regularization of u**2
class CasadiActionModelDoublePendulum:
    dt = 0.02
    length = .3  # pendulum elongated dimension
    def __init__(self,model,weights):
        self.weights = weights.copy()

        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.nu = nv

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        self.xdot = casadi.Function('xdot', [cx,cu], [ casadi.vertcat(cx[nq:], cpin.aba(cmodel,cdata,cx[:nq],cx[nq:],cu)) ])

        # The self.tip will be a casadi function mapping: state -> X-Z position of the pendulum tip
        cpin.framesForwardKinematics(self.cmodel,self.cdata,cx[:nq])
        self.tip = casadi.Function('tip', [cx], [ self.cdata.oMf[-1].translation[[0,2]] ])

    def calc(self,x, u):
        # Runge-Kutta 4 integration
        F = self.xdot; dt = self.dt
        k1 = F(x,           u)
        k2 = F(x + dt/2*k1, u)
        k3 = F(x + dt/2*k2, u)
        k4 = F(x + dt*k3,   u)
        xnext = x + dt/6*(k1+2*k2+2*k3+k4)

        cost = u.T@u
        return xnext,cost
    
### PROBLEM
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ CasadiActionModelDoublePendulum(model,costWeightsRunning) for t in range(T) ]
terminalModel = CasadiActionModelDoublePendulum(model,costWeightsTerminal)

# Decision variables
xs = [ opti.variable(model.nx) for model in runningModels+[terminalModel] ]     # state variable
us = [ opti.variable(model.nu) for model in runningModels ]                     # control variable

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(xs[0] == x0)
for t in range(T):
    xnext,rcost = runningModels[t].calc(xs[t], us[t])
    opti.subject_to(xs[t + 1] == xnext )
    totalcost += rcost
    opti.subject_to(opti.bounded(-.05, us[t][0], .05)) # control is limited
    
# Additional terminal constraint
opti.subject_to(xs[-1][model.nq:] == 0)  # 0 terminal value
opti.subject_to(terminalModel.tip(xs[T])==[0,terminalModel.length]) # tip of pendulum at max altitude

### SOLVE
opti.minimize(totalcost)
# Warm start with a previously computed solution using no torque bound.
try:
    xs0,us0=np.load(WARMSTART,allow_pickle=True)
    for x,xg in zip(xs,xs0): opti.set_initial(x,xg)
    for u,ug in zip(us,us0): opti.set_initial(u,ug)
except:
    print('No warm start file provided, searching from scratch (cold start)')
opti.solver("ipopt") # set numerical backend
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.value(u) for u in us ])
except:
    print('ERROR in convergence, plotting debug info.')
    xs_sol = np.array([ opti.debug.value(x) for x in xs ])
    us_sol = np.array([ opti.debug.value(u) for u in us ])

### PLOT AND VIZ
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3) #, constrained_layout=True)
ax0.plot(xs_sol[:,:model.nq])
ax0.set_ylabel('q')
ax0.legend(['1','2'])
ax1.plot(xs_sol[:,model.nq:])
ax1.set_ylabel('v')
ax1.legend(['1','2'])
ax2.plot(us_sol)
ax2.set_ylabel('u')
ax2.legend(['1','2'])
if viz: viz.play(xs_sol[:,:model.nq].T, CasadiActionModelDoublePendulum.dt)

### SAVE
#np.save(open(WARMSTART,'wb'),[xs_sol,us_sol])
