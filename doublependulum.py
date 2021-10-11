import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data
import matplotlib.pyplot as plt; plt.ion()
from pinocchio.visualize import GepettoVisualizer

### HYPER PARAMETERS
T = 100
dt = 0.02
x0 = np.array([-np.pi,0., 0., 0.])
costWeightsRunning = np.array([])  # sin, 1-cos, y, ydot, thdot, f
costWeightsTerminal = np.array([])

### LOAD AND DISPLAY PENDULUM
robot_model = example_robot_data.load('double_pendulum')
robot_model.model.jointPlacements[1] = pin.SE3(pin.utils.rotate('z',-np.pi/2),np.zeros(3))
for g in robot_model.visual_model.geometryObjects:
    if g.parentJoint == 0:
        M = g.placement
        M.rotation = pin.utils.rotate('z',-np.pi/2)
        g.placement = M
viz = pin.visualize.GepettoVisualizer(robot_model.model, robot_model.collision_model, robot_model.visual_model)
try:
    viz.initViewer()
    viz.loadViewerModel()
    viz.viewer.gui.setBackgroundColor1(viz.windowID,[0.,0.,0.,1.])
    viz.viewer.gui.setBackgroundColor2(viz.windowID,[0.,0.,0.,1.])
except ImportError as err:
    print("Error while initializing the viewer. It seems you should do something about it")
    viz = None
  
### MODEL
model = robot_model.model

class CasadiActionModelDoublePendulum:
    def __init__(self,model,weights):
        self.weights = weights.copy()

        self.model = model
        #data = model.createData()

        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()

        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        ctau = casadi.SX.sym("tau",cmodel.nv,1)
        x = casadi.vertcat(cq, cv)
        u = casadi.vertcat(ctau)
        #dt = casadi.SX.sym("dt", 1)
        
        # defining the forward dynamics expression
        self.acc = a = cpin.aba(cmodel,cdata,cq,cv,ctau)
        v_next = cv + a*dt
        # q_next = cpin.integrate(cmodel,cq,v_next*dt)
        x_dot = casadi.vertcat(v_next,a)
        # defining a casadi function that outputs the forward dynamics
        self.xdot = casadi.Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])
                
    def calc(self,x, u):
        # Runge-Kutta 4 integration
        F = self.xdot
        k1 = F(x,           u)
        k2 = F(x + dt/2*k1, u)
        k3 = F(x + dt/2*k2, u)
        k4 = F(x + dt*k3,   u)
        xnext = x + dt/6*(k1+2*k2+2*k3+k4)

        cost = u.T@u
        return xnext,cost
    
### PROBLEM
opti = casadi.Opti()
X = opti.variable(model.nq+model.nv, T + 1) # state trajectory
U = opti.variable(model.nv, T)     # control trajectory

runningModels = [ CasadiActionModelDoublePendulum(model,costWeightsRunning) for t in range(T) ]
terminalModel = CasadiActionModelDoublePendulum(model,costWeightsTerminal)

totalcost = 0
opti.subject_to(X[:, 0] == x0)
for t in range(T):
    xnext,rcost = runningModels[t].calc(X[:, t], U[:, t])
    opti.subject_to(X[:, t + 1] == xnext )
    totalcost += rcost
#totalcost += terminalModel.calc(X[:,-1],None)[1]
opti.subject_to(X[:, -1] == np.zeros(4))
#opti.subject_to(opti.bounded(-10, U, 10)) # control is limited

opti.minimize(totalcost)

### SOLVE
opti.set_initial(X, 0)
opti.solver("ipopt") # set numerical backend
try:
    sol = opti.solve_limited()
    xs = sol.value(X)
    us = sol.value(U)
except:
    print('ERROR in convergence, plotting debug info.')
    xs = opti.debug.value(X)
    us = opti.debug.value(U)

### PLOT AND VIZ
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3) #, constrained_layout=True)
ax0.plot(xs[:model.nq,:].T)
ax0.set_ylabel('q')
ax1.plot(xs[model.nq:,:].T)
ax1.set_ylabel('v')
ax2.plot(us.T)
ax2.set_ylabel('u')

if viz:
    viz.play(xs[:model.nq,:], dt)
