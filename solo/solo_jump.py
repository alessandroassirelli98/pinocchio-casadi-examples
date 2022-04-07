'''
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min X,U    sum_t  ||u_t-u0||**2  + || diff(x_t,x0) ||**2 + || orientation(base(x_t)) ||**2
s.t
        x_0 = x0
        x_t+1  = EULER(x_t,u_t |  f=pin.constraintDynamics with 4 feet in contact )
        v_T = x_T [nq:]  = 0
        com(q_t)[2] = com(x_T[:nq])[2] = 0.1
        orientation(base(x_T)) == 0

So the robot should just bend to reach altitude COM 10cm while stoping at the end of the movement.

Takes ~200 iteration for IpOpt ... :(

'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from pinocchio.visualize import GepettoVisualizer
from time import time
                        


plt.style.use('seaborn')

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
DT = 0.015
z_target = 0.5
v_target = 4

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('solo12')
try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
except:
    print('No viewer')

# The pinocchio model is what we are really interested by.
model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()
viz.display(robot.q0)

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
u0 = np.zeros(robot.nv - 6)

contactIds = [ i for i,f in enumerate(cmodel.frames) if "FOOT" in f.name ]
all_contacts = contactIds
contact_frames = { i:f for i,f in enumerate(cmodel.frames) if "FOOT" in f.name }
contact_models = [ cpin.RigidConstraintModel(cpin.ContactType.CONTACT_3D,cmodel,frame.parentJoint,frame.placement)
                    for frame in contact_frames.values() ]
baseId = model.getFrameId('base_link')
contact_models_dict = { str(idx) : contact_models[i] for i,idx in enumerate(contactIds) }

prox_settings = cpin.ProximalSettings(0,1e-9,1)
for c in contact_models:
    c.corrector.Kd=.00  
    
### ACTION MODEL
# The action model stores the computation of the dynamic -f- and cost -l- functions, both return by
# calc as self.calc(x,u) -> [xnext=f(x,u),cost=l(x,u)]
# The dynamics is obtained by RK4 integration of the pinocchio ABA function.
# The cost is a sole regularization of u**2
class CasadiActionModel:
    dt = DT
    def __init__(self,cmodel,contactIds,prox_settings):

        self.contactIds = contactIds
        self.freeIds = []
        [self.freeIds.append(idf) for idf in all_contacts if idf not in contactIds ]

        self.cmodel = cmodel
        self.contact_models = []
        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.ndx = 2*nv
        self.nu = nv-6
        self.ntau = nv

        self.prox_settings = prox_settings
        [self.contact_models.append(contact_models_dict[str(idx)] ) for idx in contactIds]
        self.contact_datas = [ m.createData() for m in self.contact_models ]
        reference_frame = cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        cpin.initConstraintDynamics(cmodel,cdata, self.contact_models)
        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        
        cpin.forwardKinematics(cmodel,cdata,cx[:nq],cx[nq:])
        cpin.updateFramePlacements(cmodel,cdata)
        acc = cpin.constraintDynamics(cmodel,cdata,cx[:nq],cx[nq:],ctau,
                                      self.contact_models,self.contact_datas,self.prox_settings)

        ### Casadi MX functions
        # acceleration(x,u)
        self.acc = casadi.Function('acc', [cx,ctau], [ acc ])
        # xdot(x,u) = [ v,acc ]
        self.xdot = casadi.Function('xdot', [cx,ctau], [ casadi.vertcat(cx[nq:],acc) ])
        # com(x) = cpin.centerOfMass(q)
        self.com = casadi.Function('com', [cx],[ cpin.centerOfMass(cmodel,cdata,cx[:nq]) ])
        # Base link position
        self.base_translation = casadi.Function('base_translation', [cx], [ cdata.oMf[baseId].translation ])
        # Base link velocity
        self.base_velocity = casadi.Function('base_velocity',
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata, baseId, pin.LOCAL_WORLD_ALIGNED ).linear])
        self.freeFeet = [ casadi.Function('free_foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in self.freeIds ]

        self.feet = [ casadi.Function('foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in contactIds ]

        self.vfeet = [ casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in contactIds ]

        # integrate(x,dx)
        self.integrate = casadi.Function('plus', [cx,cdx],
                                        [ casadi.vertcat(cpin.integrate(self.cmodel,cx[:nq],cdx[:nv]),
                                                         cx[-nv:]+cdx[-nv:]) ])
        # integrate(q,v)
        self.integrate_q = casadi.Function('qplus', [cq,cv],
                                           [ cpin.integrate(self.cmodel,cq,cv) ])
        # Lie difference(x1,x2) = [ pin.difference(q1,q2),v2-v1 ]
        self.difference = casadi.Function('minus', [cx,cx2],
                                          [ casadi.vertcat(cpin.difference(self.cmodel,cx2[:nq],cx[:nq]),
                                                         cx2[nq:]-cx[nq:]) ])
        
        self.forces = [ casadi.Function('force'+cmodel.frames[idf].name,
                                       [cx,ctau],[c.contact_force.linear]) for idf,c in enumerate(self.contact_datas) ]


        
    def calc(self,x, u, ocp, mid_jump = False, start_jump = False):
        # Return xnext,cost
        
        dt = self.dt
        nq,nv = self.cmodel.nq,self.cmodel.nv
        tau = casadi.vertcat( np.zeros(6), u )

        vnext = x[nq:] + self.acc(x,tau)*dt # This is the velocity at the next timestep (Euler integration)
        qnext = self.integrate_q(x[:nq], vnext*dt)  # This is the joint position at the next timestep
                                                    # Note that here the integration has been done with 
                                                    # pin.integrate(actual joint conf, displacement of joint)
        xnext = casadi.vertcat(qnext,vnext)
        
        cost = 0
        cost += 1e-1 *casadi.sumsqr(u)
        cost += 1e1 * casadi.sumsqr( self.difference(x,x0) ) * self.dt
        #cost += 1e3 * casadi.sumsqr(x[3:6]) # Keep base flat

        if(self.freeFeet and mid_jump):
            cost += 1e5 * casadi.sumsqr(self.base_translation(x)[2] - z_target )
           #ocp.subject_to(self.base_translation(x)[2] > z_target)

        """ if(start_jump):
            cost += 1e3 * casadi.sumsqr(self.base_velocity(x)[2] - v_target )
            #ocp.subject_to(self.base_velocity(x)[2] == 3) """

        return xnext,cost
    
# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
preload_steps = 15
in_air_steps = 15
contactPattern = [] \
    + [ [ 1,1,1,1 ] ] * preload_steps \
    + [ [ 0,0,0,0 ] ] * in_air_steps  \
    + [ [ 1,1,1,1 ] ] * 5 \
    + [ [ 1,1,1,1] ] 
T = len(contactPattern)-1

timings = casadi.SX.sym('t', 2)
    
def patternToId(pattern):
    return tuple( contactIds[i] for i,c in enumerate(pattern) if c==1 )

### PROBLEM
opt_start_time  = time() 
opti = casadi.Opti()

contactSequence = [ patternToId(p) for p in contactPattern ]
casadiActionModels = { contacts: CasadiActionModel(cmodel,contacts, prox_settings=prox_settings)  for contacts in set(contactSequence) }

# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ casadiActionModels[contactSequence[t]] for t in range(T) ]
terminalModel =  casadiActionModels[contactSequence[T]]

# Decision variables
dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
us = [ opti.variable(model.nu) for model in runningModels ]                     # control variable
xs = [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(dxs[0] == 0)
for t in range(T):
    print(contactPattern[t])

    """ if (t == preload_steps): # If in the mid of the jump phase
        xnext,rcost = runningModels[t].calc(xs[t], us[t], opti, start_jump=True)
        print('Jump')
        elif (t == preload_steps + in_air_steps): # If in the mid of the jump phase
        xnext,rcost = runningModels[t].calc(xs[t], us[t], opti)
        for foot in terminalModel.feet:
            opti.subject_to(foot(xs[t])[2] == foot(x0)[2] )
        for vfoot in terminalModel.vfeet:
            opti.subject_to(vfoot(xs[t]) == 0 )
        print('Landing') """
    if (t == preload_steps + int(in_air_steps/2) -1): # If in the mid of the jump phase
        xnext,rcost = runningModels[t].calc(xs[t], us[t], opti, mid_jump=True)
        print('Jump')
    elif (t == preload_steps + in_air_steps): # If in the mid of the jump phase
        xnext,rcost = runningModels[t].calc(xs[t], us[t], opti)
        for foot in terminalModel.feet:
            opti.subject_to(foot(xs[t])[2] == foot(x0)[2] )
        for vfoot in terminalModel.vfeet:
            opti.subject_to(vfoot(xs[t]) == 0 )
        print('Landing') 
    else: 
        xnext,rcost = runningModels[t].calc(xs[t], us[t], opti)
    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*cmodel.nv) )  # x' = f(x,u)
    opti.subject_to(opti.bounded(-model.effortLimit[6 :],  us[t], model.effortLimit[6 :] ))
    totalcost += rcost

opti.subject_to( xs[T][cmodel.nq:] == 0 )  # v_T = 0
""" for foot in terminalModel.feet:
    opti.subject_to(foot(xs[T])[2] == foot(x0)[2] )
for vfoot in terminalModel.vfeet:
    opti.subject_to(vfoot(xs[T]) == 0 ) """
#opti.subject_to( xs[T][3:6] == 0 ) # Base flat
#opti.subject_to( terminalModel.base_translation(xs[T])[2] == z_target ) # Base at target height


### SOLVE
opti.minimize(totalcost)
for x in dxs: opti.set_initial(x,np.zeros(terminalModel.ndx))
for u in us: opti.set_initial(u,u0)

try:
    ### Warm start
    guesses = np.load("/tmp/sol.npy",allow_pickle=True).item()
    xs_g = guesses['xs']
    us_g = guesses['us']
    acs_g = guesses['acs']
    fs_g = guesses['fs']
    
    def xdiff(x1,x2):
        nq = model.nq
        return np.concatenate([
            pin.difference(model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

    for x,xg in zip(dxs,xs_g): opti.set_initial(x, xdiff(x0,xg))
    for u,ug in zip(us,us_g): opti.set_initial(u,ug)
    
except:
    print( 'No warm start' )


cost_log = []
def call(i):
    global cost_log
    cost_log += [opti.debug.value(totalcost)]

opti.callback(call)
opti.solver("ipopt") # set numerical backend

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve()
    dxs_sol = np.array([ opti.value(x) for x in dxs ])
    xs_sol = np.array([ opti.value(x) for x in xs ])
    q_sol = xs_sol[:, :robot.nq]
    v_sol = xs_sol[:, robot.nq :]
    us_sol = np.array([ opti.value(u) for u in us ])
    base_log = []
    [base_log.append(terminalModel.base_translation(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
    base_log = np.array(base_log)
    base_vel_log = []
    [base_vel_log.append(terminalModel.base_velocity(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
    base_vel_log = np.array(base_vel_log)
except:
    print('ERROR in convergence, plotting debug info.')
    xs_sol = np.array([ opti.debug.value(x) for x in xs ])
    us_sol = np.array([ opti.debug.value(u) for u in us ])

print("TOTAL OPTIMIZATION TIME: ", time() - opt_start_time)
### CHECK
### CHECK
### CHECK

contact_frames = { i:f for i,f in enumerate(model.frames) if "FOOT" in f.name }
contact_models = [ pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,model,frame.parentJoint,frame.placement)
                    for frame in contact_frames.values() ]
prox_settings = pin.ProximalSettings(0,1e-9,5)
for c in contact_models:
    c.corrector.Kd=0#1e-3
contact_datas = [ m.createData() for m in contact_models ]
pin.initConstraintDynamics(model,data,contact_models)
nq,nv = model.nq,model.nv


""" hiter = []

# Check that all constraints are respected
for t,(m,x1,u,x2) in enumerate(zip(runningModels,xs_sol[:-1],us_sol,xs_sol[1:])):
    tau = np.concatenate([np.zeros(6),u])
    q1,v1 = x1[:nq],x1[nq:]
    a = pin.constraintDynamics(model,data,x1[:nq],x1[nq:],tau,contact_models,contact_datas,prox_settings)
    vnext = v1+a*m.dt
    qnext = pin.integrate(model,q1,vnext*m.dt)
    xnext = np.concatenate([qnext,vnext])
    assert( np.linalg.norm(xnext-x2) < 1e-6 )
    #assert( prox_settings.iter<=2 )
    hiter.append(prox_settings.iter) 
 """

### ------------------------------------------------ ###
# PLOT

plt.figure(figsize=(12, 6), dpi = 90)
plt.subplot(1,2,1)
plt.title('Residuals')
plt.semilogy(opti.stats()['iterations']['inf_du'])
plt.semilogy(opti.stats()['iterations']['inf_pr'])
plt.legend(['dual', 'primal'])
plt.subplot(1,2,2)
plt.title('cost')
plt.plot(cost_log)
plt.draw()

viz.play(q_sol.T, terminalModel.dt)

legend = ['x', 'y', 'z']
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('Base link position_' + legend[i])
    plt.plot(base_log[:, i])
    if i == 2:
        plt.axhline(y = z_target, color = 'black', linestyle = '--')
plt.draw()

legend = ['x', 'y', 'z']
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('Base link velocity_' + legend[i])
    plt.plot(base_vel_log[:, i])
    if i == 2:
        plt.axhline(y = v_target, color = 'black', linestyle = '--')

plt.draw()

plt.show()

np.save(open("/tmp/sol.npy", "wb"),
        {
            "xs": xs_sol,
            "us": us_sol,
        })
