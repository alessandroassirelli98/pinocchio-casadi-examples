'''
OCP with constrained dynamics

Simple example of jump

min X,U,A,F    sum_t  ||u||**2  + || diff(x_t,x0) ||**2 + || com_z(t*) - z_target ||**2     # t* is the middle time of the flying phase
s.t
        x_0 = x0
        x_t+1  = EULER(x_t, a_t) 
        a_t = aba(q_t,v_t,u_t, f_t) # The forces f_t can be either 4 vectors or none,
                                    # Depending on the phase of the timestep
        a_feet(tc) = 0              # tc is the time in which the constraint dynamics is active
        base_link_z(t) >= 0.03            # to avoid hitting the ground with the base
        umin <= u(t) <= umax
        v_feet(t_landing) = 0
        p_feet(t_landing) = p_feet(x_0)
        v_T = x_T [nq:]  = 0
        orientation(base(x_T)) = 0
        base_link_z(T) >= 0.1
        || f_t ||**2 <= mu * || f_n ||**2   # f_t and f_n are the tangential and orthogonal component of the contact force

It takes around 1300 iter to converge without warmstart...
If warmstarted with the solution obtained by the ocp without warmstart then it takes 900 iter
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
z_target = .5
mu = 1

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('solo12')

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

# The pinocchio model is what we are really interested by.
model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
# quasi static for x0, used for warm-start and regularization
u0 = np.zeros(robot.nv)

contactNames = [ f.name for f in cmodel.frames if "FOOT" in f.name ]
allContactIds = [ i for i,f in enumerate(cmodel.frames) if "FOOT" in f.name ]
baseId = model.getFrameId('base_link')
pin.framesForwardKinematics(model,data,x0[:model.nq])
robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]

effort_limit = np.ones(robot.nv - 6) *3

########################################################################################################
### ACTION MODEL #######################################################################################
########################################################################################################

# The action model stores the computation of the dynamic -f- and cost -l- functions, both return by
# calc as self.calc(x,u) -> [xnext=f(x,u),cost=l(x,u)]
# The dynamics is obtained by RK4 integration of the pinocchio ABA function.
class CasadiActionModel:
    dt = DT
    def __init__(self,cmodel,contactIds):
        '''
        cmodel: casadi pinocchio model
        contactIds: list of contact frames IDs.
        '''
    
        
        self.cmodel = cmodel
        self.contactIds = contactIds
        self.freeIds = []
        [self.freeIds.append(idf) for idf in allContactIds if idf not in contactIds ]

        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.ndx = 2*nv
        self.nu = nv-6
        self.nv = nv
        self.ntau = nv

        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ca = casadi.SX.sym("a",self.cmodel.nv,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        cfs = [ casadi.SX.sym("f"+cmodel.frames[idf].name,3,1) for idf in self.contactIds ]
        
        ### Build force list for ABA
        forces = [ cpin.Force.Zero() for _ in self.cmodel.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        for f,idf in zip(cfs,self.contactIds):
            # Contact forces introduced in ABA as spatial forces at joint frame.
            forces[cmodel.frames[idf].parentJoint] = cmodel.frames[idf].placement * cpin.Force(f,0*f)
        self.forces = cpin.StdVec_Force()
        for f in forces:
            self.forces.append(f)
           
        acc = cpin.aba( cmodel,cdata, cx[:nq],cx[nq:],ctau,self.forces )
        
        ### Casadi MX functions
        # acceleration(x,u,f)  = ABA(q,v,tau,f) with x=q,v, tau=u, and f built using StdVec_Force syntaxt
        self.acc = casadi.Function('acc', [cx,ctau]+cfs, [ acc ])
        # com(x) = centerOfMass(x[:nq])
        self.com = casadi.Function('com', [cx],[ cpin.centerOfMass(cmodel,cdata,cx[:nq]) ])
        # integrate(x,dx) =   [q+dq,v+dv],   with the q+dq function implemented with pin.integrate.
        self.integrate = casadi.Function('xplus', [cx,cdx],
                                        [ casadi.vertcat(cpin.integrate(self.cmodel,cx[:nq],cdx[:nv]),
                                                         cx[-nv:]+cdx[-nv:]) ])
        # integrate_q(q,dq) = pin.integrate(q,dq)
        self.integrate_q = casadi.Function('qplus', [cq,cv],
                                           [ cpin.integrate(self.cmodel,cq,cv) ])
        # Lie difference(x1,x2) = [ pin.difference(q1,q2),v2-v1 ]
        self.difference = casadi.Function('xminus', [cx,cx2],
                                          [ casadi.vertcat(cpin.difference(self.cmodel,cx2[:nq],cx[:nq]),
                                                         cx2[nq:]-cx[nq:]) ])

        cpin.forwardKinematics(cmodel,cdata,cx[:nq],cx[nq:],ca)
        cpin.updateFramePlacements(cmodel,cdata)
        # Base link position
        self.base_translation = casadi.Function('base_translation', [cx], [ cdata.oMf[baseId].translation ])
        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = [ casadi.Function('foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in contactIds ]
        self.freeFeet = [ casadi.Function('free_foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in self.freeIds ]
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = [ casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[self.cdata.oMf[idf].rotation]) for idf in contactIds ]
        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = [ casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in contactIds ]
        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = [ casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameClassicalAcceleration( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in contactIds ]

        
    def calc(self,x, u, a, fs, ocp, mid_jump = False):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        fs = [ fs[3 * i : 3 * i + 3] for i,_ in enumerate(self.contactIds) ]   # split
        # Split q,v from x
        nq,nv = self.cmodel.nq,self.cmodel.nv
        # Formulate tau = [0_6,u]
        tau = casadi.vertcat( np.zeros(6), u )

        # Euler integration, using directly the acceleration <a> introduced as a slack variable.
        vnext = x[nq:] + a*dt
        qnext = self.integrate_q(x[:nq], vnext*dt)
        xnext = casadi.vertcat(qnext,vnext)
        # The acceleration <a> is then constrained to follow ABA.
        ocp.subject_to( self.acc(x,tau,*fs ) == a )

        # Cost functions:
        cost = 0
        cost += 1e-1 *casadi.sumsqr(u)
        cost += 1e2 * casadi.sumsqr( self.difference(x,x0) ) * self.dt
        
        ### OCP additional constraints

        if(mid_jump):
            cost += 1e2 * casadi.sumsqr(self.com(x)[2] - z_target )

        if not self.freeFeet:
            for afoot in self.afeet:
                ocp.subject_to( afoot(x,a) == 0 ) # Stiff contact

            for f,R in zip(fs,self.Rfeet):   # Cone constrains (flat terrain)
                fw = R(x) @ f
                ocp.subject_to(fw[2] >= 0)
                ocp.subject_to(mu**2 * fw[2]**2 >= casadi.sumsqr(fw[0:2]))
                #ocp.subject_to(mu*fw[2] >= casadi.sqrt(fw[0]**2) )     # For linear constraints
                #ocp.subject_to(mu*fw[2] >= casadi.sqrt(fw[1]**2) )     # For linear constraints


            
        return xnext,cost


########################################################################################################
### OCP PROBLEM ########################################################################################
########################################################################################################

# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
preload_steps = 10
in_air_steps = 25
contactPattern = [] \
    + [ [ 1,1,1,1 ] ] * preload_steps \
    + [ [ 0,0,0,0 ] ] * in_air_steps  \
    + [ [ 1,1,1,1 ] ] * 20  \
    + [ [ 1,1,1,1] ] 
T = len(contactPattern)-1
    
def patternToId(pattern):
    return tuple( allContactIds[i] for i,c in enumerate(pattern) if c==1 )

# In order to avoid creating to many casadi action model, we store in a dict one model for each contact pattern.
contactSequence = [ patternToId(p) for p in contactPattern ]
casadiActionModels = { contacts: CasadiActionModel(cmodel,contacts)  for contacts in set(contactSequence) }

### PROBLEM
opt_start_time  = time() 
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ casadiActionModels[contactSequence[t]] for t in range(T) ]
terminalModel = casadiActionModels[contactSequence[T]]

# Decision variables
dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
acs = [ opti.variable(model.nv) for model in runningModels ]                      # acceleration
us =  [ opti.variable(model.nu) for model in runningModels ]                      # control variable
fs =  [ opti.variable(3*len(model.contactIds) ) for model in runningModels ]      # contact force
xs =  [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0

opti.subject_to(dxs[0] == 0)
for t in range(T):
    print(contactPattern[t])
    
    if (t == preload_steps + int(in_air_steps/2) -1): # If in the mid of the jump phase
        xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti, mid_jump=True)
        print('Mid of the jump')

    elif (t == preload_steps + in_air_steps): # If it is landing
        xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti)
        for foot in runningModels[t].feet:
            opti.subject_to(foot(xs[t])[2] == foot(x0)[2] )
        for vfoot in runningModels[t].vfeet:
            opti.subject_to(vfoot(xs[t]) == 0 )
        print('Landing') 

    else:
        xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti)

    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*cmodel.nv) )  # x' = f(x,u)
    opti.subject_to(runningModels[t].base_translation(xs[t])[2] >= 0.03)
    opti.subject_to(opti.bounded(-effort_limit,  us[t], effort_limit ))
    totalcost += rcost
        
opti.subject_to( xs[T][cmodel.nq:] == 0 ) # v_T = 0
opti.subject_to(terminalModel.base_translation(xs[T])[2] >= 0.1)
opti.subject_to( xs[T][3:6] == 0 ) # Base flat

opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend

# Callback to store the history of the cost
cost_log = []
def call(i):
    global cost_log
    cost_log += [opti.debug.value(totalcost)]
opti.callback(call)

# Try to warmstart the problem
try:
    guesses = np.load("/tmp/sol.npy",allow_pickle=True).item()
    xs_g = guesses['xs']
    us_g = guesses['us']
    acs_g = guesses['acs']
    #fs_g = guesses['fs']

    def xdiff(x1,x2):
        nq = model.nq
        return np.concatenate([
            pin.difference(model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

    for x,xg in zip(dxs,xs_g): opti.set_initial(x, xdiff(x0,xg))
    for u,ug in zip(us,us_g): opti.set_initial(u,ug)
    print("Got warm start")
except:
    print("No warm start")

### SOLVE
sol = opti.solve()

# Get optimization variables
dxs_sol = np.array([ opti.value(x) for x in dxs ])
xs_sol = np.array([ opti.value(x) for x in xs ])
q_sol = xs_sol[:,: robot.nq]
us_sol = np.array([ opti.value(u) for u in us ])
acs_sol = np.array([ opti.value(a) for a in acs ])
fsol_filled = []
for i in range(len(fs)):
    if(opti.value(fs[i]).any()):
        fsol_filled += [opti.value(fs[i])]
    else:
        fsol_filled += [fsol_filled[0] *0 ]
fs_sol = [ np.split(f,f.shape[0]//3) for f in fsol_filled ]
com_log = []
[com_log.append(terminalModel.com(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
com_log = np.array(com_log)

# Gather forces in world frame
fs_world = []
for t,m in enumerate(runningModels):
    pin.framesForwardKinematics(model, data, xs_sol[t, : robot.nq])
    fs_world.append( np.concatenate([  data.oMf[idf].rotation @ fsol_filled[t][3*i:3*i+3] for i,idf in enumerate(allContactIds) ]) )
fs_world = np.array(fs_world)


print("TOTAL OPTIMIZATION TIME: ", time() - opt_start_time)

### -------------------------------------------------------------------------------------- ###
### CHECK CONTRAINTS

ha = []

nq,nv = model.nq,model.nv
# Check that all constraints are respected
for t,(m,x1,u,f,x2) in enumerate(zip(runningModels,xs_sol[:-1],us_sol,fs_sol,xs_sol[1:])):
    tau = np.concatenate([np.zeros(6),u])
    q1,v1 = x1[:nq],x1[nq:]

    vecfs = pin.StdVec_Force()
    for _ in model.joints:
        vecfs.append(pin.Force.Zero())
    for i,idf in enumerate(m.contactIds):
        frame = model.frames[idf]
        vecfs[frame.parentJoint] = frame.placement * pin.Force(f[i],np.zeros(3))

    a = pin.aba(model,data,x1[:nq],x1[nq:],tau,vecfs)
    ha.append(a.copy())
    vnext = v1+a*m.dt
    qnext = pin.integrate(model,q1,vnext*m.dt)
    xnext = np.concatenate([qnext,vnext])
    assert( np.linalg.norm(xnext-x2) < 1e-6 )


    ### Check 0 velocity of contact points
    pin.forwardKinematics(model,data,q1,v1,a)
    pin.updateFramePlacements(model,data)
    for idf in m.contactIds:
        vf = pin.getFrameVelocity(model,data,idf)
        #assert( sum(vf.linear**2) < 1e-8 )
        af = pin.getFrameClassicalAcceleration(model,data,idf,pin.LOCAL_WORLD_ALIGNED)
        assert( sum(af.linear**2) < 1e-8 )

### ------------------------------------------------------------------- ###
### PLOT

viz.play(q_sol.T, terminalModel.dt)
t_scale = np.linspace(0, (T+1)*DT, T+1)

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

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('COM position ' + legend[i])
    plt.plot(t_scale, com_log[:, i])
    if i == 2:
        plt.axhline(y = z_target, color = 'black', linestyle = '--')

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint velocity of ' + contactNames[i])
    [plt.plot(t_scale, xs_sol[:, nq + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
    plt.ylabel('Velocity [Deg/s]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint torques of ' + contactNames[i])
    [plt.plot(t_scale[:-1], us_sol[:, (3*i+jj)]) for jj in range(3) ]
    plt.axhline(y=effort_limit[0], color= 'black', linestyle = '--')
    plt.axhline(y=-effort_limit[0], color= 'black', linestyle = '--')
    plt.ylabel('Torque [N/m]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()

legend = ['F_x', 'F_y', 'F_z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Forces on ' + contactNames[i])
    [plt.plot(t_scale[:-1], fs_world[:, (3*i+jj)]) for jj in range(3) ]
    plt.ylabel('Force [N]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()


plt.show()

plt.show()

np.save(open("/tmp/sol.npy", "wb"),
        {
            "xs": xs_sol,
            "us": us_sol,
            "acs": acs_sol,
            "fs": np.array(fsol_filled)
        })
