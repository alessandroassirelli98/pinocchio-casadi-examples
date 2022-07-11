'''
OCP with constrained dynamics

Simple example of walk  

It takes around 22 iter to converge without warmstart
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
import os


plt.style.use('seaborn')
path = os.getcwd()

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
DT = 0.015
mu = 1
kx = 20
ky = 20
k = np.array([kx, ky])
lin_vel_weight = np.array([10, 10, 10])
ang_vel_weight = np.array([10, 10, 10])
force_reg_weight = 1e-2
control_weight = 1e1
base_reg_cost = 1e1
joints_reg_cost = 1e2
sw_feet_reg_cost = 1e1

timestep_per_phase = 18
v_lin_target = np.array([2, 0, 0])
v_ang_target = np.array([0, 0, 0])
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

model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
#x0[3:7] = np.array([0,0,1,0])
# quasi static for x0, used for warm-start and regularization
u0 =  np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                0.25720939, -0.51441314])  ### quasi-static for x0
a0 = np.zeros(robot.nv)
fs0 = [np.ones(3)*0, np.ones(3) *0]


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
        self.nq = nq = cmodel.nq
        self.nv = nv =cmodel.nv
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
        reference_frame = pin.LOCAL
        # Base link position
        self.baseTranslation = casadi.Function('base_translation', [cx], [ cdata.oMf[baseId].translation ])
        self.baseRotation = casadi.Function('base_rotation', [cx], [ cdata.oMf[baseId].rotation ])
        # Base velocity
        self.baseVelocityLin = casadi.Function('base_velocity_linear', [cx], [cpin.getFrameVelocity( cmodel,cdata,baseId,reference_frame).linear])
        self.baseVelocityAng = casadi.Function('base_velocity_angular', [cx], [cpin.getFrameVelocity( cmodel,cdata,baseId,reference_frame).angular])
        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = {idf : casadi.Function('foot'+cmodel.frames[idf].name,
                                     [cx], [cdata.oMf[idf].translation]) for idf in allContactIds }
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = {idf: casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[self.cdata.oMf[idf].rotation]) for idf in allContactIds }
        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = {idf: casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in allContactIds }
        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = {idf: casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameClassicalAcceleration( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in allContactIds}

    def cost(self, x, u, fs):
        cost = 0
        cost += control_weight *casadi.sumsqr(u - u0) *self.dt
        cost += base_reg_cost *casadi.sumsqr(x[3:7] - x0[3:7]) * self.dt
        cost += joints_reg_cost *casadi.sumsqr(x[7 : self.cmodel.nq] - x0[7: self.cmodel.nq]) *self.dt

        for i, stFoot in enumerate(self.contactIds):
            # Friction cone
            R = self.Rfeet[stFoot](x)
            f = fs[i]
            fw = R @ f
            cost += force_reg_weight * casadi.sumsqr(fw[2] - robotweight/len(self.contactIds)) * self.dt
        for sw_foot in self.freeIds:
            cost += sw_feet_reg_cost * casadi.sumsqr(self.vfeet[sw_foot](x)[0:2]) * self.dt

        cost += casadi.sumsqr(lin_vel_weight*(self.baseVelocityLin(x) - v_lin_target)) * self.dt
        cost += casadi.sumsqr(ang_vel_weight*(self.baseVelocityAng(x) - v_ang_target)) * self.dt
        return cost

    def calc(self, x, a, u, fs, ocp):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        # Split q,v from x
        nq,nv = self.cmodel.nq,self.cmodel.nv
        # Formulate tau = [0_6,u]
        tau = casadi.vertcat( np.zeros(6), u )

        # Euler integration, using directly the acceleration <a> introduced as a slack variable.
        vnext = x[nq:] + a*dt
        qnext = self.integrate_q(x[:nq], vnext*dt)
        xnext = casadi.vertcat(qnext,vnext)

        # The acceleration <a> is then constrained to follow ABA.
        ocp.subject_to( self.acc(x,tau, *fs ) == a )

        # Cost functions:
        cost = self.cost(x, u, fs)

        # Contact constraints
        for i, stFoot in enumerate(self.contactIds):
            ocp.subject_to(self.afeet[stFoot](x,a) == 0) # stiff contact

            # Friction cone
            R = self.Rfeet[stFoot](x)
            f = fs[i]
            fw = R @ f
            ocp.subject_to(fw[2] >= 0)
            ocp.subject_to(mu**2 * fw[2]**2 >= casadi.sumsqr(fw[0:2]))
            
        
        for sw_foot in self.freeIds:
            ocp.subject_to(self.feet[sw_foot](x)[2] >= self.feet[sw_foot](x0)[2])
            ocp.subject_to(self.vfeet[sw_foot](x)[0:2] <= k* self.feet[sw_foot](x)[2])
            
        return xnext,cost


########################################################################################################
### OCP PROBLEM ########################################################################################
########################################################################################################

# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
contactPattern = [] \
    + [ [ 1,0,0,1 ] ] * timestep_per_phase  \
    + [ [ 0,1,1,0 ] ] * timestep_per_phase

contactPattern = contactPattern*3
#contactPattern = np.roll(contactPattern, -6, axis=0)

T = len(contactPattern) - 1
    
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
dxs = [ opti.variable(m.ndx) for m in runningModels+[terminalModel] ]     # state variable
acs = [ opti.variable(m.nv) for m in runningModels ]                      # acceleration
us =  [ opti.variable(m.nu) for m in runningModels ]                      # control variable
fs = []
for m in runningModels:
    f_tmp = [opti.variable(3) for _ in range(len(m.contactIds)) ]
    fs.append(f_tmp)

xs =  [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0

opti.subject_to(dxs[0] == 0)
for t in range(T):
    print(contactSequence[t])

    if (contactSequence[t] != contactSequence [t-1] and t >=1): # If it is landing
        for stFoot in runningModels[t].contactIds:
            opti.subject_to(runningModels[t].feet[stFoot](xs[t])[2] == runningModels[t].feet[stFoot](x0)[2] )
        for stFoot in runningModels[t].contactIds:
            opti.subject_to(runningModels[t].vfeet[stFoot](xs[t]) == 0)

        print('Landing on ', str(runningModels[t].contactIds)) 

    xnext,rcost = runningModels[t].calc(xs[t], acs[t], us[t], fs[t], opti)
    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*cmodel.nv) )  # x' = f(x,u)
    opti.subject_to(opti.bounded(-effort_limit,  us[t], effort_limit ))
    totalcost += rcost

opti.subject_to(xs[T][cmodel.nq :] == 0)
opti.minimize(totalcost)

p_opts = {"expand": True}
s_opts = {}
opti.solver("ipopt",p_opts,
                    s_opts)


# Callback to store the history of the cost
cost_log = []
def call(i):
    global cost_log
    cost_log += [opti.debug.value(totalcost)]
opti.callback(call)

# Try to warmstart the problem
try:
    guesses = np.load(path + '/sol.npy',allow_pickle=True).item()
    xs_g = guesses['xs']
    us_g = guesses['us']
    acs_g = guesses['acs']
    fs_g = guesses['fs']

    def xdiff(x1,x2):
        nq = model.nq
        return np.concatenate([
            pin.difference(model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

    for x,xg in zip(dxs,xs_g): opti.set_initial(x, xdiff(x0,xg))
    for a,ag in zip(acs,acs_g): opti.set_initial(a, ag)
    for u,ug in zip(us,us_g): opti.set_initial(u,ug)
    for f, fg in zip(fs, fs_g):
        [opti.set_initial(f[i], fg[i]) for i in range(len(f)) ]
    print('Got warmstart')

except:
    print('No warmstart')

### SOLVE
sol = opti.solve_limited()

# Get optimization variables
dxs_sol = np.array([ opti.value(x) for x in dxs ])
xs_sol = np.array([ opti.value(x) for x in xs ])
q_sol = xs_sol[:,: robot.nq]
us_sol = np.array([ opti.value(u) for u in us ])
acs_sol = np.array([ opti.value(a) for a in acs ])
fsol = {name: [] for name in allContactIds}
fsol_to_ws = []
for t in range(T):
    for i, (st_foot, sw_foot) in enumerate(\
        zip(runningModels[t].contactIds, runningModels[t].freeIds )):
        fsol[st_foot].append(opti.value(fs[t][i]))
        fsol[sw_foot].append(np.zeros(3))
    fsol_to_ws.append([opti.value(fs[t][i]) for i in range(len(fs[t]))])

for foot in fsol: fsol[foot] = np.array(fsol[foot])

base_pos_log = []
[base_pos_log.append(terminalModel.baseTranslation(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
base_pos_log = np.array(base_pos_log)

base_vel_log = []
[base_vel_log.append(terminalModel.baseVelocityLin(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
base_vel_log = np.array(base_vel_log)

feet_log = {i:[] for i in allContactIds}
for foot in feet_log:
    tmp = []
    for i in range(len(xs_sol)):
        tmp += [ terminalModel.feet[foot](xs_sol[i]).full()[:, 0] ]
    feet_log[foot] = np.array(tmp)

feet_vel_log = {i:[] for i in allContactIds}
for foot in feet_vel_log:
    tmp = []
    for i in range(len(xs_sol)):
        tmp += [ terminalModel.vfeet[foot](xs_sol[i]).full()[:, 0] ]
    feet_vel_log[foot] = np.array(tmp)


# Gather forces in world frame
model = robot.model
fs_world = {name: [] for name in allContactIds}
for t in range(T):
    pin.framesForwardKinematics(model, data, xs_sol[t, : robot.nq])
    [fs_world[foot].append(data.oMf[foot].rotation @ fsol[foot][t]) for foot in fs_world]
for foot in fs_world: fs_world[foot] = np.array(fs_world[foot])


print("TOTAL OPTIMIZATION TIME: ", time() - opt_start_time)

### -------------------------------------------------------------------------------------- ###
### CHECK CONTRAINTS

ha = []

nq,nv = model.nq,model.nv
# Check that all constraints are respected
for t,(m,x1,u,f,x2) in enumerate(zip(runningModels,xs_sol[:-1],us_sol,fsol_to_ws,xs_sol[1:])):
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
    plt.title('Base position ' + legend[i])
    plt.plot(t_scale, base_pos_log[:, i])

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('Base velocity ' + legend[i])
    plt.plot(t_scale, base_vel_log[:, i])
    if i == 0:
        plt.axhline(y=v_lin_target[0], color= 'black', linestyle = '--')

legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in allContactIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot position on ' + legend[i])
        plt.plot(t_scale, feet_log[foot][:, i])
        plt.legend(contactNames)
        
legend = ['x', 'y', 'z']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(3):
    for foot in allContactIds:
        plt.subplot(3,1,i+1)
        plt.title('Foot velocity on ' + legend[i])
        plt.plot(t_scale, feet_vel_log[foot][:, i])
        plt.legend(contactNames)

legend = ['Hip', 'Shoulder', 'Knee']
plt.figure(figsize=(12, 6), dpi = 90)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('Joint velocity of ' + contactNames[i])
    [plt.plot(t_scale, xs_sol[:, robot.nq + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
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
for i, foot in enumerate(fs_world):
    plt.subplot(2,2,i+1)
    plt.title('Forces on ' + contactNames[i])
    [plt.plot(t_scale[:-1], fs_world[foot][:, jj]) for jj in range(3) ]
    plt.ylabel('Force [N]')
    plt.xlabel('t[s]')
    plt.legend(legend)
plt.draw()


plt.show()


np.save(open(path + '/sol.npy', "wb"),
        {
            "xs": xs_sol,
            "us": us_sol,
            "acs": acs_sol,
            "fs": np.array(fsol_to_ws)
        })
