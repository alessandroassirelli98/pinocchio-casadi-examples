'''
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min X,U,A,F    sum_t  ||u_t-u0||**2  + || diff(x_t,x0) ||**2 + || orientation(base(x_t)) ||**2
s.t
        x_0 = x0
        x_t+1  = EULER(x_t,u_t,f_t |  f=a_t )
        a_t = aba(q_t,v_t,u_t)
        a_feet(t) = 0
        v_T = x_T [nq:]  = 0
        com(q_t)[2] = com(x_T[:nq])[2] = 0.1
        orientation(base(x_T)) == 0

So the robot should just bend to reach altitude COM 80cm while stoping at the end of the movement.
The acceleration is introduced as an explicit (slack) variable to simplify the formation of the contact constraint.
Contacts are 6d, yet only with normal forces positive, and 6d acceleration constraints. 

'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from numpy.linalg import norm,inv,pinv,svd,eig
plt.style.use('seaborn')
pin.SE3.__repr__=pin.SE3.__str__

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
DT = 0.010
z_target = 1.05
z_target = .8

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('talos_legs')


try:
    viz = pin.visualize.GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

# The pinocchio model is what we are really interested by.
model = robot.model
model.q0 = robot.q0
cmodel = cpin.Model(robot.model)
data = model.createData()

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
# quasi static for x0, used for warm-start and regularization
u0 =  np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
               0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
               0.25720939, -0.51441314])

contactIds = [ i for i,f in enumerate(cmodel.frames) if "sole_link" in f.name ]
baseId = model.getFrameId('root_joint')
pin.framesForwardKinematics(model,data,x0[:model.nq])
robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]

# pin.forwardKinematics(model,data,model.q0)
# fs0 = pin.StdVec_Force (model.njoints,pin.Force.Zero())
# for cid in contactIds:
#     jid = model.frames[cid].parentJoint
#     fs0[jid] = pin.Force(np.concatenate([data.oMi[jid].rotation.T@np.array([0,0,totalmass*9.81/2]),
#                                          [0.545,-2.93,-0.005]]))
# tau0 = pin.rnea(model,data,model.q0,np.zeros(model.nv),np.zeros(model.nv),fs0)
# assert(norm(tau0[:6])<0.1 and "Heuristic for tau0 is wrong, modify it" )

########################################################################################################
### HELPERS ############################################################################################
########################################################################################################

def abadict(model,data,q,v,tau,fdict):
    '''fdict are given with frame indexes'''
    fvec = pin.StdVec_Force (model.njoints,pin.Force.Zero())
    for cid,fi in fdict.items():
        jid = model.frames[cid].parentJoint
        fvec[jid] = model.frames[cid].placement * pin.Force(fi)
    return pin.aba(model,data,q,v,tau,fvec)

def torqueReg(q,cids,check=False):
    '''
    Compute torque for 0 movement with some contacts
    return tau,[f1,...fnc]
    '''
    pin.computeAllTerms(model,data,q,np.zeros(model.nv))
    J = np.vstack([ pin.getFrameJacobian(model,data,cid,pin.LOCAL)  for cid in cids ])


    ### min || Ax-b|| st Cx=d    =>>> x = xc + pinv(AP) ( b-A xc) with  xc = pinv(C)*d and P=I-pinv(C)*C
    ###### with x=(tau,f), C=[S.T J.T], d=nle, A = [0 I_6nc] and b = [0,0,mg/nc,0,0,0]*nc
    C = np.hstack([ np.eye(model.nv)[6:,:].T, J.T ])
    d = data.nle
    A = np.eye(C.shape[1])[model.nv-6:,:]
    b = np.concatenate( [ np.array([0,0,robotweight/len(cids),0,0,0]) ]*len(cids) )
    w = [1,1,0.1,1000,1000,2]
    W = np.diag(w*len(cids))
    A=W@A
    b=W@b
    
    Cp = pinv(C)
    xopt = Cp@d
    P = np.eye(model.nv-6+len(cids)*6) - Cp@C
    xopt += pinv(A@P,1e-1)@( b-A@xopt )

    tau = xopt[:model.nv-6]
    forces = [ xopt[model.nv-6+6*i:model.nv+6*i] for i in range(len(cids)) ]

    if check:
        assert(norm(abadict(model,data,q,np.zeros(model.nv),np.concatenate([np.zeros(6),tau]),
                            { cid: fi for cid,fi in zip(cids,forces)})) < 1e-6 )

    return tau,forces


check_u0,check_fs0 = torqueReg(model.q0,contactIds)
check_fvec0 = pin.StdVec_Force (model.njoints,pin.Force.Zero())
for fi,cid in zip(check_fs0,contactIds):
    jid = model.frames[cid].parentJoint
    check_fvec0[jid] = model.frames[cid].placement * pin.Force(fi)
check_acc = pin.aba(model,data,model.q0,np.zeros(model.nv),np.concatenate([np.zeros(6),check_u0]),check_fvec0)
assert(norm(check_acc)<1e-6)




########################################################################################################
### ACTION MODEL #######################################################################################
########################################################################################################

# The action model stores the computation of the dynamic -f- and cost -l- functions, both return by
# calc as self.calc(x,u) -> [xnext=f(x,u),cost=l(x,u)]
# The dynamics is obtained by RK4 integration of the pinocchio ABA function.
# The cost is a sole regularization of u**2
class CasadiActionModel:
    dt = DT
    def __init__(self,cmodel,contactIds):
        '''
        cmodel: casadi pinocchio model
        contactIds: list of contact frames IDs.
        '''
    
        
        self.cmodel = cmodel
        self.contactIds = contactIds
        
        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.ndx = 2*nv
        self.nu = nv-6
        self.nv = nv
        self.ntau = nv

        self.u0, self.fs0 = torqueReg(model.q0,contactIds)
        self.uerror = u0
        
        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ca = casadi.SX.sym("a",self.cmodel.nv,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        cfs = [ casadi.SX.sym("f"+cmodel.frames[idf].name,6,1) for idf in self.contactIds ]
        
        ### Build force list for ABA
        forces = [ cpin.Force.Zero() for __j in self.cmodel.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        for f,idf in zip(cfs,self.contactIds):
            # Contact forces introduced in ABA as spatial forces at joint frame.
            print('force = ' ,f)
            forces[cmodel.frames[idf].parentJoint] = cmodel.frames[idf].placement * cpin.Force(f)
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
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = [ casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[self.cdata.oMf[idf].rotation]) for idf in contactIds ]
        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = [ casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds ]
        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = [ casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameAcceleration( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds ]

        
    def calc(self,x, u, a, fs, ocp):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        fs = [ fs[6 * i : 6 * i + 6] for i,_ in enumerate(self.contactIds) ]   # split
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
        cost += 1e-2*casadi.sumsqr(u-self.u0) * self.dt
        cost += 1e-1*casadi.sumsqr( self.difference(x,x0) ) * self.dt
        cost += 1e2 * casadi.sumsqr(x[3:6]) # Keep base flat
        cost += 1e-3*sum( [ casadi.sumsqr(f-fref) for f,fref in zip(fs,self.fs0) ] ) * self.dt
        
        ### OCP additional constraints
        for afoot in self.afeet:
            ocp.subject_to( afoot(x,a) == 0 )
            pass
            
        for f,R in zip(fs,self.Rfeet):   # for cone constrains (flat terrain)
            fw = R(x) @ f[:3]
            ocp.subject_to(fw[2]>=0)
            
        return xnext,cost


########################################################################################################
### OCP PROBLEM ########################################################################################
########################################################################################################

# [FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT]
contactPattern = [] \
    + [ [ 1,1 ] ] * 1 \
    + [ [ 1,1 ] ] * 50  \
    + [ [ 1,1 ] ] * 0 \
    + [ [ 1,1 ] ]
T = len(contactPattern)-1
    
def patternToId(pattern):
    return tuple( contactIds[i] for i,c in enumerate(pattern) if c==1 )

# In order to avoid creating to many casadi action model, we store in a dict one model for each contact pattern.
contactSequence = [ patternToId(p) for p in contactPattern ]
casadiActionModels = { contacts: CasadiActionModel(cmodel,contacts)  for contacts in set(contactSequence) }

### PROBLEM
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ casadiActionModels[contactSequence[t]] for t in range(T) ]
terminalModel = casadiActionModels[contactSequence[T]]

# Decision variables
dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
acs = [ opti.variable(model.nv) for model in runningModels ]                      # acceleration
us =  [ opti.variable(model.nu) for model in runningModels ]                      # control variable
fs =  [ opti.variable(6*len(model.contactIds) ) for model in runningModels ]      # contact force
xs =  [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(dxs[0] == 0)

for t in range(T):
    
    xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti )
    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == 0 ) # x' = f(x,u)

    totalcost += rcost

    for i,c in enumerate(runningModels[t].contactIds):
        if c not in runningModels[t-1].contactIds:
            print(f'Impact for foot {i}:{c} at time {t}')
            opti.subject_to( runningModels[t].feet[i](xs[t])[2] == 0)
    
        
#opti.subject_to( xs[T-1][cmodel.nq:] == 0 ) # v_T = 0
#opti.subject_to( xs[T][cmodel.nq:] == 0 ) # v_T = 0
totalcost += 10000*casadi.sumsqr( xs[T][cmodel.nq:] )
#opti.subject_to( terminalModel.base_translation(xs[T])[2] == z_target ) # z_com(T) = ref
totalcost += 1000*casadi.sumsqr( terminalModel.base_translation(xs[T])[2] - z_target )
#opti.subject_to( xs[T][3:6] == 0 ) # Base flat

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend

cost_log = []
def call(i):
    global cost_log
    cost_log += [opti.debug.value(totalcost)]

opti.callback(call)

for dx in dxs:
    opti.set_initial(dx,np.zeros(model.nv*2))
for f,u,r in zip(fs,us,runningModels):
    opti.set_initial(u,r.u0)
    nc=len(r.contactIds)
    opti.set_initial(f, np.concatenate(r.fs0))
#for u in us:
#    opti.set_initial(u,u0)
#for f,r in zip(fs,runningModels):
#    nc=len(r.contactIds)
#    fguess = np.array([0,0,totalmass/nc,0,0,0])
#    opti.set_initial(f, np.concatenate([fguess]*nc))
    
    
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve()
    dxs_sol = np.array([ opti.value(x) for x in dxs ])
    xs_sol = np.array([ opti.value(x) for x in xs ])
    q_sol = xs_sol[:,: robot.nq]
    us_sol = np.array([ opti.value(u) for u in us ])
    acs_sol = np.array([ opti.value(a) for a in acs ])
    fs_sol = [ np.split(opti.value(f),f.shape[0]//6) for f in fs ]
    base_log = []
    [base_log.append(terminalModel.base_translation(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
    base_log = np.array(base_log)
    ### We reorganize fs_sol to have 4 contacts for each timestep, adding a 0 force when needed.
    fs_sol0 = [ np.concatenate([ \
                                f[runningModels[t].contactIds.index(c) ] if c in runningModels[t].contactIds
                                else np.zeros(3)
                                for i,c in enumerate(contactIds)   ])
               for t,f in enumerate(fs_sol) ]
    
except:
    print('ERROR in convergence, plotting debug info.')
    dxs_sol = np.array([ opti.debug.value(x) for x in dxs ])
    xs_sol = np.array([ opti.value(x) for x in xs ])
    q_sol = xs_sol[:,: robot.nq]
    us_sol = np.array([ opti.debug.value(u) for u in us ])
    fs_sol = [ np.split(opti.debug.value(f),f.shape[0]//6) for f in fs ]


viz.play(q_sol.T, terminalModel.dt)


### CHECK ######################################################################################################
### CHECK ######################################################################################################
### CHECK ######################################################################################################

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
        vecfs[frame.parentJoint] = frame.placement * pin.Force(f[i])

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
        af = pin.getFrameAcceleration(model,data,idf,pin.LOCAL_WORLD_ALIGNED).vector
        assert( sum(af**2) < 1e-8 )


# Gather forces in world frame
fs_world = []
for t,m in enumerate(runningModels):
    pin.framesForwardKinematics(model, data, xs_sol[t, :nq])
    fs_world.append( np.concatenate([  data.oMf[idf].rotation @ fs_sol0[t][3*i:3*i+3] for i,idf in enumerate(contactIds) ]) )
fs_world = np.array(fs_world)


### ------------------------------------------------------------------- ###
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

legend = ['x', 'y', 'z']
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('Base link position_' + legend[i])
    plt.plot(base_log[:, i])
    if i == 2:
        plt.axhline(y = z_target, color = 'black', linestyle = '--')

plt.show()

np.save(open("/tmp/sol.npy", "wb"),
        {
            "xs": xs_sol,
            "us": us_sol,
            "acs": acs_sol,
            "fs": np.array(fs_sol0)
        })
