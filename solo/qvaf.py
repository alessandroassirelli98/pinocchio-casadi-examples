'''
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min X,U,A,F    sum_t  ||u_t-u0||**2  + || diff(x_t,x0) ||**2
s.t
        x_0 = x0
        x_t+1  = EULER(x_t,u_t,f_t |  f=a_t )
        a_t = aba(q_t,v_t,u_t)
        a_feet(t) = J a = 0
        v_T = x_T [nq:]  = 0
        com(q_t)[2] = com(x_T[:nq])[2] = 0.1

So the robot should just bend to reach altitude COM 10cm while stoping at the end of the movement.
The acceleration is introduced as an explicit (slack) variable to simplify the formation of the contact constraint.


'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
#import matplotlib.pyplot as plt; plt.ion()
from pinocchio.visualize import GepettoVisualizer

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
T = 10
DT = 0.05

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('solo12')

try:
    viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
except:
    print("No viewer"  )
    
# The pinocchio model is what we are really interested by.
model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()

# Initial config, also used for warm start
x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
# quasi static for x0, used for warm-start and regularization
u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
               0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
               0.25720939, -0.51441314])

#contact_frames = { i:f for i,f in enumerate(cmodel.frames) if "FOOT" in f.name }
contactIds = [ i for i,f in enumerate(cmodel.frames) if "FOOT" in f.name ]

robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]

### ACTION MODEL
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

        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ca = casadi.SX.sym("a",self.cmodel.nv,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        cfs = [ casadi.SX.sym("f"+cmodel.frames[idf].name,3,1) for idf in self.contactIds ]
        
        # acc = cpin.constraintDynamics(cmodel,cdata,cx[:nq],cx[nq:],ctau,
        #                               self.contact_models,self.contact_datas,self.prox_settings)

        ### Build force list for ABA
        forces = [ cpin.Force.Zero() for _ in self.cmodel.joints ]
        # I am supposing here that all contact frames are on separate joints. This should be asserted.
        assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        for f,idf in zip(cfs,self.contactIds):
            forces[cmodel.frames[idf].parentJoint] = cmodel.frames[idf].placement * cpin.Force(f,0*f)
        self.forces = cpin.StdVec_Force()
        for f in forces:
            self.forces.append(f)
           
        acc = cpin.aba( cmodel,cdata, cx[:nq],cx[nq:],ctau,self.forces )
        
        ### Casadi MX functions
        # acceleration(x,u)
        self.acc = casadi.Function('acc', [cx,ctau]+cfs, [ acc ])
        self.com = casadi.Function('com', [cx],[ cpin.centerOfMass(cmodel,cdata,cx[:nq]) ])
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

        #cpin.framesForwardKinematics(cmodel,cdata,cx[:nq],cx[nq:])
        cpin.forwardKinematics(cmodel,cdata,cx[:nq],cx[nq:],ca)
        cpin.updateFramePlacements(cmodel,cdata)
        self.feet = [ casadi.Function('foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in contactIds ]
        self.Rfeet = [ casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[self.cdata.oMf[idf].rotation]) for idf in contactIds ]
        self.vfeet = [ casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in contactIds ]
        self.afeet = [ casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameClassicalAcceleration( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).linear])
                       for idf in contactIds ]

        
    def calc(self,x, u, a, fs, ocp):
        # Return xnext,cost
        fs = [ fs[3 * i : 3 * i + 3] for i,_ in enumerate(self.contactIds) ]   # split
        
        dt = self.dt
        nq,nv = self.cmodel.nq,self.cmodel.nv
        tau = casadi.vertcat( np.zeros(6), u )

        # Euler integration
        vnext = x[nq:] + a*dt
        qnext = self.integrate_q(x[:nq], vnext*dt)
        xnext = casadi.vertcat(qnext,vnext)
        
        cost = 0
        cost += 1e-2*casadi.sumsqr(u-u0) * self.dt
        cost += 1e-1*casadi.sumsqr( self.difference(x,x0) ) * self.dt
        #cost += sum( [ casadi.sumsqr(f) for f in fs ] )
        
        ### OCP additional constraints
        #for vfoot in self.vfeet:
        #    ocp.subject_to( vfoot(x) == 0 )
        for afoot in self.afeet:
            ocp.subject_to( afoot(x,a) == 0 )

        ocp.subject_to( self.acc(x,tau,*fs ) == a )
            
        #for f,R in zip(fs,self.Rfeet):   # for cone constrains
            #fw = R(x) @ f
            #opti.subject_to(fw[2]>=0)
        
        return xnext,cost

seq = [ True ]*T
 
### PROBLEM
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ CasadiActionModel(cmodel, contactIds if seq[t] else []) for t in range(T) ]
terminalModel = CasadiActionModel(cmodel,contactIds)

# Decision variables
dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
acs = [ opti.variable(model.nv) for model in runningModels ]                   # acceleration
us = [ opti.variable(model.nu) for model in runningModels ]                     # control variable
fs = [ opti.variable(3*len(model.contactIds) ) for model in runningModels ]                     # contact force
xs = [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(dxs[0] == np.zeros(2*cmodel.nv))
for t in range(T):
    
    xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti )
    #opti.subject_to( xs[t + 1] == xnext )
    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*cmodel.nv) )  # x' = f(x,u)

    totalcost += rcost
        
opti.subject_to( xs[T][cmodel.nq:] == 0 )              # v_T = 0
#opti.subject_to( xs[T][:3] == x0[:3] )              # x_T,y_T,z_T = x0,y0,z0
#opti.subject_to( terminalModel.com(xs[T])[2] == .1 )   # z_com(T) = ref
#totalcost += 1000 * (terminalModel.com(xs[T])[2] - .1)**2
#totalcost += 1 * (terminalModel.com(xs[T//2])[2] - .3)**2
opti.subject_to( terminalModel.com(xs[T])[2] == .1 )   # z_com(T) = ref
#opti.subject_to( xs[T][3:6] == 0 )   # orientation flat


### SOLVE
opti.minimize(totalcost)
for x in dxs: opti.set_initial(x,np.zeros(terminalModel.ndx))
for u in us: opti.set_initial(u,u0)
opti.solver("ipopt") # set numerical backend
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    dxs_sol = np.array([ opti.value(x) for x in dxs ])
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.value(u) for u in us ])
    acs_sol = np.array([ opti.value(a) for a in acs ])
    fs_sol = [ opti.value(f) for f in fs ]
except:
    print('ERROR in convergence, plotting debug info.')
    dxs_sol = np.array([ opti.debug.value(x) for x in dxs ])
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.debug.value(u) for u in us ])
    fs_sol = [ opti.value(f) for f in fs ]

### CHECK
### CHECK
### CHECK
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
        vecfs[frame.parentJoint] = frame.placement * pin.Force(f[3*i:3*i+3],np.zeros(3))

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
        #assert( sum(vf.linear**2) < 1e-3 )
        af = pin.getFrameClassicalAcceleration(model,data,idf,pin.LOCAL_WORLD_ALIGNED)
        assert( sum(af.linear**2) < 1e-8 )
        

    # for idf in m.contactIds:
    #     assert( abs(data.oMf[idf].translation[2] )<1e-6 )


# Gather forces in world frame
f0s = []
for t,m in enumerate(runningModels):
    f = { i: np.zeros(3) for i in contactIds}
    for i,idf in enumerate(m.contactIds):
        pin.framesForwardKinematics(model, data, xs_sol[t, :nq])
        f[idf] = data.oMf[idf].rotation @ fs_sol[t][3 * i : 3 * i + 3]
    f0s.append( list(f.values()) )
f0s = np.array([ np.concatenate(f) for f in f0s])

np.save(open("/tmp/sol.npy", "wb"),
        {
            "xs": xs_sol,
            "us": us_sol,
            "acs": acs_sol,
            "fs": np.array(fs_sol)})
