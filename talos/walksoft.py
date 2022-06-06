'''
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min X,U,A,F    sum_t  
                (q-q0)**2 + v**2 ## state regularisation
                + vcom**2
                + sum_foot cop**2
                        + distance(f,cone central axis)
                        + (f-f_t^*)**2

s.t
        x_0 = x0
        x_t+1  = EULER(x_t,u_t,f_t |  f=a_t )
        a_t = aba(q_t,v_t,u_t)
        a_feet(t) = 0
        v_T = x_T [nq:]  = 0
        com(q_t)[2] = com(x_T[:nq])[2] = X_TARGET

        forall t:
                forall foot: 
                        v_foot**2/v_0 < z_foot/z_0      ### Fly high
                        cop \in foot_box                ### COP constraint
                        f_z >= 0                        ### Not pull on ground
                        z_foot>=0                       ### not ground collision
                distance(right foot, left foot) >= 17cm
        forall impact ti:
                z_foot = 0
                v_foot = 0_6
                roll_foot = pitch_foot = 0

So the robot should just bend to reach altitude COM 80cm while stoping at the end of the movement.
The acceleration is introduced as an explicit (slack) variable to simplify the formation of the contact constraint.
Contacts are 6d, yet only with normal forces positive, and 6d acceleration constraints. 

'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt; plt.ion()
from numpy.linalg import norm,inv,pinv,svd,eig
# Local imports
import talos_low
import standard_config
from weight_share import weightShareSmoothProfile,switch_tanh,switch_linear

pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=10000)

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.

DT = 0.010 
X_TARGET = .35
FOOT_SIZE = .05

### LOAD AND DISPLAY SOLO
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = talos_low.load()

contactIds = [ i for i,f in enumerate(robot.model.frames) if "sole_link" in f.name ]
ankleToTow=0.1
ankleToHeel=-0.1
for cid in contactIds:
    f = robot.model.frames[cid]
    robot.model.addFrame(
        pin.Frame(f'{f.name}_tow',f.parentJoint,f.parentFrame,
                  f.placement*pin.SE3(np.eye(3),np.array([ankleToTow,0,0])),pin.FrameType.OP_FRAME))
    robot.model.addFrame(
        pin.Frame(f'{f.name}_heel',f.parentJoint,f.parentFrame,
                  f.placement*pin.SE3(np.eye(3),np.array([ankleToHeel,0,0])),pin.FrameType.OP_FRAME))
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
x0 = np.concatenate([model.q0,np.zeros(model.nv)])

# Some key elements of the model
towIds = { idf: model.getFrameId(f'{model.frames[idf].name}_tow') for idf in contactIds }
heelIds = { idf: model.getFrameId(f'{model.frames[idf].name}_heel') for idf in contactIds }
baseId = model.getFrameId('root_joint')
robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]
com0 = pin.centerOfMass(model,data,model.q0)

pin.framesForwardKinematics(model,data,x0[:model.nq])
# Load key postures
referencePostures,referenceTorques = standard_config.load(robot.model)


# #################################################################################################
# ## HELPERS ######################################################################################
# #################################################################################################

def listOfForcesToArrayWithZeros( forceList ):
    '''Convert a list of list of forces into a numpy array where missing forces are set to 0.'''
    forceArray = [ np.concatenate([ \
                                    f[runningModels[t].contactIds.index(c) ] if c in runningModels[t].contactIds
                                    else np.zeros(6)
                                    for i,c in enumerate(contactIds)   ])
                   for t,f in enumerate(forceList) ]
    return forceArray

sym_R = casadi.SX.sym('R', 3, 3)
sym_R_ref = casadi.SX.sym('R_ref', 3, 3)
so3_diff = casadi.Function('so3_diff', [sym_R, sym_R_ref], [cpin.log3(sym_R.T @ sym_R_ref)])
so3_log = casadi.Function('so3_log', [sym_R], [cpin.log3(sym_R)])
sym_y = casadi.SX.sym('y', 1)
sigm = casadi.Function('sigm',[sym_y],[ 1/(1+casadi.exp(sym_y)) ] )
v_activ = casadi.Function('v_activ',[sym_y],[casadi.exp(-sym_y) ] )
###################################################################################################
### TUNING ########################################################################################
###################################################################################################

# In the code, cost terms with 0 weight are commented for reducing execution cost
# An example of working weight value is then given as comment at the end of the line.
# When setting them to >0, take care to uncomment the corresponding line.
# All these lines are marked with the tag ##0##.

basisQWeight = [0,0,0,50,50,0]
legQWeight =  [3,3,1,2,1,1]
torsoQWeight = [10,10]
armQWeight = [3,3]
basisVWeight = [0,0,0,3,3,1] ### was 003331
legVWeight =  [1]*6
torsoVWeight = [20]*2
armVWeight = [2]*2

STATE_WEIGHT = 1*np.array(  \
    basisQWeight+legQWeight+legQWeight+armQWeight \
    +basisVWeight+legVWeight+legVWeight+armVWeight)
assert(len(STATE_WEIGHT)==model.nv*2)

### Gains for force continuity: wfref for tracking the reference, wfcont for time difference
contiTorqueWeight = 0
refTorqueWeight = 0
refStateWeight = 5e-2
flatBaseWeight = 0 # 1e-1
minForceWeight = 1e-2*np.array([1,1,.1,10,10,2])
comWeight = 0 # 1e-1
vcomWeight = 8e-1
acomWeight = 0 # 8e-1*DT
copWeight = 1e-2
verticalFootVelWeight = 0.1
footVelWeight = 0 # 0.1
footAccWeight = 0 # 0.01
refFootFlyingAltitude = 3e-2
footMinimalDistance = .1 # 0.1  (.17 is the max value wrt initial config)x
lowbandwidthweight = 0 # 1e-3
minTorqueDiffWeight = 0 # 1e-4

MAIN_JOINTS = [ 0,1,3 ]
MAIN_JOINTS = [ i+7 for i in MAIN_JOINTS ] + [ i+13 for i in MAIN_JOINTS ]
refMainJointsAtImpactWeight = 1e2
refForceWeight = 5e-2
contiForceWeight = 0

# #################################################################################################
# ## ACTION MODEL #################################################################################
# #################################################################################################

# The action model stores the computation of the dynamic -f- and cost -l- functions, both return by
# calc as self.calc(x,u) -> [xnext=f(x,u),cost=l(x,u)]
# The dynamics is obtained by RK4 integration of the pinocchio ABA function.
# The cost is a sole regularization of u**2
class CasadiActionModel:
    dt = DT
    def __init__(self,cmodel,activeContactIds):
        '''
        cmodel: casadi pinocchio model
        activeContactIds: list of contact frames IDs.
        '''
          
        self.cmodel = cmodel
        self.contactIds = activeContactIds
        
        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.ndx = 2*nv
        self.nu = nv-6
        self.nv = nv
        self.ntau = nv

        tauref = referenceTorques[tuple(self.contactIds)]
        assert(norm(tauref[:6])<1e-4)
        self.u0 = tauref[6:].copy()
        self.fs0 = [ np.array([0,0,robotweight/len(self.contactIds),0,0,0]) for __c in self.contactIds ]

        # Symbols used to define casadi functions
        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ca = casadi.SX.sym("a",self.cmodel.nv,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        cfs = [ casadi.SX.sym("f"+cmodel.frames[idf].name,6,1) for idf in self.contactIds ]
        cw = casadi.SX.sym("w",self.ndx,1)
        
        ### Build force list for ABA
        forces = [ cpin.Force.Zero() for __j in self.cmodel.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        assert( len( set( [ cmodel.frames[idf].parentJoint for idf in self.contactIds ]) ) == len(self.contactIds) )
        for f,idf in zip(cfs,self.contactIds):
            # Contact forces introduced in ABA as spatial forces at joint frame.
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
        # Weighted lie difference
        self.wdifference = casadi.Function('xwminus', [cx,cx2,cw],
                                           [ cw*casadi.vertcat(cpin.difference(self.cmodel,cx2[:nq],cx[:nq]),
                                                         cx2[nq:]-cx[nq:]) ])

        cpin.forwardKinematics(cmodel,cdata,cx[:nq],cx[nq:],ca)
        cpin.updateFramePlacements(cmodel,cdata)
        # Base link position
        self.base_translation = casadi.Function('base_translation', [cx], [ cdata.oMf[baseId].translation ])
        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = { idf: casadi.Function('foot'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[idf].translation]) for idf in contactIds }
        # tow[c](x) = Tow positions 
        self.tows = { idf: casadi.Function('tow'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[towIds[idf]].translation]) for idf in contactIds }
        # heels[c](x) = Heel positions
        self.heels = { idf: casadi.Function('heel'+cmodel.frames[idf].name,
                                      [cx],[self.cdata.oMf[heelIds[idf]].translation]) for idf in contactIds }
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = {idf: casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[self.cdata.oMf[idf].rotation]) for idf in contactIds }
        # vfeet[c](x) =  6d velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = {idf: casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds }
        # vtows[c](x) = 6d velocity of the tow
        self.vtows = { idf: casadi.Function('vtow'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,towIds[idf],pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds }
        # vheels[c](x) = 6d velocity of the heel
        self.vheels = { idf: casadi.Function('vheel'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel,cdata,heelIds[idf],pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds }
        # afeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = { idf: casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameAcceleration( cmodel,cdata,idf,pin.LOCAL_WORLD_ALIGNED ).vector])
                       for idf in contactIds }

        cpin.centerOfMass(cmodel,self.cdata,cx[:nq],cx[nq:],ca)
        # com(x) = com position
        self.com = casadi.Function('com',[cx],[self.cdata.com[0]])
        # acom(x) = com velocity
        self.vcom = casadi.Function('vcom',[cx],[self.cdata.vcom[0]])
        # acom(x,a) = com acceleration
        self.acom = casadi.Function('acom',[cx,ca],[self.cdata.acom[0]])
        
    def calc(self,x, u, a, fs, ocp):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        fs = [ fs[6 * i : 6 * i + 6] for i,__c in enumerate(self.contactIds) ]   # split
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
        # Follow reference torque
        # ##0## cost += refTorqueWeight*casadi.sumsqr(u-self.u0) * self.dt
        # Follow reference state
        cost += refStateWeight*casadi.sumsqr( self.wdifference(x,x0,STATE_WEIGHT) ) * self.dt
        # Keep basis horizontal
        # ##0## cost += flatBaseWeight * casadi.sumsqr(x[3:6]) # Keep base flat
        # Keep 6d contact forces at cone center, ie only f[2] should be nonzero
        cost += sum( [ casadi.sumsqr((f*minForceWeight)[[0,1,3,4,5]]) for f in fs ] ) * self.dt
        # Penalize com and derivatives of com
        # ##0## cost += comWeight* casadi.sumsqr( self.com(x)[2]-com0[2] )
        cost += vcomWeight* casadi.sumsqr( self.vcom(x)[2] )
        # ##0## cost += acomWeight* casadi.sumsqr( self.acom(x,a)[2] )
        
        ### OCP additional constraints

        # Zero acceleration at contact (baumgart not working ... why???)
        for cid in self.contactIds:
            ocp.subject_to( self.afeet[cid](x,a) == 0)#-100*self.vfeet[cid](x) )

        # Contact force with positive normal and bounded COP
        for f,cid in zip(fs,self.contactIds):
            R = self.Rfeet[cid]
            fw = R(x) @ f[:3]
            tauw = R(x) @ f[3:]
            # ocp.subject_to(fw[2]>=1)
            # ocp.subject_to( tauw[:2]/fw[2] <= FOOT_SIZE )
            # ocp.subject_to( tauw[:2]/fw[2] >= -FOOT_SIZE )
            
            cost += casadi.if_else(fw[2]>=1,0,(fw[2]-1)**2)
            cost += casadi.if_else( tauw[0] <= fw[2]*FOOT_SIZE, 0, (tauw[0]-fw[2]*FOOT_SIZE)**2 )
            cost += casadi.if_else( tauw[0] >= -fw[2]*FOOT_SIZE, 0, (tauw[0]+fw[2]*FOOT_SIZE)**2 )
            cost += casadi.if_else( tauw[1] <= fw[2]*FOOT_SIZE, 0, (tauw[1]-fw[2]*FOOT_SIZE)**2 )
            cost += casadi.if_else( tauw[1] >= -fw[2]*FOOT_SIZE, 0, (tauw[1]+fw[2]*FOOT_SIZE)**2 )

            # Minimize COP deviation as well
            cost += copWeight*casadi.sumsqr(tauw[:2]/fw[2]/FOOT_SIZE)

        for fid in contactIds:
            if fid in self.contactIds: continue
            cost += verticalFootVelWeight*casadi.sumsqr(self.vfeet[fid](x)[2])
            # ##0## cost += footVelWeight*casadi.sumsqr(self.vfeet[fid](x))
            # ##0## cost += footAccWeight*casadi.sumsqr(self.afeet[fid](x,a)[:3])

            # Foot altitude should be above ground
            #ocp.subject_to( self.feet[fid](x)[2]>=0 )
            # Foot horizontal velocity should be lower than its (normalized) altitude
            #ocp.subject_to( casadi.sumsqr(self.vfeet[fid](x)[:2]) <= (self.feet[fid](x)[2]/refFootFlyingAltitude))
            cost += casadi.if_else( self.tows[fid](x)[2]>=0,0,self.tows[fid](x)[2]**2 )
            cost += .1*v_activ( 5*self.tows[fid](x)[2]/refFootFlyingAltitude ) \
                *  casadi.sumsqr(self.vtows[fid](x)[:2])
            cost += casadi.if_else( self.heels[fid](x)[2]>=0,0,self.heels[fid](x)[2]**2 )
            cost += .1*v_activ( 5*self.heels[fid](x)[2]/refFootFlyingAltitude ) \
                *  casadi.sumsqr(self.vheels[fid](x)[:2])
            
            #ocp.subject_to( casadi.sumsqr(self.vtows[fid](x)[:2]) <= (self.tows[fid](x)[2]/refFootFlyingAltitude))
            #ocp.subject_to( casadi.sumsqr(self.vheels[fid](x)[:2]) <= (self.heels[fid](x)[2]/refFootFlyingAltitude))

            ### Avoid collision between feet
            for cid in self.contactIds:
                ocp.subject_to( casadi.sumsqr(self.feet[fid](x)[:2]-self.feet[cid](x)[:2]) >= footMinimalDistance**2)
            
        return xnext,cost


# ##################################################################################################
# ## OCP PROBLEM ###################################################################################
# ##################################################################################################

# Contact are specified with the order chosen in <contactIds>
contactPattern = [] \
    + [ [ 1,1 ] ] * 30 \
    + [ [ 1,0 ] ] * 50  \
    + [ [ 1,1 ] ] * 11  \
    + [ [ 0,1 ] ] * 50  \
    + [ [ 1,1 ] ] * 50 \
    + [ [ 1,1 ] ]
T = len(contactPattern)-1
    
def patternToId(pattern):
    '''Return the tuple of active contact from a pattern like [0,1], [1,0] or [1,1].'''
    return tuple( contactIds[i] for i,c in enumerate(pattern) if c==1 )

# In order to avoid creating too many casadi action model, we store in a dict one model for each contact pattern.
contactSequence = [ patternToId(p) for p in contactPattern ]
casadiActionModels = { contacts: CasadiActionModel(cmodel,contacts)  for contacts in set(contactSequence) }

### PROBLEM
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ casadiActionModels[contactSequence[t]] for t in range(T) ]
terminalModel = casadiActionModels[contactSequence[T]]

# Decision variables
# State increment with respect to an arbitrary x0=(q0,v0)
dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
# Acceleration a=vdot
acs = [ opti.variable(model.nv) for model in runningModels ]                      # acceleration
# Torque of actuated joints
us =  [ opti.variable(model.nu) for model in runningModels ]                      # control variable
# Contact forces (number depends on the phase)
fs =  [ opti.variable(6*len(model.contactIds) ) for model in runningModels ]      # contact force
# For convenience, we work with x=x0+dx as the real decision varible.
xs =  [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(dxs[0] == 0)

for t in range(T):
    
    xnext,rcost = runningModels[t].calc(xs[t], us[t], acs[t], fs[t], opti )
    opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == 0 ) # x' = f(x,u)

    totalcost += rcost

    # Additional cost at impact
    for i,c in enumerate(runningModels[t].contactIds):
        if c not in runningModels[t-1].contactIds:
            print(f'Impact for foot {c} (contact #{i}) at time {t}')
            #opti.subject_to( runningModels[t].feet[c](xs[t])[2] == 0) # Altitude is 0
            #opti.subject_to( runningModels[t].vfeet[c](xs[t]) == 0)   # 6d velocity at impact is 0
            #opti.subject_to( runningModels[t].Rfeet[c](xs[t])[0,2] == 0) # No rotation on roll
            #opti.subject_to( runningModels[t].Rfeet[c](xs[t])[1,2] == 0) # No rotation on pitch

            totalcost+=impactAltitudeWeight* casadi.sumsqr(runningModels[t].feet[c](xs[t])[2])
            totalcost+=impactVelocityWeight* casadi.sumsqr(runningModels[t].vfeet[c](xs[t]))
            totalcost+=impactRotationWeight* casadi.sumsqr(runningModels[t].Rfeet[c](xs[t])[0,2])
            totalcost+=impactRotationWeight* casadi.sumsqr(runningModels[t].Rfeet[c](xs[t])[1,2])

            # Keep main joints in standard configuration at impact.
            totalcost +=  refMainJointsAtImpactWeight*casadi.sumsqr( xs[t][MAIN_JOINTS] - model.q0[MAIN_JOINTS] )

    # Bandwidth cost
    if t>1:
        '''
        tau'  = (1-alpha) tau + alpha ref      with alpha ~=~ 1 
        '''
        ALPHA = 0.75

        ### Sebastien formulation ... lead to poooooor results (reference torque likely too bad)
        # ##0## totalcost += lowBandwidthWeight*casadi.sumsqr((us[t][3:6]-(1-ALPHA)*us[t-1][3:6])/ALPHA - runningModels[t].u0[3:6])
        # ##0## totalcost += lowBandwidthWeight*casadi.sumsqr((us[t][8:12]-(1-ALPHA)*us[t-1][8:12])/ALPHA - runningModels[t].u0[8:12])

        ### Min diff torque, working soso, could work with some tuning, might not be needed.
        # ##0## totalcost += minTorqueDiffWeight*casadi.sumsqr((us[t][3:6]-us[t-1][3:6]))
        # ##0## totalcost += minTorqueDiffWeight*casadi.sumsqr((us[t][8:12]-us[t-1][8:12]))
            

# ### FORCE COST ###
# The force costs are defined using a reference (smooth) force.
# Search the contact phase of minimal duration (typically double support)
contactState=[]
dur=mindur=len(contactPattern)
for t,s in enumerate(contactPattern):
    dur+=1
    if s!=contactState:
        contactState=s
        mindur=min(mindur,dur)
        dur=0
# Select the smoothing transition to be smaller than half of the minimal duration.
transDuration=(mindur-1)//2
# Compute contact importance, ie how much of the weight should be supported by each
# foot at each time.
contactImportance = weightShareSmoothProfile(contactPattern,transDuration,switch=switch_linear)
# Contact reference forces are set to contactimportance*weight
weightReaction = np.array([0,0,robotweight,0,0,0])
referenceForces = [
    [ weightReaction*contactImportance[t,contactIds.index(cid)] for cid in runningModels[t].contactIds ]
      for t in range(T) ]
# Take care, we suppose here that foot normal is vertical.

### Make forces track the reference (smooth) trajectory
for t,(refs,r) in enumerate(zip(referenceForces,runningModels)):
    for k,f in enumerate(refs):
        totalcost += casadi.sumsqr( (fs[t][6*k:6*k+6] - f)/robotweight ) * refForceWeight

### Make the forces time smooth
for t in range(1,T-1):
    for k,cid in enumerate(runningModels[t].contactIds):
        if cid in runningModels[t-1].contactIds:
            kprev = runningModels[t-1].contactIds.index(cid)
            totalcost += casadi.sumsqr( (fs[t][6*k:6*k+6] - fs[t-1][6*kprev:6*kprev+6])/robotweight )*contiForceWeight/2
        else:
            totalcost += casadi.sumsqr( fs[t][6*k:6*k+6]/robotweight )*contiForceWeight/2
        if cid in runningModels[t+1].contactIds:
            knext = runningModels[t+1].contactIds.index(cid)
            totalcost += casadi.sumsqr( (fs[t][6*k:6*k+6] - fs[t+1][6*knext:6*knext+6])/robotweight )*contiForceWeight/2
        else:
            totalcost += casadi.sumsqr( fs[t][6*k:6*k+6]/robotweight )*contiForceWeight/2

### Terminal cost
totalcost += 1000*casadi.sumsqr( xs[T][cmodel.nq:] )
totalcost += 1000*casadi.sumsqr( terminalModel.base_translation(xs[T])[0] - X_TARGET )

### SOLVE
opti.minimize(totalcost)
s_opt = {
    'tol' : 1e-4,
    'acceptable_tol': 1e-4,
    'max_iter': 200,
}
opti.solver("ipopt", {}, s_opt) # set numerical backend

cost_log = []
def call(i):
    global cost_log
    cost_log += [opti.debug.value(totalcost)]

opti.callback(call)

# Try to load the initial guess from a file.
try:
    GUESS_FILE = None
    GUESS_FILE = '/tmp/sol.npy'
    #if True: ### Load warmstart from file
    guess = np.load(GUESS_FILE,allow_pickle=True)[()]
    if len(guess['xs'])!=len(contactPattern):
        print('Guess file does not have the right time horizon. Trash it')
        raiseAnError
    t=0
    for dx,x_g in zip(dxs,guess['xs']):
        opti.set_initial(dx,np.concatenate([
            pin.difference(model,model.q0,x_g[:model.nq]),
            x_g[model.nq:]]))
    for f,f_g,m in zip(fs,guess['fs'],runningModels):
        fshort_g = np.concatenate([ f_g[6*rank:6*(rank+1)] for rank,cid in enumerate(contactIds) if cid in m.contactIds ])
        #print(f_g,fshort_g)
        opti.set_initial(f,fshort_g)
    for u,u_g in zip(us,guess['us']):
        opti.set_initial(u,u_g)
    for ac,ac_g in zip(acs,guess['acs']):
        opti.set_initial(ac,ac_g)
    print('Done with reading warm start from file')
except:
    # If initial guess cannot be red properly, build it as quasi-static start.
    print('Warm start from file failed, now building a quasistatic guess')
    for dx in dxs:
        opti.set_initial(dx,np.zeros(model.nv*2))
    for f,u,r in zip(fs,us,runningModels):
        opti.set_initial(u,r.u0)
        nc=len(r.contactIds)
        opti.set_initial(f, np.concatenate(r.fs0))

# ### SOLVE ####################################################################
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    optivalue = opti.value
except:
    print('ERROR in convergence, plotting debug info.')
    optivalue = opti.debug.value

# ### SOLUTION #################################################################
# Get solution as numy arrays.
dxs_sol = np.array([ opti.value(x) for x in dxs ])
xs_sol = np.array([ opti.value(x) for x in xs ])
q_sol = xs_sol[:,: model.nq]
us_sol = np.array([ opti.value(u) for u in us ])
acs_sol = np.array([ opti.value(a) for a in acs ])
fs_sol = [ np.split(opti.value(f),f.shape[0]//6) for f in fs ]
base_log = []
[base_log.append(terminalModel.base_translation(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
base_log = np.array(base_log)
### We reorganize fs_sol to have 4 contacts for each timestep, adding a 0 force when needed.
fs_sol0 = [ np.concatenate([ 
    f[runningModels[t].contactIds.index(c) ] if c in runningModels[t].contactIds
    else np.zeros(6)
    for i,c in enumerate(contactIds)   ])
            for t,f in enumerate(fs_sol) ]
# Gather forces in world frame
fs_world = []
for t,m in enumerate(runningModels):
    pin.framesForwardKinematics(model, data, xs_sol[t, :model.nq])
    fs_world.append( np.concatenate([  data.oMf[idf].rotation @ fs_sol0[t][3*i:3*i+3] for i,idf in enumerate(contactIds) ]) )
fs_world = np.array(fs_world)

# ## CHECK ########################################################################################
# ## CHECK ########################################################################################
# ## CHECK ########################################################################################

# Sanity check of the feasibility of the solution (just to be sure, ipOpt should
# enforce that).

ha = []

# Check that all constraints are respected
for t,(m,x1,u,f,x2) in enumerate(zip(runningModels,xs_sol[:-1],us_sol,fs_sol,xs_sol[1:])):
    tau = np.concatenate([np.zeros(6),u])
    q1,v1 = x1[:model.nq],x1[model.nq:]

    vecfs = pin.StdVec_Force()
    for __j in model.joints:
        vecfs.append(pin.Force.Zero())
    for i,idf in enumerate(m.contactIds):
        frame = model.frames[idf]
        vecfs[frame.parentJoint] = frame.placement * pin.Force(f[i])

    a = pin.aba(model,data,x1[:model.nq],x1[model.nq:],tau,vecfs)
    ha.append(a.copy())
    vnext = v1+a*m.dt
    qnext = pin.integrate(model,q1,vnext*m.dt)
    xnext = np.concatenate([qnext,vnext])
    if not( np.linalg.norm(xnext-x2) < 1e-5 ):
        print('Error: inconsistant trajectory xnext!=x+f(x,u)')


    ### Check 0 velocity of contact points
    pin.forwardKinematics(model,data,q1,v1,a)
    pin.updateFramePlacements(model,data)
    for idf in m.contactIds:
        vf = pin.getFrameVelocity(model,data,idf)
        af = pin.getFrameAcceleration(model,data,idf,pin.LOCAL_WORLD_ALIGNED).vector
        if not ( sum(af**2) < 1e-5 ): print("Warning with af**2")

### Check warm start is close to solution
for dx,x_g in zip(dxs_sol,guess['xs']):
    dx_g = np.concatenate([
        pin.difference(model,model.q0,x_g[:model.nq]),
        x_g[model.nq:]])
    if not (norm(dx_g-dx)<1e-2):
        print('Large change in <x> since warm start')
        break
for f,f_g,m in zip(fs_sol,guess['fs'],runningModels):
    change = False
    fshort_g = [ f_g[6*rank:6*(rank+1)] for rank,cid in enumerate(contactIds) if cid in m.contactIds ]
    for fi,fi_g in zip(f,fshort_g):
        if not (norm(fi-fi_g)<1e-2):
            print('Large change in <f> since warm start')
            change = True
            break
    if change: break
for u,u_g in zip(us_sol,guess['us']):
    if not norm(u-u_g)<1e-2:
        print('Large change in <u> since warm start')
        break
for ac,ac_g in zip(acs_sol,guess['acs']):
    if not (norm(ac-ac_g)<1e-2):
        print('Large change in <a> since warm start')
        break

# ### DISPLAY ######################################################################
# Play solution in gepetto viewer
viz.play(q_sol.T, terminalModel.dt)

# ### PLOT #########################################################################

# Solver convergence
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

# Robot basis movement
legend = ['x', 'y', 'z']
plt.figure('Basis move')
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('Base link position_' + legend[i])
    plt.plot(base_log[:, i])
    if i == 0:
        plt.axhline(y = X_TARGET, color = 'black', linestyle = '--')

# Cop of each foot vs time
plt.figure('cop time local')
for ifig,cid in enumerate(contactIds):
    plt.subplot(len(contactIds)+1,1,ifig+1)
    ftraj = [ [t,f[r.contactIds.index(cid)]] for t,(f,r) in enumerate(zip(fs_sol,runningModels)) if cid in r.contactIds ] 
    cop = [ [t,  [ f[4]/f[2], -f[3]/f[2] ] ] for (t,f) in ftraj ]
    plt.plot([ t for t,p in cop ], [ p for t,p in cop ],'.')

# Cop of each foot in x-vs-y (with limits)
plt.figure(figsize=(12,6))
plt.title('cop local')
l_foot = np.array([ [-FOOT_SIZE,-FOOT_SIZE,0,1],[-FOOT_SIZE,FOOT_SIZE,0,1],[FOOT_SIZE,FOOT_SIZE,0,1],[FOOT_SIZE,-FOOT_SIZE,0,1],[-FOOT_SIZE,-FOOT_SIZE,0,1] ]).T
for ifig,cid in enumerate(contactIds):
    plt.subplot(1,len(contactIds),ifig+1)
    ARENA_SIZE = .6
    plt.axis([-ARENA_SIZE/4,ARENA_SIZE*3/4,-ARENA_SIZE/2,ARENA_SIZE/2])
    plt.xlabel(model.frames[cid].name)
    for t,r in enumerate(runningModels):
        if cid not in r.contactIds: continue
        f = fs_sol[t][r.contactIds.index(cid)]
        l_cop =  np.array([ f[4]/f[2], -f[3]/f[2],0 ])
        pin.framesForwardKinematics(model,data,xs_sol[t][:model.nq])
        w_cop = data.oMf[cid] * l_cop
        plt.plot(w_cop[0],w_cop[1],'r.')
        w_foot = data.oMf[cid].homogeneous @ l_foot
        plt.plot(w_foot[0,:],w_foot[1,:],'grey')

# Forces and reference forces wrt time
plt.figure('forces')
frefplot = np.array(listOfForcesToArrayWithZeros(referenceForces))
fs0plot=np.array(fs_sol0)
plt.subplot(211)
plt.plot(fs0plot[:,2])
plt.plot(frefplot[:,2])
plt.xlabel(model.frames[contactIds[0]].name)
plt.subplot(212)
plt.plot(fs0plot[:,8])
plt.plot(frefplot[:,8])
plt.xlabel(model.frames[contactIds[1]].name)

# COM position and velocity (x+y separated from z)
plt.figure('com',figsize=(6,8))
complot = []
vcomplot = []
for x in xs_sol:
    pin.centerOfMass(model,data,x[:model.nq],x[model.nq:])
    complot.append(data.com[0].copy())
    vcomplot.append(data.vcom[0].copy())
complot = np.array(complot)
vcomplot = np.array(vcomplot)
plt.subplot(411)
plt.plot(complot[:,:2])
plt.ylabel('pos x-y')
plt.subplot(412)
plt.plot(complot[:,2])
plt.ylabel('pos z')
ax=plt.axis()
plt.axis((ax[0],ax[1],com0[2]-2.5e-2, com0[2]+2.5e-2)) # yaxis is 5cm around 0 position
plt.subplot(413)
plt.plot(vcomplot[:,:2])
plt.ylabel('vel x-y')
plt.legend([ 'x', 'y'])
plt.subplot(414)
plt.plot(vcomplot[:,2])
plt.ylabel('vel z')

# Foot position and velocity
plt.figure('foot')
foottraj = []
footvtraj = []
for x in xs_sol:
    pin.forwardKinematics(model,data,x[:model.nq],x[model.nq:])
    pin.updateFramePlacements(model,data)
    foottraj.append( np.concatenate([  data.oMf[cid].translation for cid in contactIds ]))
    footvtraj.append( np.concatenate([  pin.getFrameVelocity(model,data,cid).vector for cid in contactIds ]))
foottraj = np.array(foottraj)
footvtraj = np.array(footvtraj)
plt.subplot(311)
hplot = []
names = []
for i,cid in enumerate(contactIds):
    hplot.extend(plt.plot(foottraj[:,3*i+2]))
    names.append(model.frames[cid].name)
plt.legend(hplot,names)
plt.ylabel('altitude')
plt.subplot(313)
hplot = []
for i,cid in enumerate(contactIds):
    hplot.extend(plt.plot(foottraj[:,3*i],foottraj[:,3*i+2]))
plt.legend(hplot,names)
plt.ylabel('x-z traj')
plt.subplot(312)
hplot = []
for i,cid in enumerate(contactIds):
    hplot.extend(plt.plot(np.sqrt(np.sum(footvtraj[:,6*i:6*i+2]**2,1))))
plt.legend(hplot,names)
plt.ylabel('horz vel')

plt.figure('foot collision')
h1=plt.plot([ f[0] for f in foottraj], [f[1] for f in foottraj])
h2=plt.plot([ f[3] for f in foottraj], [f[4] for f in foottraj])
plt.legend(h1+h2 ,['left','right'])
for t in range(T):
    a = foottraj[t][:2]
    b = foottraj[t][3:5]
    m = (a+b)/2
    d = (b-a)
    d /= norm(d)
    aa = m+d*footMinimalDistance/2
    bb = m-d*footMinimalDistance/2
    plt.plot([aa[0],bb[0]],[aa[1],bb[1]],'grey')
plt.axis([ -.1,0.4,-.25,.25 ])

# ### SAVE #####################################################################
#plt.show()
def save():
    np.save(open("/tmp/sol.npy", "wb"),
            {
                "xs": xs_sol,
                "us": us_sol,
                "acs": acs_sol,
                "fs": np.array(fs_sol0)
        })
    

    
