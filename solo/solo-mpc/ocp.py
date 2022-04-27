import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from pinocchio.visualize import GepettoVisualizer
import conf
from time import time
import os
plt.style.use('seaborn')
path = os.getcwd()


class OCP():
    def __init__(self, robot, gait):

        self.robot = robot
        self.model = model = robot.model
        self.cmodel = cmodel = cpin.Model(robot.model)

        self.iterationTime = 0

        self.allContactIds = [ i for i,f in enumerate(cmodel.frames) if "FOOT" in f.name ]
        self.contactNames = [ f.name for f in cmodel.frames if "FOOT" in f.name ]
        self.effort_limit = np.ones(robot.nv - 6) *3   

        contactSequence = [ self.patternToId(p) for p in gait]
        self.casadiActionModels = { contacts: ShootingNode( cmodel=cmodel, model=model, q0=robot.q0, allContactIds=self.allContactIds,\
                                                            contactIds=contacts)  \
                                    for contacts in set(contactSequence) } # build the different shooting nodes

        # Initial config, also used for warm start
        

    def patternToId(self, gait):
        return tuple(self.allContactIds[i] for i,c in enumerate(gait) if c==1 )
    
    def warmstart(self, guess=None):
        try:
            xs_g = guess['xs']
            us_g = guess['us']
            acs_g = guess['acs']
            fs_g = guess['fs']

            def xdiff(x1,x2):
                nq = self.model.nq
                return np.concatenate([
                    pin.difference(self.model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

            for x,xg in zip(self.dxs,xs_g): self.opti.set_initial(x, xdiff(self.x0,xg))
            for a,ag in zip(self.acs,acs_g): self.opti.set_initial(a, ag)
            for u,ug in zip(self.us,us_g): self.opti.set_initial(u,ug)
            for f, fg in zip(self.fs, fs_g):
                [self.opti.set_initial(f[i], fg[i]) for i in range(len(f)) ]
            print("Got warm start")
        except:
            print("No warm start")
    
    
    def get_results(self):
        xs_sol = np.array([ self.opti.value(x) for x in self.xs ])
        us_sol = np.array([ self.opti.value(u) for u in self.us ])
        acs_sol = np.array([ self.opti.value(a) for a in self.acs ])
        fsol = {name: [] for name in self.allContactIds}
        fsol_to_ws = []
        for t in range(self.T):
            for i, (st_foot, sw_foot) in enumerate(\
                zip(self.runningModels[t].contactIds, self.runningModels[t].freeIds )):
                fsol[st_foot].append(self.opti.value(self.fs[t][i]))
                fsol[sw_foot].append(np.zeros(3))
            fsol_to_ws.append([self.opti.value(self.fs[t][i]) for i in range(len(self.fs[t]))])
        for foot in fsol: fsol[foot] = np.array(fsol[foot])

        return xs_sol, acs_sol, us_sol, fsol_to_ws, fsol


    def solve(self, gait, x0, x_ref, u_ref, v_lin_target, v_ang_target, guess=None):

        self.x0 = x0

        contactSequence = [ self.patternToId(p) for p in gait]
        self.T = T = len(gait) - 1

        self.runningModels = runningModels = [ self.casadiActionModels[contactSequence[t]] for t in range(T) ]
        self.terminalModel = terminalModel = self.casadiActionModels[contactSequence[T]]
        
        self.opti = opti = casadi.Opti()

        # Optimization variables
        self.dxs = dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]   
        self.acs = acs = [ opti.variable(model.nv) for model in runningModels ]
        self.us = us =  [ opti.variable(model.nu) for model in runningModels ]
        self.xs = xs =  [ m.integrate(x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]
        self.fs = fs = []
        for m in runningModels:
            f_tmp = [opti.variable(3) for _ in range(len(m.contactIds)) ]
            fs.append(f_tmp)
        self.fs = fs

        totalcost = 0
        opti.subject_to(dxs[0] == 0)
        for t in range(T):
            print(contactSequence[t])
            print(t)

            if (contactSequence[t] != contactSequence[t-1] and t >= 1): # If it is landing
                print('Landing on ', str(runningModels[t].contactIds)) 
                runningModels[t].constraint_landing_feet(xs[t], opti, x_ref)

            xnext,rcost = runningModels[t].calc(x=xs[t], u=us[t], a=acs[t],\
                                                    fs=fs[t], ocp=opti, x_ref=x_ref,\
                                                    u_ref=u_ref, v_lin_target=v_lin_target, \
                                                    v_ang_target=v_ang_target)

            opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*runningModels[t].nv) )  # x' = f(x,u)
            opti.subject_to(opti.bounded(-self.effort_limit,  us[t], self.effort_limit ))
            totalcost += rcost
    
        if (contactSequence[T] != contactSequence[T-1]): # If it is landing
            print('Landing on ', str(terminalModel.contactIds)) 
            terminalModel.constraint_landing_feet(xs[t], opti, x_ref)

        opti.subject_to(xs[T][terminalModel.nq :] == 0)
        opti.minimize(totalcost)
        opti.solver("ipopt") # set numerical backend

        self.warmstart(guess)

        ### SOLVE
        start_time = time() 
        opti.solve()
        self.iterationTime = time() - start_time


class ShootingNode():
    def __init__(self, cmodel, model, q0, allContactIds, contactIds):
        
        self.dt = conf.dt

        self.contactIds = contactIds
        self.freeIds = []

        self.baseId = baseId = model.getFrameId('base_link')
        
        self.cdata = cdata = cmodel.createData()
        data = model.createData()

        self.nq = nq = cmodel.nq
        self.nv = nv = cmodel.nv
        self.nx = nq+nv
        self.ndx = 2*nv
        self.nu = nv-6
        self.ntau = nv

        pin.framesForwardKinematics(model,data, q0)
        self.robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]
        [self.freeIds.append(idf) for idf in allContactIds if idf not in contactIds ]

        cx = casadi.SX.sym("x",self.nx,1)
        cq = casadi.SX.sym("q",cmodel.nq,1)
        cv = casadi.SX.sym("v",cmodel.nv,1)
        cx2 = casadi.SX.sym("x2",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        ca = casadi.SX.sym("a",cmodel.nv,1)
        ctau = casadi.SX.sym("tau",self.ntau,1)
        cdx = casadi.SX.sym("dx",self.ndx,1)
        cfs = [ casadi.SX.sym("f"+cmodel.frames[idf].name,3,1) for idf in self.contactIds ]
        R = casadi.SX.sym('R', 3, 3)
        R_ref = casadi.SX.sym('R_ref', 3, 3)

        
        ### Build force list for ABA
        forces = [ cpin.Force.Zero() for _ in cmodel.joints ]
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
        self.com = casadi.Function('com', [cx],[ cpin.centerOfMass(cmodel, cdata, cx[:nq]) ])
        # integrate(x,dx) =   [q+dq,v+dv],   with the q+dq function implemented with pin.integrate.
        self.integrate = casadi.Function('xplus', [cx,cdx],
                                        [ casadi.vertcat(cpin.integrate(cmodel, cx[:nq], cdx[:nv]),
                                                         cx[-nv:]+cdx[-nv:]) ])
        # integrate_q(q,dq) = pin.integrate(q,dq)
        self.integrate_q = casadi.Function('qplus', [cq,cv],
                                           [ cpin.integrate(cmodel,cq,cv) ])
        # Lie difference(x1,x2) = [ pin.difference(q1,q2),v2-v1 ]
        self.difference = casadi.Function('xminus', [cx, cx2],
                                          [ casadi.vertcat(cpin.difference(cmodel, cx2[:nq], cx[:nq]),
                                                         cx2[nq:]-cx[nq:]) ])

        cpin.forwardKinematics(cmodel,cdata,cx[:nq],cx[nq:],ca)
        cpin.updateFramePlacements(cmodel,cdata)
        # Base link position
        self.baseTranslation = casadi.Function('base_translation', [cx], [ cdata.oMf[baseId].translation ])
        self.baseRotation = casadi.Function('base_rotation', [cx], [ cdata.oMf[baseId].rotation ])
        self.log3 = casadi.Function('log', [R, R_ref], [cpin.log3(R.T @ R_ref)])
        # Base velocity
        self.baseVelocityLin = casadi.Function('base_velocity_linear', [cx], \
                                            [cpin.getFrameVelocity( cmodel, cdata, baseId,pin.LOCAL_WORLD_ALIGNED ).linear])
        self.baseVelocityAng = casadi.Function('base_velocity_angular', [cx],\
                                            [cpin.getFrameVelocity( cmodel,cdata,baseId,pin.LOCAL_WORLD_ALIGNED ).angular])
        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = {idf : casadi.Function('foot'+cmodel.frames[idf].name,
                                     [cx], [cdata.oMf[idf].translation]) \
                    for idf in allContactIds }
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = {idf: casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                       [cx],[cdata.oMf[idf].rotation]) \
                    for idf in allContactIds }
        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = {idf: casadi.Function('vfoot'+cmodel.frames[idf].name,
                                       [cx],[cpin.getFrameVelocity( cmodel, cdata, idf, pin.LOCAL_WORLD_ALIGNED ).linear])
                    for idf in allContactIds }
        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = {idf: casadi.Function('afoot'+cmodel.frames[idf].name,
                                       [cx,ca],[cpin.getFrameClassicalAcceleration( cmodel, cdata, idf, \
                                                pin.LOCAL_WORLD_ALIGNED ).linear])
                    for idf in allContactIds}

        
    def calc(self,x, u, a, fs, ocp, x_ref, u_ref, v_lin_target, v_ang_target):
        '''
        This function return xnext,cost
        '''

        dt = self.dt
        self.ocp = ocp

        # First split the concatenated forces in 3d vectors.
        # Split q,v from x
        nq = self.nq
        # Formulate tau = [0_6,u]
        tau = casadi.vertcat( np.zeros(6), u )

        # Euler integration, using directly the acceleration <a> introduced as a slack variable.
        vnext = x[nq:] + a*dt
        qnext = self.integrate_q(x[:nq], vnext*dt)
        xnext = casadi.vertcat(qnext,vnext)

        # Constraints
        self.constraint_standing_feet(x=x, a=a, f=fs)
        self.constraint_swing_feet(x=x, x_ref=x_ref, k = conf.k)
        self.constraint_dynamics(x=x, a=a, tau=tau, f=fs)

        # Cost functions:
        self.cost = 0
        self.force_reg_cost(x, fs)
        self.control_cost(u, u_ref)
        self.body_reg_cost(x=x, x_ref=x_ref)
        self.target_cost(x=x, v_lin_target=v_lin_target, v_ang_target=v_ang_target)
        self.sw_feet_cost(x)

        return xnext, self.cost



    def constraint_landing_feet(self, x, ocp, x_ref):
        for stFoot in self.contactIds:
            ocp.subject_to(self.feet[stFoot](x)[2] ==  \
                            self.feet[stFoot](x_ref)[2] )
            ocp.subject_to(self.vfeet[stFoot](x) == 0)
    
    def constraint_standing_feet(self, x, a, f):
        for i, stFoot in enumerate(self.contactIds):
            self.ocp.subject_to(self.afeet[stFoot](x,a) == 0) # stiff contact
            
            # Friction cone
            R = self.Rfeet[stFoot](x)
            f_ = f[i]
            fw = R @ f_
            self.ocp.subject_to(fw[2] >= 0)
            self.ocp.subject_to(conf.mu**2 * fw[2]**2 >= casadi.sumsqr(fw[0:2]))

    def constraint_dynamics(self, x, a, tau, f):
        self.ocp.subject_to( self.acc(x,tau, *f ) == a )

    def constraint_swing_feet(self, x, x_ref, k):
        for sw_foot in self.freeIds:
            self.ocp.subject_to(self.feet[sw_foot](x)[2] >= self.feet[sw_foot](x_ref)[2])
            #cost += 1e5 * casadi.sumsqr(self.feet[sw_foot](x)[2] - step_height) *self.dt
            self.ocp.subject_to(self.vfeet[sw_foot](x)[0:2] <= k* self.feet[sw_foot](x)[2])

    def force_reg_cost(self, x, f):
        for i, stFoot in enumerate(self.contactIds):
            R = self.Rfeet[stFoot](x)
            f_ = f[i]
            fw = R @ f_
            self.cost += conf.force_reg_weight * casadi.sumsqr(fw[2] - \
                                                self.robotweight/len(self.contactIds)) * self.dt
    
    def control_cost(self, u, u_ref):
        self.cost += conf.control_weight *casadi.sumsqr(u - u_ref) *self.dt

    def body_reg_cost(self, x, x_ref):
        self.cost += conf.base_reg_cost * casadi.sumsqr(x[3:7] - x_ref[3:7]) * self.dt
        #self.cost += conf.base_reg_cost * casadi.sumsqr( self.log3(self.baseRotation(x), self.baseRotation(x_ref)) ) * self.dt
        self.cost += conf.joints_reg_cost * casadi.sumsqr(x[7 : self.nq] - x_ref[7: self.nq]) *self.dt

    def target_cost(self, x, v_lin_target, v_ang_target):
        self.cost += casadi.sumsqr(conf.lin_vel_weight*(self.baseVelocityLin(x) - v_lin_target)) * self.dt
        self.cost += casadi.sumsqr(conf.ang_vel_weight*(self.baseVelocityAng(x) - v_ang_target)) * self.dt

    def sw_feet_cost(self, x):
        for sw_foot in self.freeIds:
            self.cost += conf.sw_feet_reg_cost * casadi.sumsqr(self.vfeet[sw_foot](x)[0:2]) *self.dt
        