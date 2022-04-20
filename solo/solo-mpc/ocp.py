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

        self.allContactIds = [ i for i,f in enumerate(cmodel.frames) if "FOOT" in f.name ]

        self.x0 = x0 = np.concatenate([robot.q0,np.zeros(model.nv)])
        self.u0 = u0 = np.zeros(robot.nv)

        self.effort_limit = np.ones(robot.nv - 6) *3   

        contactSequence = [ self.patternToId(p) for p in gait]
        self.casadiActionModels = { contacts: ShootingNode( cmodel=cmodel, model=model, allContactIds=self.allContactIds,\
                                                            contactIds=contacts, x0=x0 )  \
                                    for contacts in set(contactSequence) } # build the different shooting nodes

        # Initial config, also used for warm start
        

    def patternToId(self, gait):
        return tuple(self.allContactIds[i] for i,c in enumerate(gait) if c==1 )

    def solve(self, gait, x_ref, v_lin_target, v_ang_target):

        contactSequence = [ self.patternToId(p) for p in gait]
        T = len(gait) - 1

        runningModels = [ self.casadiActionModels[contactSequence[t]] for t in range(T) ]
        terminalModel = self.casadiActionModels[contactSequence[T]]
        
        opti = casadi.Opti()
        dxs = [ opti.variable(model.ndx) for model in runningModels+[terminalModel] ]     # state variable
        acs = [ opti.variable(model.nv) for model in runningModels ]                      # acceleration
        us =  [ opti.variable(model.nu) for model in runningModels ]                      # control variable
        fs =  [ opti.variable(3*len(model.contactIds) ) for model in runningModels ]      # contact force
        xs =  [ m.integrate(self.x0,dx) for m,dx in zip(runningModels+[terminalModel],dxs) ]
        
        totalcost = 0

        opti.subject_to(dxs[0] == 0)
        for t in range(T):
            print(contactSequence[t])

            if (contactSequence[t] != contactSequence[t-1] and t >=1): # If it is landing

                for stFoot in runningModels[t].contactIds:
                    opti.subject_to(runningModels[t].feet[stFoot](xs[t])[2] ==  \
                                    runningModels[t].feet[stFoot](self.x0)[2] )
                
                for stFoot in runningModels[t].contactIds:
                    opti.subject_to(runningModels[t].vfeet[stFoot](xs[t]) == 0)

                print('Landing on ', str(runningModels[t].contactIds)) 

            else:
                xnext,rcost = runningModels[t].calc(x=xs[t], u=us[t], a=acs[t],\
                                                    fs=fs[t], ocp=opti, x_ref=x_ref,\
                                                    v_lin_target=v_lin_target, v_ang_target=v_ang_target)

            opti.subject_to( runningModels[t].difference(xs[t + 1],xnext) == np.zeros(2*runningModels[t].nv) )  # x' = f(x,u)
            opti.subject_to(opti.bounded(-self.effort_limit,  us[t], self.effort_limit ))
            totalcost += rcost

        opti.minimize(totalcost)
        opti.solver("ipopt") # set numerical backend

        try:
            guesses = np.load(path + '/sol.npy',allow_pickle=True).item()
            xs_g = guesses['xs']
            us_g = guesses['us']
            acs_g = guesses['acs']
            #fs_g = guesses['fs']

            def xdiff(x1,x2):
                nq = self.model.nq
                return np.concatenate([
                    pin.difference(self.model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

            for x,xg in zip(dxs,xs_g): opti.set_initial(x, xdiff(self.x0,xg))
            for u,ug in zip(us,us_g): opti.set_initial(u,ug)
            print("Got warm start")
        except:
            print("No warm start")

        ### SOLVE
        sol = opti.solve()

        xs_sol = np.array([ opti.value(x) for x in xs ])
        us_sol = np.array([ opti.value(u) for u in us ])
        return xs_sol, us_sol


class ShootingNode():
    def __init__(self, cmodel, model, allContactIds, contactIds, x0):
        
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

        pin.framesForwardKinematics(model,data, x0[:model.nq])
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

        
    def calc(self,x, u, a, fs, ocp, x_ref, v_lin_target, v_ang_target):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        fs = [ fs[3 * i : 3 * i + 3] for i,_ in enumerate(self.contactIds) ]   # split
        # Split q,v from x
        nq = self.nq
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
        cost += 1e1 *casadi.sumsqr(u) *self.dt
        cost += 1e1 *casadi.sumsqr(x[3:7] - x_ref[3:7]) * self.dt
        cost += 1e2 *casadi.sumsqr(x[7 : nq] - x_ref[7: nq]) *self.dt

        #cost += casadi.sumsqr(position_weight*(self.baseTranslation(x) - self.baseTranslation(x0))) *self.dt
        cost += casadi.sumsqr(conf.lin_vel_weight*(self.baseVelocityLin(x) - v_lin_target)) * self.dt
        cost += casadi.sumsqr(conf.ang_vel_weight*(self.baseVelocityAng(x) - v_ang_target)) * self.dt
        
        # Contact constraints
        for i, stFoot in enumerate(self.contactIds):
            ocp.subject_to(self.afeet[stFoot](x,a) == 0) # stiff contact

            # Friction cone
            R = self.Rfeet[stFoot](x)
            f = fs[i]
            fw = R @ f
            ocp.subject_to(fw[2] >= 0)
            ocp.subject_to(conf.mu**2 * fw[2]**2 >= casadi.sumsqr(fw[0:2]))
            cost += conf.force_reg_weight * casadi.sumsqr(fw[2] - self.robotweight/len(self.contactIds)) * self.dt
        
        for sw_foot in self.freeIds:
            ocp.subject_to(self.feet[sw_foot](x)[2] >= self.feet[sw_foot](x_ref)[2])
            #cost += 1e5 * casadi.sumsqr(self.feet[sw_foot](x)[2] - step_height) *self.dt
            ocp.subject_to(self.vfeet[sw_foot](x)[0:2] <= conf.k* self.feet[sw_foot](x)[2])


        return xnext,cost