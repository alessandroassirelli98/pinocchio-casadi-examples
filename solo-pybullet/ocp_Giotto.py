import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from pinocchio.visualize import GepettoVisualizer
import ocp_parameters_conf as conf
from time import time
import os

import proxnlp
from proxnlp.manifolds import MultibodyPhaseSpace, VectorSpace
from proxnlp.utils import CasadiFunction, plot_pd_errs


plt.style.use('seaborn')
path = os.getcwd()


class ShootingNode():
    def __init__(self, cmodel, model, q0, allContactIds, contactIds, dt):

        self.dt = dt

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

        pin.framesForwardKinematics(model, data, q0)
        self.robotweight = - \
            sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]
        [self.freeIds.append(idf)
         for idf in allContactIds if idf not in contactIds]

        cx = casadi.SX.sym("x", self.nx, 1)
        cq = casadi.SX.sym("q", cmodel.nq, 1)
        cv = casadi.SX.sym("v", cmodel.nv, 1)
        cx2 = casadi.SX.sym("x2", self.nx, 1)
        cu = casadi.SX.sym("u", self.nu, 1)
        ca = casadi.SX.sym("a", cmodel.nv, 1)
        ctau = casadi.SX.sym("tau", self.ntau, 1)
        cdx = casadi.SX.sym("dx", self.ndx, 1)
        cfs = [casadi.SX.sym("f"+cmodel.frames[idf].name, 3, 1)
               for idf in self.contactIds]
        R = casadi.SX.sym('R', 3, 3)
        R_ref = casadi.SX.sym('R_ref', 3, 3)

        # Build force list for ABA
        forces = [cpin.Force.Zero() for _ in cmodel.joints]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        assert(len(set([cmodel.frames[idf].parentJoint for idf in contactIds])) == len(
            contactIds))
        for f, idf in zip(cfs, self.contactIds):
            # Contact forces introduced in ABA as spatial forces at joint frame.
            forces[cmodel.frames[idf].parentJoint] = cmodel.frames[idf].placement * \
                cpin.Force(f, 0*f)
        self.forces = cpin.StdVec_Force()
        for f in forces:
            self.forces.append(f)

        acc = cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], ctau, self.forces)

        # Casadi MX functions
        # acceleration(x,u,f)  = ABA(q,v,tau,f) with x=q,v, tau=u, and f built using StdVec_Force syntaxt
        self.acc = casadi.Function('acc', [cx, ctau]+cfs, [acc])
        # com(x) = centerOfMass(x[:nq])
        self.com = casadi.Function(
            'com', [cx], [cpin.centerOfMass(cmodel, cdata, cx[:nq])])
        # integrate(x,dx) =   [q+dq,v+dv],   with the q+dq function implemented with pin.integrate.
        self.integrate = casadi.Function('xplus', [cx, cdx],
                                         [casadi.vertcat(cpin.integrate(cmodel, cx[:nq], cdx[:nv]),
                                                         cx[-nv:]+cdx[-nv:])])
        # integrate_q(q,dq) = pin.integrate(q,dq)
        self.integrate_q = casadi.Function('qplus', [cq, cv],
                                           [cpin.integrate(cmodel, cq, cv)])
        # Lie difference(x1,x2) = [ pin.difference(q1,q2),v2-v1 ]
        self.difference = casadi.Function('xminus', [cx, cx2],
                                          [casadi.vertcat(cpin.difference(cmodel, cx2[:nq], cx[:nq]),
                                                          cx2[nq:]-cx[nq:])])

        vel_reference_frame = pin.LOCAL

        cpin.forwardKinematics(cmodel, cdata, cx[:nq], cx[nq:], ca)
        cpin.updateFramePlacements(cmodel, cdata)
        # Base link position
        self.baseTranslation = casadi.Function(
            'base_translation', [cx], [cdata.oMf[baseId].translation])
        self.baseRotation = casadi.Function(
            'base_rotation', [cx], [cdata.oMf[baseId].rotation])
        self.log3 = casadi.Function(
            'log', [R, R_ref], [cpin.log3(R.T @ R_ref)])
        # Base velocity
        self.baseVelocityLin = casadi.Function('base_velocity_linear', [cx],
                                               [cpin.getFrameVelocity(cmodel, cdata, baseId, vel_reference_frame).linear])
        self.baseVelocityAng = casadi.Function('base_velocity_angular', [cx],
                                               [cpin.getFrameVelocity(cmodel, cdata, baseId, vel_reference_frame).angular])
        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = {idf: casadi.Function('foot'+cmodel.frames[idf].name,
                                          [cx], [cdata.oMf[idf].translation])
                     for idf in allContactIds}
        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = {idf: casadi.Function('Rfoot'+cmodel.frames[idf].name,
                                           [cx], [cdata.oMf[idf].rotation])
                      for idf in allContactIds}
        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = {idf: casadi.Function('vfoot'+cmodel.frames[idf].name,
                                           [cx], [cpin.getFrameVelocity(cmodel, cdata, idf, pin.LOCAL_WORLD_ALIGNED).linear])
                      for idf in allContactIds}
        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = {idf: casadi.Function('afoot'+cmodel.frames[idf].name,
                                           [cx, ca], [cpin.getFrameClassicalAcceleration(cmodel, cdata, idf,
                                                                                         pin.LOCAL_WORLD_ALIGNED).linear])
                      for idf in allContactIds}

    def init(self, x, a, u, fs):
        self.x = x
        self.a = a
        self.u = u
        self.tau = casadi.vertcat(np.zeros(6), u)
        self.fs = fs

    def calc(self, x_ref, u_ref, target):
        '''
        This function return xnext,cost
        '''

        dt = self.dt

        # First split the concatenated forces in 3d vectors.
        nq = self.nq

        # Euler integration, using directly the acceleration <a> introduced as a slack variable.
        use_rk2 = False
        if not use_rk2:
            vnext = self.x[nq:] + self.a*dt
            qnext = self.integrate_q(self.x[:nq], vnext*dt)
        else:
            # half-dt step over x=(q,v)
            vm = self.x[nq:] + self.a*.5*dt
            qm = self.integrate_q(self.x[:nq], .5 * self.x[nq:]*dt)
            xm = casadi.vertcat(qm, vm)
            amid = self.acc(xm, self.tau, *self.fs)

            # then simple Euler step over (qm, vm)
            qnext = self.integrate_q(qm, vm*dt)
            vnext = vm + amid*dt

        xnext = casadi.vertcat(qnext, vnext)

        # Cost functions:
        self.compute_cost(x_ref, u_ref, target)

        return xnext, self.cost

    def constraint_landing_feet_eq(self, x_ref):
        eq = []
        for stFoot in self.contactIds:
            eq.append(self.feet[stFoot](self.x)[2] -
                      self.feet[stFoot](x_ref)[2])
            eq.append(self.vfeet[stFoot](self.x))
        return(casadi.vertcat(*eq))

    def constraint_standing_feet_eq(self):
        eq = []
        for stFoot in self.contactIds:
            eq.append(self.afeet[stFoot](self.x, self.a))  # stiff contact

        return(casadi.vertcat(*eq))

    def constraint_standing_feet_ineq(self):
        # Friction cone
        ineq = []
        for i, stFoot in enumerate(self.contactIds):
            R = self.Rfeet[stFoot](self.x)
            f_ = self.fs[i]
            fw = R @ f_
            ineq.append(-fw[2])
            ineq.append( fw[0] - conf.mu*fw[2] )
            ineq.append( -fw[0] - conf.mu*fw[2] )
            ineq.append(  fw[1] - conf.mu*fw[2] )
            ineq.append( -fw[1] - conf.mu*fw[2] )

        return(casadi.vertcat(*ineq))

    def constraint_dynamics_eq(self):
        eq = []
        eq.append(self.acc(self.x, self.tau, *self.fs) - self.a)
        return(casadi.vertcat(*eq))

    def constraint_swing_feet_ineq(self, x_ref, k):
        ineq = []
        for sw_foot in self.freeIds:
            ineq.append(-self.feet[sw_foot](self.x)
                        [2] + self.feet[sw_foot](x_ref)[2])
            ineq.append(self.vfeet[sw_foot](self.x)[
                        0:2] - k[0] * self.feet[sw_foot](self.x)[2])

        return(casadi.vertcat(*ineq))

    def force_reg_cost(self):
        for i, stFoot in enumerate(self.contactIds):
            R = self.Rfeet[stFoot](self.x)
            f_ = self.fs[i]
            fw = R @ f_
            self.cost += conf.force_reg_weight * casadi.sumsqr(fw[2] -
                                                               self.robotweight/len(self.contactIds)) * self.dt

    def control_cost(self, u_ref):
        self.cost += conf.control_weight * \
            casadi.sumsqr(self.u - u_ref) * self.dt

    def body_reg_cost(self, x_ref):
        self.cost += conf.base_reg_cost * \
            casadi.sumsqr(self.x[3:7] - x_ref[3:7]) * self.dt
        self.cost += conf.base_translation_weight * casadi.sumsqr(self.baseTranslation(self.x) - self.baseTranslation(x_ref)) * self.dt
        self.cost += conf.joints_reg_cost * \
            casadi.sumsqr(self.x[7: self.nq] - x_ref[7: self.nq]) * self.dt
        self.cost += conf.joints_vel_reg_cost * \
            casadi.sumsqr(self.x[self.nq + 6:] - x_ref[self.nq + 6:]) * self.dt

    def target_cost(self, target):
        #self.cost += casadi.sumsqr(conf.lin_vel_weight*(self.baseVelocityLin(self.x) - v_lin_target)) * self.dt
        #self.cost += casadi.sumsqr(conf.ang_vel_weight*(self.baseVelocityAng(self.x) - v_ang_target)) * self.dt
        self.cost += casadi.sumsqr(conf.foot_tracking_cost *
                                   (self.feet[self.freeIds[0]](self.x) - target)) * self.dt

    def sw_feet_cost(self):
        for sw_foot in self.freeIds:
            self.cost += conf.sw_feet_reg_cost * \
                casadi.sumsqr(self.vfeet[sw_foot](self.x)[0:2]) * self.dt
    
    def st_feet_cost(self):
        for stFoot in self.contactIds:
            self.cost += conf.stiff_contact_weight *casadi.sumsqr(self.afeet[stFoot](self.x, self.a)) * self.dt # stiff contact
            self.cost += conf.stiff_contact_weight *casadi.sumsqr(self.vfeet[stFoot](self.x)) * self.dt # stiff contact

    def compute_cost(self, x_ref, u_ref, target):
        self.cost = 0
        self.force_reg_cost()
        self.control_cost(u_ref)
        self.body_reg_cost(x_ref=x_ref)
        #self.st_feet_cost()
        self.target_cost(target=target)

        return self.cost


class OCP():
    def __init__(self, robot, gait, x0, x_ref, u_ref, target, dt=0.015):
        self.robot = robot
        self.model = model = robot.model
        self.cmodel = cmodel = cpin.Model(robot.model)

        self.nq = robot.nq
        self.nv = robot.nv
        self.nu = robot.nv - 6

        self.x0 = x0
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.target = target
        self.dt = dt

        self.T = len(gait) - 1

        self.iterationTime = 0

        self.allContactIds = [i for i, f in enumerate(
            cmodel.frames) if "FOOT" in f.name]
        self.contactNames = [f.name for f in cmodel.frames if "FOOT" in f.name]
        self.effort_limit = np.ones(robot.nv - 6) * 3

        self.contactSequence = [self.patternToId(p) for p in gait]
        self.casadiActionModels = {contacts: ShootingNode(cmodel=cmodel, model=model, q0=robot.q0, allContactIds=self.allContactIds,
                                                          contactIds=contacts, dt=self.dt)
                                   for contacts in set(self.contactSequence)}  # build the different shooting nodes

    def patternToId(self, gait):
        return tuple(self.allContactIds[i] for i, c in enumerate(gait) if c == 1)

    def warmstart(self, guess=None):
        for g in guess:
            if guess[g] == []:
                print("No warmstart provided")
                return 0    
        try:
            xs_g = guess['xs']
            us_g = guess['us']
            acs_g = guess['acs']
            fs_g = guess['fs']

            def xdiff(x1, x2):
                nq = self.model.nq
                return np.concatenate([
                    pin.difference(self.model, x1[:nq], x2[:nq]), x2[nq:]-x1[nq:]])

            for x, xg in zip(self.dxs, xs_g):
                self.opti.set_initial(x, xdiff(self.x0, xg))
            for a, ag in zip(self.acs, acs_g):
                self.opti.set_initial(a, ag)
            for u, ug in zip(self.us, us_g):
                self.opti.set_initial(u, ug)
            for f, fg in zip(self.fs, fs_g):
                fgsplit = np.split(fg, len(f))
                fgc = []
                [fgc.append(f) for f in fgsplit]
                [self.opti.set_initial(f[i], fgc[i])
                    for i in range(len(f))]
            print("Got warm start")
        except:
            print("Can't load warm start")

    def get_results(self):

        dxs_sol = np.array([self.opti.value(dx) for dx in self.dxs])
        xs_sol = np.array([self.opti.value(x) for x in self.xs])
        us_sol = np.array([self.opti.value(u) for u in self.us])
        acs_sol = np.array([self.opti.value(a) for a in self.acs])
        fsol = {name: [] for name in self.allContactIds}
        fsol_to_ws = []
        for t in range(self.T):
            for i, st_foot in enumerate(self.runningModels[t].contactIds):
                fsol[st_foot].append(self.opti.value(self.fs[t][i]))
            fsol_to_ws.append(np.concatenate([self.opti.value(
                self.fs[t][j]) for j in range(len(self.terminalModel.contactIds))]))
        for foot in fsol:
            fsol[foot] = np.array(fsol[foot])

        return dxs_sol, xs_sol, acs_sol, us_sol, fsol_to_ws

    def get_feet_position(self, xs_sol):
        feet_log = {i: [] for i in self.allContactIds}
        for foot in feet_log:
            tmp = []
            for i in range(len(xs_sol)):
                tmp += [self.terminalModel.feet[foot](xs_sol[i]).full()[:, 0]]
            feet_log[foot] = np.array(tmp)

        return feet_log

    def get_feet_velocities(self, xs_sol):
        feet_log = {i: [] for i in self.allContactIds}
        for foot in feet_log:
            tmp = []
            for i in range(len(xs_sol)):
                tmp += [self.terminalModel.vfeet[foot](xs_sol[i]).full()[:, 0]]
            feet_log[foot] = np.array(tmp)

        return feet_log

    def get_base_log(self, xs_sol):

        base_pos_log = []
        [base_pos_log.append(self.terminalModel.baseTranslation(
            xs_sol[i]).full()[:, 0]) for i in range(len(xs_sol))]
        base_pos_log = np.array(base_pos_log)
        return base_pos_log

    def make_ocp(self):
        totalcost = 0
        eq = []
        ineq = []

        eq.append(self.dxs[0])
        for t in range(self.T):
            print(self.contactSequence[t])
            self.runningModels[t].init(
                self.xs[t], self.acs[t], self.us[t], self.fs[t])

            xnext, rcost = self.runningModels[t].calc(x_ref=self.x_ref, u_ref=self.u_ref,
                                                      target=self.target[t])

            # Constraints
            eq.append(self.runningModels[t].constraint_standing_feet_eq())
            eq.append(self.runningModels[t].constraint_dynamics_eq())
            eq.append(self.runningModels[t].difference(
                self.xs[t + 1], xnext) - np.zeros(2*self.runningModels[t].nv))

            ineq.append(self.runningModels[t].constraint_standing_feet_ineq())
            ineq.append(self.us[t] - self.effort_limit)
            ineq.append(-self.us[t] - self.effort_limit)

            totalcost += rcost

        eq.append(self.xs[self.T][self.terminalModel.nq:])

        eq_constraints = casadi.vertcat(*eq)
        ineq_constraints = casadi.vertcat(*ineq)

        return totalcost, eq_constraints, ineq_constraints

    def use_ipopt_solver(self, guess=None):

        opti = casadi.Opti()
        self.opti = opti
        # Optimization variables
        self.dxs = [opti.variable(model.ndx)
                    for model in self.runningModels+[self.terminalModel]]
        self.acs = [opti.variable(model.nv) for model in self.runningModels]
        self.us = [opti.variable(model.nu) for model in self.runningModels]
        self.xs = [m.integrate(self.x0, dx) for m, dx in zip(
            self.runningModels+[self.terminalModel], self.dxs)]
        self.fs = []
        for m in self.runningModels:
            f_tmp = [opti.variable(3) for _ in range(len(m.contactIds))]
            self.fs.append(f_tmp)
        self.fs = self.fs

        cost, eq_constraints, ineq_constraints = self.make_ocp()

        opti.minimize(cost)

        opti.subject_to(eq_constraints == 0)
        #opti.subject_to(ineq_constraints <= 0)

        p_opts = {}
        s_opts = { "tol": 1e-4,
             "acceptable_tol":1e-4,
            # "max_iter": 21,
            # "compl_inf_tol": 1e-2,
            # "constr_viol_tol": 1e-2
            # "resto_failure_feasibility_threshold": 1
            # "linear_solver": "ma57"
        }

        opti.solver("ipopt", p_opts,
                    s_opts)

        self.warmstart(guess)

        # SOLVE
        opti.solve_limited()

    def solve(self, guess=None):

        self.runningModels = [
            self.casadiActionModels[self.contactSequence[t]] for t in range(self.T)]
        self.terminalModel = self.casadiActionModels[self.contactSequence[self.T]]

        start_time = time()
        self.use_ipopt_solver(guess)


        self.iterationTime = time() - start_time
