import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt
from numpy.linalg import norm,inv,pinv,svd,eig
import talos_low


def computeStaticPosture(model,contactIds,refConfig):
    # The pinocchio model is what we are really interested by.
    model.q0 = refConfig
    cmodel = cpin.Model(robot.model)
    data = model.createData()
    cdata = cmodel.createData()

    pin.framesForwardKinematics(model,data,model.q0)
    robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]

    ### HELPERS
    cq = casadi.SX.sym("q",cmodel.nq,1)
    cdq = casadi.SX.sym("q",cmodel.nv,1)
    cv = casadi.SX.sym("v",cmodel.nv,1)
    ca = casadi.SX.sym("a",cmodel.nv,1)
    ctau = casadi.SX.sym("a",cmodel.nv,1)
    
    cpin.framesForwardKinematics(cmodel,cdata,cq)
    cpin.centerOfMass(cmodel,cdata,cq)
    
    com = cdata.com[0]
    fref = np.array([0,0,robotweight/len(contactIds)])
    
    qint = casadi.Function('qint', [cq,cdq], [cpin.integrate(cmodel,cq,cdq)])
    footPos = { cid: casadi.Function('footPos',[cq],[cdata.oMf[cid].translation]) for cid in contactIds }
    footRot = { cid: casadi.Function('footRot',[cq],[cdata.oMf[cid].rotation]) for cid in contactIds }
    lever = { cid: casadi.Function('lever',[cq],[com-cdata.oMf[cid].translation]) for cid in contactIds }
    torque = { cid: casadi.Function('lever',[cq],[casadi.cross(com-cdata.oMf[cid].translation,fref)]) for cid in contactIds }
    ###
    '''
    min_q   ||q-qref||  
    st
        contact flat on the ground
        aba(q,0,tau,f0) = 0
    '''

    opti = casadi.Opti()
    
    var_dq =  opti.variable(model.nv)
    var_q =  qint(model.q0,var_dq)

    totalcost = casadi.sumsqr(var_dq[6:])

    # totaltorque = 0
    # for cid in contactIds:
    #     totaltorque += casadi.cross(com-data.oMf[cid].translation,fref)
    # totaltorque = casadi.Function('totaltorque',[cq],[totaltorque])
    #opti.subject_to(totaltorque(var_q) == 0)
    opti.subject_to(sum( [ t(var_q) for t in  torque.values() ]) == 0)

    for cid in contactIds:
        # Altitude = 0
        opti.subject_to(footPos[cid](var_q)[2] == 0)
        # Orientation = 0
        opti.subject_to(footRot[cid](var_q)[0,2] == 0)
        opti.subject_to(footRot[cid](var_q)[1,2] == 0)

        
    # ## SOLVE ######################################################################
    opti.minimize(totalcost)
    s_opt = {
        'tol' : 1e-6,
        'acceptable_tol': 1e-6,
        'max_iter': 400,
    }
    opti.solver("ipopt", {}, s_opt) # set numerical backend
    
    sol = opti.solve_limited()
    q_opt = opti.value(var_q)
    return q_opt

# ###############################################################################
def computeStaticTorque(model,q,contactIds):
    data = model.createData()

    pin.framesForwardKinematics(model,data,q)
    robotweight = -sum([Y.mass for Y in model.inertias]) * model.gravity.linear[2]
    fref = pin.Force(np.array([0,0,robotweight/len(contactIds)]),np.zeros(3))

    forces = pin.StdVec_Force(model.njoints,pin.Force.Zero())
    for cid in contactIds:
        forces[model.frames[cid].parentJoint] = model.frames[cid].placement*fref

    return pin.rnea(model,data,q,np.zeros(model.nv),np.zeros(model.nv),forces)


# ###############################################################################
# From https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
from itertools import chain, combinations

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


def save(postures,torques,model):
    filename = f"static-postures-{model.name}-{model.njoints}.npy"
    np.save(open(filename, "wb"),
            {'postures':postures, 'torques': torques})
    print(f"Saved in '{filename}'!")

def load(model):
    filename = f"static-postures-{model.name}-{model.njoints}.npy"
    print(f"Load posture from '{filename}'!")
    d = np.load(filename, allow_pickle=True)[()]
    return d['postures'],d['torques']

# ###############################################################################
# ### MAIN ######################################################################
# ###############################################################################

if __name__ == "__main__":
    pin.SE3.__repr__=pin.SE3.__str__
    np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=10000)

    robot = talos_low.load()
    
    try:
        viz = pin.visualize.GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        gv = viz.viewer.gui
    except:
        print("No viewer"  )

    contactIds = [ i for i,f in enumerate(robot.model.frames) if "sole_link" in f.name ]
        
    postures = {
        tuple(cids): computeStaticPosture(robot.model,cids,robot.q0) for cids in powerset(contactIds)
    }
    torques = {
        tuple(cids): computeStaticTorque(robot.model,postures[cids],cids) for cids in powerset(contactIds)
    }
    
    save(postures,torques,robot.model)
    load(robot.model)
    
