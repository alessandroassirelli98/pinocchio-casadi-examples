'''
Minimal example program using Casadi with pinocchio algorithmic derivatives.
The problem solved is:
min_q   || q - q_0 ||**2  st p(q) = p_0,   q_0-3<=q<=q_0+3

where p(q) \in R^3 is the end-effector position and q_0 is a reference (target) configuration
The joint limits are added to force the unicity of the solution.
'''

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
from numpy.linalg import norm,inv,pinv,eig,svd

### ROBOT MODELS
### Load the robot from example robot data, create pinocchio and casadi model and data
robot = robex.load('talos_arm')
model = robot.model
data = model.createData()

cmodel = cpin.Model(model)
cdata = cmodel.createData()

### NLP PARAMETERS
### Choose the paramters so that the NLP problem is a reasonible inverse-geometry problem
# Target endeffector position
target = np.array([.1,.2,.3])
# Target robot configuration
target_q = robot.q0
# Frame index of the end effector
fid = model.getFrameId('gripper_left_fingertip_2_link')
# Initial guess for the NLP solvers
#guess_q = target_q  ### With this initial value, we are more prone to local minima that make the final check impossible
guess_q = np.array([-1, -1, -2,  0. , -0, -0, -0. ])  ### This one is closer to the global (?) optima.

### CASADI WRAPPERS
### Define casadi function that wraps the forward geometry.
sq = casadi.SX.sym("q",model.nq)
cpin.framesForwardKinematics(cmodel,cdata,sq)

# First wrapper without explicit jacobian (it will be computed symbolically by casadi)
endeff = casadi.Function('endeff', [ sq ], [ cdata.oMf[fid].translation ])

# Second wrapper with explicit jacobian, computed by pinocchio
MULTERROR = 1  ### Make me not 1 to artificially introduce error in the derivative.
cpin.computeJointJacobians(cmodel,cdata,sq)
sp = casadi.SX.sym('p',3)
Jendeff = casadi.Function('Jendeff', [ sq, sp ],
                          [ MULTERROR*cpin.getFrameJacobian(cmodel,cdata,fid,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:] ])
endeff_withdiff = casadi.Function('endeff', [ sq ], [ cdata.oMf[fid].translation ],
                                  {"custom_jacobian": Jendeff, "jac_penalty":0})

### CHECK DERIVATIVES
### Sanity check to see that, at a random configuration, both Casadi and Pinocchio derivatives match.
q_check = pin.randomConfiguration(model)
p_check = endeff(q_check)
Jsym = endeff.jacobian()(q_check,p_check)
Jalg = Jendeff(q_check,p_check)/MULTERROR   ### If multerror is not 1, correct the result to recover the exact derivative.
assert(casadi.sumsqr(Jsym-Jalg)<1e-8)


### PROBLEM with symbolic diff
### Definition of a first NLP with symbolic (casadi) derivatives.
opti = casadi.Opti()
var_q =  opti.variable(model.nq)
opti.subject_to(endeff(var_q) == target )
opti.subject_to(  var_q <= target_q + 3 )
opti.subject_to(  var_q >= target_q - 3 )
opti.minimize( casadi.sumsqr(var_q-target_q) )

opti.solver("ipopt") # set numerical backend
opti.set_initial(var_q,guess_q)
sol = opti.solve_limited()
opt_q = opti.value(var_q)


### PROBLEM with algorithmic diff
### Definition of a second NLP with algorithmic (pinocchio) derivatives.
opti2 = casadi.Opti()
var2_q =  opti2.variable(model.nq)
opti2.subject_to(endeff_withdiff(var2_q) == target )
opti2.subject_to(  var2_q <= target_q + 3 )
opti2.subject_to(  var2_q >= target_q - 3 )
opti2.minimize( casadi.sumsqr(var2_q-target_q) )

opti2.solver("ipopt") # set numerical backend
opti2.set_initial(var2_q,guess_q)
sol2 = opti2.solve_limited()
opt2_q = opti2.value(var2_q)

### FINAL CHECK
### Check that the two solutions match... 
assert( norm(opt2_q-opt_q)<1e-6 )
 
