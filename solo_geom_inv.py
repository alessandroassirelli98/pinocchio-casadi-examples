''' 
Inverse kinematics with friction cone constraint.

min Dq, f || q - q0 ||**2 + || f ||**2

subject to  q = pin.integrate(q0, Dq)
            sum(f_i) == weight
            com_z >= avg(p_z) # com height higher than the average height of the feet
            f_i X (com_pos - p_i) == 0 # zero angular momentum at the com 

            # The friction cone constraint can be one of the following:
            0)  || f_t ||**2 <= mu * || f_n ||**2   # in this case mu = 1. f_t and f_n are the tangential and orthogonal component of the contact force
            1)  || f @ k.T @ k - k @ k.T @ f ||**2 <= k.T @ f @ f.T @ k @k.T @ k    # k is the vector normal to the ground, while f is the vector of contact force
            2)  || f.T @ k ||**2 >= (cos(alpha_k))**2 || f ||**2 * || k ||**2   # here alpha_k is the angle of the friction cone, and it's set to 45 degree,
                                                                                which is equivalent to mu = 1

'''

import matplotlib
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex 
from pinocchio.visualize import GepettoVisualizer

### LOAD AND DISPLAY SOLO12
robot = robex.load('solo12')
model = robot.model
data = model.createData()

# Either 0, 1 or 2 This is used to choose between the 3 methods 
# They are equivalent, but the convergence is slower for some of them
FRICTION_CONE_CONTRAINT_TYPE = 2 

try:
    viz = pin.visualize.GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(robot.q0)
except:
    viz=None


def ground(xy):
    return (np.sin(xy[0]*3)/5 + np.cos(xy[1]**2*3)/20 + np.sin(xy[1]*xy[0]*5)/10)

def vizGround(viz,elevation,space,name='ground',color=[1,1,1,1]):
    space = 1e-1
    gv = viz.viewer.gui
    for i,x in enumerate(np.arange(-1,1,space)):
        gv.deleteNode(f'world/pinocchio/visuals/{name}_cx{i}',True)
        gv.addCurve(f'world/pinocchio/visuals/{name}_cx{i}',
                    [ [x,y,elevation([x,y])] for y in np.arange(-1,1,space) ], color)
    for i,y in enumerate(np.arange(-1,1,space)):
        gv.deleteNode(f'world/pinocchio/visuals/{name}_cy{i}',True)
        gv.addCurve(f'world/pinocchio/visuals/{name}_cy{i}',
                    [ [x,y,elevation([x,y])] for x in np.arange(-1,1,space) ], color)

if viz is not None: vizGround(viz,ground,.1,color=[1,1,1,1])

cxyz = casadi.SX.sym("xyz",3,1)

groundNormal = casadi.Function('groundNormal',[cxyz],[casadi.gradient(cxyz[2]-ground(cxyz[:2]),cxyz)])

feet = { idx: f.name for idx,f in enumerate(model.frames) if 'FOOT' in f.name }
mass = sum( [ y.mass for y in model.inertias ] )
grav = np.linalg.norm(model.gravity.linear)

### ---------------------------------------------------------------------- ###
### ACTION MODEL
class MX2SX:
    length = .3  # pendulum elongated dimension
    def __init__(self,model):
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()

        cq = casadi.SX.sym("q",model.nq,1)
        cdq = casadi.SX.sym("dq",model.nv,1)

        cpin.framesForwardKinematics(self.cmodel,self.cdata,cq)
        self.feet = \
            { idx: casadi.Function(f'{name}_pos', [cq], [cdata.oMf[idx].translation])  for idx,name in feet.items() }

        self.com = casadi.Function('com',[cq],[cpin.centerOfMass(cmodel,cdata,cq)])

        self.integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(cmodel,cq,cdq) ])

### PROBLEM
opti = casadi.Opti()
mx2sx = MX2SX(model)

# Decision variables
# Note that the contact forces are optimization variables
dq = opti.variable(model.nv)
q = mx2sx.integrate(robot.q0,dq)
fs = [ opti.variable(3) for _ in feet.values() ]

# constraint the com_xy position
opti.subject_to(mx2sx.com(q)[:2] == np.array([.1,.2]))
rq = q[7:]-robot.q0[7:]
totalcost = rq.T@rq

for f in mx2sx.feet.values():
    opti.subject_to( f(q)[2] == ground(f(q)[:2]) )  ### Foot in contact with the ground

torque = 0
com = mx2sx.com(q)
for idx,force in zip(mx2sx.feet.keys(),fs):
    pos = mx2sx.feet[idx](q)
    perp = groundNormal(pos)

    if (FRICTION_CONE_CONTRAINT_TYPE == 0):
            normal = perp/ casadi.norm_2(perp) 
            fn = force.T@normal
            ft = force - fn*normal
            opti.subject_to(  ft.T@ft <= fn.T@fn  )

    elif (FRICTION_CONE_CONTRAINT_TYPE == 1):
        fp = (force@perp.T@perp- perp@perp.T@force)
        opti.subject_to( fp.T@fp <=  (perp.T@force@force.T@perp@perp.T@perp))

    elif (FRICTION_CONE_CONTRAINT_TYPE == 2):
        opti.subject_to( (force.T@perp)@perp.T@force >= 1/2 * (force.T@force) * (perp.T@perp)) 
    else:
        raise ValueError("Please select a valid type of friction constraint")

    torque += casadi.cross(force,pos-com)
    
opti.subject_to( sum(fs) == np.array([ 0,0,mass*grav ]))  # Sum of forces is weight
opti.subject_to( torque == 0 )  # Sum of torques around COM is 0
totalcost += sum( [ f.T@f for f in fs ] )/10 # Add penalty on forces
opti.subject_to( mx2sx.com(q)[2] >= 
                (sum([ f(q)[2] for f in mx2sx.feet.values() ]))/len(feet))  # Center of mass above the feet

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
sol = opti.solve()
qopt = opti.value(q)
dqopt = opti.value(dq)
try:
    fs_opt = [ opti.value(f) for f in fs ]
except:
    fs_opt = [ np.zeros(3) for _ in feet ]
positions = [ np.array( p(qopt) ).T[0] for p in mx2sx.feet.values() ]
perps = [ np.array(groundNormal(p)).T[0] for p in positions ]  
normals = [ n/np.linalg.norm(n) for n in perps ]


### ---------------------------------------------------------------------- ###
## Visualization

if viz is not None: gv=viz.viewer.gui
for idx,force in zip(mx2sx.feet.keys(),fs_opt):
    #opti.subject_to( f[:2].T@f[:2] <= f[2]**2 )
    pos = np.array(mx2sx.feet[idx](qopt))
    perp = np.array(groundNormal(pos))
    print(perp.T)
    normal = perp/(perp.T@perp)

    if viz:
        gv.deleteNode(f'world/pinocchio/visuals/normal{idx}',True)
        gv.addCurve(f'world/pinocchio/visuals/normal{idx}', [ list(pos.flat), list((pos+normal).flat) ] , [ 0,0,1,1 ])
        
        gv.deleteNode(f'world/pinocchio/visuals/force{idx}',True)
        gv.addCurve(f'world/pinocchio/visuals/force{idx}', [ list(pos.flat), list(pos.flat+force/mass/grav) ] , [ 1,0,0,1 ])

if viz is not None:
    viz.display(qopt)
