import example_robot_data
import pinocchio as pin
import numpy as np

def create_double_pendulum_model(with_display=True):
    robot = example_robot_data.load('double_pendulum')
    # Next 6 lines are to have the pendulum align to face default front plane of the viewer.
    robot.model.jointPlacements[1] = pin.SE3(pin.utils.rotate('z',-np.pi/2),np.zeros(3))
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint == 0:
            M = g.placement
            M.rotation = pin.utils.rotate('z',-np.pi/2)
            g.placement = M
    # Next 10 lines are to initialize the viewer with the pendulum model.
    robot.model.addFrame(pin.Frame('tip',2,5,pin.SE3(np.eye(3),np.array([0,0,0.2])),pin.OP_FRAME))
    viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    try:
        viz.initViewer()
        viz.loadViewerModel()
        viz.viewer.gui.setBackgroundColor1(viz.windowID,[0.,0.,0.,1.])
        viz.viewer.gui.setBackgroundColor2(viz.windowID,[0.,0.,0.,1.])
        viz.display(robot.q0)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should do something about it")
        viz = None

    return robot,viz
