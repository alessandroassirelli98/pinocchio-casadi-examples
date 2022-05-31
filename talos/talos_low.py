import example_robot_data as robex
import pinocchio as pin

def load():
    robot = robex.load('talos')
    jointToLockNames =     [
        # "universe",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        #"arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        #"arm_right_4_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        "gripper_left_joint",
        "gripper_right_joint",
        "head_1_joint",
        "head_2_joint",
        "torso_1_joint",
        "torso_2_joint",
    ]
    jointToLockIds = [i for (i, n) in enumerate(robot.model.names) if n in jointToLockNames]
    robot.model, [robot.collision_model, robot.visual_model] = pin.buildReducedModel(
        robot.model,
        [robot.collision_model, robot.visual_model],
        jointToLockIds,
        robot.q0,
    )
    robot.q0 = robot.model.referenceConfigurations['half_sitting']
    robot.rebuildData()

    return robot
