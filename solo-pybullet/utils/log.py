import numpy as np

class Log:
    def __init__(self):
        self.foot_tracking_cost = 0
        self.force_reg_weight = 0
        self.control_weight = 0
        self.base_reg_cost = 0
        self.base_translation_weight = 0
        self.joints_reg_cost = 0
        self.joints_vel_reg_cost = 0
        self.terminal_cost = 0

        self.constraints = {'terminal_v_zero': False,\
                            'sw_foot_geq_ref': False,\
                            'effort_limit': False }