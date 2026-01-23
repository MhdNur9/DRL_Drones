import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


base_dir = os.path.dirname(os.path.abspath(__file__))
# usd_path = os.path.join(base_dir, "/home/mirpalab-sim/RL_catch/assets/drone/firefly.usd")
usd_path = os.path.join(base_dir, "/home/mirpalab-sim/RL_catch/assets/drone/firefly_with_basket2.usd")
firefly_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        scale=(2.0, 2.0, 2.0),
        # scale=(1.0, 1.0, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "rotor_0_joint": 200.0,
            "rotor_1_joint": -200.0,
            "rotor_2_joint": 200.0,
            "rotor_3_joint": -200.0,
            "rotor_4_joint": 200.0,
            "rotor_5_joint": -200.0,
        },
        # joint_vel={
        #     ".*": 0.0,
        # },

    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
