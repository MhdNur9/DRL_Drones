# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from envs.throw_T3_sc3.lift_env_cfg import LiftEnvCfg

from envs.throw_T3_sc3 import mdp as mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.Target_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.0], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
        self.scene.person = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/person",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/1X/Neo/Neo.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.65, -1.5, -0.2),
        joint_pos={".*": 0.0},
        # choose ONE:
        rot=(0.707, 0, 0, 0.707),   # 90° about X  (what you currently have)
        # rot=(0, 0, 0.707, 0.707), # 90° about Z
        # rot=(0, 1, 0, 0),         # 180° flip around Y
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # acts on all unless you split groups
            stiffness={
                # torso/neck
                r"j_spine_.*": 20.0,
                r"j_neck_[xyz]": 10.0,

                # shoulders & arms
                r"j_[lr]_shoulder_[xyz]": 10.0,   # was upper_arm
                r"j_[lr]_elbow_[yz]": 2.0,        # was lower_arm
                r"j_[lr]_wrist_[xy]": 2.0,

                # hips & legs
                r"j_[lr]_hip_[xyz]": 10.0,        # was thigh:*
                r"j_[lr]_(upperknee|lowerknee)_y": 5.0,  # was shin/thigh buckets
                r"j_[lr]_ankle_[xy]": 2.0,        # was foot
            },
            damping={
                r"j_spine_.*": 5.0,
                r"j_neck_[xyz]": 5.0,

                r"j_[lr]_shoulder_[xyz]": 5.0,
                r"j_[lr]_elbow_[yz]": 1.0,
                r"j_[lr]_wrist_[xy]": 1.0,

                r"j_[lr]_hip_[xyz]": 5.0,
                r"j_[lr]_(upperknee|lowerknee)_y": 0.1,
                r"j_[lr]_ankle_[xy]": 1.0,
            },
        ),
    },
)
        


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
