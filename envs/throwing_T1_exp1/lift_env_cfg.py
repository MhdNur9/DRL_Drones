# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from envs.throwing_T1_exp1 import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0], rot=[-0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",scale=[0.3,0.3,1.0]),
    # )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Basket Table
    bs_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/basket_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.8, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Basket
    basket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.5, 0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path="/home/mirpalab-sim/RL_catch/assets/basket/basket.usd",scale=[1.2,1.2,1.2]),
    )

    # basket1 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Basket1",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1.4, 0, 0.0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path="/home/mirpalab-sim/RL_catch/assets/basket/basket.usd",scale=[0.7,1.7,1.2]),
    # )

    # basket2 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Basket2",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1.4, 0.4, 0.0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path="/home/mirpalab-sim/RL_catch/assets/basket/basket.usd",scale=[0.7,1.7,1.2]),
    # )

    # basket3 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Basket3",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1.4, -0.4, 0.0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path="/home/mirpalab-sim/RL_catch/assets/basket/basket.usd",scale=[0.7,1.7,1.2]),
    # )

    # belt = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/belt",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.38, 0.35, -1.24], rot=[-0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A06.usd",scale=[0.5,0.5,0.7]),
    # )

    # # # Person 1
    # person1 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/person1",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.8, -0.1], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/1X/Neo/Neo.usd",scale=[1.1,1.1,1.1],
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.9), metallic=0.2),
    #     ),
    # )

    # person2 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/person2",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.8, 0.8, -0.1], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/1X/Neo/Neo.usd",scale=[1.1,1.1,1.1],
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.9), metallic=0.2),
    #     ),
    # )

    # person3 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/person3",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.5, 0.0, -0.1], rot=[0.0, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/1X/Neo/Neo.usd",
    #                      scale=[1.1,1.1,1.1], 
    #                      activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.9), metallic=0.2),
    #     ),
    # )
    # collision_sensor0: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0", update_period=0.0, history_length=6, debug_vis=True
    # )

    # collision_sensor1: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link1", update_period=0.0, history_length=6, debug_vis=True
    # )

    # collision_sensor2: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link2", update_period=0.0, history_length=6, debug_vis=True
    # )

    # collision_sensor3: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link3", update_period=0.0, history_length=6, debug_vis=True
    # )

    # collision_sensor4: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link4", update_period=0.0, history_length=6, debug_vis=True
    # )
    
    # collision_sensor5: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link5", update_period=0.0, history_length=6, debug_vis=True
    # )
    
    # collision_sensor6: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link6", update_period=0.0, history_length=6, debug_vis=True
    # )
    
    # collision_sensor7: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link7", update_period=0.0, history_length=6, debug_vis=True
    # )

##
# MDP settings
##
   

@configclass
# the commands is used to help the robot throwing
class CommandsCfg:
    """Command terms for the MDP."""

    Target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(3.0, 3.0),
        # resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(1.2, 1.2),
            # pos_y=(-0.45, +0.45),
            pos_y=(0.0, 0.0),
            pos_z=(0.4, 0.4),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_vel = ObsTerm(func=mdp.object_vel_in_robot_root_frame)
        obj_ee_tracking=ObsTerm(func=mdp.object_ee_distance_obs, params={"std": 0.1})
        Obj_bsk_tracking_with_bsk_pos=ObsTerm(func=mdp.obj_bsk_distance_obs)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "Target_pose"})
               
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.2, 0.5), "y": (-0.25, 0.25), "z": (0.1, 0.1)}, 
            # "pose_range": {"x": (0.35, 0.35), "y": (-0.25, 0.25), "z": (0.1, 0.1)}, 
            # "pose_range": {"x": (0.35, 0.35), "y": (-0.35, -0.35), "z": (0.1, 0.1)}, 
            "velocity_range": {},
            
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # interval
    # belt_moving = EventTerm(
    #     func=mdp.moving_cube,
    #     mode="interval",
    #     interval_range_s=(0.1, 0.1),
    #     params={
    #         "velocity_range": {
    #             "x": (0.0,0.0), 
    #             "y": (+0.6,+0.6), 
    #             "z": (+0.1, +0.1), 
    #             "roll": (0.0,0.0), 
    #             "pitch": (0.0,0.0), 
    #             "yaw": (0.0,0.0)
    #         }
    #     },
    # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (1.5, 1.5), "y": (-0.27, -0.27), "z": (0.2, 0.2)}, # there is a shift in X by 0.5
    #         "velocity_range": {        
    #             "x": (0.0, 0.0),
    #             "y": (0.0,0.0),
    #             "z": (0.0,0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # minimal height do not change
 
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=10)

    # # # # distance error between basket and object
    obj_bsk_tracking_distance= RewTerm(func=mdp.obj_bsk_distance,weight=10.0,
        params={"std": 1, "minimal_height": 0.04, "command_name": "Target_pose"})
    
    obj_bsk_tracking_tunes= RewTerm(func=mdp.obj_bsk_distance,weight=5.0,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "Target_pose"})
    
    releasing_cube=RewTerm(func=mdp.reward_gripper_release_mid_throw,params={"minimal_height": 0.24}, weight=150)
 
    vel_check=RewTerm(func=mdp.obj_vel_release_check,params={"Vxy_velocity": 1.5},  weight=50)
    
    # # # adding later as bonus
    object_inside_basket=RewTerm(func=mdp.object_inside_basket,weight=+150)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2,weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # obj_inside_bsk=DoneTerm(func=mdp.object_inside_basket_terminate)

    # # Collision with the humanoid
    # collision0 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor0"), "threshold": 0.01})
    # collision1 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor1"), "threshold": 0.01})
    # collision2 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor2"), "threshold": 0.01})
    # collision3 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor3"), "threshold": 0.01})
    # collision4 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor4"), "threshold": 0.01})
    # collision5 = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor5"), "threshold": 0.01})



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625