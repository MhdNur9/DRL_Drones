import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from envs.Hovering.utils_scripts.Rough import ROUGH_TERRAINS_CFG

import envs.Hovering.mdp as mdp
from assets.drone.firefly import firefly_CFG,object_CFG

from .utils_scripts.track_gen import gen_track

@configclass
class HoverSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    object1: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    object2: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.0, 0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    object3: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.5, 0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    object4: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.75, 0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )

    object5: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object5",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[2.0, 0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    
    object6: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object6",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.0, 0.5, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    
    object7: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object7",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.0, 1.0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    object8: RigidObjectCfg = object_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object8",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.0, 1.0, 0.0],   
            rot=[1, 0, 0, 0]
        ),
    )
    
    # track
    track: RigidObjectCollectionCfg = gen_track(
        track_config={
        # --- Bottom square (Z = 1.0) ---
        "1":  {"pos": (1.0, 1.0, 1.0), "yaw": 0.0, "color": (0.0, 1.0, 0.0)},
        "2":  {"pos": (4.0, 1.0, 1.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        "3":  {"pos": (4.0, 4.0, 1.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        "4":  {"pos": (1.0, 4.0, 1.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        # --- Move up ---
        "5":  {"pos": (1.0, 4.0, 3.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        # --- Top square (Z = 3.0) ---
        "6":  {"pos": (4.0, 4.0, 3.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        "7":  {"pos": (4.0, 1.0, 3.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        "8":  {"pos": (1.0, 1.0, 3.0), "yaw": 0.0, "color": (1.0, 1.0, 0.0)},
        # --- Go back down ---
        "9":  {"pos": (1.0, 1.0, 1.0), "yaw": 0.0, "color": (0.0, 0.0, 1.0)},
        "10":  {"pos": (0.1, 0.1, 0.5), "yaw": 0.0, "color": (0.0, 1.0, 0.0)},
        }
    )
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body/camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.14, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.FisheyeCameraCfg(),
    #     width=1000,
    #     height=1000,
    # )


    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     # visual_material=  sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.9), metallic=0.2),  
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     # visual_material=sim_utils.MdlFileCfg(
    #     #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #     #     project_uvw=True,
    #     #     texture_scale=(0.25, 0.25),            
    #     # ),
    #     visual_material=sim_utils.PreviewSurfaceCfg(
    #         diffuse_color=(0.25, 0.15, 0.08),
    #         metallic=0.0,
    #         roughness=0.9,
    #     ),   

    #     debug_vis=False,
    # )
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(color=[0.75, 0.75, 0.75]),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.5, 0.75, 1.0), intensity=3000.0),
    )


    # drone
    robot: ArticulationCfg = firefly_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    body_torque_control_action: mdp.BodyTorqueControlActionCfg = mdp.BodyTorqueControlActionCfg(
        asset_name="robot",
        use_motor_model=True,
        control_level="thrust",
    )

@configclass
class CommandsCfg:
    target_pos = mdp.TargetPosFromTrackCommandCfg(
        track_name="track",
        start_index=0,
        resampling_time_range=(1e9, 1e9),  # don't auto-resample; we'll drive it via event
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # (1) Full state
        position = ObsTerm(func=mdp.root_pos_w)
        orientation = ObsTerm(func=mdp.root_rotmat_w_track)
        linear_vel = ObsTerm(func=mdp.root_lin_vel_b)
        angular_vel = ObsTerm(func=mdp.root_ang_vel_b)
        # (2) Hover specific
        # target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"target_pos": [2.0, 2.0, 2.5]})
        target_pos_b_track = ObsTerm(func=mdp.target_pos_b_track, params={"command_name": "target_pos"})
        # (3) Other
        actions = ObsTerm(func=mdp.last_action)
        thrusts = ObsTerm(func=mdp.last_thrust)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_event,
        mode="reset",
        params={
            "pose_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (0.01, 0.015),
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.01, 0.02),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
   
    advance_target = EventTerm(
        func=mdp.advance_track_target_event,
        mode="interval",
        interval_range_s=(5.0, 5.0),  
        params={"command_name": "target_pos", "step": 1},
    )

    object_releasing = EventTerm(
        func=mdp.reset_obj_releaseing_track_event,
        mode="interval",
        interval_range_s=(48, 48.1),
    )

    place_objects_seq = EventTerm(
        func=mdp.place_objects_sequentially_event,
        mode="interval",
        interval_range_s=(0.05, 0.05),  # call frequently so timing is accurate
    )

    # # Disturbances
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque_event,
    #     mode="interval",
    #     interval_range_s=(0.1, 0.5),
    #     params={
    #         "force_range": (-0.5, 0.5),
    #         "torque_range": (-0.05, 0.05),
    #     },
    # )

    # steady wind
    # wind_robot = EventTerm(
    #     func=mdp.apply_user_wind_event,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.05),
    #     params={
    #         "wind_force_w": (0.4, 0.4, 0.0),
    #         "wind_torque_w": (0.0, 0.0, 0.0),
    #         "wind_freq": 0.5,
    #         "phase": 0.0,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #     },
    # )

    # randomize wrench map parameters
    # randomize_wrench_map = EventTerm(
    #     func=mdp.randomize_wrench_map,
    #     mode="reset",
    #     params={
    #         "action": "body_torque_control_action",
    #         "randomization_params": {
    #             "kf": (0.8, 1.2),
    #             "kd": (0.8, 1.2),
    #             "length": (0.8, 1.2),
    #             "alpha": (0.8, 1.2),
    #         },
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     }
    # )

    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="interval",
    #     interval_range_s=(7, 8),
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.0, 0.0),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    terminating = RewTerm(func=mdp.is_terminated_reward, weight=-500.0)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_reward, weight=-0.005)

    motor_balance = RewTerm(func=mdp.motor_balance_band_reward, weight=4.0,
        params={"tol": 0.10, "bonus": 0.2, "penalty_scale": 0.2, "p": 2.0},)
    
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2_reward, weight=-5.0)

    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-1.0)
    
    pos_error_tanh1 = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
                              params={"command_name": "target_pos", "std": 2.0},)
    
    pos_error_tanh2 = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
                    params={"command_name": "target_pos", "std": 1.0},)
    
    pos_error_tanh3 = RewTerm(func=mdp.pos_error_tanh, weight=25.0,
                    params={"command_name": "target_pos", "std": 0.3},)
    
    pos_error_tanh4 = RewTerm(func=mdp.pos_error_tanh, weight=35.0,
                    params={"command_name": "target_pos", "std": 0.1},)
    
    pos_error_tanh5 = RewTerm(func=mdp.pos_error_tanh, weight=55.0,
                    params={"command_name": "target_pos", "std": 0.05},)
    
    # pos_error_tanh6 = RewTerm(func=mdp.pos_error_tanh, weight=35.0,
    #                 params={"command_name": "target_pos", "std": 0.01},)
       
    vel_toward = RewTerm(func=mdp.vel_toward_target, weight=55.0,
                    params={"command_name": "target_pos"},)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # flyaway = DoneTerm(func=mdp.flyaway, params={"target_pos": [2.0, 2.0, 2.5], "distance": 5.0})
    flip = DoneTerm(func=mdp.flip, params={"angle": 60.0})


@configclass
class HoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone environment."""

    # Scene settings
    scene: HoverSceneCfg = HoverSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 50
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
