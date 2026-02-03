import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from envs.Hovering.Rough import ROUGH_TERRAINS_CFG

import envs.Hovering.mdp as mdp
from assets.drone.firefly import firefly_CFG


@configclass
class HoverSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    # scale=(0.5, 0.5, 0.5),
                                        scale=(1.0, 1.0, 1.0),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                ),
                )

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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # (1) Full state
        position = ObsTerm(func=mdp.root_pos_w)
        orientation = ObsTerm(func=mdp.root_rotmat_w)
        linear_vel = ObsTerm(func=mdp.root_lin_vel_b)
        angular_vel = ObsTerm(func=mdp.root_ang_vel_b)
        # (2) Hover specific
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"target_pos": [0.0, 0.0, 2.0]})
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
                "x": (-3.0, 3.0),
                "y": (-3.0, 3.0),
                "z": (0.05, 5.0),
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
    
    # randomize rigid body inertia
    randomize_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        }
    )

    # Disturbances
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_event,
        mode="interval",
        interval_range_s=(0.1, 0.5),
        params={
            "force_range": (-0.5, 0.5),
            "torque_range": (-0.05, 0.05),
        },
    )
    object_releasing = EventTerm(
        func=mdp.reset_obj_releaseing_event,
        mode="interval",
        interval_range_s=(8, 9),
    )
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

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    pos_error_tanh_far = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
                    params={"target_pos": [0.0, 0.0, 2.0], "std": 2.0})
    
    pos_error_tanh_fine_tune_mid = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
                                            params={"target_pos": [0.0, 0.0, 2.0], "std": 0.3})
    
    pos_error_tanh_very_fine = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
                                   params={"target_pos":[0,0,2.0], "std": 0.1})
    
    # pos_error_tanh_very_near = RewTerm(func=mdp.pos_error_tanh, weight=1.0,
    #                                params={"target_pos":[0,0,2.0], "std": 0.02})
    
    # pos_far  = RewTerm(func=mdp.pos_error_tanh, weight=8.0, params={"target_pos":[0,0,2.0], "std":2.0})
    # pos_near = RewTerm(func=mdp.pos_error_tanh, weight=4.0, params={"target_pos":[0,0,2.0], "std":0.3})

    vel_toward = RewTerm(func=mdp.vel_toward_target, weight=13.0,
                     params={"target_pos":[0.0, 0.0, 2.0]}) 
    motor_balance = RewTerm(
        func=mdp.motor_balance_band_reward,
        weight=1.0,
        params={"tol": 0.10, "bonus": 0.2, "penalty_scale": 0.2, "p": 2.0},
        )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"target_pos": [0.0, 0.0, 2.0], "distance": 3.0})
    flip = DoneTerm(func=mdp.flip, params={"angle": 60.0})


@configclass
class HoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone environment."""

    # Scene settings
    scene: HoverSceneCfg = HoverSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
