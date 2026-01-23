import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import envs.Hovering.mdp_original as mdp
from assets.drone.fiberthex import FIBERTHEX_CFG


@configclass
class HoverSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # drone
    robot: ArticulationCfg = FIBERTHEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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
        # target_pos_b_modifird = ObsTerm(func=mdp.target_pos_b_modified, params={"target_pos": [0.0, 0.0, 2.0]})
        # (3) Other
        actions = ObsTerm(func=mdp.last_action)
        thrusts = ObsTerm(func=mdp.last_thrust)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 10
            # self.flatten_history_dim = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (1.0, 3.0),
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

    # Disturbances
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.1, 0.5),
        params={
            "force_range": (-0.5, 0.5),
            "torque_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    pos_error_tanh_far = RewTerm(func=mdp.pos_error_tanh, weight=5.0, params={"target_pos": [0.0, 0.0, 2.0], "std": 2.0})
    pos_error_tanh_fine_tune_mid = RewTerm(func=mdp.pos_error_tanh, weight=15.0, params={"target_pos": [0.0, 0.0, 2.0], "std": 0.3})
    # pos_error_tanh_very_fine = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
    #                                params={"target_pos":[0,0,2.0], "std": 0.1})
    # pos_error_tanh_very_near = RewTerm(func=mdp.pos_error_tanh, weight=15.0,
    #                                params={"target_pos":[0,0,2.0], "std": 0.05})

    # vel_toward = RewTerm(func=mdp.vel_toward_target, weight=13.0,
    #                  params={"target_pos":[0.0, 0.0, 2.0]}) 
    
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
        self.episode_length_s = 10
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
