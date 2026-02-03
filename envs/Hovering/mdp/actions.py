from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal, Union

import copy
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from envs.Hovering.mdp.controllers.attitude import AttitudeController
from envs.Hovering.mdp.controllers.body_rate import BodyRateController
from envs.Hovering.mdp.dynamics.allocation import compute_allocation
# from envs.Hovering.mdp.dynamics.motor_model import MotorModel
from envs.Hovering.mdp.dynamics.motor_model import MotorModelV2
from envs.Hovering.mdp.utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BodyTorqueControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands which could represent:
    1. Actuator thrust setpoints
    2. Angular rates setpoints
    3. Attitude setpoints

    """

    cfg: BodyTorqueControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: BodyTorqueControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg
        self.cfg_default = copy.deepcopy(cfg)

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        # Rotor joints (for visuals)
        self._rotor_joint_ids = self._robot.find_joints("rotor_.*_joint")
        print(" self._rotor_joint_ids = ",        self._rotor_joint_ids)
        rotor_joint_ids, rotor_joint_names = self._robot.find_joints("rotor_.*_joint")
        self._rotor_joint_ids = rotor_joint_ids

        # (optional debug)
        print("Rotor joints:", rotor_joint_ids, rotor_joint_names)

        self._spin_signs = torch.tensor([1, -1, 1, -1, 1, -1], device=self.device).view(1, 6)


        self._body_id = self._robot.find_bodies("base_link")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._rotor_angle = torch.zeros(self.num_envs, 6,device=self.device,dtype=self._raw_actions.dtype)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._actual_thrust = torch.zeros_like(self._raw_actions)

        self._kf = torch.full((self.num_envs, 1), self.cfg.kf, device=self.device)
        self._kd = torch.full((self.num_envs, 1), self.cfg.kd, device=self.device)
        self._length = torch.full((self.num_envs, 1), self.cfg.length, device=self.device)
        self._alpha = torch.full((self.num_envs, 1), self.cfg.alpha, device=self.device)

        self._allocation_matrix = compute_allocation(
            self._kf, self._kd, self._length, self._alpha, self.num_envs, device=self.device
        ).to(self.device, dtype=self._raw_actions.dtype)
        # self._motor_model = MotorModel(
        #             self.num_envs,
        #     self.cfg.taus,
        #     self.cfg.init_thrust,
        #     self.cfg.max_thrust,
        #     self.cfg.max_thrust_rate,
        #     self.cfg.min_thrust_rate,
        #     env.physics_dt,
        #     self.cfg.use_motor_model,
        #     self.device,
        # )
        self._motor_model = MotorModelV2(
            num_envs=self.num_envs,
            num_motors=self.cfg.num_motors,
            dt=env.physics_dt,
            device=torch.device(self.device),   # or just self.device
            KF=self.cfg.max_thrust,             # treat max_thrust as "KF" scale (see note below)
            tau_up=0.43,
            tau_down=0.43,
            init_throttle=0.0,
            noise_scale=0.0,
            use=self.cfg.use_motor_model,
        )




        self._rate_controller = BodyRateController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_rates,
            self.device,
        )
        self._attitude_controller = AttitudeController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_attitude,
            torch.eye(3) * self.cfg.k_rates,
            self.device,
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
        return self.cfg.num_motors

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    # Observations
    @property
    def last_thrust(self) -> torch.Tensor:
        return self._actual_thrust
    
    # Radomization
    @property
    def config(self) -> BodyTorqueControlActionCfg:
        return self.cfg
    
    @property
    def kf(self) -> torch.Tensor:
        return self._kf

    @property
    def kd(self) -> torch.Tensor:
        return self._kd

    @property
    def length(self) -> torch.Tensor:
        return self._length

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha
    
    @kf.setter
    def kf(self, value: torch.Tensor) -> None:
        self._kf = value

    @kd.setter
    def kd(self, value: torch.Tensor) -> None:
        self._kd = value

    @length.setter
    def length(self, value: torch.Tensor) -> None:
        self._length = value

    @alpha.setter
    def alpha(self, value: torch.Tensor) -> None:
        self._alpha = value

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        log(self._env, ["a1", "a2", "a3", "a4", "a5", "a6"], actions)

        self._raw_actions[:] = actions
        clamped = self._raw_actions.clamp_(-1.0, 1.0)

        if self.cfg.control_level == "thrust":
            thrusts_ref = (
                (clamped + 1.0)
                * torch.tensor(self.cfg.max_thrust, device=self.device, dtype=self._raw_actions.dtype)
                / 2.0
            )
        elif self.cfg.control_level == "rates":
            # Clamp rates setpoint and 3D force
            # Calculate wrench based on rates setpoint
            # Calculate thrust setpoint based on wrench and allocation matrix inverse
            # Clamp thrust setpoint
            clamped[:, :3] *= torch.tensor(self.cfg.max_force, device=self.device, dtype=self._raw_actions.dtype)
            clamped[:, 3:] *= torch.tensor(self.cfg.max_ang_vel, device=self.device, dtype=self._raw_actions.dtype)
            clamped[:, 3:] = self._rate_controller.compute_moment(clamped[:, 3:], self._robot.data.root_ang_vel_b)
            thrusts_ref = torch.bmm(self._allocation_matrix.inverse(), clamped.unsqueeze(-1)).squeeze(-1)            

        elif self.cfg.control_level == "attitude":
            # Clamp orientation setpoint and 3D force
            # Calculate wrench based on orientation setpoint
            # Calculate thrust setpoint based on wrench and allocation matrix inverse
            # Clamp thrust setpoint
            clamped[:, :3] *= torch.tensor(self.cfg.max_force, device=self.device, dtype=self._raw_actions.dtype)
            clamped[:, 3:] *= torch.tensor(self.cfg.max_attitude, device=self.device, dtype=self._raw_actions.dtype)
            clamped[:, 3:] = self._attitude_controller.compute_moment(
                clamped[:, 3:], self._robot.data.root_quat_w, self._robot.data.root_ang_vel_b
            )
            thrusts_ref = torch.bmm(self._allocation_matrix.inverse(), clamped.unsqueeze(-1)).squeeze(-1)

        # self._actual_thrust = self._motor_model.update_thrust(thrusts_ref)
        if self.cfg.control_level == "thrust":
            self._actual_thrust = self._motor_model.update_thrust(clamped)
        else:
            self._actual_thrust = thrusts_ref

        self._processed_actions = torch.bmm(self._allocation_matrix, self._actual_thrust.unsqueeze(-1)).squeeze(-1)

    def apply_actions(self):
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        forces[:, 0, :] = self._processed_actions[:, :3]
        torques[:, 0, :] = self._processed_actions[:, 3:]

        self._robot.set_external_force_and_torque(forces, torques, body_ids=self._body_id)
        self._robot.update(self._env.physics_dt)
        # --- Visual rotor spinning (independent from physics thrust control) ---
        k_vis = 8.0e-7 
        omega = torch.sqrt(torch.clamp(self._actual_thrust, min=0.0) / k_vis)  # (num_envs, 6)
        omega = torch.clamp(omega, 0.0, 1500.0)  # cap for visuals
        omega = omega * self._spin_signs

        # integrate angle
        self._rotor_angle = (self._rotor_angle + omega * self._env.physics_dt) % (2 * torch.pi)
        # print("self._rotor_angle  = ",self._rotor_angle )

        # WRITE POSITIONS (this WILL rotate the mesh)
        self._robot.write_joint_position_to_sim(
            self._rotor_angle,
            joint_ids=self._rotor_joint_ids
        )
        self._robot.write_data_to_sim()

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        # Reset robot joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._allocation_matrix = compute_allocation(
            self._kf, self._kd, self._length, self._alpha, self.num_envs, device=self.device
        ).to(self.device, dtype=self._raw_actions.dtype)
        self._motor_model.reset(env_ids)


@configclass
class BodyTorqueControlActionCfg(ActionTermCfg):
    """
    See :class:`BodyTorqueControlAction` for more details.
    """

    class_type: type[ActionTerm] = BodyTorqueControlAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    # Physical parameters
    num_motors: int = 6
    """Number of motors in the drone."""
    kf: float = 11.75e-4
    """Thrust coefficient."""
    kd: float = 2.388e-5
    """Drag coefficient."""
    length: float = 0.215
    """Arm length."""
    alpha: float = 0.0
    """Tilt angle."""

    # Motor related
    init_thrust: float = 3.75
    """Initial thrust."""
    max_thrust: list[float] = [14.0, 14.0, 14.0, 14.0, 14.0, 14.0]
    """Maximum thrust."""
    max_thrust_rate: list[float] = [25, 25, 25, 25, 25, 25]
    """Maximum thrust rate."""
    min_thrust_rate: list[float] = [-15, -15, -15, -15, -15, -15]
    """Minimum thrust rate."""
    taus: list[float] = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    """Time constants."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""

    # Controller related
    max_force: list[float] = [40.0, 40.0, 40.0]
    """Maximum force."""
    max_ang_vel: list[float] = [10.0, 10.0, 10.0]
    """Maximum angular velocity."""
    max_attitude: list[float] = [torch.pi, torch.pi, torch.pi]
    """Maximum angular velocity."""
    k_attitude: float = 100.0
    """Proportional gain for attitude error."""
    k_rates: float = 10.0
    """Proportional gain for angular velocity error."""
    ControlLevel = Union[Literal["thrust"], Literal["rates"], Literal["attitude"]]
    control_level: ControlLevel = "thrust"
    """Control level."""
