
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation,RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# envs/quadcopter_mb/mdp.py

from typing import Optional, Tuple, Dict
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


# -----------------------
# Small math helpers
# -----------------------

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    # q: (w,x,y,z)
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=-1)

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((w, x, y, z), dim=-1)

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v' = q*(0,v)*q_conj
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat((zeros, v), dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:, 1:]


# -----------------------
# Shared desired target buffer
# -----------------------

def _get_or_create_desired_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_desired_pos_w"):
        env._desired_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    return env._desired_pos_w


# -----------------------
# Action term: EXACT Direct mapping
# -----------------------

@configclass
class BodyTorqueFrom4ActionCfg(ActionTermCfg):
    """Actions: [a0, a1, a2, a3] in [-1,1]
    Direct mapping:
      thrust_z = thrust_to_weight * robot_weight * (a0+1)/2
      moment   = moment_scale * [a1,a2,a3]
    """
    class_type: type[ActionTerm] = None  # filled below

    asset_name: str = "robot"
    body_name: str = "body"
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01


class BodyTorqueFrom4Action(ActionTerm):
    cfg: BodyTorqueFrom4ActionCfg

    def __init__(self, cfg: BodyTorqueFrom4ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]

        body_ids, _ = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"Body name '{cfg.body_name}' must match exactly one body. Got: {body_ids}")
        self._body_id = body_ids[0]

        # buffers (match Direct shapes: [N,1,3])
        self._actions = torch.zeros((env.num_envs, 4), device=env.device)
        self._thrust = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._moment = torch.zeros((env.num_envs, 1, 3), device=env.device)

        # Direct: robot_weight computed from mass * gravity norm
        mass = self._asset.root_physx_view.get_masses()[0].sum()
        gmag = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        self._robot_weight = (mass * gmag).item()

        self._action_dim = 4

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, env_ids: Optional(torch.Tensor) = None) -> None:
        if env_ids is None:
            self._actions.zero_()
            self._thrust.zero_()
            self._moment.zero_()
        else:
            self._actions[env_ids] = 0.0
            self._thrust[env_ids] = 0.0
            self._moment[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._actions = actions.clamp(-1.0, 1.0)
        return self._actions

    def apply_actions(self) -> None:
        # EXACT Direct mapping :contentReference[oaicite:6]{index=6}
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        self._asset.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)


BodyTorqueFrom4ActionCfg.class_type = BodyTorqueFrom4Action


# -----------------------
# Observation terms (exact Direct signals)
# -----------------------

def _robot(env: ManagerBasedRLEnv, name: str = "robot"):
    return env.scene[name]

def root_lin_vel_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _robot(env).data.root_lin_vel_b

def root_ang_vel_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _robot(env).data.root_ang_vel_b

def projected_gravity_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _robot(env).data.projected_gravity_b

def desired_pos_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    r = _robot(env)
    desired_w = _get_or_create_desired_pos_w(env)
    delta_w = desired_w - r.data.root_pos_w
    # body frame: R^T * delta_w = rotate by q_conj
    return quat_rotate(quat_conj(r.data.root_quat_w), delta_w)


# -----------------------
# Reward terms (exact Direct formula + dt scaling)
# -----------------------

def lin_vel_l2_scaled_dt(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Direct: sum(square(root_lin_vel_b))*scale*dt :contentReference[oaicite:7]{index=7}
    lin = torch.sum(torch.square(_robot(env).data.root_lin_vel_b), dim=1)
    # scales from Direct cfg: -0.05 :contentReference[oaicite:8]{index=8}
    return lin * (-0.05) * env.step_dt

def ang_vel_l2_scaled_dt(env: ManagerBasedRLEnv) -> torch.Tensor:
    ang = torch.sum(torch.square(_robot(env).data.root_ang_vel_b), dim=1)
    # Direct scale: -0.01 :contentReference[oaicite:9]{index=9}
    return ang * (-0.01) * env.step_dt

def distance_to_goal_tanh_scaled_dt(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Direct:
    #   dist = norm(desired - root_pos)
    #   mapped = 1 - tanh(dist/0.8)
    #   reward = mapped * 15.0 * dt :contentReference[oaicite:10]{index=10}
    r = _robot(env)
    desired_w = _get_or_create_desired_pos_w(env)
    dist = torch.linalg.norm(desired_w - r.data.root_pos_w, dim=1)
    mapped = 1.0 - torch.tanh(dist / 0.8)
    return mapped * 15.0 * env.step_dt


# -----------------------
# Terminations (exact Direct)
# -----------------------

def died(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Direct: z < 0.1 or z > 2.0 :contentReference[oaicite:11]{index=11}
    z = _robot(env).data.root_pos_w[:, 2]
    return torch.logical_or(z < 0.1, z > 2.0)

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    # When time_out=True in DoneTerm, IsaacLab handles it.
    # Return all False here.
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


# -----------------------
# Events (reset + target sampling)
# -----------------------

def reset_robot_to_default(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    r = _robot(env)

    # reset robot internal buffers
    r.reset(env_ids)

    # default root state + origins (same as Direct) :contentReference[oaicite:12]{index=12}
    default_root_state = r.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]

    r.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    r.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    # joints
    try:
        joint_pos = r.data.default_joint_pos[env_ids]
        joint_vel = r.data.default_joint_vel[env_ids]
        r.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    except Exception:
        pass

    # reset action buffers
    env.action_manager.reset(env_ids)

def sample_desired_pos_w(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    xy_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> None:
    desired_w = _get_or_create_desired_pos_w(env)

    # Direct sampling: xy uniform(-2,2) + env_origins; z uniform(0.5,1.5) :contentReference[oaicite:13]{index=13}
    desired_w[env_ids, :2] = torch.zeros((env_ids.numel(), 2), device=env.device).uniform_(xy_range[0], xy_range[1])
    desired_w[env_ids, :2] += env.scene.env_origins[env_ids, :2]
    desired_w[env_ids, 2] = torch.zeros((env_ids.numel(),), device=env.device).uniform_(z_range[0], z_range[1])


# OPTIONAL “wind/push” (matches the *idea* in the wind cfg you uploaded: interval force/torque) :contentReference[oaicite:14]{index=14}
def apply_external_force_torque(
    env: ManagerBasedRLEnv,
    force_range: Tuple[float, float],
    torque_range: Tuple[float, float],
    asset_name: str = "robot",
    body_name: str = "body",
) -> None:
    r = env.scene[asset_name]
    body_ids, _ = r.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Body name '{body_name}' must match exactly one body. Got: {body_ids}")
    body_id = body_ids[0]

    forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
    torques = torch.zeros((env.num_envs, 1, 3), device=env.device)

    forces[:, 0, :] = torch.empty((env.num_envs, 3), device=env.device).uniform_(force_range[0], force_range[1])
    torques[:, 0, :] = torch.empty((env.num_envs, 3), device=env.device).uniform_(torque_range[0], torque_range[1])

    r.set_external_force_and_torque(forces, torques, body_ids=body_id)
