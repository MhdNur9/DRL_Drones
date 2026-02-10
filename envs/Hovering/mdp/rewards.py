from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_error_l2(
    env: ManagerBasedRLEnv,
    target_pos: list,
    target_attitude: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pose from its target pose using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_pos_tensor = (
        torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
        + env.scene.env_origins
    )
    target_attitude_tensor = torch.tensor(target_attitude, dtype=torch.float32, device=asset.device).repeat(
        env.num_envs, 1
    )

    pos_error, att_error = math_utils.compute_pose_error(
        asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor, target_attitude_tensor
    )

    # Compute sum of squared errors
    return torch.sum(torch.square(torch.cat([pos_error, att_error[:, 1:]], dim=1)), dim=1)


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def att_error_mag(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_attitude: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset attitude from its target attitude using angular error between target quaternion in radians."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_attitude is None:
        target_attitude = env.command_manager.get_term(command_name).command
        target_attitude_tensor = target_attitude[:, 3:]
    else:
        target_attitude_tensor = torch.tensor(target_attitude, dtype=torch.float32, device=asset.device).repeat(
            env.num_envs, 1
        )

    quat_error = math_utils.quat_error_magnitude(asset.data.root_quat_w, target_attitude_tensor)

    return quat_error


def yaw_error(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_attitude: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset yaw from its target yaw using angular error between target quaternion."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_attitude is None:
        target_attitude = env.command_manager.get_term(command_name).command
        target_attitude_tensor = target_attitude[:, 3:]
    else:
        target_attitude_tensor = torch.tensor(target_attitude, dtype=torch.float32, device=asset.device).repeat(
            env.num_envs, 1
        )

    # Compute the yaw from the asset's quaternion
    asset_yaw = math_utils.yaw_quat(asset.data.root_quat_w)
    # Compute the yaw from the target quaternion
    target_yaw = math_utils.yaw_quat(target_attitude_tensor)

    return torch.norm((torch.square(asset_yaw - target_yaw)), dim=1)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

def vel_toward_target(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # target position in world (per-env)
    if target_pos is None:
        tgt = env.command_manager.get_term(command_name).command[:, :3]
    else:
        tgt = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device)
            .repeat(env.num_envs, 1) + env.scene.env_origins
        )

    # direction from drone to target (world frame- Do not change it)
    to_tgt = tgt - asset.data.root_pos_w
    dir_w = torch.nn.functional.normalize(to_tgt, dim=1)
    # world-frame linear velocity (fallback if only body-frame is available)
    try:
        v_w = asset.data.root_lin_vel_w
    except AttributeError:
        # rotate body-frame velocity into world frame
        v_w = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.root_lin_vel_b)
    # component of velocity toward target
    v_along = torch.sum(v_w * dir_w, dim=1)  # m/s; >0 when moving toward target
    # keep reward scale bounded (for moothing only -  need further trials)
    # gate: 1 when far, 0 when near
    distance = torch.norm(to_tgt, dim=1)

    gate_far = torch.clamp((distance - 0.3) / (1.5 - 0.3), 0.0, 1.0)


    r = torch.tanh(v_along / 2.0)            # ≈[-1, 1], smooth

    return r* gate_far


def motor_balance_band_reward(
    env: ManagerBasedRLEnv,
    tol: float = 0.10,          # 10% band
    bonus: float = 1.0,         # positive reward if all within band
    penalty_scale: float = 1.0, # how strong the penalty is
    p: float = 2.0,             # penalty curvature: 1=linear, 2=quadratic
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Reward motor balance:
    +bonus if all actions are within ±tol of mean (relative),
    else negative penalty proportional to how far outside the band.
    """
    # print("*****************************")
    # Use physical thrusts (N): shape (N, 6), nonnegative
    thrust = env.action_manager.get_term(
        "body_torque_control_action"
    ).last_thrust  # (N, 6)

    # print("thrust = ", thrust[0])

    mu = thrust.mean(dim=1, keepdim=True)  # (N, 1)
    # print("mu = ", mu[0])

    # relative deviation from mean
    rel_dev = torch.abs(thrust - mu) / (mu + eps)  # (N, 6)
    # print("rel_dev = ", rel_dev[0])

    max_dev = rel_dev.max(dim=1).values  # (N,)
    # print("max_dev = ", max_dev[0])

    # inside band?
    inside = (max_dev <= tol)
    # print("inside = ", inside[0])

    # how far outside band (0 if inside)
    exceed = torch.clamp(max_dev - tol, min=0.0)
    # print("exceed = ", exceed[0])

    # normalize exceed
    norm_exceed = exceed / tol
    # print("norm_exceed = ", norm_exceed[0])

    penalty = -penalty_scale * torch.pow(norm_exceed, p)
    # print("penalty = ", penalty[0])

    reward = torch.where(
        inside,
        torch.full_like(max_dev, bonus),
        penalty,
    )
    return reward
