from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from envs.Hovering.mdp.utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    # print("lin_vel = ",lin_vel)
    log(env, ["vx", "vy", "vz"], lin_vel)
    return lin_vel


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    log(env, ["wx", "wy", "wz"], ang_vel)
    return ang_vel


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    object1: RigidObject = env.scene["object"]
    
    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    #####
    q_d = asset.data.root_quat_w      # (N,4) wxyz
    q_c = object1.data.root_quat_w     # (N,4) wxyz

    q_rel = math_utils.quat_mul(math_utils.quat_inv(q_d), q_c)

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(q_rel)  # each (N,)
    yaw_err = yaw  # already relative yaw (cube w.r.t drone)

    # print("yaw_err =", yaw_err)

    log(env, ["yaw_err"], yaw_err.unsqueeze(-1)-0.44)
    # log as (1,) tensor for env0 (or log whole batch if your logger supports it)
    # log(env, ["cube_ori_err_rad"], theta.view(-1, 1))
  
    #####
    return flat_rotmat


def root_rotmat_w_track(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    object1: RigidObject = env.scene["object1"]
    object2: RigidObject = env.scene["object2"]
    object3: RigidObject = env.scene["object3"]
    object4: RigidObject = env.scene["object4"]
    object5: RigidObject = env.scene["object5"]
    object6: RigidObject = env.scene["object6"]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    #####
    q_d = asset.data.root_quat_w      # (N,4) wxyz
    q_c = object1.data.root_quat_w     # (N,4) wxyz

    q_rel = math_utils.quat_mul(math_utils.quat_inv(q_d), q_c)

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(q_rel)  # each (N,)
    yaw_err = yaw  # already relative yaw (cube w.r.t drone)

    # print("yaw_err =", yaw_err)

    log(env, ["yaw_err"], yaw_err.unsqueeze(-1)-0.44)
    # log as (1,) tensor for env0 (or log whole batch if your logger supports it)
    # log(env, ["cube_ori_err_rad"], theta.view(-1, 1))
  
    #####
    return flat_rotmat


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    return position


def pose_error(
    env: ManagerBasedRLEnv, target_pos: list, target_attitude: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Pose error in the world frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos_tensor = (
        torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
        + env.scene.env_origins
    )
    target_attitude_tensor = torch.tensor(target_attitude, dtype=torch.float32, device=asset.device).repeat(
        env.num_envs, 1
    )

    pos_error, att_error = math_utils.compute_pose_error(
        asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor, target_attitude_tensor, rot_error_type="quat"
    )
    return torch.cat([pos_error, att_error[:, 1:]], dim=1)


def pos_error(
    env: ManagerBasedRLEnv, target_pos: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Position error in the world frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos_tensor = (
        torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
        + env.scene.env_origins
    )

    pos_error = asset.data.root_pos_w - target_pos_tensor
    return pos_error


def att_error(
    env: ManagerBasedRLEnv, target_attitude: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Attitude error in the world frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    target_attitude_tensor = torch.tensor(target_attitude, dtype=torch.float32, device=asset.device).repeat(
        env.num_envs, 1
    )

    source_quat_norm = math_utils.quat_mul(asset.data.root_quat_w, math_utils.quat_conjugate(asset.data.root_quat_w))[
        :, 0
    ]
    source_quat_inv = math_utils.quat_conjugate(asset.data.root_quat_w) / source_quat_norm.unsqueeze(-1)
    quat_error = math_utils.quat_mul(target_attitude_tensor, source_quat_inv)

    return quat_error[:, 1:]


def last_thrust(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Last thrust commands sent to actuators"""
    thrust = env.action_manager.get_term("body_torque_control_action").last_thrust
    log(env, ["t1", "t2", "t3", "t4", "t5", "t6"], thrust)
    return thrust


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    # print("****************")
    # print("Robot details")
    # print(asset.data.default_mass)
    # print(asset.data.default_inertia)

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)
    log(env, ["pxd", "pyd", "pzd"], target_pos_tensor)

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)

    return pos_b






def target_pos_b_track(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]


    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
        # print("target_pos = ",target_pos)
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )
    log(env, ["pxd", "pyd", "pzd"], target_pos_tensor)

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)
    # print("pos_b = ",pos_b)

    return pos_b