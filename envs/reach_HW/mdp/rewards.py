# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    ######
    env.extras['des_pos_w'] =des_pos_b 
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    env.extras['ee_frame_pos']= ee_frame_pos
    

    
    ######
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    # print("command = ",des_quat_w)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    # print("Robot = ",curr_quat_w)
    # print("quat_error_magnitude = ",quat_error_magnitude(curr_quat_w, des_quat_w))
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def reset_joints_limit(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    # Scale joint position limits:
    new_joint_pos_limits = joint_pos_limits.clone()
    # Multiply lower limits by new_position_range[0]
    new_joint_pos_limits[:, :, 0] = joint_pos_limits[:, :, 0] * position_range[0]
    # Multiply upper limits by new_position_range[1]
    new_joint_pos_limits[:, :, 1] = joint_pos_limits[:, :, 1] * position_range[1]
    joint_pos = joint_pos.clamp_(new_joint_pos_limits[..., 0], new_joint_pos_limits[..., 1])

    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    # Scale joint velocities:
    new_joint_vel = joint_vel_limits * velocity_range[1]
    joint_vel = joint_vel.clamp_(-new_joint_vel, new_joint_vel)
    # Writing the new values
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_joint_limits_to_sim(new_joint_pos_limits,env_ids=env_ids)
