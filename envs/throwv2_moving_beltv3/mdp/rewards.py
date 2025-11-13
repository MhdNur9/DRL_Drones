# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation,RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from envs.throwv2_moving_beltv3.mdp.observations import bsk_pos_in_robot_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)
    
    # Compute the reward
    return 1 - torch.tanh(object_ee_distance / std)

def obj_bsk_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def object_inside_basket(env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    obj_pos=object.data.root_pos_w
    # Define box boundaries:
    xmin, xmax = 1.25, 1.75
    ymin, ymax = -0.25, 0.25
    z_max = 0.05  # Condition: z must be less than 0.23

    # Check conditions for each coordinate:
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    inside_z = object_pos_w[:, 2] < z_max

    # Combine all conditions:
    inside_box = (inside_x & inside_y & inside_z).int()
    # if torch.any(inside_box):
    #     print("Obj is inside the basket")

    # if obj_pos[0][0]>1.0 :
    #     if obj_pos[0][2]<0.05 :
    #          print("obj_pos = ",obj_pos)

    return inside_box

def reward_gripper_release_mid_throw(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    robot: Articulation=env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # gripper open state (assumed scalar per env)
    joint_pos=robot.data.joint_pos.clone()

    # Check if the last 2 elements are approximately 0.04
    right_gripper=joint_pos[:, -1]>0.037
    # print("before right_gripper = ",right_gripper)
    left_gripper=joint_pos[:, -2]>0.037
    # print("the gripper is open or not = ",right_gripper&left_gripper)
    results=(object.data.root_pos_w[:, 2] > minimal_height)*(right_gripper&left_gripper).int()
    results=results.int()
   
    return results

def obj_vel_release_check(
    env: ManagerBasedRLEnv,
    Vxy_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command("Target_pose")
    # compute the desired position in the world frame
    cmd_pos = command[:, :3]
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    # Required_Speed
    obj_vel_x=obj_vel[:, :1]
    obj_vel_y=obj_vel[:, [1]]
    ref_vel_x = torch.full_like(obj_vel_x, (Vxy_velocity))
    ref_vel_y = torch.full_like(obj_vel_y, Vxy_velocity/5)

    same_sign = ((obj_vel_y * cmd_pos[:, [1]]) > 0).squeeze(1)

    vel_x_state = (torch.norm(obj_vel_x, dim=1) > torch.norm(ref_vel_x, dim=1)).int()
    vel_y_state = (torch.norm(obj_vel_y, dim=1) > torch.norm(ref_vel_y, dim=1)).int()
    # print("vel_x_state = ",vel_x_state)
    # print("vel_y_state = ",vel_y_state)
    vel_y_check=vel_y_state&same_sign
    # print("vel_y_check = ",vel_y_check)
    vel_state=vel_x_state&vel_y_check
    vel_state=vel_x_state
    # print("reward obj_vel_release_check = ",vel_state)
  
    return vel_state