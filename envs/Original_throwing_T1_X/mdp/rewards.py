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
from isaaclab.utils.math import combine_frame_transforms
from envs.Original_throwing_T1_X.mdp.observations import bsk_pos_in_robot_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # print(torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0))
    
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

def obj_cmd_distance(
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

def object_inside_basket(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    obj_pos=object.data.root_pos_w
    # Define box boundaries:
    xmin, xmax = 0.95, 1.45
    xmin, xmax = 1.25, 1.75
    # ymin, ymax = -0.25, 0.25
    ymin, ymax = -0.8, 0.8
    z_max = 0.05  # Condition: z must be less than 0.23
    # Check conditions for each coordinate:
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    inside_z = object_pos_w[:, 2] < z_max
    # Combine all conditions:
    inside_box = inside_x & inside_y & inside_z
    # if inside_box:
    #     print("obj is inside the basket")
    # if obj_pos[0][0]>1.0 :
    #     if obj_pos[0][2]<0.05 :
    #          print("obj_pos = ",obj_pos)
        
    return inside_box

def obj_vel_release_check(
    env: ManagerBasedRLEnv,
    Vxy_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    # obj_vel_release_check conditions    
    # Direction alignment
    # Project both vectors into the XY plane
    # Direction alignment
    vel_xy = obj_vel[:, :2]
    to_basket_xy = bsk_pos[:, :2] - obj_pos[:, :2]
    # Normalize both vectors
    vel_unit = vel_xy / (torch.norm(vel_xy, dim=1, keepdim=True) + 1e-8)
    basket_unit = to_basket_xy / (torch.norm(to_basket_xy, dim=1, keepdim=True) + 1e-8)
    # Compute unsigned angle (in radians → degrees)
    dot = torch.sum(vel_unit * basket_unit, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)  # for numerical safety
    angle_rad = torch.acos(dot)
    angle_reward = torch.cos(angle_rad)
    Direction_alignment=torch.where(angle_reward > 0.85, 1.0, 0.0)
    
    # Required_Speed
    vel_state=torch.where((torch.norm(vel_xy, dim=1) > Vxy_velocity),1,0)

    result=vel_state*Direction_alignment
    # until now no need for direction alignment
    # for future research, the direction alignment is needed

    return vel_state

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
    left_gripper=joint_pos[:, -2]>0.037
    
    return (object.data.root_pos_w[:, 2] > minimal_height)*(right_gripper&left_gripper)

def obj_vel_release_check2(
    env: ManagerBasedRLEnv,
    Vxy_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    obj_vel=object.data.root_lin_vel_w
    # print("obj_vel = ",obj_vel)
    vel_xy = obj_vel[:, [0]]
    # vel_xy = obj_vel[:, :2]
    # print("vel_xy = ",vel_xy)

    # Required_Speed
    vel_state=torch.where((vel_xy[:, 0] > Vxy_velocity),1,0)
    # print("vel_state = ",vel_state)

    return vel_state

