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
from envs.throwing_T1_exp1.mdp.observations import bsk_pos_in_robot_root_frame
from datetime import datetime



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
    # Define box boundaries:
    xmin, xmax = 1.25, 1.75
    ymin, ymax = -0.75, 0.75
    command = env.command_manager.get_command("Target_pose")
    
    cmd_pos = command[:, :3]
    obj_pos=object_pos_w
    # print("****************")
    cmd_pos_y=cmd_pos[:, 1]
    obj_pos_y=obj_pos[:, [1]]
    # Prepare lists to store ymin and ymax
    ymin_list = []
    ymax_list = []

    # Iterate through each element and apply conditions
    for y in cmd_pos_y:
        if -0.55 <= y.item() <= -0.2:
            ymin_list.append(-0.55)
            ymax_list.append(-0.25)
        elif -0.2 < y.item() <= 0.2:
            ymin_list.append(-0.15)
            ymax_list.append(0.15)
        elif 0.2 < y.item() <= 0.55:
            ymin_list.append(0.25)
            ymax_list.append(0.55)
        else:
            ymin_list.append(float('nan'))
            ymax_list.append(float('nan'))

    # # Convert to tensors if needed
    # ymin_tensor = torch.tensor(ymin_list, device='cuda:0')
    # ymax_tensor = torch.tensor(ymax_list, device='cuda:0')

    # print("ymin = ", ymin_list)
    # print("ymax = ", ymax_list)
    # Squeeze obj_pos_y to shape (5,) for comparison
    obj_pos_y_flat = obj_pos_y.squeeze(1)
    # print("obj_pos_y_flat = ", obj_pos_y_flat)

    # Condition: check if each obj_pos_y is between its ymin and ymax
    in_range = (obj_pos_y_flat >= ymin) & (obj_pos_y_flat <= ymax)

    # print("In range mask:", in_range)

    z_max = 0.05  # Condition: z must be less than 0.23

    # Check conditions for each coordinate:
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    ymin, ymax = -0.258, 0.258
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    # inside_y=in_range
    inside_z = object_pos_w[:, 2] < z_max
    # print("inside_x = ",inside_x)
    # print("inside_y = ",inside_y,object_pos_w[:, 1])
    # print("inside_z = ",inside_z)

    # Combine all conditions:
    inside_box = (inside_x & inside_y & inside_z).int()
    # print("object_pos_w = ",object_pos_w)

    # if torch.any(inside_box):
    #     print("Obj is inside the basket",obj_pos)
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
    # print("joint_pos = ",joint_pos)

    # Check if the last 2 elements are approximately 0.04
    right_gripper=joint_pos[:, -1]>0.037
    # print("before right_gripper = ",right_gripper)
    left_gripper=joint_pos[:, -2]>0.037
    # print("the gripper is open or not = ",right_gripper&left_gripper)
    cond = (object.data.root_pos_w[:, 2] > minimal_height) * (right_gripper & left_gripper)
    print(datetime.now())

    if object.data.root_pos_w[:, 2] .item() > 0.270:
        print("lifted")



    if cond.item():
        print("reward_gripper_release_mid_throw =", cond.item())

    results=(object.data.root_pos_w[:, 2] > minimal_height)*(right_gripper&left_gripper).int()
    results=results.int()
   
    return results

def obj_vel_release_check(
    env: ManagerBasedRLEnv,
    Vxy_velocity: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command("Target_pose")
    # compute the desired position in the world frame
    cmd_pos = command[:, :3]
    obj_vel=object.data.root_lin_vel_w
    # Required_Speed
    # print("obj vel = ",obj_vel[:, :3])
    # print("vel_xy = ",vel_xy)
    obj_vel_x=obj_vel[:, :1]
    obj_vel_y=obj_vel[:, [1]]
    # print("obj pos = ",obj_pos)
    # print("command pos = ",cmd_pos)
    # print("obj_vel_x = ",obj_vel_x)
    # print("obj_vel_y = ",obj_vel_y)
    # ref_vel_x = torch.full_like(obj_vel_x, (Vxy_velocity)*3/4)
    ref_vel_x = torch.full_like(obj_vel_x, (Vxy_velocity)*3/4)
    ref_vel_y = torch.full_like(obj_vel_y, Vxy_velocity/5)
    # print("cmd_pos[:, [1]] = ",cmd_pos[:, [1]])
    same_sign = ((obj_vel_y * cmd_pos[:, [1]]) > 0).squeeze(1)
    # print("ref_vel_x = ",ref_vel_x)
    # print("ref_vel_y = ",ref_vel_y)
    # print("same_sign = ",same_sign)
    vel_x_state = (torch.norm(obj_vel_x, dim=1) > torch.norm(ref_vel_x, dim=1)).int()
    vel_y_state = (torch.norm(obj_vel_y, dim=1) > torch.norm(ref_vel_y, dim=1)).int()
    # print("vel_x_state = ",vel_x_state)
    # print("vel_y_state = ",vel_y_state)
    vel_y_check=vel_y_state&same_sign
    # print("vel_y_check = ",vel_y_check)
    vel_state=vel_x_state&vel_y_check
    # print("reward obj_vel_release_check = ",vel_state)
  
    return vel_state