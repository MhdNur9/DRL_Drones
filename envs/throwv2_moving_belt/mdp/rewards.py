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
from envs.throwv2_moving_belt.mdp.observations import bsk_pos_in_robot_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene["robot"]
    obj_pos=object.data.root_pos_w
    # print("before obj_pos = ",obj_pos)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj_pos)
    # print("after obj_pos = ",des_pos_w)
    # print("minimal_height = ",minimal_height)
    # print("result = ",torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0))
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
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos = object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
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
    command = env.command_manager.get_command("Target_pose")
    # print("***********")
    z_max = 0.05  # Condition: z must be less than 0.23
    # print("command[:, :3] = ",command[:, :3])
    # print("obj pos = ",object_pos_w)
    xmin= command[:, 0]+0.2
    xmax=command[:, 0]+0.6
    ymin=command[:, 1]-0.25
    ymax=command[:, 1]+0.25
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    inside_z = object_pos_w[:, 2] < z_max
    # print("inside = ",inside_x,inside_y,inside_z,inside_x & inside_y & inside_z)
    # Combine all conditions:
    inside_box = inside_x & inside_y & inside_z
    # print("inside_box = ",inside_box)
    if torch.any(inside_box):
        print("Obj is inside the basket")
    
    return inside_box

def object_inside_basket2(env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    # Define box boundaries:
    xmin, xmax = 0.95, 1.45
    ymin, ymax = -0.25, 0.25
    z_max = 0.05  # Condition: z must be less than 0.23

    # Check conditions for each coordinate:
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    inside_z = object_pos_w[:, 2] < z_max

    # Combine all conditions:
    inside_box = inside_x & inside_y & inside_z
    # print("inside_box = ",inside_box)
    if torch.any(inside_box):
        print("Obj is inside the basket")
    return inside_box

#######

def obj_vel_release_check(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    ee_distance_release: float,
    Vxy_velocity: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    # obj_vel_release_check conditions    
    # Direction alignment
    # Project both vectors into the XY plane
    # Direction alignment
    vel_xy = obj_vel[:, :2]
    to_basket_xy = bsk_pos[:, :2] - obj_pos[:, :2]
    # print("vel_xy = ", vel_xy)
    # print("mag vel_xy = ",(torch.norm(vel_xy, dim=1) > Vxy_velocity),object.data.root_pos_w[:, 2])
    # print("to_basket_xy = ", to_basket_xy)
    # # Normalize both vectors
    vel_unit = vel_xy / (torch.norm(vel_xy, dim=1, keepdim=True) + 1e-8)
    # print("vel_unit = ",vel_unit)
    basket_unit = to_basket_xy / (torch.norm(to_basket_xy, dim=1, keepdim=True) + 1e-8)
    # print("basket_unit = ",basket_unit)
    # # Compute unsigned angle (in radians → degrees)
    dot = torch.sum(vel_unit * basket_unit, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)  # for numerical safety
    # print("dot = ",dot)
    angle_rad = torch.acos(dot)
    # print("angle_rad = ",angle_rad)
    # # # ✅ INSERT HERE:
    angle_reward = torch.cos(angle_rad)
    # print("angle_reward = ",angle_reward)
    Direction_alignment=torch.where(angle_reward > 0.85, 1.0, 0.0)
    # print("Direction_alignment = ",Direction_alignment)

    # Required_Speed
    # vel_state=torch.where(torch.norm(obj_vel, dim=1)>2.2,1.0,0.0)
    vel_state=torch.where((torch.norm(vel_xy, dim=1) > Vxy_velocity),1,0)
    # print("vel_state = ",vel_state)
    # print("Direction_alignment = ",Direction_alignment)
    result=vel_state*Direction_alignment
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
    # Set a tolerance for comparison
    tol = 1e-6
    # Check if the last 2 elements are approximately 0.04
    right_gripper=joint_pos[:, -1]>0.037
    # print("before right_gripper = ",right_gripper)
    left_gripper=joint_pos[:, -2]>0.037
    # print("the gripper is open or not = ",right_gripper&left_gripper)
    results=(object.data.root_pos_w[:, 2] > minimal_height)*(right_gripper&left_gripper)
    return results



