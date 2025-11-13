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


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    # robot: Articulation=env.scene[robot_cfg.name]
    # joint_pos=robot.data.joint_pos.clone()
    # print("joint pos = ", joint_pos)
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
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
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
    # print("*****************")
    # print("object.data.root_pos_w = ",object.data.root_pos_w)
    # print("object.data.root_vel_w = ",object.data.root_lin_vel_w)
    # print("robot joints = ",robot.data.joint_pos)
    # print("Gripper joints = ",robot.data.joint_pos[:, -2:])
    gripper_joints=robot.data.joint_pos[:, -2:]
    condition = gripper_joints > 0.038
    all_true = torch.all(gripper_joints > 0.038).item()
    # print("Gripper condition = ",condition,all_true)
    results=0.0
    # print("results = ",results)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_throwing_check(
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
    # object pos & vel
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    # checking the conditions
    # object_is_lifeted and gripper is openned and has_velocity
    # if these conditions are met meanning the object is tossed
    # print("********************************************************************")
    # # checking if the object is lifted
    object_lifted=(object.data.root_pos_w[:, 2] > minimal_height)
    # checking if the gripper is opened
    robot_joints = robot.data.joint_pos
    right_gripper=robot_joints[:, -1]>0.037
    left_gripper=robot_joints[:, -2]>0.037
    # right_gripper=right_gripper>0.037
    # left_gripper=left_gripper>0.037
    # print("obj is lifted = ",object_lifted)
    # print("right_gripper is open = ",right_gripper)
    # print("left_gripper is open = ",left_gripper)
    # checking if the object has velocity
    # print("obj vel = ",torch.norm(obj_vel, dim=1))
    has_velocity = (torch.norm(obj_vel, dim=1) > 1.5)
    # print("has velocity = ",has_velocity)
    # Combined condition for throwing:
    throw_state = torch.all(object_lifted) and torch.all(right_gripper)and torch.all(left_gripper)and torch.all(has_velocity)
    throw_state = object_lifted & right_gripper & left_gripper & has_velocity
    # print("Throw state:", throw_state)
    # Combined condition for throwing:
    results=1.0
    # print("results = ",throw_state*results)    
    results=throw_state*results

    return results

def object_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    goal_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.015
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(goal_name)

    # goal in robot frame -> world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # object's current position and linear velocity
    obj_pos = object.data.root_pos_w
    obj_vel = object.data.root_lin_vel_w
    # direction to goal
    goal_dir = des_pos_w - obj_pos
    goal_dir_norm = torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6
    goal_dir_unit = goal_dir / goal_dir_norm
    # projection of velocity onto goal direction
    forward_velocity = torch.sum(obj_vel * goal_dir_unit, dim=1)
    # reward only if object is lifted
    lifted = (object.data.root_pos_w[:, 2] > min_height)
    result =lifted * torch.clamp(forward_velocity, min=0.0)
    result=0.0 
    return result


def reward_gripper_release_mid_throw(
    env: ManagerBasedRLEnv,
    goal_name: str,
    ee_frame_cfg: str = "ee_frame",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_threshold: float = 0.05,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg]
    robot: Articulation=env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]   
    command = env.command_manager.get_command(goal_name)
    # goal in robot frame -> world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # gripper open state (assumed scalar per env)
    joint_pos=robot.data.joint_pos.clone()
    # Set a tolerance for comparison
    tol = 1e-6
    # Check if the last 2 elements are approximately 0.04
    gripper_open = torch.all(torch.abs(joint_pos[..., -2:] - 0.04) < tol, dim=-1)
    # cubic pos +vel
    obj_pos = object.data.root_pos_w
    obj_vel = object.data.root_lin_vel_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    dist_to_gripper = torch.norm(obj_pos - ee_w, dim=1)
    # airborne condition
    released = gripper_open & (dist_to_gripper > release_threshold)
    # velocity projection toward goal
    goal_dir = des_pos_w - obj_pos
    goal_dir_norm = torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6
    goal_dir_unit = goal_dir / goal_dir_norm
    velocity_toward_goal = torch.sum(obj_vel * goal_dir_unit, dim=1)
    # reward only if object is moving forward while released
    return released.float() * torch.clamp(velocity_toward_goal, min=0.0)

###################
###################
###################

# def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current orientations
#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
#     curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
#     # print("quat_error_magnitude = ",quat_error_magnitude(curr_quat_w, des_quat_w))
#     return quat_error_magnitude(curr_quat_w, des_quat_w)




def object_inside_basket(env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    bs_pos=torch.tensor([[1.5, 0, 0.105]]) # basket position
    bs_pos = bs_pos.to(object_pos_w.device)
    # print("bs_pos = ", bs_pos)
    # print("object_pos_w = ", object_pos_w)
    

    # Compute squared Euclidean distance
    distance_squared = torch.sum((object_pos_w - bs_pos) ** 2, dim=-1)
    # Reward is negative squared distance
    distance_squared = abs(distance_squared)
    # print("object_inside_basket")
    # print("distance_squared = ", distance_squared)
    # extract the asset (to enable type hinting)
    # asset: RigidObject = env.scene['robot']

    return (torch.where(distance_squared < 0.15, 30.0, 0.0))

# def reward_gripper_release_near_basket(env: ManagerBasedRLEnv) -> torch.Tensor:
#     object: RigidObject = env.scene["object"]
#     gripper_joint_pos = env.scene["robot"].data.joint_pos[:, -2:]  # last two joints (fingers)
#     object_pos = object.data.root_pos_w[:, :3]
#     basket_pos = torch.tensor([[1.0, 0.0, 0.105]], device=object_pos.device)

#     distance = torch.norm(object_pos - basket_pos, dim=-1)
#     gripper_open = (gripper_joint_pos > 0.03).all(dim=-1)  # threshold for open gripper

#     return torch.where((distance < 0.1) & gripper_open, 50.0, 0.0)





