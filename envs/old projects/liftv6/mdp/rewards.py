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
from envs.liftv6.mdp.observations import bsk_pos_in_robot_root_frame

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
    # print("object position = ",obj_pos)
    # print("obj vel = ",obj_vel)
    # print("obj ma vel = ",torch.norm(obj_vel, dim=1))
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_w)
    # print("before ee_w = ",ee_w)
    # print("after ee_w = ",des_pos_w)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)
    # print("ee distance = ",object_ee_distance)
    # print("object_ee_distance = ", torch.norm(obj_pos - ee_w, dim=1))
    # print("************")
    # Compute the reward
    return 1 - torch.tanh(object_ee_distance / std)


def object_inside_basket(env: ManagerBasedRLEnv,
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
    return inside_box

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
    # bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    des_pos_b = command[:, :3]
    # print("************")
    # print("before command = ",command[:, :3])
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # print("command pos = ",des_pos_w)
    # print("object pos = ",object.data.root_pos_w[:, :3])
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # print("distance = ",distance, std)
    # print("reward = ",(object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std)))
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


#######
def object_throwing_check(
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
    print("********************************************************************")
    print("********************************************************************")
    print("reward function object_throwing_check")
    
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    print("obj_pos = ",obj_pos)
    print("obj_vel = ",obj_vel)
    print("obj vel mag = ",torch.norm(obj_vel, dim=1))
    print("speed condition = ",torch.where(torch.norm(obj_vel, dim=1)>2.2,1.0,0.0))
    print("bsk_pos = ",bsk_pos)
    
    # throwing conditions
    # 1-Releasing the object from the End-effector
    # 2-Direction alignment
    # 3-Required_Speed

    # 1-Releasing the object from the End-effector
    print("******** Releasing the object from the End-effector ********")
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)

    obj_released=(object.data.root_pos_w[:, 2] > minimal_height)*torch.where(object_ee_distance > ee_distance_release, 1.0, 0.0)
    print("ee_w = ",ee_w)
    print("object_ee_distance = ",object_ee_distance,object.data.root_pos_w[:, 2] > minimal_height)
    print("obj_released condition = ",obj_released)

    print("******** Direction alignment ********")

    # 2-Direction alignment
    # Project both vectors into the XY plane
    vel_xy = obj_vel[:, :2]
    to_basket_xy = bsk_pos[:, :2] - obj_pos[:, :2]
    print("vel_xy = ", vel_xy)
    print("mag vel_xy = ",(torch.norm(vel_xy, dim=1) > Vxy_velocity),object.data.root_pos_w[:, 2])
    print("to_basket_xy = ", to_basket_xy)
    # # Normalize both vectors
    vel_unit = vel_xy / (torch.norm(vel_xy, dim=1, keepdim=True) + 1e-8)
    print("vel_unit = ",vel_unit)
    basket_unit = to_basket_xy / (torch.norm(to_basket_xy, dim=1, keepdim=True) + 1e-8)
    print("basket_unit = ",basket_unit)
    # # Compute unsigned angle (in radians → degrees)
    dot = torch.sum(vel_unit * basket_unit, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)  # for numerical safety
    print("dot = ",dot)
    angle_rad = torch.acos(dot)
    print("angle_rad = ",angle_rad)
    # # # ✅ INSERT HERE:
    angle_reward = torch.cos(angle_rad)
    print("angle_reward = ",angle_reward)
    Direction_alignment=torch.where(angle_reward > 0.85, 1.0, 0.0)
    print("Direction_alignment = ",Direction_alignment)
    print("******** Required Speed ********")
    required_speed=torch.where(torch.norm(obj_vel, dim=1)>2.2,1.0,0.0)

    # angle_deg = angle_rad * (180.0 / torch.pi)
    # print("angle_deg = ",angle_deg)


    # print("Unsigned angle (deg) between velocity and basket direction:", angle_deg)
    # Assuming object_pos_w is a tensor of shape (num_envs, 3)
    # z_below_zero = obj_pos[:, 2] <= 0.0

    # if torch.any(z_below_zero):
    #     print("⚠️ Object Z position is ≤ 0 in some environments:")
    #     print(obj_pos)
    #     print(obj_pos[z_below_zero])
    # # Combined condition for throwing:
    # checking if the object has velocity
    # Combined condition for throwing:
    # throw_state = object_lifted & right_gripper & left_gripper & has_velocity
    throw_state=obj_released*Direction_alignment*required_speed
    print("throw_state = ",throw_state)
    

    return throw_state

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
    print("vel_state = ",vel_state)
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

#############################


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     # print("*****************")
#     # print("object.data.root_pos_w = ",object.data.root_pos_w)
#     # print("object.data.root_vel_w = ",object.data.root_lin_vel_w)
#     # print("robot joints = ",robot.data.joint_pos)
#     # print("Gripper joints = ",robot.data.joint_pos[:, -2:])
#     gripper_joints=robot.data.joint_pos[:, -2:]
#     condition = gripper_joints > 0.038
#     all_true = torch.all(gripper_joints > 0.038).item()
#     # print("Gripper condition = ",condition,all_true)
#     results=0.0
#     # print("results = ",results)
#     # Compute the reward conditionally (object must be lifted above minimal_height)
#     lifted_mask = object.data.root_pos_w[:, 2] > minimal_height
#     reward_tensor = lifted_mask * (1 - torch.tanh(distance / std))

#     # Check if any reward value is greater than 0 and print
#     # if torch.any(reward_tensor > 0):
#     #     print("reward function object_goal_distance: ", reward_tensor)
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

# def object_throwing_check2(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # object pos & vel
#     obj_pos=object.data.root_pos_w
#     obj_vel=object.data.root_lin_vel_w
#     # checking the conditions
#     # object_is_lifeted and gripper is openned and has_velocity
#     # if these conditions are met meanning the object is tossed
#     # print("********************************************************************")
#     # # checking if the object is lifted
#     object_lifted=(object.data.root_pos_w[:, 2] > minimal_height)
#     # checking if the gripper is opened
#     robot_joints = robot.data.joint_pos
#     right_gripper=robot_joints[:, -1]>0.037
#     left_gripper=robot_joints[:, -2]>0.037
#     # right_gripper=right_gripper>0.037
#     # left_gripper=left_gripper>0.037
#     # print("obj is lifted = ",object_lifted)
#     # print("right_gripper is open = ",right_gripper)
#     # print("left_gripper is open = ",left_gripper)
#     # checking if the object has velocity
#     # print("obj vel = ",torch.norm(obj_vel, dim=1))
#     has_velocity = (torch.norm(obj_vel, dim=1) > 1.5)
#     # print("has velocity = ",has_velocity)
#     # Combined condition for throwing:
#     throw_state = torch.all(object_lifted) and torch.all(right_gripper)and torch.all(left_gripper)and torch.all(has_velocity)
#     throw_state = object_lifted & right_gripper & left_gripper & has_velocity
#     # print("Throw state:", throw_state)
#     # Combined condition for throwing:
#     results=1.0
#     # print("results = ",throw_state*results)    
#     results=throw_state*results

#     return results

# def object_velocity_toward_goal(
#     env: ManagerBasedRLEnv,
#     goal_name: str,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     min_height: float = 0.015
# ) -> torch.Tensor:
#     object: RigidObject = env.scene[object_cfg.name]
#     robot: RigidObject = env.scene[robot_cfg.name]
#     command = env.command_manager.get_command(goal_name)

#     # goal in robot frame -> world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # object's current position and linear velocity
#     obj_pos = object.data.root_pos_w
#     obj_vel = object.data.root_lin_vel_w
#     # direction to goal
#     goal_dir = des_pos_w - obj_pos
#     goal_dir_norm = torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6
#     goal_dir_unit = goal_dir / goal_dir_norm
#     # projection of velocity onto goal direction
#     forward_velocity = torch.sum(obj_vel * goal_dir_unit, dim=1)
#     # reward only if object is lifted
#     lifted = (object.data.root_pos_w[:, 2] > min_height)
#     result =lifted * torch.clamp(forward_velocity, min=0.0)
#     result=0.0 
#     return result




# def obj_lost_contact(
#     env: ManagerBasedRLEnv,
#     minimal_height: float,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """giving additional rewards for throwing"""
    
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene["ee_frame"]
#     # Target object position: (num_envs, 3)
#     obj_pos = object.data.root_pos_w
#     # print("object position = ",obj_pos)
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_w)
#     # print("before ee_w = ",ee_w)
#     # print("after ee_w = ",des_pos_w)
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)
#     # print("**************lost_contact**************")
#     # print("object_ee_distance = ",object_ee_distance)
#     # print("obj pos = ",obj_pos)
#     # print("results = ",(object.data.root_pos_w[:, 2] > minimal_height),torch.where(object_ee_distance > 0.3, 1.0, 0.0))
    
#     return (object.data.root_pos_w[:, 2] > minimal_height)*torch.where(object_ee_distance > 0.3, 1.0, 0.0)

