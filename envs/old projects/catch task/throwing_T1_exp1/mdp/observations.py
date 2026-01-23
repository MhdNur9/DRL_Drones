# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def object_vel_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    obj_vel=object.data.root_lin_vel_w

    return obj_vel

def bsk_pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the basket"""
    bsk_pos=torch.tensor([[1.2, 0.0, 0.0]], device='cuda:0')
    robot: RigidObject = env.scene[robot_cfg.name]
    bsk_pos, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], bsk_pos
    )
        # user code
    robot: RigidObject = env.scene["robot"]


    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    command = env.command_manager.get_command("Target_pose")
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    command_pos_b = command[:, :3]
    command_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command_pos_b)
    distance = torch.norm(ee_w - command_pos_w, dim=1)


    # saving env_variables
    env.extras['robot joint acc']= robot.data.joint_acc.clone()
    env.extras['robot joint vel']= robot.data.joint_vel.clone()
    env.extras['robot joint pos']= robot.data.joint_pos.clone()
    # print("last two joints =", robot.data.joint_pos.clone()[0, -2:])

    env.extras['robot joint applied torque']= robot.data.applied_torque.clone()
    env.extras['robot joint computed torque']= robot.data.computed_torque.clone()
    env.extras['robot joint effort limits']= robot.data.joint_effort_limits.clone()
    env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()

    env.extras['robot ee_frame']= ee_w.clone()
    env.extras['command_pos_w']= command_pos_w.clone()
    env.extras['EE - Command distance']= distance.clone()

    return bsk_pos


def object_ee_distance_obs(
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
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)
    # Compute the reward
    result=(1 - torch.tanh(object_ee_distance / std)).unsqueeze(1)
    
    return result

def obj_bsk_distance_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    """Reward the agent for decreasing the distance between the object and the basket using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # object's current position and linear velocity
    obj_pos=object.data.root_pos_w
    bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    # Distance of the basket to the object: (num_envs,)
    obj_bsk_distance = torch.norm(obj_pos - bsk_pos, dim=1).unsqueeze(1)
    
    return obj_bsk_distance


