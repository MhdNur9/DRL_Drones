# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import AssetBase, Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]

    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    obj_pos=object.data.root_pos_w
    obj_vel=object.data.root_lin_vel_w
    results=torch.norm(obj_vel, dim=1)[0]
    # print("joints = ",robot.data.joint_pos)
    # print("obj_vel = ",results)
    if results.item() > 5:
        print("danger")
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    # print("ee_frame = ",ee_frame.data.target_pos_w)

    # Get linear velocity of the last link (usually the end-effector)
    ee_lin_vel = robot.data.body_lin_vel_w[0, -1]  # shape: (3,)
    ee_speed = torch.norm(ee_lin_vel).item()
    # Get all link linear velocities
    lin_vel = robot.data.body_lin_vel_w[0]  # shape: [num_links, 3]

    # Get per-link speeds and average
    avg_speed = torch.norm(lin_vel, dim=1).mean().item()
    # Get the maximum speed
    # Extract linear velocities of all links (shape: [num_links, 3])
    lin_vel = robot.data.body_lin_vel_w[0]

    # Compute Euclidean speed (norm) for each link
    link_speeds = torch.norm(lin_vel, dim=1)

    max_speed = link_speeds.max().item()
    if max_speed>5:
        print("Maximum link speed:", max_speed, "m/s")
    if ee_speed>5:
        print("End-effector speed:", ee_speed, "m/s")

    # person: Articulation = env.scene["person1"]
    # person_pos=person.data.root_pos_w
    # env.extras['person_pos']= person_pos
    # robot_pos=robot.data.root_pos_w
    # env.extras['robot_pos']= robot_pos
    ee_lin_vel = robot.data.body_lin_vel_w[0, -1]  # shape: (3,)
    ee_speed = torch.norm(ee_lin_vel).item()
    # print("ee_speed = ",    torch.norm(object.data.root_lin_vel_w).item())
    # print("ee_speed = ",    object.data.root_lin_vel_w)
    # print("ee_speed = ",    ee_speed)
    return object_pos_b

def bsk_pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   
) -> torch.Tensor:
    """The position of the basket"""
    bsk_pos=torch.tensor([[1.2, 0.0, 0.0]], device='cuda:0') # do not change it
    # bsk_pos=torch.tensor([[1.0, 0.0, 0.0]], device='cuda:0') # do not change it
    robot: RigidObject = env.scene[robot_cfg.name]
    bsk_pos, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], bsk_pos
    )
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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos - ee_w, dim=1)

    return (1 - torch.tanh(object_ee_distance / std)).unsqueeze(1)

def obj_bsk_distance_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    """Reward the agent for decreasing the distance between the object and the basket using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # object's current position and linear velocity
    obj_pos=object.data.root_pos_w
    # bsk_pos=torch.tensor([[1.2, 0.0, 0.04]], device='cuda:0')
    bsk_pos=bsk_pos_in_robot_root_frame(env=env)
    # Distance of the basket to the object: (num_envs,)
    obj_bsk_distance = torch.norm(obj_pos - bsk_pos, dim=1).unsqueeze(1)
   
    return obj_bsk_distance


