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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_rel_f(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """

    # user code
    # saving env_variables
    robot: Articulation = env.scene["robot"]
    env.extras['robot joint acc']= robot.data.joint_acc.clone()
    env.extras['robot joint vel']= robot.data.joint_vel.clone()
    env.extras['robot joint pos']= robot.data.joint_pos.clone()
    env.extras['robot joint applied torque']= robot.data.applied_torque.clone()
    env.extras['robot joint computed torque']= robot.data.computed_torque.clone()
    env.extras['robot joint effort limits']= robot.data.joint_effort_limits.clone()
    env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    # Target object position: (num_envs, 3)
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    print("ee_w = ",ee_w)




    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]



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
    # saving env_variables
    robot: Articulation = env.scene["robot"]
    env.extras['robot joint acc']= robot.data.joint_acc.clone()
    env.extras['robot joint vel']= robot.data.joint_vel.clone()
    env.extras['robot joint pos']= robot.data.joint_pos.clone()
    env.extras['robot joint applied torque']= robot.data.applied_torque.clone()
    env.extras['robot joint computed torque']= robot.data.computed_torque.clone()
    env.extras['robot joint effort limits']= robot.data.joint_effort_limits.clone()
    env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    # Target object position: (num_envs, 3)
    obj_pos = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    print("ee_w = ",ee_w)
    print("obj_pos = ",obj_pos)
    print("results = ",obj_pos - ee_w,torch.norm(obj_pos - ee_w, dim=1))
    env.extras['robot ee_frame']= robot.data.joint_effort_target.clone()
    env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()
    
    # joint_pos=robot.data.joint_pos.clone()
    # joint_vel=robot.data.joint_vel.clone()
    # joint_acc=robot.data.joint_acc.clone()
    # print("actual joints values = ",joint_pos)
    # print("actual joints vel = ",joint_vel)
    # print("actual joints acc = ",joint_acc)
    # print("actual joint_effort_limits = ", robot.data.joint_effort_limits)
    # print("actual joint_effort_target = ", robot.data.joint_effort_target)



    return object_pos_b

def object_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    # print("obj vel = ",object.data.root_lin_vel_w)
    return object.data.root_lin_vel_w

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


