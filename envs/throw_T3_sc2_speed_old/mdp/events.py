# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation,RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_limiting_vel(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,   
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    velocity_value: torch.Tensor,     
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    # velocities
    # user coding
    robot: Articulation = env.scene[asset_cfg.name]
    person: Articulation = env.scene["person"]
    ##############   # condition
    person_pos=person.data.root_com_pos_w
    robot_pos=robot.data.root_com_pos_w
    # print("person pos = ",person_pos)
    # print("robot pos = ",robot_pos)
    # Compute Euclidean distance for each corresponding pair
    distances = torch.norm(person_pos - robot_pos, dim=1)

    # print("distances = ",distances)
    ############## 

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    # joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()
    # scale these values randomly
    # joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    # joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    # Scale joint position limits:
    new_joint_pos_limits = joint_pos_limits.clone()
    # Multiply lower limits by new_position_range[0]
    new_joint_pos_limits[:, :, 0] = joint_pos_limits[:, :, 0] * position_range[0]
    # Multiply upper limits by new_position_range[1]
    new_joint_pos_limits[:, :, 1] = joint_pos_limits[:, :, 1] * position_range[1]
    # joint_pos = joint_pos.clamp_(new_joint_pos_limits[..., 0], new_joint_pos_limits[..., 1])

    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    # Scale joint velocities:
    new_joint_vel = joint_vel_limits * velocity_range[1]
    # Create two clamping bounds: default (-limit, +limit), and safety (0, +limit)
    lower_vel_limit = -new_joint_vel.clone()
    upper_vel_limit = new_joint_vel.clone()
    # Mask for environments where person is too close
    too_close = distances <= 1.8  # boolean mask, shape: (num_envs,)

    # Apply [0, velocity_value] range where too close
    for i, close in enumerate(too_close):
        if close:
            lower_vel_limit[i] = torch.zeros_like(lower_vel_limit[i])
            upper_vel_limit[i] = torch.full_like(upper_vel_limit[i], velocity_value)


    # Clamp joint velocities
    joint_vel = joint_vel.clamp_(lower_vel_limit, upper_vel_limit)
    # print("joint_vel = ", joint_vel)
    # Writing the new values
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # asset.write_joint_limits_to_sim(new_joint_pos_limits,env_ids=env_ids)

 



def belt_moving(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def moving_cube(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    obj_pos=asset.data.root_pos_w
    obj_vel=asset.data.root_lin_vel_w
    # print("obj pos = ",obj_pos)
    # print("obj vel = ",obj_vel)
    if obj_pos[0][2]<0.0365 and obj_pos[0][0]<0.6:
        if obj_pos[0][1]<-0.05:

            # velocities
            vel_w = asset.data.root_vel_w[env_ids]

            # sample random velocities
            range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=asset.device)
            vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
            # set the velocities into the physics simulation
            asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def moving_person(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("person"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.root_pos_w[env_ids]

    person_pos=asset.data.root_pos_w
    person_vel=asset.data.root_lin_vel_w
    # print("obj pos = ",obj_pos)
    # print("obj vel = ",obj_vel)

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the positions & velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)