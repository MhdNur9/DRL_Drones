# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply_inverse, yaw_quat
import isaaclab.utils.math as math_utils
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# from envs.velocity.mdp.user_functions import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # Start of User Code
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    # print("********************************************************************************")
   
    # print("joint_torque_applied = ", robot.data.applied_torque.clone())
    # print("joint_torque_computed = ", robot.data.computed_torque.clone())
    # print("joint_pos = ",robot.data.joint_pos.clone())
    # print("joint_vel = ",robot.data.joint_vel.clone())
    # print("cmd_vel = ",env.command_manager.get_command(command_name)[:, :2])
    # print("wz_cmd    = ", env.command_manager.get_command(command_name)[:, 2])
    # print("base_pos_w = ",robot.data.root_pos_w.clone())
    # print("base_quat_w = ",robot.data.root_quat_w.clone())
    # print("base_ang_vel_b = ", robot.data.root_ang_vel_b.clone())
    # print("base_ang_vel_w = ",robot.data.root_ang_vel_w.clone())
    # print("base_lin_vel_b = ",robot.data.root_lin_vel_b.clone())
    # print("base_lin_vel_w = ",robot.data.root_lin_vel_w.clone())
    # print("yaw_err2_body =", torch.square(env.command_manager.get_command(command_name)[:, 2] -  robot.data.root_ang_vel_b[:, 2]))
    # print("projected_gravity = ",robot.data.projected_gravity_b.clone())

    # print("height_scan = ",height_sc(env))
    # print("action_rate_l2 = ",action_rate_l2_user_code(env))
    # print("action = ",env.action_manager.action.clone())
    # print("env.action_manager.action = ",env.action_manager.action)
    # print("robot mass = ",robot.data.default_mass.sum())

    # --- torques ---
    env.extras["joint_torque_applied"]  = robot.data.applied_torque.clone()
    env.extras["joint_torque_computed"] = robot.data.computed_torque.clone()
    # --- joint states ---
    env.extras["joint_pos"] = robot.data.joint_pos.clone()
    env.extras["joint_vel"] = robot.data.joint_vel.clone()
    # --- commands ---
    cmd = env.command_manager.get_command(command_name)
    env.extras["cmd_vel_xy"] = cmd[:, :2].clone()   # [vx, vy]
    env.extras["cmd_wz"]     = cmd[:, 2].clone()    # yaw-rate command
    # --- base states (world frame) ---
    env.extras["base_pos_w"]     = robot.data.root_pos_w.clone()
    env.extras["base_quat_w"]    = robot.data.root_quat_w.clone()
    env.extras["base_lin_vel_w"] = robot.data.root_lin_vel_w.clone()
    env.extras["base_ang_vel_w"] = robot.data.root_ang_vel_w.clone()
    # --- base states (body frame) ---
    env.extras["base_lin_vel_b"] = robot.data.root_lin_vel_b.clone()
    env.extras["base_ang_vel_b"] = robot.data.root_ang_vel_b.clone()
    # --- yaw tracking error (body frame) ---
    env.extras["yaw_err2_body"] = torch.square(env.extras["cmd_wz"] - env.extras["base_ang_vel_b"][:, 2])
    # --- additional observations / diagnostics ---
    env.extras["projected_gravity_b"] = robot.data.projected_gravity_b.clone()
    # env.extras["height_scan"]         = height_sc(env).clone()
    # --- action diagnostics ---
    env.extras["action_rate_l2"] = action_rate_l2_user_code(env).clone()
    env.extras["action"]         = env.action_manager.action.clone()
    # --- constants (log once is enough, but ok to store) ---
    env.extras["robot_mass"] = robot.data.default_mass.sum().clone()
    

    # for attr in dir(robot.data):
    #     if not attr.startswith("_"):
    #         try:
    #             val = getattr(robot.data, attr)
    #             print(f"{attr}: {val}")
    #         except Exception as e:
    #             print(f"{attr}: <error: {e}>")


    # End of User Code

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_bpd(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """


       # Start of User Code
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    # print("********************************************************************************")
   
    # print("joint_torque_applied = ", robot.data.applied_torque.clone())
    # print("joint_torque_computed = ", robot.data.computed_torque.clone())
    # print("joint_pos = ",robot.data.joint_pos.clone())
    # print("joint_vel = ",robot.data.joint_vel.clone())
    # print("cmd_vel = ",env.command_manager.get_command(command_name)[:, :2])
    # print("wz_cmd    = ", env.command_manager.get_command(command_name)[:, 2])
    # print("base_pos_w = ",robot.data.root_pos_w.clone())
    # print("base_quat_w = ",robot.data.root_quat_w.clone())
    # print("base_ang_vel_b = ", robot.data.root_ang_vel_b.clone())
    # print("base_ang_vel_w = ",robot.data.root_ang_vel_w.clone())
    # print("base_lin_vel_b = ",robot.data.root_lin_vel_b.clone())
    # print("base_lin_vel_w = ",robot.data.root_lin_vel_w.clone())
    # print("yaw_err2_body =", torch.square(env.command_manager.get_command(command_name)[:, 2] -  robot.data.root_ang_vel_b[:, 2]))
    # print("projected_gravity = ",robot.data.projected_gravity_b.clone())

    # print("height_scan = ",height_sc(env))
    # print("action_rate_l2 = ",action_rate_l2_user_code(env))
    # print("action = ",env.action_manager.action.clone())
    # print("env.action_manager.action = ",env.action_manager.action)
    # print("robot mass = ",robot.data.default_mass.sum())

    # --- torques ---
    env.extras["joint_torque_applied"]  = robot.data.applied_torque.clone()
    env.extras["joint_torque_computed"] = robot.data.computed_torque.clone()
    # --- joint states ---
    env.extras["joint_pos"] = robot.data.joint_pos.clone()
    env.extras["joint_vel"] = robot.data.joint_vel.clone()
    # --- commands ---
    cmd = env.command_manager.get_command(command_name)
    env.extras["cmd_vel_xy"] = cmd[:, :2].clone()   # [vx, vy]
    env.extras["cmd_wz"]     = cmd[:, 2].clone()    # yaw-rate command
    # --- base states (world frame) ---
    env.extras["base_pos_w"]     = robot.data.root_pos_w.clone()
    env.extras["base_quat_w"]    = robot.data.root_quat_w.clone()
    env.extras["base_lin_vel_w"] = robot.data.root_lin_vel_w.clone()
    env.extras["base_ang_vel_w"] = robot.data.root_ang_vel_w.clone()
    # --- base states (body frame) ---
    env.extras["base_lin_vel_b"] = robot.data.root_lin_vel_b.clone()
    env.extras["base_ang_vel_b"] = robot.data.root_ang_vel_b.clone()
    # --- yaw tracking error (body frame) ---
    env.extras["yaw_err2_body"] = torch.square(env.extras["cmd_wz"] - env.extras["base_ang_vel_b"][:, 2])
    # --- additional observations / diagnostics ---
    env.extras["projected_gravity_b"] = robot.data.projected_gravity_b.clone()
    # env.extras["height_scan"]         = height_sc(env).clone()
    # --- action diagnostics ---
    env.extras["action_rate_l2"] = action_rate_l2_user_code(env).clone()
    env.extras["action"]         = env.action_manager.action.clone()
    # --- constants (log once is enough, but ok to store) ---
    env.extras["robot_mass"] = robot.data.default_mass.sum().clone()



    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

####################### User functions
def height_sc(env: ManagerBasedEnv, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors["height_scanner"]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset



def action_rate_l2_user_code(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
