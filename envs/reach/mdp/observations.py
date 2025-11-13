
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)
    print("gripper_pos = ",torch.cat((finger_joint_1, finger_joint_2), dim=1))
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    return ee_frame_pos

def w_frame_pos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    return des_pos_b

def ee_pos_error(env: ManagerBasedRLEnv, command_name: str,ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    pos_error_scalar = torch.norm((des_pos_b - ee_frame_pos), p=2, dim=-1)
    # print("position error scalar =", pos_error_scalar)
    return (des_pos_b - ee_frame_pos)


