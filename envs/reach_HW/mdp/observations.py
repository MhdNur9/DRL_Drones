
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation,RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
import isaaclab.utils.math as math_utils

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
                # user code
    robot: RigidObject = env.scene["robot"]


    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    command = env.command_manager.get_command("ee_pose")
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    command_pos_b = command[:, :3]
    command_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command_pos_b)
    distance = torch.norm(ee_w - command_pos_w, dim=1)


    # # saving env_variables
    # env.extras['robot joint acc']= robot.data.joint_acc.clone()
    # env.extras['robot joint vel']= robot.data.joint_vel.clone()
    # env.extras['robot joint pos']= robot.data.joint_pos.clone()
    # # print("last two joints =", robot.data.joint_pos.clone()[0, -2:])

    # env.extras['robot joint applied torque']= robot.data.applied_torque.clone()
    # env.extras['robot joint computed torque']= robot.data.computed_torque.clone()
    # env.extras['robot joint effort limits']= robot.data.joint_effort_limits.clone()
    # env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()

    # env.extras['robot ee_frame']= ee_w.clone()
    # env.extras['command_pos_w']= command_pos_w.clone()
    # env.extras['EE - Command distance']= distance.clone()
    # env.extras['EE - Command distance in 3D']= (des_pos_b - ee_frame_pos).clone()
    # env.extras['action_rate_l2']= torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clone()

    return (des_pos_b - ee_frame_pos)


def joint_pos_rel_f(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def joint_vel_rel_f(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
        # user code
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
    robot: RigidObject = env.scene["robot"]


    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    command = env.command_manager.get_command("ee_pose")
    # obtain the desired and current positions
    command_pos_b = command[:, :3]
    command_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command_pos_b)
    distance = torch.norm(ee_w - command_pos_w, dim=1)


    # saving env_variables
    # env.extras['robot joint acc']= robot.data.joint_acc.clone()
    # env.extras['robot joint vel']= robot.data.joint_vel.clone()
    # env.extras['robot joint pos']= robot.data.joint_pos.clone()
    # env.extras['robot joint applied torque']= robot.data.applied_torque.clone()
    # env.extras['robot joint computed torque']= robot.data.computed_torque.clone()
    # env.extras['robot joint effort limits']= robot.data.joint_effort_limits.clone()
    # env.extras['robot joint effort target']= robot.data.joint_effort_target.clone()

    # env.extras['robot ee_frame']= ee_w.clone()
    # env.extras['command_pos_w']= command_pos_w.clone()
    # env.extras['EE - Command distance']= distance.clone()

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
