# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
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
    return distance < threshold

def object_inside_basket_terminate(env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    # Define box boundaries:
    xmin, xmax = 1.25, 1.75
    ymin, ymax = -0.75, 0.75
    command = env.command_manager.get_command("Target_pose")
    cmd_pos = command[:, :3]
    obj_pos=object_pos_w
    # print("****************")
    cmd_pos_y=cmd_pos[:, 1]
    obj_pos_y=obj_pos[:, [1]]
    # Prepare lists to store ymin and ymax
    ymin_list = []
    ymax_list = []

    # Iterate through each element and apply conditions
    for y in cmd_pos_y:
        if -0.55 <= y.item() <= -0.2:
            ymin_list.append(-0.55)
            ymax_list.append(-0.25)
        elif -0.2 < y.item() <= 0.2:
            ymin_list.append(-0.15)
            ymax_list.append(0.15)
        elif 0.2 < y.item() <= 0.55:
            ymin_list.append(0.25)
            ymax_list.append(0.55)
        else:
            ymin_list.append(float('nan'))
            ymax_list.append(float('nan'))

    # # Convert to tensors if needed
    # ymin_tensor = torch.tensor(ymin_list, device='cuda:0')
    # ymax_tensor = torch.tensor(ymax_list, device='cuda:0')

    # print("ymin = ", ymin_list)
    # print("ymax = ", ymax_list)
    # Squeeze obj_pos_y to shape (5,) for comparison
    obj_pos_y_flat = obj_pos_y.squeeze(1)
    # print("obj_pos_y_flat = ", obj_pos_y_flat)

    # Condition: check if each obj_pos_y is between its ymin and ymax
    in_range = (obj_pos_y_flat >= ymin) & (obj_pos_y_flat <= ymax)

    # print("In range mask:", in_range)

    z_max = 0.05  # Condition: z must be less than 0.23

    # Check conditions for each coordinate:
    inside_x = (object_pos_w[:, 0] >= xmin) & (object_pos_w[:, 0] <= xmax)
    ymin, ymax = -0.258, 0.258
    inside_y = (object_pos_w[:, 1] >= ymin) & (object_pos_w[:, 1] <= ymax)
    # inside_y=in_range
    inside_z = object_pos_w[:, 2] < z_max
    # print("inside_x = ",inside_x)
    # print("inside_y = ",inside_y,object_pos_w[:, 1])
    # print("inside_z = ",inside_z)

    # Combine all conditions:
    inside_box = (inside_x & inside_y & inside_z)
    # print("object_pos_w = ",object_pos_w)

    # if torch.any(inside_box):
    #     print("Obj is inside the basket",obj_pos)
        # print("obj pos = ",obj_pos)
        # if obj_pos[0][2]<0.05:
        #     print("obj pos = ",obj_pos)

    return inside_box
