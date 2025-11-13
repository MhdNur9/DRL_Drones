from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from envs.throwv2_moving_beltv2 import mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

