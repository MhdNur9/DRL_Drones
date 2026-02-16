from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import cv2
import torch


from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg,SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from envs.Hovering.mdp.utils.logger import log
from .events import reset_after_prev_gate


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TargetPosFromTrackCommand(CommandTerm):
    """Outputs a 3D position command that points to a track object indexed by `next_idx`."""

    cfg: "TargetPosFromTrackCommandCfg"

    def __init__(self, cfg: "TargetPosFromTrackCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg

        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.num_points = self.track.num_objects
        self.curr_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


        # one index per env
        self.next_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.start_index is not None:
            self.next_idx[:] = int(cfg.start_index)

        # command buffer: (N,3)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # Return current target position (world)
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        # When an env resets, set it back to first point (or configured start)
        if self.cfg.start_index is None:
            self.next_idx[env_ids] = 0
        else:
            self.next_idx[env_ids] = int(self.cfg.start_index)

    def _update_command(self):
        # track positions: (N, num_points, 3)
        pos_w = self.track.data.object_com_pos_w
        self._command = pos_w[torch.arange(self.num_envs, device=self.device), self.next_idx]

    # helper for your event
    def advance(self, env_ids: torch.Tensor, step: int = 1):
        self.next_idx[env_ids] = (self.next_idx[env_ids] + step) % self.num_points


@configclass
class TargetPosFromTrackCommandCfg(CommandTermCfg):
    class_type: type = TargetPosFromTrackCommand

    track_name: str = MISSING
    start_index: int | None = 0