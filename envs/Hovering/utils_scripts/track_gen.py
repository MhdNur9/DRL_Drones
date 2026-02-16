import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg


def gen_track(track_config: dict | None) -> RigidObjectCollectionCfg:
    track_config = track_config or {}

    return RigidObjectCollectionCfg(
        rigid_objects={
            f"reaching_point_{rp_id}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/reaching_point_{rp_id}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    # per-point color (RGB in [0,1])
                    # visual_material=sim_utils.PreviewSurfaceCfg(
                    #     diffuse_color=tuple(rp_cfg.get("color", (0.0, 1.0, 0.0)))
                    # ),
                    visual_material_path="Looks/material",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(rp_cfg.get("color", (0.0, 1.0, 0.0)))
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=rp_cfg["pos"],
                    rot=math_utils.quat_from_euler_xyz(
                        torch.tensor(0.0),
                        torch.tensor(0.0),
                        torch.tensor(rp_cfg.get("yaw", 0.0)),
                    ).tolist(),
                ),
            )
            for rp_id, rp_cfg in track_config.items()
        }
    )
