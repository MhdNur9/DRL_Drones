from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.
    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.
    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract the specific diagonal element for the specified envs and bodies
        current_inertias = inertias[env_ids[:, None], body_ids, idx]

        # Randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            current_inertias,
            inertia_distribution_params,
            torch.arange(len(env_ids), device="cpu"),  # Use sequential indices for the subset
            torch.arange(len(body_ids), device="cpu"),  # Use sequential indices for the subset
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)

def randomize_wrench_map(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action: str,
    randomization_params: dict[str, tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize control terms by adding, scaling, or setting random values.

    Args:
        env: The environment instance
        env_ids: Environment IDs to randomize (None for all environments)
        action: Name of the action term
        randomization_params: Dictionary mapping parameter names to distribution params.
                            Keys should match both config attribute names and action term attribute names.
                            For example: {"kf": (0.8, 1.2), "kd": (0.5, 2.0), ...}
        operation: Operation to perform ("add", "scale", "abs")
        distribution: Distribution type for sampling
    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    config = env.action_manager.get_term(action).config
    action_term = env.action_manager.get_term(action)

    # Loop through each parameter to randomize
    for param_name, distribution_params in randomization_params.items():
        # Get default value from config
        default_value = getattr(config, param_name)
        term_default = torch.full((env.num_envs, 1), default_value, device=env.device)

        # Get current value from action term
        term_current = getattr(action_term, param_name)
        term_current[env_ids] = term_default[env_ids]

        # Randomize the parameter
        term_new = _randomize_prop_by_op(
            term_current,
            distribution_params,
            env_ids,
            slice(None),
            operation,
            distribution,
        )

        # Set the new randomized value
        setattr(action_term, param_name, term_new)

def reset_root_state_uniform_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = robot.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=robot.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=robot.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    # print ("positions = ",positions)
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)



    ################################################################# Object
    obj: RigidObject | Articulation = env.scene["object"]

    # robot world position you already computed
    robot_pos_w = positions  # (N,3) from earlier robot reset

    # place object under robot: same x,y, lower z
    obj_pos_w = robot_pos_w.clone()
    obj_pos_w[:, 2] = robot_pos_w[:, 2] - 0.04  # or your desired offset

    # orientation: zero (identity quaternion) OR keep object's default
    obj_quat_w = torch.zeros((len(env_ids), 4), device=robot.device)
    obj_quat_w[:, 0] = 1.0  # [w,x,y,z] = [1,0,0,0]

    # write pose
    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)


    #################################################################
    # print("robot pos = ",robot.data.root_pos_w[:, :3])
    # print("object pos = ",obj.data.root_pos_w[:, :3])

def apply_external_force_torque_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)
    # print("An external force torque event is applied")
    # obj: RigidObject | Articulation = env.scene["object"]
    # print("object = ", obj.data.projected_gravity_b)
    # print("robot = ", asset.data.projected_gravity_b)

def reset_obj_releaseing_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    ################################################################# Object
    obj: RigidObject | Articulation = env.scene["object"]
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    # print("---------------------")
    # print("object pos = ",obj.data.root_pos_w[:, :3])
    # print("robot pos = ",robot.data.root_pos_w[:, :3])

    # robot world position you already computed
    # place object under robot: same x,y, lower z
    obj_pos_w = obj.data.root_pos_w.clone()
    # print("before object pos = ",obj_pos_w)
    # print("before object pos = ",obj_pos_w[:, :3])
    obj_pos_w[:, 2] = 0.04  # 
    # print("after object pos = ",obj_pos_w[:, :3])

    # # orientation: zero (identity quaternion) OR keep object's default
    obj_quat_w = obj.data.root_com_quat_w.clone()
    # print("obj_quat_w = ",obj.data.root_com_quat_w.clone())

    # write pose
    root_pose_all = torch.cat([obj.data.root_pos_w.clone(), obj.data.root_com_quat_w.clone()], dim=-1)
    root_pose_all[env_ids, 2] = 0.04
    # obj.write_root_pose_to_sim(root_pose_all[env_ids], env_ids=env_ids)

    # obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)


    #################################################################
    # print("robot pos = ",robot.data.root_pos_w[:, :3])
    # print("object pos = ",obj.data.root_pos_w[:, :3])
