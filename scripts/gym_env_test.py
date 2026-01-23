import argparse
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Gymnasium interface of custom env built in Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

# Custom envs
import envs  # noqa: F401
# Default envs included in IsaacLab
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.0

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    ####
    # 
    
    scene = env.unwrapped.scene

    print("Scene entities:", scene.keys())

    robot = scene["robot"]   # adjust key if different

    print("\n=== Articulation prim path ===")
    print(robot.cfg.prim_path)

    # ------------------------
    # LINKS / BODIES
    # ------------------------
    print("\n=== Links (Bodies) ===")
    print(f"Number of bodies: {len(robot.body_names)}")
    for i, name in enumerate(robot.body_names):
        print(f"[{i}] {name}")

    # ------------------------
    # JOINTS
    # ------------------------
    print("\n=== Joints ===")
    print(f"Number of joints: {len(robot.joint_names)}")
    for i, name in enumerate(robot.joint_names):
        print(f"[{i}] {name}") 
    # 
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # print("actions = ",actions)
            
            actions[:, :] = 0.0

            # actions[:, 0] = 0.5 # Push into wall
            #actions[:, 2] = 0.5 # Hover
            # actions[:, 5] = 0.1666 # Yaw to remain square
           
            # apply actions
            obs, reward, terminated, truncated, done = env.step(actions)
            # print("actions = ",actions)
            # print(obs)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
