import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from isaaclab.envs import ManagerBasedRLEnv
from envs.hover.hover_env_cfg import HoverEnvCfg


def main():
    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :] = 0.0
            actions[:, 2] = 0.5

            obs, rew, terminated, truncated, info = env.step(actions)
            # print(rew)
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
