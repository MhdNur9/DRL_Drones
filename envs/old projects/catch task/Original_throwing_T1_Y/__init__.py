import gymnasium as gym

from envs.Original_throwing_T1_Y import agents

gym.register(
    id="Isaac-Franka-Throw-T1-Y-original",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rlgames_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)