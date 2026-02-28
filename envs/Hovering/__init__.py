import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Hover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Hover-firefly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_env_cfg:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Hover-firefly-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_env_cfg_play:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Hover-firefly-track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_track_env_cfg:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_track_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Hover-firefly-track-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_track_env_cfg_play:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_track_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Hover-simple-drone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_grasp_env_cfg:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:simple_drone_skrl_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Hover-firefly-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_env_cfg:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_PPO_cfg.yaml", # PPO
        "skrl_trpo_cfg_entry_point": f"{agents.__name__}:firefly_skrl_trpo_cfg.yaml", # TRPO
        "skrl_rpo_cfg_entry_point": f"{agents.__name__}:firefly_skrl_rpo_cfg.yaml", # RPO
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:firefly_skrl_sac_cfg.yaml", # SAC
        "skrl_td3_cfg_entry_point": f"{agents.__name__}:firefly_skrl_td3_cfg.yaml", # TD3
    },
)

gym.register(
    id="Isaac-Hover-firefly-play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_firefly_env_cfg_play:HoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:firefly_skrl_PPO_cfg.yaml", # PPO
        "skrl_trpo_cfg_entry_point": f"{agents.__name__}:firefly_skrl_trpo_cfg.yaml", # TRPO
        "skrl_rpo_cfg_entry_point": f"{agents.__name__}:firefly_skrl_rpo_cfg.yaml", # RPO
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:firefly_skrl_sac_cfg.yaml", # SAC
        "skrl_td3_cfg_entry_point": f"{agents.__name__}:firefly_skrl_td3_cfg.yaml", # TD3
    },
)
