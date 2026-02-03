import gymnasium as gym

from envs.reach_HW import agents

gym.register(
    id="Isaac-Franka-Reach-HW",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UR10-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg_ur10:UR10ReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_UR10_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg_open_arm:OpenArmReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_open_arm_cfg.yaml",
    },
)