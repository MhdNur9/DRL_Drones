

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicModel
from skrl.noises.torch import OrnsteinUhlenbeckNoise, GaussianNoise
from skrl.agents.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.env import wrap_env
from skrl.utils.isaacgym_utils import load_isaacgym_env_preview2, load_isaacgym_env_preview3


# models
class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False) -> None:
        super().__init__(observation_space, action_space, device, clip_actions)

        self.layer_linear1 = nn.Linear(self.num_observations, 32)
        self.layer_linear2 = nn.Linear(32, 32)
        self.layer_action_linear = nn.Linear(32, self.num_actions)

        self.to(self.device)

    def compute(self, states, taken_actions):
        x = F.relu(self.layer_linear1(states))
        x = F.relu(self.layer_linear2(x))
        return torch.tanh(self.layer_action_linear(x))

class DeterministicCritic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False) -> None:
        super().__init__(observation_space, action_space, device, clip_actions)

        self.layer_linear1 = nn.Linear(self.num_observations + self.num_actions, 32)
        self.layer_linear2 = nn.Linear(32, 32)
        self.layer_action_linear = nn.Linear(32, 1)

        self.to(self.device)

    def compute(self, states, taken_actions):
        x = torch.cat([states, taken_actions], dim=1)
        x = F.relu(self.layer_linear1(x))
        x = F.relu(self.layer_linear2(x))
        return self.layer_action_linear(x)


# environment
try:
    env = load_isaacgym_env_preview3("Cartpole")
except Exception as e:
    print("Isaac Gym (preview 3) failed: {}".format(e))
    print("Trying with Isaac Gym (preview 2)")
    env = load_isaacgym_env_preview2("Cartpole")
env = wrap_env(env)

device = env.device
print(f"device: {device}")


# memory
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device, replacement=False)


# networks
networks_ddpg = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
                 "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
                 "critic": DeterministicCritic(env.observation_space, env.action_space, device),
                 "target_critic": DeterministicCritic(env.observation_space, env.action_space, device)}

for k in networks_ddpg:
    networks_ddpg[k].init_parameters(method_name="normal_", mean=0.0, std=0.1)    


# agent
cfg_ddpg = DDPG_DEFAULT_CONFIG
# cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg_ddpg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
cfg_ddpg["batch_size"] = 256
cfg_ddpg["random_timesteps"] = 100
cfg_ddpg["learning_starts"] = 100
cfg_ddpg["discount_factor"] = 0.99
cfg_ddpg["polyak"] = 0.995
cfg_ddpg["actor_learning_rate"] = 0.001
cfg_ddpg["critic_learning_rate"] = 0.001
cfg_ddpg["experiment"]["write_interval"] = 25

agent_ddpg = DDPG(env=env, networks=networks_ddpg, memory=memory, cfg=cfg_ddpg)


# trainer
cfg = {"timesteps": 20000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent_ddpg)
trainer.start()
