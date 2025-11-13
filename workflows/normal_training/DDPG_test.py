import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
import skrl.models.torch.deterministic as deterministic
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise, GaussianNoise
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG


# models
class DeterministicActor(deterministic):
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

class DeterministicCritic(deterministic):
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
print("cfg_ddpg",cfg_ddpg)
