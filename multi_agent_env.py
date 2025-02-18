import gym
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim

# Custom Multi-Agent Environment with Domain Randomization
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(num_agents * 4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * num_agents)
        self.max_steps = 200
        self.steps = 0
        self.randomize_environment()
    
    def randomize_environment(self):
        """Apply domain randomization by varying agent attributes and environment parameters."""
        self.move_range = np.random.randint(5, 15)
        self.start_bounds = np.random.randint(30, 100)
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.agent_positions = np.random.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
    
    def reset(self):
        self.randomize_environment()
        self.steps = 0
        return self.agent_states.flatten()

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        move_mask = (actions == 1)
        rewards[move_mask] = 1
        if np.sum(move_mask) > 0:
            self.agent_positions[move_mask] += np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        self.steps += 1
        done = self.steps >= self.max_steps
        return self.agent_states.flatten(), np.sum(rewards), done, {}

# Transfer Learning with PPO
env = MultiAgentEnv(num_agents=2)
try:
    model = PPO.load("ppo_multiagent")
    model.set_env(env)
    print("Loaded pre-trained model.")
except:
    model = PPO("MlpPolicy", env, verbose=1)
    print("No pre-trained model found. Training from scratch.")

model.learn(total_timesteps=10000)
model.save("ppo_multiagent")

# Meta-Learning without learn2learn
class MetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def meta_train(meta_model, env, meta_lr=0.01, inner_lr=0.001, meta_iterations=100):
    optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
    for _ in range(meta_iterations):
        env.reset()
        state = torch.tensor(env.agent_states.flatten(), dtype=torch.float32)
        optimizer.zero_grad()
        output = meta_model(state)
        loss = -output.mean()
        loss.backward()
        optimizer.step()
    return meta_model

meta_model = MetaPolicy(input_dim=8, output_dim=2)
meta_trained_model = meta_train(meta_model, MultiAgentEnv(num_agents=2))

# Performance Comparison
new_task_env = MultiAgentEnv(num_agents=2)
task_states = torch.tensor(new_task_env.reset(), dtype=torch.float32)

ppo_adaptation_reward = np.mean([model.predict(task_states.numpy())[0] for _ in range(10)])
meta_adaptation_reward = meta_trained_model(task_states).mean().item()

print(f"Transfer Learning PPO Reward: {ppo_adaptation_reward}")
print(f"Meta-Learning Reward: {meta_adaptation_reward}")
