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

# Train PPO using domain randomization across multiple environments
def train_domain_randomization(num_envs=3, timesteps=10000):
    models = []
    for i in range(num_envs):
        env = MultiAgentEnv(num_agents=2)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)
        models.append(model)
    return models

# Fine-tune an existing model on a new environment (transfer learning)
def fine_tune_transfer_learning(base_model, new_env, timesteps=5000):
    base_model.set_env(new_env)
    base_model.learn(total_timesteps=timesteps)
    return base_model

# Meta-Policy Network with Sigmoid Activation
class MetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Ensure output is between 0 and 1

# Meta-Training with MSE Loss
def meta_train(meta_model, num_tasks=3, meta_lr=0.01, meta_iterations=100):
    optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
    for _ in range(meta_iterations):
        for _ in range(num_tasks):
            env = MultiAgentEnv(num_agents=2)
            env.reset()
            state = torch.tensor(env.agent_states.flatten(), dtype=torch.float32)
            expected_rewards = torch.tensor([1.0] * 2, dtype=torch.float32)  # Simulated expected rewards
            optimizer.zero_grad()
            output = meta_model(state)
            loss = torch.nn.functional.mse_loss(output, expected_rewards)  # MSE Loss
            loss.backward()
            optimizer.step()
    return meta_model

# Train domain-randomized PPO models
randomized_models = train_domain_randomization(num_envs=3, timesteps=10000)

# Select one model for domain-randomized transfer learning
new_env = MultiAgentEnv(num_agents=2)
dr_transfer_model = fine_tune_transfer_learning(randomized_models[0], new_env, timesteps=5000)

# Train meta-learning model
meta_model = MetaPolicy(input_dim=8, output_dim=2)
meta_trained_model = meta_train(meta_model, num_tasks=3)

# Fine-tune meta-learning model (Meta-Learning + Transfer Learning)
def fine_tune_meta_learning(meta_model, new_env, meta_lr=0.001, meta_iterations=100):
    optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
    for _ in range(meta_iterations):
        new_env.reset()
        state = torch.tensor(new_env.agent_states.flatten(), dtype=torch.float32)
        expected_rewards = torch.tensor([1.0] * 2, dtype=torch.float32)  # Expected rewards for adaptation
        optimizer.zero_grad()
        output = meta_model(state)
        loss = torch.nn.functional.mse_loss(output, expected_rewards)
        loss.backward()
        optimizer.step()
    return meta_model

ml_transfer_model = fine_tune_meta_learning(meta_trained_model, new_env)

# Performance Comparison
test_env = MultiAgentEnv(num_agents=2)
test_state = torch.tensor(test_env.reset(), dtype=torch.float32)

# Evaluate Domain Randomization + Transfer Learning
dr_transfer_rewards = np.mean([dr_transfer_model.predict(test_state.numpy())[0] for _ in range(10)])

# Evaluate Meta-Learning + Transfer Learning
ml_transfer_rewards = ml_transfer_model(test_state).mean().item()

print(f"Domain Randomization + Transfer Learning Reward: {dr_transfer_rewards}")
print(f"Meta-Learning + Transfer Learning Reward: {ml_transfer_rewards}")
