import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
MAX_AGENTS = 5

# Custom Multi-Agent Environment with Domain Randomization
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * MAX_AGENTS)
        self.max_steps = 200
        self.steps = 0
        self.seed = seed
        self.randomize_environment()

    def randomize_environment(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.move_range = np.random.randint(3, 20)
        self.start_bounds = np.random.randint(20, 150)
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.agent_positions = np.random.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
        self.agent_speeds = np.random.uniform(0.5, 2.0, size=self.num_agents)
        self.goal_zones = np.random.randint(50, 550, size=(self.num_agents, 2))
        self.observation_noise_std = np.random.uniform(0, 0.2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed or self.seed)
        self.randomize_environment()
        self.steps = 0
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, {}

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        move_mask = (actions[:self.num_agents] == 1)
        rewards[move_mask] = 1.0  # Base reward per agent
        if np.sum(move_mask) > 0:
            move_amounts = np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amounts[i] = (move_amounts[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amounts
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        for i in range(self.num_agents):
            distance = np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])
            rewards[i] += max(0, 100 - distance) / 100  # Normalized distance reward
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        noisy_obs = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, truncated, {}

# Unseen MAEnv class
class UnseenMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents=2, seed=None):
        super().__init__(num_agents=num_agents, seed=seed)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)
        self.observation_noise_std = 0.5

    def randomize_environment(self):
        super().randomize_environment()
        self.move_range = np.random.randint(20, 40)

    def step(self, actions):
        if len(actions) > self.num_agents:
            actions = actions[:self.num_agents]
        rewards = np.zeros(self.num_agents)
        move_mask = (actions == 1)
        rewards[move_mask] = 1.0
        if np.sum(move_mask) > 0:
            move_amounts = np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amounts[i] = (move_amounts[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amounts
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        avg_pos = np.mean(self.agent_positions, axis=0)
        for i in range(self.num_agents):
            dist_to_goal = np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])
            dist_to_avg = np.linalg.norm(self.agent_positions[i] - avg_pos)
            rewards[i] += max(0, 100 - dist_to_goal) / 100 + 0.5 * max(0, 50 - dist_to_avg) / 50  # Reduce avg_pos impact # Dual reward
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        noisy_obs = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        avg_reward = np.sum(rewards) / self.num_agents
        return padded_obs, avg_reward, terminated, truncated, {}
    
# Lightweight Meta-Policy
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LightMetaPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# MAPPO Implementation
class MAPPOPolicy(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim * MAX_AGENTS, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def get_action(self, obs, num_agents, deterministic=False):
        agent_obs = obs.reshape(MAX_AGENTS, -1)[:num_agents]
        action_probs = self.actor(agent_obs)
        action_probs = action_probs.squeeze(-1)
        if deterministic:
            return (action_probs > 0.5).float()
        dist = torch.distributions.Bernoulli(action_probs)
        return dist.sample()
    
    def get_value(self, all_obs):
        return self.critic(all_obs)

# Simplified MAML Implementation
class MAMLPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
        
    def adapt(self, loss, lr=0.01):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads):
                param -= lr * grad
        return self

# Training Functions
def train_light_meta_policy(model, env_fn, meta_iterations=300, inner_rollouts=30, gamma=0.99, entropy_coef=0.02):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    for iteration in range(meta_iterations):
        total_loss = 0
        for _ in range(2):
            num_agents = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.1, 0.3, 0.4])
            env = env_fn(num_agents=num_agents)
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            log_probs, rewards, entropies = [], [], []
            for _ in range(inner_rollouts):
                action_probs = model(obs)
                dist = torch.distributions.Bernoulli(action_probs)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).mean()
                entropy = dist.entropy().mean()
                next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
                reward=reward * np.random.uniform(0.8, 1.2)
                obs = torch.tensor(next_obs, dtype=torch.float32)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                if terminated or truncated:
                    break
            returns = []
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + gamma * discounted_reward
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns, dtype=torch.float32)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            task_loss = -torch.sum(log_probs * returns) - entropy_coef * entropies.sum()
            total_loss += task_loss / 2
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if iteration % 10 == 0:
            print(f"Light Meta-Iter {iteration} | Avg Return: {returns.sum().item():.2f}")
    return model

def train_mappo_policy(env_fn, num_agents=2, iterations=1000, rollout_steps=200, gamma=0.99):
    env = env_fn(num_agents, seed=42)  # Fixed seed
    obs_dim = env.observation_space.shape[0] // MAX_AGENTS
    policy = MAPPOPolicy(obs_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    for iteration in range(iterations):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        rewards, log_probs, values = [], [], []
        for _ in range(rollout_steps):
            actions = policy.get_action(obs, num_agents, deterministic=False)
            value = policy.get_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            dist = torch.distributions.Bernoulli(policy.actor(obs.reshape(MAX_AGENTS, -1)[:num_agents]).squeeze(-1))
            log_prob = dist.log_prob(actions).mean()
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            obs = next_obs
            if terminated or truncated:
                break
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze(1)
        advantages = returns - values.detach()
        policy_loss = -torch.stack(log_probs) * advantages
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss.mean() + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(f"MAPPO Iter {iteration} | Return: {returns.sum().item():.2f}")
    return policy

def meta_train_maml(env_fn, meta_iterations=200, inner_steps=5, inner_rollouts=10, gamma=0.99, inner_lr=0.01):
    input_dim = MAX_AGENTS * 4
    output_dim = MAX_AGENTS
    meta_model = MAMLPolicy(input_dim, output_dim)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    for iteration in range(meta_iterations):
        env = env_fn()
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        adapted_model = MAMLPolicy(input_dim, output_dim)
        adapted_model.load_state_dict(meta_model.state_dict())
        for _ in range(inner_steps):
            action_probs = adapted_model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            loss = -dist.log_prob(actions).mean() * reward
            adapted_model = adapted_model.adapt(loss, lr=inner_lr)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated or truncated:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
        log_probs, rewards = [], []
        for _ in range(inner_rollouts):
            action_probs = adapted_model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            log_probs.append(log_prob.mean())
            rewards.append(reward)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated or truncated:
                break
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * returns)
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
        if iteration % 10 == 0:
            print(f"MAML Meta-Iter {iteration} | Return: {returns.sum().item():.2f}")
    return meta_model

# Fine-Tuning Functions
def fine_tune_light_meta_policy(model, env_fn, max_steps=25, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    env = env_fn()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    best_reward = float('-inf')
    no_improve_count = 0

    for step in range(max_steps):
        action_probs = model(obs)
        dist = torch.distributions.Bernoulli(action_probs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).mean()
        entropy = dist.entropy().mean()
        next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        reward = (reward - env.num_agents * 0.5) / env.num_agents
        loss = -log_prob * reward - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if reward > best_reward:
            best_reward = reward
            no_improve_count = 0
        else:
            no_improve_count += 1
        obs = next_obs
        if terminated or truncated or no_improve_count >= patience:
            break
    return model

def fine_tune_maml_policy(meta_model, env_fn, steps=10):  # Match LightMeta capacity
    env = env_fn()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    adapted_model = MAMLPolicy(meta_model.fc1.in_features, meta_model.fc2.out_features)
    adapted_model.load_state_dict(meta_model.state_dict())
    for _ in range(steps):
        action_probs = adapted_model(obs)
        dist = torch.distributions.Bernoulli(action_probs)
        actions = dist.sample()
        next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
        loss = -dist.log_prob(actions).mean() * reward
        adapted_model = adapted_model.adapt(loss)
        obs = torch.tensor(next_obs, dtype=torch.float32)
        if terminated or truncated:
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
    return adapted_model

# Evaluation Functions
def evaluate_meta_policy(model, env_fn, episodes=10):
    total_rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                probs = model(obs)
                dist = torch.distributions.Bernoulli(probs)
                actions = dist.sample().numpy()[:env.num_agents]  # Sample, not threshold
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def evaluate_mappo_policy(policy, env_fn, episodes=10):
    total_rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        num_agents = env.num_agents
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                actions = policy.get_action(obs, num_agents, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
            done = terminated or truncated
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def evaluate_sb3_policy(model, env_fn, episodes=10):
    total_rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if len(action) > env.num_agents:
                action = action[:env.num_agents]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

# Environment Factory Functions
def env_fn(num_agents=None, seed=None):
    if num_agents is None:
        num_agents = np.random.choice([1, 2, 3, 4, 5])  # Expanded range
    return MultiAgentEnv(num_agents=num_agents, seed=seed)

def unseen_env_fn(seed=None):
    env = UnseenMultiAgentEnv(num_agents=5, seed=seed)
    print(f"Unseen Env Noise: {env.observation_noise_std}")  # Verify distinction
    return env

def generate_unseen_envs_with_varied_agents(agent_counts, trials_per_count):
    envs = []
    for count in agent_counts:
        for trial in range(trials_per_count):
            envs.append(lambda seed=None, c=count, t=trial: UnseenMultiAgentEnv(num_agents=c, seed=seed if seed is not None else 42 + t))
    return envs

def evaluate_on_varied_unseen_envs(light_meta_model, maml_model, dr_model, ppo_model, mappo_policy, 
                                 fine_tune_steps_light=20, fine_tune_steps_maml=10, agent_counts=None):
    if agent_counts is None:
        agent_counts = [1, 2, 3, 4, 5]
    unseen_envs = generate_unseen_envs_with_varied_agents(agent_counts, trials_per_count=3)
    results = {
        "agent_counts": [],
        "ppo_rewards": [],
        "dr_rewards": [],
        "mappo_rewards": [],
        "light_meta_zero_shot": [],
        "light_meta_adapted": [],
        "maml_zero_shot": [],
        "maml_adapted": []
    }
    for env_fn in unseen_envs:
        env_instance = env_fn()
        num_agents = env_instance.num_agents
        results["agent_counts"].append(num_agents)
        print(f"\nEvaluating Unseen Environment with {num_agents} agents...")
        ppo_reward = evaluate_sb3_policy(ppo_model, env_fn, episodes=10)
        dr_reward = evaluate_sb3_policy(dr_model, env_fn, episodes=10)
        mappo_reward = evaluate_mappo_policy(mappo_policy, env_fn, episodes=10)
        light_meta_zero = evaluate_meta_policy(light_meta_model, env_fn, episodes=10)
        light_meta_adapted = evaluate_meta_policy(fine_tune_light_meta_policy(light_meta_model, env_fn, max_steps=fine_tune_steps_light), env_fn, episodes=10)
        maml_zero = evaluate_meta_policy(maml_model, env_fn, episodes=10)
        maml_adapted = evaluate_meta_policy(fine_tune_maml_policy(maml_model, env_fn, steps=fine_tune_steps_maml), env_fn, episodes=10)
        results["ppo_rewards"].append(ppo_reward)
        results["dr_rewards"].append(dr_reward)
        results["mappo_rewards"].append(mappo_reward)
        results["light_meta_zero_shot"].append(light_meta_zero)
        results["light_meta_adapted"].append(light_meta_adapted)
        results["maml_zero_shot"].append(maml_zero)
        results["maml_adapted"].append(maml_adapted)
        print(f"Results for {num_agents} agents: "
              f"PPO: {ppo_reward:.1f}, DR: {dr_reward:.1f}, MAPPO: {mappo_reward:.1f}, "
              f"LightMeta-Zero: {light_meta_zero:.1f}, LightMeta-Adapted: {light_meta_adapted:.1f}, "
              f"MAML-Zero: {maml_zero:.1f}, MAML-Adapted: {maml_adapted:.1f}")
    return results

def analyze_varied_agent_results(results):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(results)
    grouped = df.groupby('agent_counts').mean()
    
    print("\n=== Performance Summary by Agent Count ===")
    print(grouped)
    
    plt.figure(figsize=(12, 8))
    for column in grouped.columns:
        plt.plot(grouped.index, grouped[column], label=column)
    
    plt.title("Performance Across Agent Counts in Unseen Environments")
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return grouped

# --- Training Phase ---
env = MultiAgentEnv(num_agents=2, seed=42)
ppo_model = PPO("MlpPolicy", env, verbose=0)
ppo_model.learn(total_timesteps=10000)

domain_randomized_models = [PPO("MlpPolicy", MultiAgentEnv(num_agents=2, seed=42 + i), verbose=0) for i in range(3)]
for model in domain_randomized_models:
    model.learn(total_timesteps=10000)

dr_transfer_model = domain_randomized_models[0]
dr_transfer_model.set_env(env)
dr_transfer_model.learn(total_timesteps=5000)

mappo_policy = train_mappo_policy(env_fn, num_agents=2, iterations=1000)

sample_env = MultiAgentEnv(num_agents=MAX_AGENTS, seed=42)
input_dim = sample_env.observation_space.shape[0]
output_dim = sample_env.action_space.nvec.shape[0]

light_meta_model = LightMetaPolicy(input_dim=input_dim, output_dim=output_dim)
light_meta_trained_model = train_light_meta_policy(light_meta_model, env_fn, meta_iterations=300, inner_rollouts=30)

maml_model = meta_train_maml(env_fn)


# --- Evaluation Phase ---
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
print("\n--- Generalization to Larger Teams (5 Agents) ---")
ppo_reward_5 = evaluate_sb3_policy(ppo_model, lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
dr_reward_5 = evaluate_sb3_policy(dr_transfer_model, lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
mappo_reward_5 = evaluate_mappo_policy(mappo_policy, lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
light_meta_zero_shot_5 = evaluate_meta_policy(light_meta_trained_model, lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
light_meta_adapted_5 = evaluate_meta_policy(fine_tune_light_meta_policy(light_meta_trained_model, lambda: MultiAgentEnv(num_agents=5, seed=42), max_steps=20), lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
maml_zero_shot_5 = evaluate_meta_policy(maml_model, lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)
maml_adapted_5 = evaluate_meta_policy(fine_tune_maml_policy(maml_model, lambda: MultiAgentEnv(num_agents=5, seed=42), steps=20), lambda seed=None: MultiAgentEnv(num_agents=5, seed=seed), episodes=10)

print(f"PPO Reward (5 agents): {ppo_reward_5}")
print(f"Domain Randomization + Transfer Reward (5 agents): {dr_reward_5}")
print(f"MAPPO Reward (5 agents): {mappo_reward_5}")
print(f"Light Meta-Learning Zero-Shot Reward (5 agents): {light_meta_zero_shot_5}")
print(f"Light Meta-Learning Adapted Reward (5 agents): {light_meta_adapted_5}")
print(f"MAML Zero-Shot Reward (5 agents): {maml_zero_shot_5}")
print(f"MAML Adapted Reward (5 agents): {maml_adapted_5}")

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
print("\n--- Generalization to Unseen Environment ---")
ppo_unseen = evaluate_sb3_policy(ppo_model, lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
dr_unseen = evaluate_sb3_policy(dr_transfer_model, lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
mappo_unseen = evaluate_mappo_policy(mappo_policy, lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
light_meta_zero_shot_unseen = evaluate_meta_policy(light_meta_trained_model, lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
light_meta_finetuned_unseen = evaluate_meta_policy(fine_tune_light_meta_policy(light_meta_trained_model, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42), max_steps=25),lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
maml_zero_shot_unseen = evaluate_meta_policy(maml_model, lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)
maml_finetuned_unseen = evaluate_meta_policy(fine_tune_maml_policy(maml_model, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42), steps=25), lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed), episodes=10)

print(f"PPO Reward (Unseen env): {ppo_unseen}")
print(f"Domain Randomization + Transfer Reward (Unseen env): {dr_unseen}")
print(f"MAPPO Reward (Unseen env): {mappo_unseen}")
print(f"Light Meta-Learning Zero-Shot Reward (Unseen env): {light_meta_zero_shot_unseen}")
print(f"Light Meta-Learning Adapted Reward (Unseen env): {light_meta_finetuned_unseen}")
print(f"MAML Zero-Shot Reward (Unseen env): {maml_zero_shot_unseen}")
print(f"MAML Adapted Reward (Unseen env): {maml_finetuned_unseen}")

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
print("\n--- Generalization to Varied Unseen Environments ---")
unseen_results = evaluate_on_varied_unseen_envs(
    light_meta_trained_model, maml_model, dr_transfer_model, ppo_model, mappo_policy,
    fine_tune_steps_light=25, fine_tune_steps_maml=25, agent_counts=[1, 2, 3, 4, 5]
)
results_summary = analyze_varied_agent_results(unseen_results)