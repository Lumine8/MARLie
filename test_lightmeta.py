import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
MAX_AGENTS = 5
EPISODES = 20
ACTION_DIM = 5  # 0:stay, 1:up, 2:down, 3:left, 4:right

# Reward Shaping Function (unchanged)
def shape_multi_agent_rewards(rewards, agent_positions, goal_zones, num_agents):
    if num_agents <= 1:
        return rewards
    shaped_rewards = rewards.copy()
    centroid = np.mean(agent_positions, axis=0)
    avg_goal = np.mean(goal_zones, axis=0)
    centroid_goal_distance = np.linalg.norm(centroid - avg_goal)
    agent_centroid_distances = [np.linalg.norm(pos - centroid) for pos in agent_positions]
    cohesion_factor = min(1.0, centroid_goal_distance / 200.0)
    for i in range(num_agents):
        cohesion_bonus = cohesion_factor * (1.0 - agent_centroid_distances[i] / 100.0)
        exploration_factor = 1.0 - cohesion_factor
        exploration_bonus = exploration_factor * (agent_centroid_distances[i] / 100.0) * 0.5
        shaped_component = max(-0.5, min(0.5, cohesion_bonus + exploration_bonus))
        shaped_rewards[i] += shaped_component
    return shaped_rewards

# Multi-Agent Environment
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([ACTION_DIM] * MAX_AGENTS)
        self.max_steps = 200
        self.steps = 0
        self.seed = seed
        self.randomize_environment()

    def randomize_environment(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.move_amount = 5
        self.start_bounds = np.random.randint(20, 150)
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.agent_positions = np.random.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2)).astype(float)
        self.agent_speeds = np.random.uniform(0.5, 2.0, size=self.num_agents)
        self.goal_zones = np.random.randint(50, 550, size=(self.num_agents, 2))
        self.observation_noise_std = np.random.uniform(0, 0.2)
        self.prev_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed or self.seed)
        self.randomize_environment()
        self.steps = 0
        relative_distances = [(self.agent_positions[i] - self.goal_zones[i]) / 600 for i in range(self.num_agents)]
        relative_distances = np.array(relative_distances).flatten()
        obs = np.concatenate([self.agent_states.flatten(), relative_distances])
        padded_obs = np.zeros((MAX_AGENTS * 6,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, {}

    def step(self, actions):
        actions = np.array(actions).flatten()[:self.num_agents]
        rewards = np.zeros(self.num_agents)

        move_vectors = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])
        for i in range(self.num_agents):
            move_vec = move_vectors[actions[i]] * self.move_amount * self.agent_speeds[i]
            self.agent_positions[i] += move_vec

        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)

        new_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            distance = new_distances[i]
            rewards[i] += max(0, 100 - distance) / 100
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - distance) / 100)

        self.prev_distances = new_distances
        rewards = shape_multi_agent_rewards(rewards, self.agent_positions, self.goal_zones, self.num_agents)

        self.steps += 1
        terminated = self.steps >= self.max_steps

        noisy_states = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        relative_distances = [(self.agent_positions[i] - self.goal_zones[i]) / 600 for i in range(self.num_agents)]
        relative_distances = np.array(relative_distances).flatten()
        obs = np.concatenate([noisy_states.flatten(), relative_distances])
        padded_obs = np.zeros((MAX_AGENTS * 6,), dtype=np.float32)
        padded_obs[:len(obs)] = obs

        avg_goal_distance = np.mean(new_distances)
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {"avg_goal_distance": avg_goal_distance}

# Unseen Multi-Agent Environment
class UnseenMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents=2, seed=None):
        super().__init__(num_agents=num_agents, seed=seed)
        self.action_space = spaces.MultiDiscrete([ACTION_DIM] * num_agents)
        self.observation_noise_std = 0.5

    def randomize_environment(self):
        super().randomize_environment()
        self.move_amount = np.random.randint(10, 20)
# LightMetaPolicy (Unchanged)
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_channels):
        super().__init__()
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_transform = nn.Linear(self.agent_dim, 32)
        self.query_transform = nn.Linear(self.agent_dim, 32)
        self.value_transform = nn.Linear(self.agent_dim, 64)
        self.agent_relation = nn.Linear(32, 32)
        self.post_attention = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, output_channels))
        self.value_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.reshape(batch_size, MAX_AGENTS, self.agent_dim)
        keys = self.key_transform(agents); queries = self.query_transform(agents); values = self.value_transform(agents)
        attention = torch.bmm(queries, keys.transpose(1, 2)) / (32 ** 0.5)
        relation_queries = self.agent_relation(queries)
        relation_bias = torch.bmm(relation_queries, relation_queries.transpose(1, 2)) / (32 ** 0.5)
        attention = torch.softmax(attention + relation_bias, dim=-1)
        context = torch.bmm(attention, values)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.1).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        action_logits = self.post_attention(pooled).view(batch_size, MAX_AGENTS, ACTION_DIM)
        value = self.value_head(pooled)
        return action_logits, value

## UPDATED ## - Training function upgraded to PPO with tuned hyperparameters.
def train_policy(model, env_fn, meta_iterations=1000, rollout_len=256, gamma=0.99, lr=3e-4,
                 ppo_epochs=10, clip_epsilon=0.2, entropy_coef=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()

    for iteration in range(meta_iterations):
        if iteration < meta_iterations * 0.4: num_agents = np.random.choice([1, 2])
        elif iteration < meta_iterations * 0.7: num_agents = np.random.choice([3, 4])
        else: num_agents = 5

        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()

        # --- Data Collection ---
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_values, batch_dones = [], [], [], [], [], []
        steps_this_rollout = 0
        while steps_this_rollout < rollout_len:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_logits, value = model(obs_tensor)
            
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum()
            
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            done = terminated or truncated
            
            batch_obs.append(obs)
            batch_actions.append(actions.numpy().flatten())
            batch_log_probs.append(log_prob.item())
            batch_rewards.append(reward)
            batch_values.append(value.item())
            batch_dones.append(done)
            
            obs = next_obs
            steps_this_rollout += 1
            if done:
                obs, _ = env.reset()

        # --- GAE and Returns Calculation ---
        with torch.no_grad():
            _, last_value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        
        advantages = []
        gae = 0
        for i in reversed(range(rollout_len)):
            mask = 1.0 - batch_dones[i]
            next_val = batch_values[i+1] if i + 1 < rollout_len else last_value.item()
            delta = batch_rewards[i] + gamma * next_val * mask - batch_values[i]
            gae = delta + gamma * 0.95 * gae * mask
            advantages.insert(0, gae)
            
        returns = np.array(advantages) + np.array(batch_values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # --- PPO Update ---
        obs_t = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        actions_t = torch.tensor(np.array(batch_actions), dtype=torch.long)
        old_log_probs_t = torch.tensor(batch_log_probs, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        for _ in range(ppo_epochs):
            action_logits, values = model(obs_t)
            values = values.squeeze()
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions_t).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_t
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns_t)
            
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iteration % 20 == 0:
            print(f"LightMeta Iter {iteration} | Avg Reward (Rollout): {np.mean(batch_rewards):.2f} | Total Loss: {loss.item():.3f}")

    training_time = time.time() - start_time
    return model, training_time

## NEW ## - Proper PPO-based fine-tuning function.
def fine_tune_policy(model, env_fn, ft_steps=5000, lr=1e-4, gamma=0.99, ppo_epochs=4, clip_epsilon=0.2):
    print(f"Fine-tuning for {ft_steps} steps...")
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=lr)
    env = env_fn()
    obs, _ = env.reset()
    
    total_steps = 0
    while total_steps < ft_steps:
        # Collect a short rollout
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones, batch_values = [], [], [], [], [], []
        for _ in range(128): # Short rollout for fine-tuning
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_logits, value = tuned_model(obs_tensor)
            
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum()
            
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            done = terminated or truncated

            batch_obs.append(obs); batch_actions.append(actions.numpy().flatten()); batch_log_probs.append(log_prob.item())
            batch_rewards.append(reward); batch_dones.append(done); batch_values.append(value.item())
            
            obs = next_obs
            total_steps += 1
            if done: obs, _ = env.reset()

        # GAE and PPO Update (same logic as main training)
        with torch.no_grad(): _, last_value = tuned_model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        advantages = []; gae = 0
        for i in reversed(range(len(batch_rewards))):
            mask = 1.0 - batch_dones[i]
            next_val = batch_values[i+1] if i + 1 < len(batch_values) else last_value.item()
            delta = batch_rewards[i] + gamma * next_val * mask - batch_values[i]
            gae = delta + gamma * 0.95 * gae * mask
            advantages.insert(0, gae)
        returns = np.array(advantages) + np.array(batch_values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        obs_t = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        actions_t = torch.tensor(np.array(batch_actions), dtype=torch.long)
        old_log_probs_t = torch.tensor(batch_log_probs, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        
        for _ in range(ppo_epochs):
            action_logits, values = tuned_model(obs_t)
            values = values.squeeze()
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions_t).sum(dim=1)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns_t)
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tuned_model

# Evaluation Function
def evaluate_policy(model, env_fn, episodes=EPISODES):
    rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions = torch.argmax(action_logits, dim=-1).numpy()

            obs, reward, terminated, truncated, _ = env.step(actions.astype(int))
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards

# Environment Factories
def env_fn(num_agents=None, seed=None):
    if num_agents is None:
        num_agents = np.random.choice([1, 2, 3, 4, 5])
    return MultiAgentEnv(num_agents=num_agents, seed=seed)

def unseen_env_fn(seed=None):
    return UnseenMultiAgentEnv(num_agents=5, seed=seed)

# Main Test Function
def test_lightmeta():
    input_dim = MAX_AGENTS * 6
    output_channels = MAX_AGENTS * ACTION_DIM

    # Train main model
    model = LightMetaPolicy(input_dim, output_channels)
    params = sum(p.numel() for p in model.parameters())
    print("Training LightMetaPolicy with PPO...")
    trained_model, training_time = train_policy(model, env_fn, meta_iterations=500)
    print(f"Training completed in {training_time:.1f} seconds")

    agent_counts = [1, 2, 3, 4, 5]
    results = {"LightMeta": {}}

    print("\n=== MultiAgentEnv Evaluation ===")
    for num_agents in agent_counts:
        env_factory = lambda seed=None: MultiAgentEnv(num_agents=num_agents, seed=seed)
        print(f"\nEvaluating on {num_agents} agents:")

        zero_shot_mean, zero_shot_std, zero_shot_raw = evaluate_policy(trained_model, env_factory)
        adapted_model = fine_tune_policy(trained_model, lambda: MultiAgentEnv(num_agents=num_agents, seed=42))
        adapted_mean, adapted_std, _ = evaluate_policy(adapted_model, env_factory)

        print(f"LightMeta: {zero_shot_mean:.2f} ± {zero_shot_std:.2f} / Adapted: {adapted_mean:.2f}")
        results["LightMeta"][num_agents] = (zero_shot_mean, adapted_mean, zero_shot_std, zero_shot_raw)

    print("\n=== UnseenMultiAgentEnv Evaluation (5 agents) ===")
    zero_shot_u_mean, zero_shot_u_std, zero_shot_u_raw = evaluate_policy(trained_model, unseen_env_fn)
    adapted_u_model = fine_tune_policy(trained_model, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42))
    adapted_u_mean, adapted_u_std, _ = evaluate_policy(adapted_u_model, unseen_env_fn)
    print(f"LightMeta (Unseen): {zero_shot_u_mean:.2f} ± {zero_shot_u_std:.2f} / Adapted: {adapted_u_mean:.2f}")

    # Statistical Significance
    mappo_5_agents_median = 209.96
    stat, p = wilcoxon(np.array(results["LightMeta"][5][3]) - mappo_5_agents_median)
    print(f"\nLightMeta Zero-Shot vs. MAPPO median (5 agents): p-value = {p:.4f}")

    print("\n=== Compute Metrics ===")
    print(f"LightMeta: {params} params, {training_time:.1f}s")

    plt.figure(figsize=(12, 6))
    zero_shot = [v[0] for v in results["LightMeta"].values()]
    adapted = [v[1] for v in results["LightMeta"].values()]
    plt.plot(agent_counts, zero_shot, label=f"LightMeta Zero-Shot", linestyle='--', marker='o')
    plt.plot(agent_counts, adapted, label=f"LightMeta Adapted", marker='o')
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Reward")
    plt.title("LightMeta Performance Across Agent Counts (MultiAgentEnv)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_lightmeta()