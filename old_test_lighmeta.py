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

# Multi-Agent Environment (reverted with tweaks)
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 6,), dtype=np.float32)  # agent_states + relative distances
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
        move_mask = (actions == 1)
        rewards[move_mask] = 1.0
        if np.sum(move_mask) > 0:
            move_amounts = np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amounts[i] = (move_amounts[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amounts
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        new_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            distance = new_distances[i]
            rewards[i] += max(0, 100 - distance) / 100
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - distance) / 100)  # Distance reduction bonus
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

# Unseen Multi-Agent Environment (reverted with tweaks)
class UnseenMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents=2, seed=None):
        super().__init__(num_agents=num_agents, seed=seed)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)
        self.observation_noise_std = 0.5

    def randomize_environment(self):
        super().randomize_environment()
        self.move_range = np.random.randint(20, 40)

    def step(self, actions):
        actions = np.array(actions).flatten()[:self.num_agents]
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
        new_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            dist_to_goal = new_distances[i]
            dist_to_avg = np.linalg.norm(self.agent_positions[i] - avg_pos)
            rewards[i] += max(0, 100 - dist_to_goal) / 100 + 0.5 * max(0, 50 - dist_to_avg) / 50
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - dist_to_goal) / 100)
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

# LightMetaPolicy (tweaked)
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_transform = nn.Linear(self.agent_dim, 32)
        self.query_transform = nn.Linear(self.agent_dim, 32)
        self.value_transform = nn.Linear(self.agent_dim, 64)
        self.agent_relation = nn.Linear(32, 32)
        self.post_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.reshape(batch_size, MAX_AGENTS, self.agent_dim)
        keys = self.key_transform(agents)
        queries = self.query_transform(agents)
        values = self.value_transform(agents)
        attention = torch.bmm(queries, keys.transpose(1, 2)) / (32 ** 0.5)
        relation_queries = self.agent_relation(queries)
        relation_bias = torch.bmm(relation_queries, relation_queries.transpose(1, 2)) / (32 ** 0.5)
        attention = attention + relation_bias
        attention = torch.softmax(attention, dim=-1)
        context = torch.bmm(attention, values)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.1).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        action_probs = 0.5 * self.post_attention(pooled) + 0.25  # Scale to ~0.25–0.75
        value = self.value_head(pooled)
        return action_probs, value

# Training Function (tweaked)
def train_light_meta_policy(model, env_fn, meta_iterations=1000, inner_rollouts=80, gamma=0.99, entropy_coef=0.3):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    reward_buffer = []
    for iteration in range(meta_iterations):
        if iteration < meta_iterations * 0.4:
            num_agents = np.random.choice([1, 2])
        elif iteration < meta_iterations * 0.7:
            num_agents = np.random.choice([3, 4])
        else:
            num_agents = 5
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        log_probs, rewards, values, goal_distances, action_probs_list, action_counts = [], [], [], [], [], []
        steps = 0
        while steps < inner_rollouts:
            action_probs, value = model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).mean()
            next_obs, reward, terminated, truncated, info = env.step(actions.numpy().astype(int))
            done = terminated or truncated
            reward_buffer.append(reward)
            if len(reward_buffer) > 1000:
                reward_buffer.pop(0)
            reward_mean = np.mean(reward_buffer) if reward_buffer else 0
            reward_std = np.std(reward_buffer) if reward_buffer else 1
            normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
            log_probs.append(log_prob)
            rewards.append(normalized_reward)
            values.append(value.squeeze())
            goal_distances.append(info["avg_goal_distance"])
            action_probs_list.append(action_probs.mean().item())
            action_counts.append(actions.mean().item())
            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            steps += 1
            if done and steps < inner_rollouts:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        returns = []
        discounted_reward = 0
        for r in reversed(rewards[:inner_rollouts]):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = returns + [0] * (inner_rollouts - len(returns))
        values = values[:inner_rollouts] + [torch.tensor(0.0)] * (inner_rollouts - len(values))
        log_probs = log_probs[:inner_rollouts] + [torch.tensor(0.0)] * (inner_rollouts - len(log_probs))
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if iteration == 0:
            print(f"Debug: values shape: {values.shape}, returns shape: {returns.shape}")
            print(f"Debug: First value: {values[0]}, First return: {returns[0]}")
            print(f"Debug: Episode length: {len(rewards)}, Total raw reward: {sum(rewards):.2f}")
            print(f"Debug: Avg goal distance: {np.mean(goal_distances):.2f}")
            print(f"Debug: Avg action prob: {np.mean(action_probs_list):.2f}")
            print(f"Debug: Avg action count: {np.mean(action_counts):.2f}")
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if iteration % 20 == 0:
            print(f"LightMeta Iter {iteration} | Avg Return: {returns.mean().item():.2f} | Episode Length: {len(rewards)} | Avg Goal Distance: {np.mean(goal_distances):.2f} | Avg Action Prob: {np.mean(action_probs_list):.2f} | Avg Action Count: {np.mean(action_counts):.2f}")
    training_time = time.time() - start_time
    return model, training_time

# Fine-Tuning Function (tweaked)
def fine_tune_light_meta_policy(model, env_fn, max_steps=100):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0001)
    env = env_fn()
    experiences = []
    reward_buffer = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < 50:
            with torch.no_grad():
                action_probs, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions = torch.bernoulli(action_probs).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(actions.astype(int))
            reward = np.clip(reward, -5, 5)  # Clip rewards
            reward_buffer.append(reward)
            reward_mean = np.mean(reward_buffer) if reward_buffer else 0
            reward_std = np.std(reward_buffer) if reward_buffer else 1
            normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
            experiences.append((obs, actions.flatten(), normalized_reward, next_obs))
            obs = next_obs
            done = terminated or truncated
            episode_steps += 1
    for step in range(max_steps):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.float32)
            action_probs, _ = tuned_model(batch_obs)
            imitation_loss = nn.BCELoss()(action_probs, batch_actions)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            rl_loss = -torch.log(action_probs + 1e-10) * batch_actions * batch_rewards.unsqueeze(1)
            loss = (1 - step/max_steps) * imitation_loss + (step/max_steps) * rl_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuned_model.parameters(), max_norm=0.5)
            optimizer.step()
    return tuned_model

# Evaluation Function (unchanged)
def evaluate_meta_policy(model, env_fn, episodes=EPISODES):
    rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                output = model(obs)
                action_probs, _ = output if isinstance(output, tuple) else (output, None)
                actions = torch.bernoulli(action_probs).numpy()
            obs, reward, terminated, truncated, _ = env.step(np.array(actions).astype(int))
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
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

# Main Test Function (tweaked)
def test_lightmeta():
    input_dim = MAX_AGENTS * 6  # agent_states + relative distances
    output_dim = MAX_AGENTS

    # 20K Version
    model_20k = LightMetaPolicy(input_dim, output_dim)
    params_20k = sum(p.numel() for p in model_20k.parameters())
    print("Training LightMetaPolicy (20K steps)...")
    trained_model_20k, training_time_20k = train_light_meta_policy(model_20k, env_fn, meta_iterations=250)
    print(f"Training completed in {training_time_20k:.1f} seconds")

    # 40K Version
    model_40k = LightMetaPolicy(input_dim, output_dim)
    params_40k = sum(p.numel() for p in model_40k.parameters())
    print("\nTraining LightMetaPolicy (40K steps)...")
    trained_model_40k, training_time_40k = train_light_meta_policy(model_40k, env_fn, meta_iterations=500)
    print(f"Training completed in {training_time_40k:.1f} seconds")

    agent_counts = [1, 2, 3, 4, 5]
    results = {"LightMeta_20K": {}, "LightMeta_40K": {}}

    print("\n=== MultiAgentEnv Evaluation ===")
    for num_agents in agent_counts:
        env_factory = lambda seed=None: MultiAgentEnv(num_agents=num_agents, seed=seed)
        print(f"\nEvaluating on {num_agents} agents:")
        zero_shot_20k_mean, zero_shot_20k_std, zero_shot_20k_raw = evaluate_meta_policy(trained_model_20k, env_factory)
        adapted_20k_mean, adapted_20k_std, adapted_20k_raw = evaluate_meta_policy(
            fine_tune_light_meta_policy(trained_model_20k, lambda: MultiAgentEnv(num_agents=num_agents, seed=42)),
            env_factory
        )
        zero_shot_40k_mean, zero_shot_40k_std, zero_shot_40k_raw = evaluate_meta_policy(trained_model_40k, env_factory)
        adapted_40k_mean, adapted_40k_std, adapted_40k_raw = evaluate_meta_policy(
            fine_tune_light_meta_policy(trained_model_40k, lambda: MultiAgentEnv(num_agents=num_agents, seed=42)),
            env_factory
        )
        print(f"LightMeta_20K: {zero_shot_20k_mean:.2f} ± {zero_shot_20k_std:.2f} / {adapted_20k_mean:.2f}")
        print(f"LightMeta_40K: {zero_shot_40k_mean:.2f} ± {zero_shot_40k_std:.2f} / {adapted_40k_mean:.2f}")
        results["LightMeta_20K"][num_agents] = (zero_shot_20k_mean, adapted_20k_mean, zero_shot_20k_std, zero_shot_20k_raw)
        results["LightMeta_40K"][num_agents] = (zero_shot_40k_mean, adapted_40k_mean, zero_shot_40k_std, zero_shot_40k_raw)

    print("\n=== UnseenMultiAgentEnv Evaluation (5 agents) ===")
    zero_shot_20k_u_mean, zero_shot_20k_u_std, zero_shot_20k_u_raw = evaluate_meta_policy(trained_model_20k, unseen_env_fn)
    adapted_20k_u_mean, adapted_20k_u_std, adapted_20k_u_raw = evaluate_meta_policy(
        fine_tune_light_meta_policy(trained_model_20k, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42)),
        unseen_env_fn
    )
    zero_shot_40k_u_mean, zero_shot_40k_u_std, zero_shot_40k_u_raw = evaluate_meta_policy(trained_model_40k, unseen_env_fn)
    adapted_40k_u_mean, adapted_40k_u_std, adapted_40k_u_raw = evaluate_meta_policy(
        fine_tune_light_meta_policy(trained_model_40k, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42)),
        unseen_env_fn
    )
    print(f"LightMeta_20K: {zero_shot_20k_u_mean:.2f} ± {zero_shot_20k_u_std:.2f} / {adapted_20k_u_mean:.2f}")
    print(f"LightMeta_40K: {zero_shot_40k_u_mean:.2f} ± {zero_shot_40k_u_std:.2f} / {adapted_40k_u_mean:.2f}")

    # Statistical Significance
    mappo_5_agents = [209.96] * EPISODES
    stat, p = wilcoxon(zero_shot_40k_raw, mappo_5_agents)
    print(f"\nLightMeta_40K Zero-Shot vs. MAPPO (5 agents): p-value = {p:.4f}")

    print("\n=== Compute Metrics ===")
    print(f"LightMeta_20K: {params_20k} params, {training_time_20k:.1f}s")
    print(f"LightMeta_40K: {params_40k} params, {training_time_40k:.1f}s")

    plt.figure(figsize=(12, 6))
    for model_name, data in results.items():
        zero_shot = [v[0] for v in data.values()]
        adapted = [v[1] for v in data.values()]
        plt.plot(agent_counts, zero_shot, label=f"{model_name} Zero-Shot", linestyle='--')
        plt.plot(agent_counts, adapted, label=f"{model_name} Adapted")
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Reward")
    plt.title("LightMeta Performance Across Agent Counts (MultiAgentEnv)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_lightmeta()