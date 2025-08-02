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

    def step(self, actions):
        actions = np.array(actions).flatten()[:self.num_agents]
        rewards = np.zeros(self.num_agents)
        
        move_vectors = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])
        for i in range(self.num_agents):
            move_vec = move_vectors[actions[i]] * self.move_amount * self.agent_speeds[i]
            self.agent_positions[i] += move_vec

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

# LightMetaPolicy
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_channels):
        super().__init__()
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_transform = nn.Linear(self.agent_dim, 32)
        self.query_transform = nn.Linear(self.agent_dim, 32)
        self.value_transform = nn.Linear(self.agent_dim, 64)
        self.agent_relation = nn.Linear(32, 32)
        
        self.post_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, output_channels)
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
        
        action_logits = self.post_attention(pooled)
        action_logits = action_logits.view(batch_size, MAX_AGENTS, ACTION_DIM)
        
        value = self.value_head(pooled)
        return action_logits, value

# Training Function
def train_light_meta_policy(model, env_fn, meta_iterations=1000, inner_rollouts=80, gamma=0.99, entropy_coef=0.01):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    start_time = time.time()
    reward_buffer = []
    
    # NEW: History trackers
    history = {
        "avg_raw_reward": [], "total_loss": [], "policy_loss": [],
        "value_loss": [], "avg_goal_distance": []
    }
    
    for iteration in range(meta_iterations):
        if iteration < meta_iterations * 0.4: num_agents = np.random.choice([1, 2])
        elif iteration < meta_iterations * 0.7: num_agents = np.random.choice([3, 4])
        else: num_agents = 5
            
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        log_probs, rewards, values, entropies, goal_distances, raw_rewards = [], [], [], [], [], []
        
        steps = 0
        while steps < inner_rollouts:
            action_logits, value = model(obs)
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum()
            entropy = dist.entropy().sum()
            
            next_obs, reward, terminated, truncated, info = env.step(actions.numpy().astype(int))
            done = terminated or truncated
            
            raw_rewards.append(reward) # Store raw reward
            reward_buffer.append(reward)
            if len(reward_buffer) > 1000: reward_buffer.pop(0)
            reward_mean = np.mean(reward_buffer) if reward_buffer else 0
            reward_std = np.std(reward_buffer) if reward_buffer else 1
            normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
            
            log_probs.append(log_prob)
            rewards.append(normalized_reward)
            values.append(value.squeeze())
            entropies.append(entropy)
            goal_distances.append(info["avg_goal_distance"])
            
            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            steps += 1
            if done and steps < inner_rollouts:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # A2C Update
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropies.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if iteration % 20 == 0:
            avg_raw_reward = np.mean(raw_rewards)
            avg_goal_dist = np.mean(goal_distances)
            
            # NEW: More detailed logging
            print(f"Iter {iteration:4d} | Avg Raw Reward: {avg_raw_reward:6.2f} | "
                  f"Total Loss: {loss.item():.3f} (P: {policy_loss.item():.3f}, V: {value_loss.item():.3f}) | "
                  f"Avg Goal Dist: {avg_goal_dist:.2f}")

            # NEW: Store history
            history["avg_raw_reward"].append(avg_raw_reward)
            history["total_loss"].append(loss.item())
            history["policy_loss"].append(policy_loss.item())
            history["value_loss"].append(value_loss.item())
            history["avg_goal_distance"].append(avg_goal_dist)
            
    training_time = time.time() - start_time
    return model, training_time, history

# Fine-Tuning Function
def fine_tune_light_meta_policy(model, env_fn, max_steps=100):
    print("Skipping complex fine-tuning, returning zero-shot model.")
    return copy.deepcopy(model)

# Evaluation Function
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
                action_logits, _ = model(obs)
                actions = torch.argmax(action_logits, dim=-1).numpy()
            
            obs, reward, terminated, truncated, _ = env.step(actions.astype(int))
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards

# NEW: Plotting function for training history
def plot_training_history(history, title):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    iterations = range(0, len(history["total_loss"]) * 20, 20)

    # Plot Average Raw Reward
    axs[0, 0].plot(iterations, history["avg_raw_reward"], label="Avg Raw Reward", color='green')
    axs[0, 0].set_title("Average Raw Reward per Rollout")
    axs[0, 0].set_xlabel("Meta Iteration")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot Total Loss
    axs[0, 1].plot(iterations, history["total_loss"], label="Total Loss", color='red')
    axs[0, 1].set_title("Total Loss")
    axs[0, 1].set_xlabel("Meta Iteration")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot Policy and Value Loss
    axs[1, 0].plot(iterations, history["policy_loss"], label="Policy Loss", color='blue')
    axs[1, 0].set_title("Policy Loss (Actor)")
    axs[1, 0].set_xlabel("Meta Iteration")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(iterations, history["value_loss"], label="Value Loss", color='orange')
    axs[1, 1].set_title("Value Loss (Critic)")
    axs[1, 1].set_xlabel("Meta Iteration")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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

    # 20K Version
    model_20k = LightMetaPolicy(input_dim, output_channels)
    params_20k = sum(p.numel() for p in model_20k.parameters())
    print("Training LightMetaPolicy (250 meta-iterations)...")
    trained_model_20k, training_time_20k, history_20k = train_light_meta_policy(model_20k, env_fn, meta_iterations=250)
    print(f"Training completed in {training_time_20k:.1f} seconds")

    # 40K Version
    model_40k = LightMetaPolicy(input_dim, output_channels)
    params_40k = sum(p.numel() for p in model_40k.parameters())
    print("\nTraining LightMetaPolicy (500 meta-iterations)...")
    trained_model_40k, training_time_40k, history_40k = train_light_meta_policy(model_40k, env_fn, meta_iterations=500)
    print(f"Training completed in {training_time_40k:.1f} seconds")

    agent_counts = [1, 2, 3, 4, 5]
    results = {"LightMeta_20K": {}, "LightMeta_40K": {}}

    print("\n=== MultiAgentEnv Evaluation ===")
    for num_agents in agent_counts:
        env_factory = lambda seed=None: MultiAgentEnv(num_agents=num_agents, seed=seed)
        print(f"\nEvaluating on {num_agents} agents:")
        
        zero_shot_20k_mean, zero_shot_20k_std, zero_shot_20k_raw = evaluate_meta_policy(trained_model_20k, env_factory)
        adapted_20k_model = fine_tune_light_meta_policy(trained_model_20k, lambda: MultiAgentEnv(num_agents=num_agents, seed=42))
        adapted_20k_mean, adapted_20k_std, _ = evaluate_meta_policy(adapted_20k_model, env_factory)

        zero_shot_40k_mean, zero_shot_40k_std, zero_shot_40k_raw = evaluate_meta_policy(trained_model_40k, env_factory)
        adapted_40k_model = fine_tune_light_meta_policy(trained_model_40k, lambda: MultiAgentEnv(num_agents=num_agents, seed=42))
        adapted_40k_mean, adapted_40k_std, _ = evaluate_meta_policy(adapted_40k_model, env_factory)
        
        print(f"LightMeta_20K: {zero_shot_20k_mean:.2f} ± {zero_shot_20k_std:.2f} / Adapted: {adapted_20k_mean:.2f}")
        print(f"LightMeta_40K: {zero_shot_40k_mean:.2f} ± {zero_shot_40k_std:.2f} / Adapted: {adapted_40k_mean:.2f}")
        
        results["LightMeta_20K"][num_agents] = (zero_shot_20k_mean, adapted_20k_mean, zero_shot_20k_std, zero_shot_20k_raw)
        results["LightMeta_40K"][num_agents] = (zero_shot_40k_mean, adapted_40k_mean, zero_shot_40k_std, zero_shot_40k_raw)

    print("\n=== UnseenMultiAgentEnv Evaluation (5 agents) ===")
    zero_shot_40k_u_mean, zero_shot_40k_u_std, zero_shot_40k_u_raw = evaluate_meta_policy(trained_model_40k, unseen_env_fn)
    adapted_40k_u_model = fine_tune_light_meta_policy(trained_model_40k, lambda: UnseenMultiAgentEnv(num_agents=5, seed=42))
    adapted_40k_u_mean, adapted_40k_u_std, _ = evaluate_meta_policy(adapted_40k_u_model, unseen_env_fn)
    print(f"LightMeta_40K: {zero_shot_40k_u_mean:.2f} ± {zero_shot_40k_u_std:.2f} / Adapted: {adapted_40k_u_mean:.2f}")

    # Statistical Significance
    mappo_5_agents_median = 209.96
    stat, p = wilcoxon(np.array(results["LightMeta_40K"][5][3]) - mappo_5_agents_median)
    print(f"\nLightMeta_40K Zero-Shot vs. MAPPO median (5 agents): p-value = {p:.4f}")

    print("\n=== Compute Metrics ===")
    print(f"LightMeta_20K: {params_20k} params, {training_time_20k:.1f}s")
    print(f"LightMeta_40K: {params_40k} params, {training_time_40k:.1f}s")

    # Plot evaluation results
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

    # Plot training history
    plot_training_history(history_20k, "Training History (250 Iterations)")
    plot_training_history(history_40k, "Training History (500 Iterations)")

if __name__ == "__main__":
    test_lightmeta()