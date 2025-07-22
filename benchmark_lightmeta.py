import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import csv

# ==========================
#   GLOBAL CONSTANTS
# ==========================
MAX_AGENTS = 5
EPISODES = 20

# ==========================
#   SEEDING
# ==========================
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==========================
#   REWARD SHAPING
# ==========================
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


# ==========================
#   ENVIRONMENTS
# ==========================
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 6,), dtype=np.float32)
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
        self.prev_distances = np.array(
            [np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)]
        )

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


class UnseenMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents=2, seed=None):
        super().__init__(num_agents=num_agents, seed=seed)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)
        self.observation_noise_std = 0.5

    def randomize_environment(self):
        super().randomize_environment()
        self.move_range = np.random.randint(20, 40)


# ==========================
#   POLICIES
# ==========================
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
        action_probs = 0.5 * self.post_attention(pooled) + 0.25
        value = self.value_head(pooled)
        return action_probs, value


class SimpleMLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x), None


class RandomPolicy:
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        return torch.rand((batch_size, MAX_AGENTS)), None


class RuleBasedGreedyPolicy:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        actions = torch.zeros((batch_size, MAX_AGENTS))
        for b in range(batch_size):
            rel_dists = x[b, -MAX_AGENTS*2:].reshape(MAX_AGENTS, 2)
            distances = torch.norm(rel_dists, dim=1)
            actions[b] = (distances > self.threshold).float()
        return actions, None


# ==========================
#   EVALUATION
# ==========================
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
                action_probs, _ = model.forward(obs)
                actions = torch.bernoulli(action_probs).numpy()
            obs, reward, terminated, truncated, _ = env.step(np.array(actions).astype(int))
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards


# ==========================
#   MAIN TEST
# ==========================
def test_lightmeta():
    input_dim = MAX_AGENTS * 6
    output_dim = MAX_AGENTS

    models = {
        "LightMetaPolicy": LightMetaPolicy(input_dim, output_dim),
        "SimpleMLPPolicy": SimpleMLPPolicy(input_dim, output_dim),
        "RandomPolicy": RandomPolicy(),
        "RuleBasedGreedyPolicy": RuleBasedGreedyPolicy()
    }

    agent_counts = [1, 2, 3, 4, 5]
    results = []

    print("\n=== Model Evaluation Across Agent Counts ===")
    for num_agents in agent_counts:
        env_factory = lambda seed=None: MultiAgentEnv(num_agents=num_agents, seed=seed)
        for model_name, model in models.items():
            mean_reward, std_reward, _ = evaluate_meta_policy(model, env_factory)
            results.append({
                "Agents": num_agents,
                "Model": model_name,
                "Avg Reward": mean_reward,
                "Std Reward": std_reward
            })
            print(f"{model_name} | Agents {num_agents} | Avg: {mean_reward:.2f} ± {std_reward:.2f}")

    final_rows = []
    overall_efficiency = {}
    for num_agents in agent_counts:
        subset = [r for r in results if r["Agents"] == num_agents]
        max_reward = max(r["Avg Reward"] for r in subset)
        for r in subset:
            efficiency = (r["Avg Reward"] / max_reward) * 100 if max_reward > 0 else 0
            final_rows.append({**r, "Efficiency (%)": efficiency})
            overall_efficiency[r["Model"]] = overall_efficiency.get(r["Model"], 0) + efficiency

    for model in overall_efficiency:
        overall_efficiency[model] /= len(agent_counts)
        print(f"Overall Efficiency - {model}: {overall_efficiency[model]:.2f}%")

    csv_file = "model_comparison_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Agents", "Model", "Avg Reward", "Std Reward", "Efficiency (%)"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nResults saved to {csv_file}")
    return final_rows, overall_efficiency


# ==========================
#   ENTRYPOINT
# ==========================
if __name__ == "__main__":
    table, overall_efficiency = test_lightmeta()
    print("\n=== Final Comparison Table ===")
    for row in table:
        print(row)

    # Line plot of efficiencies
    plt.figure(figsize=(10, 6))
    for model_name in set(row["Model"] for row in table):
        subset = [row for row in table if row["Model"] == model_name]
        plt.plot([row["Agents"] for row in subset], [row["Efficiency (%)"] for row in subset],
                 label=model_name, marker='o')
    plt.title("Model Efficiency Comparison")
    plt.xlabel("Number of Agents")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.show()

    # Bar chart of average rewards
    plt.figure(figsize=(10, 6))
    models = sorted(set(row["Model"] for row in table))
    for i, model_name in enumerate(models):
        subset = [row for row in table if row["Model"] == model_name]
        plt.bar([x + i * 0.2 for x in range(len(subset))],
                [row["Avg Reward"] for row in subset], width=0.2, label=model_name)
    plt.xticks([x + 0.3 for x in range(len(subset))], [row["Agents"] for row in subset])
    plt.title("Average Reward per Model per Agent Count")
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
