import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_spread_v3
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3 import PPO as SB3PPO

# ------------- Suppress Warnings, Set Seeds ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
MAX_AGENTS = 5
EPISODES = 5
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ------------- Custom MultiAgentEnv + Policy ----------
def shape_multi_agent_rewards(rewards, agent_positions, goal_zones, num_agents):
    if num_agents <= 1:
        return rewards
    shaped = rewards.copy()
    centroid = np.mean(agent_positions, axis=0)
    avg_goal = np.mean(goal_zones, axis=0)
    centroid_goal_distance = np.linalg.norm(centroid - avg_goal)
    agent_centroid_dists = [np.linalg.norm(pos - centroid) for pos in agent_positions]
    cohesion_factor = min(1.0, centroid_goal_distance / 200.0)
    for i in range(num_agents):
        cohesion_bonus = cohesion_factor * (1.0 - agent_centroid_dists[i] / 100.0)
        exploration_factor = 1.0 - cohesion_factor
        exploration_bonus = exploration_factor * (agent_centroid_dists[i] / 100.0) * 0.5
        shaped_component = max(-0.5, min(0.5, cohesion_bonus + exploration_bonus))
        shaped[i] += shaped_component
    return shaped

class MultiAgentEnv:
    def __init__(self, num_agents=MAX_AGENTS, seed=None):
        self.num_agents = num_agents
        self.max_steps = 200
        self.steps = 0
        self.seed = seed
        self.randomize_environment()
    def randomize_environment(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.move_range = np.random.randint(3, 20)
        self.start_bounds = np.random.randint(20, 150)
        self.agent_states = np.random.uniform(-1, 1, (self.num_agents, 4))
        self.agent_positions = np.random.randint(self.start_bounds, 600-self.start_bounds, (self.num_agents, 2))
        self.agent_speeds = np.random.uniform(0.5, 2.0, self.num_agents)
        self.goal_zones = np.random.randint(50, 550, (self.num_agents, 2))
        self.observation_noise_std = np.random.uniform(0, 0.2)
        self.prev_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
    def reset(self, seed=None):
        self.randomize_environment()
        self.steps = 0
        rel_dist = [(self.agent_positions[i] - self.goal_zones[i]) / 600 for i in range(self.num_agents)]
        rel_dist = np.array(rel_dist).flatten()
        obs = np.concatenate([self.agent_states.flatten(), rel_dist])
        padded_obs = np.zeros(MAX_AGENTS * 6, dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, {}
    def step(self, actions):
        actions = np.array(actions).flatten()[:self.num_agents]
        rewards = np.zeros(self.num_agents)
        move_mask = (actions == 1)
        rewards[move_mask] = 1.0
        if np.sum(move_mask) > 0:
            move_amt = np.random.randint(-self.move_range, self.move_range, (np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amt[i] = (move_amt[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amt
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        new_dist = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            dist = new_dist[i]
            rewards[i] += max(0, 100 - dist) / 100
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - dist) / 100)
        self.prev_distances = new_dist
        rewards = shape_multi_agent_rewards(rewards, self.agent_positions, self.goal_zones, self.num_agents)
        self.steps += 1
        terminated = self.steps >= self.max_steps
        noisy_states = self.agent_states + np.random.normal(0, self.observation_noise_std, self.agent_states.shape)
        rel_dist = [(self.agent_positions[i] - self.goal_zones[i]) / 600 for i in range(self.num_agents)]
        rel_dist = np.array(rel_dist).flatten()
        obs = np.concatenate([noisy_states.flatten(), rel_dist])
        padded_obs = np.zeros(MAX_AGENTS * 6, dtype=np.float32)
        padded_obs[:len(obs)] = obs
        avg_goal_distance = np.mean(new_dist)
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {"avg_goal_distance": avg_goal_distance}

class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_transform = nn.Linear(self.agent_dim, 32)
        self.query_transform = nn.Linear(self.agent_dim, 32)
        self.value_transform = nn.Linear(self.agent_dim, 64)
        self.agent_relation = nn.Linear(32, 32)
        self.post_attention = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, output_dim), nn.Sigmoid())
        self.value_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
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

def train_light_meta_policy(model, env_fn, meta_iterations=150, inner_rollouts=60, gamma=0.99):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for iteration in range(meta_iterations):
        num_agents = MAX_AGENTS
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        log_probs, rewards, values = [], [], []
        steps = 0
        while steps < inner_rollouts:
            action_probs, value = model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).mean()
            next_obs, reward, terminated, truncated, info = env.step(actions.numpy().astype(int))
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze())
            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            steps += 1
            if (terminated or truncated) and steps < inner_rollouts:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, None

def evaluate_meta_policy(model, num_agents=MAX_AGENTS, episodes=EPISODES):
    rewards = []
    for ep in range(episodes):
        env = MultiAgentEnv(num_agents=num_agents, seed=ep)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action_probs, _ = model(obs)
                actions = torch.bernoulli(action_probs).numpy()
            obs, reward, terminated, truncated, _ = env.step(np.array(actions).astype(int))
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards

# --- PettingZoo Adapter for LightMetaPolicy ---
def flatten_obs_from_dict(obs_dict, all_agents, obs_size):
    flat = np.zeros((MAX_AGENTS, obs_size), dtype=np.float32)
    for i, agent in enumerate(all_agents):
        if agent in obs_dict:
            flat[i, :len(obs_dict[agent])] = obs_dict[agent]
    return flat.flatten()

def evaluate_lightmeta_on_pettingzoo(model, env, episodes=5, fine_tune=False, ft_iters=2000):
    all_agents = [f"agent_{i}" for i in range(MAX_AGENTS)]
    obs_size = env.observation_space(all_agents[0]).shape[0]
    model_to_use = model
    optimizer = torch.optim.Adam(model_to_use.parameters(), lr=2e-4)
    device = next(model_to_use.parameters()).device
    # Fine-tune (optional)
    if fine_tune:
        print(f"Fine-tuning LightMetaPolicy on PettingZoo env for {ft_iters} steps...")
        for ft_step in range(ft_iters):
            obs_dict, _ = env.reset()
            term = {a: False for a in env.agents}
            trun = {a: False for a in env.agents}
            while not (all(term.values()) or all(trun.values())):
                obs_flat = flatten_obs_from_dict(obs_dict, all_agents, obs_size)
                obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
                act_probs, _ = model_to_use(obs_tensor)
                actions = torch.bernoulli(act_probs).squeeze().detach().cpu().numpy().astype(int)
                action_dict = {agent: actions[i] for i, agent in enumerate(all_agents) if agent in obs_dict}
                obs_next_dict, rewards, term, trun, _ = env.step(action_dict)
                reward = sum(rewards.values())
                loss = -torch.log(torch.clamp(act_probs, 1e-4,1-1e-4)).mean() * float(reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                obs_dict = obs_next_dict
    rewards_list = []
    for ep in range(episodes):
        obs_dict, _ = env.reset()
        total_reward = 0.0
        term = {a: False for a in env.agents}
        trun = {a: False for a in env.agents}
        while not (all(term.values()) or all(trun.values())):
            obs_flat = flatten_obs_from_dict(obs_dict, all_agents, obs_size)
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act_probs, _ = model_to_use(obs_tensor)
                actions = torch.bernoulli(act_probs).squeeze().detach().cpu().numpy().astype(int)
            action_dict = {agent: actions[i] for i, agent in enumerate(all_agents) if agent in obs_dict}
            obs_dict, rewards, term, trun, _ = env.step(action_dict)
            total_reward += sum(rewards.values())
        rewards_list.append(total_reward)
    avg, std = np.mean(rewards_list), np.std(rewards_list)
    print(f"LightMetaPolicy on PettingZoo (fine_tuned={fine_tune}): {avg:.2f} ± {std:.2f}")
    return avg, std, rewards_list

# --------------- RLlib + SB3 Benchmarks -------------------
def register_pettingzoo_env():
    tune.register_env(
        "simple_spread",
        lambda config: PettingZooEnv(
            simple_spread_v3.env(
                N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False
            )
        ),
    )
def get_common_config():
    return {
        "num_workers": 0, "num_gpus": 0,
        "rollout_fragment_length": 100,
        "train_batch_size": 5000, "log_level": "ERROR",
        "framework": "torch",
    }
def train_rllib_ppo():
    print("\n=== Training MAPPO (Ray RLlib PPO) on simple_spread ===")
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_pettingzoo_env()
    config = (
        PPOConfig()
        .environment(env="simple_spread")
        .update_from_dict(get_common_config())
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    algo = config.build()
    for i in range(60):
        result = algo.train()
        if (i+1) % 10 == 0 or i == 0:
            reward = result.get("episode_reward_mean", float("nan"))
            print(f"PPO Iter {i+1}: avg reward = {reward:.2f}")
    env = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    total_reward = 0
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}
        while not all(done.values()):
            actions = {agent: algo.compute_single_action(obs[agent]) for agent in env.agents}
            obs, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(rewards.values())
            done = {a: terminations[a] or truncations[a] for a in env.agents}
    ray.shutdown()
    avg_reward = total_reward / EPISODES
    print(f"MAPPO Eval Avg Reward: {avg_reward:.2f}")
    return avg_reward
def train_rllib_appo():
    print("\n=== Training APPO (Ray RLlib) on simple_spread ===")
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_pettingzoo_env()
    config = (
        APPOConfig()
        .environment(env="simple_spread")
        .update_from_dict(get_common_config())
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    algo = config.build()
    for i in range(60):
        result = algo.train()
        if (i+1) % 10 == 0 or i == 0:
            reward = result.get("episode_reward_mean", float("nan"))
            print(f"APPO Iter {i+1}: avg reward = {reward:.2f}")
    env = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    total_reward = 0
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}
        while not all(done.values()):
            actions = {agent: algo.compute_single_action(obs[agent]) for agent in env.agents}
            obs, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(rewards.values())
            done = {a: terminations[a] or truncations[a] for a in env.agents}
    ray.shutdown()
    avg_reward = total_reward / EPISODES
    print(f"APPO Eval Avg Reward: {avg_reward:.2f}")
    return avg_reward

def train_ippo():
    print("\n=== Training IPPO (SB3 PPO) on simple_spread ===")
    env = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    model = SB3PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=50000)
    eval_env = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    total_reward = 0
    for _ in range(EPISODES):
        obs, _ = eval_env.reset()
        terminations = {agent: False for agent in eval_env.agents}
        truncations = {agent: False for agent in eval_env.agents}
        while not (all(terminations.values()) or all(truncations.values())):
            actions = {agent: model.predict(obs[agent], deterministic=True)[0] for agent in eval_env.agents}
            obs, rewards, terminations, truncations, _ = eval_env.step(actions)
            total_reward += sum(rewards.values())
    avg_reward = total_reward / EPISODES
    print(f"IPPO Eval Avg Reward: {avg_reward:.2f}")
    return avg_reward

#########################
# COMPARISON MAIN
#########################
if __name__ == "__main__":
    print("\n==== Training LightMetaPolicy (custom env obs) ====")
    input_dim_custom = MAX_AGENTS * 6      # custom env: per-agent obs = 6
    output_dim = MAX_AGENTS
    model_custom = LightMetaPolicy(input_dim_custom, output_dim)
    trained_model, _ = train_light_meta_policy(
        model_custom,
        lambda num_agents=None, seed=None: MultiAgentEnv(num_agents=num_agents, seed=seed),
        meta_iterations=150  # slight boost, still short
    )
    custom_mean, custom_std, _ = evaluate_meta_policy(trained_model, num_agents=MAX_AGENTS, episodes=EPISODES)
    print(f"LightMetaPolicy on custom env (5 agents): {custom_mean:.2f} ± {custom_std:.2f}")

    # NEW: Evaluate/fine-tune LightMetaPolicy on PettingZoo obs shape
    env_pz = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    obs_dim_pz = env_pz.observation_space("agent_0").shape[0]
    input_dim_pz = MAX_AGENTS * obs_dim_pz
    model_pz = LightMetaPolicy(input_dim_pz, output_dim)
    print("\n==== LightMetaPolicy (PettingZoo obs format, Zero-Shot) ====")
    zs_mean, zs_std, _ = evaluate_lightmeta_on_pettingzoo(model_pz, env_pz, episodes=EPISODES, fine_tune=False)
    print("\n==== LightMetaPolicy (PettingZoo obs format, Fine-Tuned) ====")
    env_pz_ft = simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5, continuous_actions=False)
    ft_mean, ft_std, _ = evaluate_lightmeta_on_pettingzoo(model_pz, env_pz_ft, episodes=EPISODES, fine_tune=True, ft_iters=2000)

    mappo_score = train_rllib_ppo()
    appo_score = train_rllib_appo()
    ippo_score = train_ippo()

    models = [
        ("LightMetaPolicy (custom env)", custom_mean, custom_std),
        ("LightMetaPolicy (PettingZoo ZS)", zs_mean, zs_std),
        ("LightMetaPolicy (PettingZoo FT)", ft_mean, ft_std),
        ("MAPPO (RLlib)", mappo_score, 0),
        ("APPO (RLlib)", appo_score, 0),
        ("IPPO (SB3 PPO)", ippo_score, 0)
    ]
    max_score = max(x[1] for x in models)
    print("\nComparison Table:")
    print(f"{'Model':38s} {'AvgRwd':>9} {'Std':>8} {'Eff(%)':>7}")
    for name, avg, std in models:
        eff = 100 * avg / max_score if max_score != 0 else 0
        print(f"{name:38s} {avg:9.2f} {std:8.2f} {eff:7.2f}")

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.bar([x[0] for x in models], [x[1] for x in models], color='slateblue')
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Average Reward (5 agents)')
    plt.title('Multi-Agent Benchmark Comparison')
    plt.tight_layout()
    plt.show()
