import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import time
from scipy.stats import wilcoxon

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
MAX_AGENTS = 5
USE_SIMPLE_SPREAD = False
TRAIN_ALL_AGENTS = True
EPISODES = 20

# --- Environments ---
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
        for i in range(self.num_agents):
            distance = np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])
            rewards[i] += max(0, 100 - distance) / 100
        self.steps += 1
        terminated = self.steps >= self.max_steps
        noisy_obs = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

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
        for i in range(self.num_agents):
            dist_to_goal = np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])
            dist_to_avg = np.linalg.norm(self.agent_positions[i] - avg_pos)
            rewards[i] += max(0, 100 - dist_to_goal) / 100 + 0.5 * max(0, 50 - dist_to_avg) / 50
        self.steps += 1
        terminated = self.steps >= self.max_steps
        noisy_obs = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class SimpleSpreadWrapper(gym.Env):
    def __init__(self, num_agents=5, seed=None):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=200, continuous_actions=False)
        self.num_agents = num_agents
        self.seed = seed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_AGENTS * 22,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5] * MAX_AGENTS)
        self.reset()

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else self.seed
        obs_dict, infos = self.env.reset(seed=effective_seed)
        self.agents = list(obs_dict.keys())
        return self._flatten_obs(obs_dict), infos

    def step(self, actions):
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        next_obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos = self.env.step(action_dict)
        self.agents = list(next_obs_dict.keys())
        reward = sum(rewards_dict.values()) / len(rewards_dict) if rewards_dict else 0
        terminated = any(terminateds_dict.values())
        truncated = any(truncateds_dict.values())
        return self._flatten_obs(next_obs_dict), reward, terminated, truncated, infos

    def _flatten_obs(self, obs_dict):
        padded_obs = np.zeros((MAX_AGENTS * 22,), dtype=np.float32)
        for i, agent in enumerate([f'agent_{j}' for j in range(MAX_AGENTS)]):
            obs = obs_dict.get(agent, np.zeros(22))
            start_idx = i * 22
            padded_obs[start_idx:start_idx + 22] = obs
        return padded_obs

# --- Models ---
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
        action_probs = self.post_attention(pooled)
        value = self.value_head(pooled)
        return action_probs, value

class MAPPOPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim=1):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim * MAX_AGENTS, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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

# --- Training Functions ---
def train_light_meta_policy(model, env_fn, meta_iterations=1000, inner_rollouts=40, gamma=0.99, entropy_coef=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([1, 2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        log_probs, rewards, values = [], [], []
        for _ in range(inner_rollouts):
            action_probs, value = model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).mean()
            next_obs, reward, terminated, truncated, _ = env.step(actions.numpy().astype(int))
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if done:
                break
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze(-1)
        advantages = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if iteration % 100 == 0:
            print(f"LightMeta Iter {iteration} | Avg Return: {returns.mean().item():.2f}")
    training_time = time.time() - start_time
    return model, training_time

def train_mappo_policy(env_fn, iterations=2500, rollout_steps=200, gamma=0.99):
    obs_dim = 4 if not USE_SIMPLE_SPREAD else 22
    action_dim = 1 if not USE_SIMPLE_SPREAD else 5
    policy = MAPPOPolicy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(iterations):
        num_agents = np.random.choice([1, 2, 3, 4, 5]) if TRAIN_ALL_AGENTS else 2
        env = env_fn(num_agents=num_agents, seed=42 + iteration)
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
            print(f"MAPPO Iter {iteration} | Avg Return: {returns.mean().item():.2f}")
    training_time = time.time() - start_time
    return policy, training_time

def train_ippo_policy(env_fn, timesteps=100000):
    policies = {}
    start_time = time.time()
    if TRAIN_ALL_AGENTS:
        for num_agents in [1, 2, 3, 4, 5]:
            env = env_fn(num_agents, seed=42)
            policies[num_agents] = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
            policies[num_agents].learn(total_timesteps=timesteps // 5)
    else:
        env = env_fn(2, seed=42)
        policies[2] = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
        policies[2].learn(total_timesteps=timesteps)
    training_time = time.time() - start_time
    return policies, training_time

def meta_train_maml(env_fn, meta_iterations=750, inner_steps=10, inner_rollouts=50, gamma=0.99, inner_lr=0.05):
    input_dim = MAX_AGENTS * (22 if USE_SIMPLE_SPREAD else 4)
    output_dim = MAX_AGENTS * (5 if USE_SIMPLE_SPREAD else 1)
    meta_model = MAMLPolicy(input_dim, output_dim)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([1, 2, 3, 4, 5])
        env = env_fn(num_agents)
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
        if iteration % 100 == 0:
            print(f"MAML Iter {iteration} | Avg Return: {returns.mean().item():.2f}")
    training_time = time.time() - start_time
    return meta_model, training_time

# --- Fine-Tuning and Evaluation ---
def fine_tune_light_meta_policy(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0001)
    env = env_fn()
    num_agents = env.num_agents
    experiences = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < 50:
            with torch.no_grad():
                action_probs, _ = model(torch.tensor(obs, dtype=torch.float32))
                actions = torch.bernoulli(action_probs).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(actions.astype(int))
            experiences.append((obs, actions.flatten(), reward, next_obs))
            obs = next_obs
            done = terminated or truncated
            episode_steps += 1
    for step in range(max_steps):
        if len(experiences) > 32:
            batch = random.sample(experiences, 32)
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

def fine_tune_maml_policy(meta_model, env_fn, steps=40):
    env = env_fn()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    input_dim = MAX_AGENTS * (22 if USE_SIMPLE_SPREAD else 4)
    output_dim = MAX_AGENTS * (5 if USE_SIMPLE_SPREAD else 1)
    adapted_model = MAMLPolicy(input_dim, output_dim)
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

def evaluate_policy(model, env_fn, episodes=EPISODES, is_mappo=False, is_ippo=False, ippo_policies=None):
    rewards = []
    for ep in range(episodes):
        env = env_fn(seed=42 + ep)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                if is_mappo:
                    actions = model.get_action(obs, env.num_agents, deterministic=True)
                elif is_ippo:
                    num_agents = env.num_agents
                    policy = ippo_policies.get(num_agents, ippo_policies[min(ippo_policies.keys())])
                    action, _ = policy.predict(obs.numpy(), deterministic=True)
                    actions = action[:num_agents]
                else:
                    output = model(obs)
                    if isinstance(output, tuple):
                        action_probs, _ = output
                    else:
                        action_probs = output
                    actions = torch.bernoulli(action_probs).numpy()
            obs, reward, terminated, truncated, _ = env.step(np.array(actions).astype(int))
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards

# --- Main Comparison ---
def run_comparison():
    custom_env_fn = lambda num_agents=None, seed=None: MultiAgentEnv(num_agents=num_agents or np.random.choice([1, 2, 3, 4, 5]), seed=seed)
    unseen_env_fn = lambda seed=None: UnseenMultiAgentEnv(num_agents=5, seed=seed)
    ss_env_fn = lambda num_agents=None, seed=None: SimpleSpreadWrapper(num_agents=num_agents or np.random.choice([1, 2, 3, 4, 5]), seed=seed)

    # Training
    print("Training LightMetaPolicy (20K steps)...")
    light_meta_model_20k = LightMetaPolicy(input_dim=MAX_AGENTS * 4, output_dim=MAX_AGENTS)
    light_meta_trained_20k, light_meta_time_20k = train_light_meta_policy(light_meta_model_20k, custom_env_fn, meta_iterations=500)

    print("Training LightMetaPolicy (40K steps)...")
    light_meta_model_40k = LightMetaPolicy(input_dim=MAX_AGENTS * 4, output_dim=MAX_AGENTS)
    light_meta_trained_40k, light_meta_time_40k = train_light_meta_policy(light_meta_model_40k, custom_env_fn, meta_iterations=1000)

    print("Training MAPPO (500K steps)...")
    mappo_policy, mappo_time = train_mappo_policy(custom_env_fn, iterations=2500, rollout_steps=200)

    print("Training IPPO...")
    ippo_policies, ippo_time = train_ippo_policy(custom_env_fn, timesteps=100000)

    print("Training MAML...")
    maml_model, maml_time = meta_train_maml(custom_env_fn, inner_steps=10, inner_rollouts=50, inner_lr=0.05)

    light_meta_params = sum(p.numel() for p in light_meta_model_20k.parameters())
    mappo_params = sum(p.numel() for p in mappo_policy.parameters())
    ippo_params = sum(p.numel() for p in list(ippo_policies.values())[0].policy.parameters())
    maml_params = sum(p.numel() for p in maml_model.parameters())

    if USE_SIMPLE_SPREAD:
        print("Training LightMetaPolicy (Simple Spread)...")
        light_meta_model_ss = LightMetaPolicy(input_dim=MAX_AGENTS * 22, output_dim=MAX_AGENTS * 5)
        light_meta_trained_ss, _ = train_light_meta_policy(light_meta_model_ss, ss_env_fn)
        print("Training MAPPO (Simple Spread)...")
        mappo_policy_ss, _ = train_mappo_policy(ss_env_fn, iterations=2500, rollout_steps=200)

    # Evaluation
    agent_counts = [1, 2, 3, 4, 5]
    results = {"LightMeta_20K": {}, "LightMeta_40K": {}, "MAPPO": {}, "IPPO": {}, "MAML": {}}

    print("\n=== Custom Env Evaluation ===")
    for num_agents in agent_counts:
        env_fn = lambda seed=None: custom_env_fn(num_agents=num_agents, seed=seed)
        print(f"\nEvaluating on {num_agents} agents:")
        lm_20k_zero, lm_20k_var, lm_20k_raw = evaluate_policy(light_meta_trained_20k, env_fn)
        lm_20k_adapted, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained_20k, env_fn), env_fn)
        lm_40k_zero, lm_40k_var, lm_40k_raw = evaluate_policy(light_meta_trained_40k, env_fn)
        lm_40k_adapted, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained_40k, env_fn), env_fn)
        mappo_reward, mappo_var, mappo_raw = evaluate_policy(mappo_policy, env_fn, is_mappo=True)
        ippo_reward, ippo_var, ippo_raw = evaluate_policy(None, env_fn, is_ippo=True, ippo_policies=ippo_policies)
        maml_zero, maml_var, maml_raw = evaluate_policy(maml_model, env_fn)
        maml_adapted, _, _ = evaluate_policy(fine_tune_maml_policy(maml_model, env_fn), env_fn)
        print(f"LightMeta_20K: {lm_20k_zero:.2f} ± {lm_20k_var:.2f} / {lm_20k_adapted:.2f}")
        print(f"LightMeta_40K: {lm_40k_zero:.2f} ± {lm_40k_var:.2f} / {lm_40k_adapted:.2f}")
        print(f"MAPPO: {mappo_reward:.2f} ± {mappo_var:.2f}")
        print(f"IPPO: {ippo_reward:.2f} ± {ippo_var:.2f}")
        print(f"MAML: {maml_zero:.2f} ± {maml_var:.2f} / {maml_adapted:.2f}")
        results["LightMeta_20K"][num_agents] = (lm_20k_zero, lm_20k_adapted, lm_20k_var, lm_20k_raw)
        results["LightMeta_40K"][num_agents] = (lm_40k_zero, lm_40k_adapted, lm_40k_var, lm_40k_raw)
        results["MAPPO"][num_agents] = (mappo_reward, mappo_var, mappo_raw)
        results["IPPO"][num_agents] = (ippo_reward, ippo_var, ippo_raw)
        results["MAML"][num_agents] = (maml_zero, maml_adapted, maml_var, maml_raw)

    print("\n=== Unseen Env Evaluation (5 agents) ===")
    lm_20k_zero_u, lm_20k_var_u, lm_20k_raw_u = evaluate_policy(light_meta_trained_20k, unseen_env_fn)
    lm_20k_adapted_u, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained_20k, unseen_env_fn), unseen_env_fn)
    lm_40k_zero_u, lm_40k_var_u, lm_40k_raw_u = evaluate_policy(light_meta_trained_40k, unseen_env_fn)
    lm_40k_adapted_u, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained_40k, unseen_env_fn), unseen_env_fn)
    mappo_u, mappo_var_u, mappo_raw_u = evaluate_policy(mappo_policy, unseen_env_fn, is_mappo=True)
    ippo_u, ippo_var_u, ippo_raw_u = evaluate_policy(None, unseen_env_fn, is_ippo=True, ippo_policies=ippo_policies)
    maml_zero_u, maml_var_u, maml_raw_u = evaluate_policy(maml_model, unseen_env_fn)
    maml_adapted_u, _, _ = evaluate_policy(fine_tune_maml_policy(maml_model, unseen_env_fn), unseen_env_fn)
    print(f"LightMeta_20K: {lm_20k_zero_u:.2f} ± {lm_20k_var_u:.2f} / {lm_20k_adapted_u:.2f}")
    print(f"LightMeta_40K: {lm_40k_zero_u:.2f} ± {lm_40k_var_u:.2f} / {lm_40k_adapted_u:.2f}")
    print(f"MAPPO: {mappo_u:.2f} ± {mappo_var_u:.2f}")
    print(f"IPPO: {ippo_u:.2f} ± {ippo_var_u:.2f}")
    print(f"MAML: {maml_zero_u:.2f} ± {maml_var_u:.2f} / {maml_adapted_u:.2f}")

    if USE_SIMPLE_SPREAD:
        print("\n=== Simple Spread Evaluation (5 agents) ===")
        ss_eval_fn = lambda seed=None: ss_env_fn(num_agents=5, seed=seed)
        lm_20k_zero_ss, _, _ = evaluate_policy(light_meta_trained_ss, ss_eval_fn)
        lm_20k_adapted_ss, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained_ss, ss_eval_fn), ss_eval_fn)
        mappo_ss, _, _ = evaluate_policy(mappo_policy_ss, ss_eval_fn, is_mappo=True)
        print(f"LightMeta_20K: {lm_20k_zero_ss:.2f} / {lm_20k_adapted_ss:.2f}")
        print(f"MAPPO: {mappo_ss:.2f}")

    # Compute Metrics
    print("\n=== Compute Metrics ===")
    print(f"LightMeta_20K: {light_meta_params} params, {light_meta_time_20k:.1f}s")
    print(f"LightMeta_40K: {light_meta_params} params, {light_meta_time_40k:.1f}s")
    print(f"MAPPO: {mappo_params} params, {mappo_time:.1f}s")
    print(f"IPPO: {ippo_params} params (per policy), {ippo_time:.1f}s")
    print(f"MAML: {maml_params} params, {maml_time:.1f}s")

    # Statistical Significance (5 agents)
    stat, p = wilcoxon([x - y for x, y in zip(results["LightMeta_40K"][5][3], results["MAPPO"][5][2])])
    print(f"LightMeta_40K Zero-Shot vs. MAPPO (5 agents): p-value = {p:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    for model, data in results.items():
        if "LightMeta" in model or model == "MAML":
            zero_shot = [v[0] for v in data.values()]
            adapted = [v[1] for v in data.values()]
            plt.plot(agent_counts, zero_shot, label=f"{model} Zero-Shot", linestyle='--')
            plt.plot(agent_counts, adapted, label=f"{model} Adapted")
        else:
            rewards = [v[0] for v in data.values()]
            plt.plot(agent_counts, rewards, label=model)
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Reward")
    plt.title("Performance Across Agent Counts (Custom Env)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_comparison()