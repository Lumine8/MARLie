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
import collections
import multiprocessing as mp

# --- SCRIPT CONFIGURATION ---
# Set to True to run the Simple Spread benchmark. Set to False to run the custom environment.
USE_SIMPLE_SPREAD = True
# Set seeds for the entire experiment for reproducibility
SEEDS = [42, 123, 888, 2024, 7] # Using 5 seeds for final paper results
# Set the number of CPU cores to use for parallel execution
NUM_WORKERS = 6

# --- Global Constants ---
MAX_AGENTS = 5
EPISODES = 20

# --- Environments ---
class CustomMultiAgentEnv(gym.Env):
    """
    Our custom multi-agent environment designed to test generalization.
    """
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 4,), dtype=np.float32)
        # Action space is binary (move or not) for each agent
        self.action_space = spaces.MultiDiscrete([2] * MAX_AGENTS)
        self.max_steps = 200
        self.steps = 0
        self.seed = seed
        self.randomize_environment()

    def randomize_environment(self):
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(3, 20)
        self.start_bounds = local_random_state.randint(20, 150)
        self.agent_positions = local_random_state.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
        self.agent_speeds = local_random_state.uniform(0.5, 2.0, size=self.num_agents)
        self.goal_zones = local_random_state.randint(50, 550, size=(self.num_agents, 2))
        self.observation_noise_std = local_random_state.uniform(0, 0.2)
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.prev_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)
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
        if np.sum(move_mask) > 0:
            move_amounts = np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
            for i, idx in enumerate(np.where(move_mask)[0]):
                move_amounts[i] = (move_amounts[i] * self.agent_speeds[idx]).astype(int)
            self.agent_positions[move_mask] += move_amounts
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        new_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
        for i in range(self.num_agents):
            rewards[i] += max(0, 100 - new_distances[i]) / 100
            rewards[i] += 0.5 * max(0, (self.prev_distances[i] - new_distances[i]) / 100)
        self.prev_distances = new_distances
        self.steps += 1
        terminated = self.steps >= self.max_steps
        noisy_obs = self.agent_states + np.random.normal(0, self.observation_noise_std, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class UnseenCustomMultiAgentEnv(CustomMultiAgentEnv):
    """
    An out-of-distribution version of the custom environment to test true generalization.
    It features different dynamics and reward structures.
    """
    def __init__(self, num_agents=5, seed=None):
        super().__init__(num_agents=num_agents, seed=seed)

    def randomize_environment(self):
        # Inherit randomization but change key parameters
        super().randomize_environment()
        local_random_state = np.random.RandomState(self.seed)
        # Faster, more erratic movement
        self.move_range = local_random_state.randint(25, 50)
        # Much higher observation noise
        self.observation_noise_std = 0.4

    def step(self, actions):
        # Get the standard step result from the parent class
        padded_obs, reward, terminated, truncated, info = super().step(actions)

        # Add a new reward component: a cohesion penalty
        centroid = np.mean(self.agent_positions, axis=0)
        distances_from_centroid = np.linalg.norm(self.agent_positions - centroid, axis=1)
        # Penalize agents for being too far from the group average
        cohesion_penalty = -0.1 * np.mean(distances_from_centroid) / 100.0
        
        # Return the observation and info, but with the modified reward
        return padded_obs, reward + cohesion_penalty, terminated, truncated, info

class SimpleSpreadWrapper(gym.Env):
    """Wrapper for the PettingZoo Simple Spread environment."""
    
    # Helper to dynamically get the max obs dim without creating a full instance in __main__
    @staticmethod
    def get_max_obs_dim(max_agents):
        temp_env = simple_spread_v3.parallel_env(N=max_agents)
        max_dim = temp_env.observation_space("agent_0").shape[0]
        temp_env.close()
        return max_dim

    def __init__(self, num_agents=5, seed=None):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=100, continuous_actions=False)
        self.num_agents = num_agents
        self.seed = seed

        # Dynamically get the maximum possible observation size for any agent
        self.max_obs_dim_per_agent = self.get_max_obs_dim(MAX_AGENTS)
        self.action_dim = self.env.action_space("agent_0").n
        
        # The observation space for the wrapper should be the maximum possible size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.action_dim] * MAX_AGENTS)
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.reset(seed=self.seed)

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else self.seed
        obs_dict, infos = self.env.reset(seed=effective_seed)
        return self._flatten_obs(obs_dict), infos

    def step(self, actions):
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        next_obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos = self.env.step(action_dict)
        reward = sum(rewards_dict.values()) / self.num_agents if self.num_agents > 0 else 0
        terminated = all(terminateds_dict.values())
        truncated = all(truncateds_dict.values())
        return self._flatten_obs(next_obs_dict), reward, (terminated or truncated), False, infos

    def _flatten_obs(self, obs_dict):
        # Always pad to the max size to ensure consistent input for the models
        padded_obs = np.zeros((MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        for i, agent_name in enumerate(self.agents):
            if agent_name in obs_dict:
                obs = obs_dict[agent_name]
                # Place the agent's observation at the start of its allocated block
                start_idx = i * self.max_obs_dim_per_agent
                end_idx = start_idx + len(obs)
                padded_obs[start_idx:end_idx] = obs
        return padded_obs

# --- Models ---
class LightMetaPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions):
        super().__init__()
        self.agent_dim = agent_obs_dim
        self.num_actions = num_actions
        self.key_transform = nn.Linear(self.agent_dim, 32)
        self.query_transform = nn.Linear(self.agent_dim, 32)
        self.value_transform = nn.Linear(self.agent_dim, 64)
        self.agent_relation = nn.Linear(32, 32)
        output_logit_dim = MAX_AGENTS * self.num_actions
        self.post_attention = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, output_logit_dim))
        self.value_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.view(batch_size, MAX_AGENTS, self.agent_dim)
        keys, queries, values = self.key_transform(agents), self.query_transform(agents), self.value_transform(agents)
        attention = torch.bmm(queries, keys.transpose(1, 2)) / (32 ** 0.5)
        relation_queries = self.agent_relation(queries)
        attention += torch.bmm(relation_queries, relation_queries.transpose(1, 2)) / (32 ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        context = torch.bmm(attention, values)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        action_logits = self.post_attention(pooled).view(batch_size, MAX_AGENTS, self.num_actions)
        value = self.value_head(pooled)
        return action_logits, value

class MAPPOPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(agent_obs_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_actions))
        self.critic = nn.Sequential(nn.Linear(agent_obs_dim * MAX_AGENTS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def get_action(self, obs, num_agents, deterministic=False):
        agent_obs = obs.view(MAX_AGENTS, -1)[:num_agents]
        dist = torch.distributions.Categorical(logits=self.actor(agent_obs))
        return dist.sample() if not deterministic else torch.argmax(dist.logits, dim=-1)

    def get_value(self, all_obs):
        return self.critic(all_obs)

class MAMLPolicy(nn.Module):
    def __init__(self, input_dim, output_logit_dim):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, 128), nn.Linear(128, 128), nn.Linear(128, output_logit_dim)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    def adapt(self, loss, lr=0.01):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads):
                param -= lr * grad
        return self

# --- Training & Evaluation Functions ---
def train_light_meta_policy(model, env_fn, meta_iterations=400, inner_rollouts=128, n_epochs=4, gamma=0.99, entropy_coef=0.01, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        
        # --- Collect Rollout ---
        all_obs, all_rewards, all_actions = [], [], []
        for _ in range(inner_rollouts):
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            
            all_obs.append(obs)
            all_actions.append(actions.numpy())
            all_rewards.append(reward)
            
            obs = next_obs
            if terminated:
                obs, _ = env.reset()
        
        # --- Prepare Tensors for Update ---
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)
        
        returns = []
        discounted_reward = 0
        for r in reversed(all_rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        for _ in range(n_epochs):
            action_logits, value_pred = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            new_log_probs = dist.log_prob(actions_tensor)

            advantages = returns - value_pred.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -(new_log_probs.mean(dim=-1) * advantages).mean()
            value_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        if verbose and iteration % 100 == 0: print(f"  LightMeta Iter {iteration} | Avg Return: {returns.mean().item():.2f}")
    return model, time.time() - start_time

def train_mappo_policy(model, env_fn, iterations=1000, rollout_steps=512, n_epochs=4, gamma=0.99, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        
        # --- Collect Rollout ---
        all_obs, all_rewards, all_actions = [], [], []
        for _ in range(rollout_steps):
            with torch.no_grad():
                actions = model.get_action(torch.tensor(obs, dtype=torch.float32), num_agents)
            next_obs, reward, terminated, _, _ = env.step(actions.numpy())
            
            all_obs.append(obs)
            all_actions.append(actions.numpy())
            all_rewards.append(reward)
            
            obs = next_obs
            if terminated:
                obs, _ = env.reset()
        
        # --- Prepare Tensors for Update ---
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)
        
        returns = []
        discounted_reward = 0
        for r in reversed(all_rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(n_epochs):
            dist = torch.distributions.Categorical(logits=model.actor(obs_tensor.view(-1, MAX_AGENTS, model.actor[0].in_features)[:,:num_agents,:]))
            new_log_probs = dist.log_prob(actions_tensor)
            value_pred = model.get_value(obs_tensor)
            
            advantages = returns - value_pred.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(new_log_probs.mean(dim=-1) * advantages).mean()
            value_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if verbose and iteration % 250 == 0: print(f"  MAPPO Iter {iteration} | Avg Return: {returns.mean().item():.2f}")
    return model, time.time() - start_time

def train_ippo_policy(env_fn, timesteps=100000):
    policies = {}
    start_time = time.time()
    for num_agents in [2, 3, 4, 5]:
        env = env_fn(num_agents=num_agents)
        policies[num_agents] = PPO("MlpPolicy", env, verbose=0, n_steps=2048, learning_rate=0.0003, device='cpu')
        policies[num_agents].learn(total_timesteps=timesteps // 4)
    return policies, time.time() - start_time

def meta_train_maml(model, env_fn, meta_iterations=750, inner_steps=10, inner_rollouts=50, gamma=0.99, inner_lr=0.01, verbose=True):
    meta_optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        adapted_model = copy.deepcopy(model)
        for _ in range(inner_steps):
            action_logits = adapted_model(obs)
            dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:,:num_agents,:])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            loss = -dist.log_prob(actions).mean() * reward
            adapted_model = adapted_model.adapt(loss, lr=inner_lr)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated: obs, _ = env.reset(); obs = torch.tensor(obs, dtype=torch.float32)
        log_probs, rewards = [], []
        for _ in range(inner_rollouts):
            action_logits = adapted_model(obs)
            dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:,:num_agents,:])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            log_probs.append(dist.log_prob(actions).mean())
            rewards.append(reward)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            if terminated: break
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        loss = -torch.sum(torch.stack(log_probs) * torch.tensor(returns, dtype=torch.float32))
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
        if verbose and iteration % 100 == 0: print(f"  MAML Iter {iteration} | Avg Return: {np.mean(returns):.2f}")
    return model, time.time() - start_time

def fine_tune_light_meta_policy(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0001)
    env = env_fn()
    experiences = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
                actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            experiences.append((obs, actions.numpy().flatten(), reward, next_obs))
            obs = next_obs
            done = terminated
    for step in range(max_steps):
        if len(experiences) > 64: 
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
            log_probs = dist.log_prob(batch_actions)
            loss = - (log_probs * batch_rewards.unsqueeze(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return tuned_model

def fine_tune_maml_policy(meta_model, env_fn, steps=40):
    env = env_fn()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    adapted_model = copy.deepcopy(meta_model)
    for _ in range(steps):
        action_logits = adapted_model(obs)
        dist = torch.distributions.Categorical(logits=action_logits.view(1, MAX_AGENTS, -1)[:,:env.num_agents,:])
        actions = dist.sample()
        next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
        loss = -dist.log_prob(actions).mean() * reward
        adapted_model = adapted_model.adapt(loss)
        obs = torch.tensor(next_obs, dtype=torch.float32)
        if terminated: obs, _ = env.reset(); obs = torch.tensor(obs, dtype=torch.float32)
    return adapted_model

def evaluate_policy(model, env_fn, episodes=EPISODES, is_mappo=False, is_ippo=False, ippo_policies=None):
    rewards = []
    for ep in range(episodes):
        env = env_fn(seed=1000 + ep)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                if is_mappo:
                    actions = model.get_action(torch.tensor(obs, dtype=torch.float32), env.num_agents, deterministic=True)
                    actions = actions.numpy()
                elif is_ippo:
                    actions, _ = ippo_policies[env.num_agents].predict(obs, deterministic=True)
                else: # LightMeta or MAML
                    output = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    if isinstance(output, tuple):
                        action_logits, _ = output
                    else:
                        action_logits = output
                    if action_logits.dim() == 2:
                        num_actions = action_logits.shape[1] // MAX_AGENTS
                        action_logits = action_logits.view(1, MAX_AGENTS, num_actions)
                    actions = torch.argmax(action_logits[:, :env.num_agents, :], dim=-1).numpy().flatten()
            obs, reward, terminated, _, _ = env.step(actions)
            total_reward += reward
            done = terminated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards

# --- Main Comparison ---
def run_comparison(args):
    seed, env_config, unseen_env_config = args
    print(f"\n--- Running Comparison for Seed: {seed} on {env_config['name']} ---")
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    env_fn, agent_obs_dim, num_actions = env_config['fn'], env_config['agent_obs_dim'], env_config['num_actions']
    input_dim = MAX_AGENTS * agent_obs_dim
    run_results = {
        "in_distribution": collections.defaultdict(dict), 
        "out_of_distribution": collections.defaultdict(dict),
        "compute_metrics": collections.defaultdict(dict)
    }
    
    # --- Training ---
    print(f"[{seed}] Training LightMetaPolicy..."); light_meta_model = LightMetaPolicy(agent_obs_dim, num_actions); light_meta_trained, t = train_light_meta_policy(light_meta_model, env_fn, verbose=True); run_results["compute_metrics"]["LightMeta"] = (sum(p.numel() for p in light_meta_model.parameters()), t)
    print(f"[{seed}] Training MAPPO..."); mappo_policy = MAPPOPolicy(agent_obs_dim, num_actions); mappo_trained, t = train_mappo_policy(mappo_policy, env_fn, verbose=True); run_results["compute_metrics"]["MAPPO"] = (sum(p.numel() for p in mappo_policy.parameters()), t)
    print(f"[{seed}] Training IPPO..."); ippo_policies, t = train_ippo_policy(env_fn); run_results["compute_metrics"]["IPPO"] = (sum(p.numel() for p in list(ippo_policies.values())[0].policy.parameters()), t)
    print(f"[{seed}] Training MAML..."); maml_model = MAMLPolicy(input_dim, MAX_AGENTS * num_actions); maml_trained, t = meta_train_maml(maml_model, env_fn, verbose=True); run_results["compute_metrics"]["MAML"] = (sum(p.numel() for p in maml_model.parameters()), t)
    
    # --- In-Distribution Evaluation ---
    print(f"[{seed}] Evaluating models on in-distribution tasks...")
    for num_agents in [2, 3, 4, 5]:
        eval_env_fn = lambda seed=None: env_fn(num_agents=num_agents, seed=seed)
        lm_zero, lm_var, lm_raw = evaluate_policy(light_meta_trained, eval_env_fn); lm_adapted, _, _ = evaluate_policy(fine_tune_light_meta_policy(light_meta_trained, eval_env_fn), eval_env_fn); run_results["in_distribution"]["LightMeta"][num_agents] = (lm_zero, lm_adapted, lm_var, lm_raw)
        mappo_reward, mappo_var, mappo_raw = evaluate_policy(mappo_trained, eval_env_fn, is_mappo=True); run_results["in_distribution"]["MAPPO"][num_agents] = (mappo_reward, mappo_var, mappo_raw)
        ippo_reward, ippo_var, ippo_raw = evaluate_policy(None, eval_env_fn, is_ippo=True, ippo_policies=ippo_policies); run_results["in_distribution"]["IPPO"][num_agents] = (ippo_reward, ippo_var, ippo_raw)
        maml_zero, maml_var, maml_raw = evaluate_policy(maml_trained, eval_env_fn); maml_adapted, _, _ = evaluate_policy(fine_tune_maml_policy(maml_trained, eval_env_fn), eval_env_fn); run_results["in_distribution"]["MAML"][num_agents] = (maml_zero, maml_adapted, maml_var, maml_raw)

    # --- Out-of-Distribution Evaluation ---
    if not USE_SIMPLE_SPREAD:
        print(f"[{seed}] Evaluating models on out-of-distribution task...")
        unseen_env_fn = lambda seed=None: unseen_env_config['fn'](seed=seed)
        run_results["out_of_distribution"]["LightMeta"] = evaluate_policy(light_meta_trained, unseen_env_fn)
        run_results["out_of_distribution"]["MAPPO"] = evaluate_policy(mappo_trained, unseen_env_fn, is_mappo=True)
        run_results["out_of_distribution"]["IPPO"] = evaluate_policy(None, unseen_env_fn, is_ippo=True, ippo_policies=ippo_policies)
        run_results["out_of_distribution"]["MAML"] = evaluate_policy(maml_trained, unseen_env_fn)
    
    print(f"--- Finished Seed: {seed} ---")
    return run_results

def process_and_display_results(all_runs_results, env_name):
    print(f"\n\n{'='*20} AGGREGATED RESULTS ({env_name}) {'='*20}")
    agent_counts = [2, 3, 4, 5]
    model_names = ["LightMeta", "MAML", "MAPPO", "IPPO"]
    
    # --- In-Distribution Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    for model in model_names:
        if model in ["LightMeta", "MAML"]:
            zero_means = [np.mean([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            zero_stds = [np.std([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            adapted_means = [np.mean([r["in_distribution"][model][n][1] for r in all_runs_results]) for n in agent_counts]
            adapted_stds = [np.std([r["in_distribution"][model][n][1] for r in all_runs_results]) for n in agent_counts]
            plt.plot(agent_counts, zero_means, 'o--', label=f"{model} (Zero-Shot)")
            plt.fill_between(agent_counts, np.array(zero_means)-np.array(zero_stds), np.array(zero_means)+np.array(zero_stds), alpha=0.1)
            plt.plot(agent_counts, adapted_means, 'o-', label=f"{model} (Adapted)")
            plt.fill_between(agent_counts, np.array(adapted_means)-np.array(adapted_stds), np.array(adapted_means)+np.array(adapted_stds), alpha=0.2)
        else:
            means = [np.mean([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            stds = [np.std([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            plt.plot(agent_counts, means, '^-', label=model)
            plt.fill_between(agent_counts, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.1)
    plt.title(f"In-Distribution Performance on {env_name} (Aggregated over {len(SEEDS)} Seeds)", fontsize=16)
    plt.xlabel("Number of Agents", fontsize=14); plt.ylabel("Average Total Reward", fontsize=14)
    plt.xticks(agent_counts); plt.legend(fontsize=12); plt.grid(True); plt.tight_layout(); plt.show()

    # --- Out-of-Distribution Results ---
    if not USE_SIMPLE_SPREAD:
        print("\n" + "="*20 + " OUT-OF-DISTRIBUTION (OOD) RESULTS " + "="*20)
        print(f"{'Model':<20} | {'Mean Reward':<15} | {'Std Dev':<15}")
        print("-"*55)
        for model in model_names:
            # Check if OOD results exist for this run before trying to access them
            if "out_of_distribution" in all_runs_results[0] and model in all_runs_results[0]["out_of_distribution"]:
                rewards = [r["out_of_distribution"][model][0] for r in all_runs_results]
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                print(f"{model:<20} | {mean_reward:<15.2f} | {std_reward:<15.2f}")


if __name__ == "__main__":
    # Get the correct max obs dim once and pass it, preventing re-instantiation
    max_ss_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    ss_action_dim = SimpleSpreadWrapper(num_agents=2).action_dim # Action dim is constant

    ENV_CONFIGS = {
        "SimpleSpread": {"name": "Simple Spread", "fn": SimpleSpreadWrapper, "agent_obs_dim": max_ss_obs_dim, "num_actions": ss_action_dim},
        "CustomEnv": {"name": "Custom Environment", "fn": CustomMultiAgentEnv, "agent_obs_dim": 4, "num_actions": 2}
    }
    UNSEEN_ENV_CONFIG = {
        "UnseenCustomEnv": {"name": "Unseen Custom", "fn": UnseenCustomMultiAgentEnv}
    }

    config = ENV_CONFIGS["SimpleSpread"] if USE_SIMPLE_SPREAD else ENV_CONFIGS["CustomEnv"]
    unseen_config = UNSEEN_ENV_CONFIG["UnseenCustomEnv"]

    # --- Parallel Execution ---
    pool = mp.Pool(processes=NUM_WORKERS)
    args_for_pool = [(seed, config, unseen_config) for seed in SEEDS]
    
    all_runs_results = pool.map(run_comparison, args_for_pool)
    
    pool.close()
    pool.join()

    process_and_display_results(all_runs_results, config["name"])
