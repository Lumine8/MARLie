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
USE_SIMPLE_SPREAD = True
SEEDS = [42, 123, 888, 2024, 7]
NUM_WORKERS = 6
MAX_AGENTS = 5
EPISODES = 20
META_ITERATIONS = 500

# --- LoRASA Layer ---
class LoRASALayer(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.linear = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None: self.linear.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_scale = nn.Parameter(torch.ones(out_features))
        self.lora_shift = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
    def forward(self, x):
        frozen_output = self.linear(x)
        adapter_output = (x @ self.lora_A @ self.lora_B) * self.lora_scale + self.lora_shift
        return frozen_output + adapter_output

# --- Environments ---
class CustomMultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, seed=None):
        super().__init__()
        self.num_agents = num_agents
        self.observation_space = spaces.Box(low=-1, high=1, shape=(MAX_AGENTS * 4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * MAX_AGENTS)
        self.max_steps = 200; self.steps = 0; self.seed = seed
        self.randomize_environment()
    def randomize_environment(self):
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(3, 20)
        self.start_bounds = local_random_state.randint(20, 150)
        self.agent_positions = local_random_state.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
        self.agent_speeds = local_random_state.uniform(0.5, 2.0, size=self.num_agents)
        self.goal_zones = local_random_state.randint(50, 550, size=(self.num_agents, 2))
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.prev_distances = np.array([np.linalg.norm(self.agent_positions[i] - self.goal_zones[i]) for i in range(self.num_agents)])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.seed)
        self.randomize_environment(); self.steps = 0
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
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
        self.steps += 1; terminated = self.steps >= self.max_steps
        noisy_obs = self.agent_states + np.random.normal(0, 0.2, size=self.agent_states.shape)
        obs = noisy_obs.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class UnseenCustomMultiAgentEnv(CustomMultiAgentEnv):
    def randomize_environment(self):
        super().randomize_environment()
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(25, 50)
        self.observation_noise_std = 0.4
    def step(self, actions):
        padded_obs, reward, terminated, truncated, info = super().step(actions)
        centroid = np.mean(self.agent_positions, axis=0)
        cohesion_penalty = -0.1 * np.mean(np.linalg.norm(self.agent_positions - centroid, axis=1)) / 100.0
        return padded_obs, reward + cohesion_penalty, terminated, truncated, info

class SimpleSpreadWrapper(gym.Env):
    @staticmethod
    def get_max_obs_dim(max_agents):
        temp_env = simple_spread_v3.parallel_env(N=max_agents)
        max_dim = temp_env.observation_space("agent_0").shape[0]
        temp_env.close()
        return max_dim
    def __init__(self, num_agents=5, seed=None):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=100, continuous_actions=False)
        self.num_agents = num_agents; self.seed = seed
        self.max_obs_dim_per_agent = self.get_max_obs_dim(MAX_AGENTS)
        self.action_dim = self.env.action_space("agent_0").n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.action_dim] * MAX_AGENTS)
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.reset(seed=self.seed)
    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed if seed is not None else self.seed)
        return self._flatten_obs(obs_dict), infos
    def step(self, actions):
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        next_obs_dict, rewards_dict, terminateds_dict, _, _ = self.env.step(action_dict)
        reward = sum(rewards_dict.values()) / self.num_agents if self.num_agents > 0 else 0
        terminated = all(terminateds_dict.values())
        return self._flatten_obs(next_obs_dict), reward, terminated, False, {}
    def _flatten_obs(self, obs_dict):
        padded_obs = np.zeros((MAX_AGENTS * self.max_obs_dim_per_agent,), dtype=np.float32)
        for i, agent_name in enumerate(self.agents):
            if agent_name in obs_dict:
                obs = obs_dict[agent_name]
                padded_obs[i*self.max_obs_dim_per_agent : i*self.max_obs_dim_per_agent + len(obs)] = obs
        return padded_obs

# --- Models ---
class LightMetaPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions, use_layer_norm=True, use_residual=True):
        super().__init__()
        self.agent_dim, self.num_actions = agent_obs_dim, num_actions
        self.use_layer_norm, self.use_residual = use_layer_norm, use_residual
        self.d_model = 64
        self.input_proj = nn.Linear(self.agent_dim, self.d_model)
        self.key_transform = nn.Linear(self.d_model, self.d_model)
        self.query_transform = nn.Linear(self.d_model, self.d_model)
        self.value_transform = nn.Linear(self.d_model, self.d_model)
        if self.use_layer_norm: self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        if self.use_layer_norm: self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.action_head = nn.Linear(self.d_model, MAX_AGENTS * self.num_actions)
        self.value_head = nn.Sequential(nn.Linear(self.d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.view(batch_size, MAX_AGENTS, self.agent_dim)
        projected_agents = self.input_proj(agents)
        queries = self.query_transform(projected_agents).unsqueeze(2)
        keys = self.key_transform(projected_agents).unsqueeze(2)
        values = self.value_transform(projected_agents).unsqueeze(2)
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values).squeeze(2)
        context = self.fc_out(context)
        if self.use_residual: context = context + projected_agents
        if self.use_layer_norm: context = self.layer_norm1(context)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        ffn_out = self.ffn(pooled)
        if self.use_residual: ffn_out = ffn_out + pooled
        if self.use_layer_norm: ffn_out = self.layer_norm2(ffn_out)
        action_logits = self.action_head(ffn_out).view(batch_size, MAX_AGENTS, self.num_actions)
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
    def get_value(self, all_obs): return self.critic(all_obs)

class MAMLPolicy(nn.Module):
    def __init__(self, input_dim, output_logit_dim):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, 128), nn.Linear(128, 128), nn.Linear(128, output_logit_dim)
    def forward(self, x): return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    def adapt(self, loss, lr=0.01):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads): param -= lr * grad
        return self

def inject_lorasa(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRASALayer(module, rank=rank))
        elif len(list(module.children())) > 0:
            inject_lorasa(module, rank=rank)
    return model

def freeze_base_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

# --- Training & Fine-Tuning Functions ---
def train_light_meta_policy(model, env_fn, meta_iterations=400, inner_rollouts=128, n_epochs=4, gamma=0.99, entropy_coef=0.01, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    learning_curve = []
    total_steps = 0
    for iteration in range(meta_iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        all_obs, all_rewards, all_actions = [], [], []
        for _ in range(inner_rollouts):
            with torch.no_grad():
                action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
            actions = dist.sample()
            next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
            all_obs.append(obs); all_actions.append(actions.numpy()); all_rewards.append(reward)
            obs = next_obs
            total_steps += 1
            if terminated: obs, _ = env.reset()
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
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if iteration % 20 == 0:
            eval_reward, _, _ = evaluate_policy(model, env_fn)
            learning_curve.append((total_steps, eval_reward))
    return model, time.time() - start_time, learning_curve

def train_mappo_policy(model, env_fn, iterations=1000, rollout_steps=512, n_epochs=4, gamma=0.99, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    learning_curve = []
    total_steps = 0
    for iteration in range(iterations):
        num_agents = np.random.choice([2, 3, 4, 5])
        env = env_fn(num_agents=num_agents)
        obs, _ = env.reset()
        all_obs, all_rewards, all_actions = [], [], []
        for _ in range(rollout_steps):
            with torch.no_grad():
                actions = model.get_action(torch.tensor(obs, dtype=torch.float32), num_agents)
            next_obs, reward, terminated, _, _ = env.step(actions.numpy())
            all_obs.append(obs); all_actions.append(actions.numpy()); all_rewards.append(reward)
            obs = next_obs
            total_steps += 1
            if terminated: obs, _ = env.reset()
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
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if iteration % 50 == 0:
            eval_reward, _, _ = evaluate_policy(model, env_fn, is_mappo=True)
            learning_curve.append((total_steps, eval_reward))
    return model, time.time() - start_time, learning_curve

def train_ippo_policy(env_fn, timesteps=100000):
    policies = {}
    start_time = time.time()
    for num_agents in [2, 3, 4, 5]:
        env = env_fn(num_agents=num_agents)
        policies[num_agents] = PPO("MlpPolicy", env, verbose=0, n_steps=2048, learning_rate=0.0003, device='cpu')
        policies[num_agents].learn(total_timesteps=timesteps // 4)
    return policies, time.time() - start_time

def meta_train_maml(model, env_fn, meta_iterations=750, inner_steps=10, inner_rollouts=50, gamma=0.99, inner_lr=0.01, verbose=False):
    meta_optimizer = optim.Adam(model.parameters(), lr=0.0003)
    start_time = time.time()
    for _ in range(meta_iterations):
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
        meta_optimizer.zero_grad(); loss.backward(); meta_optimizer.step()
    return model, time.time() - start_time

def fine_tune_standard(model, env_fn, max_steps=40):
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
            obs = next_obs; done = terminated
    for _ in range(max_steps):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
            log_probs = dist.log_prob(batch_actions)
            loss = - (log_probs * batch_rewards.unsqueeze(1)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tuned_model

def fine_tune_with_lorasa(model, env_fn, max_steps=40):
    tuned_model = copy.deepcopy(model)
    inject_lorasa(tuned_model)
    freeze_base_parameters(tuned_model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, tuned_model.parameters()), lr=0.001)
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
            obs = next_obs; done = terminated
    for _ in range(max_steps):
        if len(experiences) > 64:
            batch = random.sample(experiences, 64)
            batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
            batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
            batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
            action_logits, _ = tuned_model(batch_obs)
            dist = torch.distributions.Categorical(logits=action_logits[:, :env.num_agents, :])
            log_probs = dist.log_prob(batch_actions)
            loss = - (log_probs * batch_rewards.unsqueeze(1)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
        total_reward = 0; done = False
        while not done:
            with torch.no_grad():
                if is_mappo:
                    actions = model.get_action(torch.tensor(obs, dtype=torch.float32), env.num_agents, deterministic=True).numpy()
                elif is_ippo:
                    actions, _ = ippo_policies[env.num_agents].predict(obs, deterministic=True)
                else: 
                    output = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action_logits = output[0] if isinstance(output, tuple) else output
                    if action_logits.dim() == 2:
                        num_actions = action_logits.shape[1] // MAX_AGENTS
                        action_logits = action_logits.view(1, MAX_AGENTS, num_actions)
                    actions = torch.argmax(action_logits[:, :env.num_agents, :], dim=-1).numpy().flatten()
            obs, reward, terminated, _, _ = env.step(actions)
            total_reward += reward; done = terminated
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
        "compute_metrics": collections.defaultdict(dict),
        "learning_curves": collections.defaultdict(list)
    }
    
    print(f"[{seed}] Training LightMetaPolicy..."); light_meta_model = LightMetaPolicy(agent_obs_dim, num_actions); light_meta_trained, t, curve = train_light_meta_policy(light_meta_model, env_fn); run_results["compute_metrics"]["LightMeta"] = (sum(p.numel() for p in light_meta_model.parameters()), t); run_results["learning_curves"]["LightMeta"] = curve
    print(f"[{seed}] Training MAPPO..."); mappo_policy = MAPPOPolicy(agent_obs_dim, num_actions); mappo_trained, t, curve = train_mappo_policy(mappo_policy, env_fn); run_results["compute_metrics"]["MAPPO"] = (sum(p.numel() for p in mappo_policy.parameters()), t); run_results["learning_curves"]["MAPPO"] = curve
    print(f"[{seed}] Training IPPO..."); ippo_policies, t = train_ippo_policy(env_fn); run_results["compute_metrics"]["IPPO"] = (sum(p.numel() for p in list(ippo_policies.values())[0].policy.parameters()), t)
    print(f"[{seed}] Training MAML..."); maml_model = MAMLPolicy(input_dim, MAX_AGENTS * num_actions); maml_trained, t = meta_train_maml(maml_model, env_fn); run_results["compute_metrics"]["MAML"] = (sum(p.numel() for p in maml_model.parameters()), t)
    
    lorasa_model_for_counting = inject_lorasa(copy.deepcopy(light_meta_trained))
    run_results["compute_metrics"]["LightMeta_LoRASA"] = (sum(p.numel() for p in lorasa_model_for_counting.parameters() if p.requires_grad), 0)

    print(f"[{seed}] Evaluating models on in-distribution tasks...")
    for num_agents in [2, 3, 4, 5]:
        eval_env_fn = lambda seed=None: env_fn(num_agents=num_agents, seed=seed)
        
        lm_zero, _, lm_raw_zero = evaluate_policy(light_meta_trained, eval_env_fn)
        lm_full, _, lm_raw_full = evaluate_policy(fine_tune_standard(light_meta_trained, eval_env_fn), eval_env_fn)
        lm_lorasa, _, lm_raw_lorasa = evaluate_policy(fine_tune_with_lorasa(light_meta_trained, eval_env_fn), eval_env_fn)
        run_results["in_distribution"]["LightMeta"][num_agents] = (lm_zero, lm_full, lm_lorasa, lm_raw_lorasa) # Added raw for sig test

        mappo_reward, _, mappo_raw = evaluate_policy(mappo_trained, eval_env_fn, is_mappo=True); run_results["in_distribution"]["MAPPO"][num_agents] = (mappo_reward, mappo_raw)
        ippo_reward, _, ippo_raw = evaluate_policy(None, eval_env_fn, is_ippo=True, ippo_policies=ippo_policies); run_results["in_distribution"]["IPPO"][num_agents] = (ippo_reward, ippo_raw)
        maml_zero, _, maml_raw_zero = evaluate_policy(maml_trained, eval_env_fn); maml_adapted, _, maml_raw_adapted = evaluate_policy(fine_tune_maml_policy(maml_trained, eval_env_fn), eval_env_fn); run_results["in_distribution"]["MAML"][num_agents] = (maml_zero, maml_adapted, maml_raw_adapted)

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
    
    # --- In-Distribution Results Table ---
    print("\n" + "="*20 + " IN-DISTRIBUTION PERFORMANCE (TABLE) " + "="*20)
    header_parts = [f"{f'N={n} Agents':<18}" for n in agent_counts]
    header = f"{'Model':<28} | " + " | ".join(header_parts)
    print(header); print("-" * len(header))
    for i, method in enumerate(["Zero-Shot", "Adapted (Full)", "Adapted (LoRASA)"]):
        label = f"LightMeta ({method})"
        row_data = [label.ljust(28)]
        for n in agent_counts:
            rewards = [r["in_distribution"]["LightMeta"][n][i] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    for i, method in enumerate(["Zero-Shot", "Adapted"]):
        label = f"MAML ({method})"
        row_data = [label.ljust(28)]
        for n in agent_counts:
            rewards = [r["in_distribution"]["MAML"][n][i] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    for model in ["MAPPO", "IPPO"]:
        row_data = [f"{model:<28}"]
        for n in agent_counts:
            rewards = [r["in_distribution"][model][n][0] for r in all_runs_results]
            row_data.append(f"{np.mean(rewards):<8.2f} ± {np.std(rewards):<7.2f}")
        print(" | ".join(row_data))
    
    # --- In-Distribution Results Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    lm_zero_means = [np.mean([r["in_distribution"]["LightMeta"][n][0] for r in all_runs_results]) for n in agent_counts]
    lm_zero_stds = [np.std([r["in_distribution"]["LightMeta"][n][0] for r in all_runs_results]) for n in agent_counts]
    lm_full_means = [np.mean([r["in_distribution"]["LightMeta"][n][1] for r in all_runs_results]) for n in agent_counts]
    lm_full_stds = [np.std([r["in_distribution"]["LightMeta"][n][1] for r in all_runs_results]) for n in agent_counts]
    lm_lorasa_means = [np.mean([r["in_distribution"]["LightMeta"][n][2] for r in all_runs_results]) for n in agent_counts]
    lm_lorasa_stds = [np.std([r["in_distribution"]["LightMeta"][n][2] for r in all_runs_results]) for n in agent_counts]
    
    plt.plot(agent_counts, lm_zero_means, 'o--', label="LightMeta (Zero-Shot)", color='blue')
    plt.fill_between(agent_counts, np.array(lm_zero_means)-np.array(lm_zero_stds), np.array(lm_zero_means)+np.array(lm_zero_stds), alpha=0.1, color='blue')
    plt.plot(agent_counts, lm_full_means, 'o-', label="LightMeta (Adapted - Full)", color='darkorange')
    plt.fill_between(agent_counts, np.array(lm_full_means)-np.array(lm_full_stds), np.array(lm_full_means)+np.array(lm_full_stds), alpha=0.1, color='darkorange')
    plt.plot(agent_counts, lm_lorasa_means, 'o-', label="LightMeta (Adapted - LoRASA)", color='cyan')
    plt.fill_between(agent_counts, np.array(lm_lorasa_means)-np.array(lm_lorasa_stds), np.array(lm_lorasa_means)+np.array(lm_lorasa_stds), alpha=0.2, color='cyan')

    for model in ["MAML", "MAPPO", "IPPO"]:
        if model == "MAML":
            zero_means = [np.mean([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            zero_stds = [np.std([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            adapted_means = [np.mean([r["in_distribution"][model][n][1] for r in all_runs_results]) for n in agent_counts]
            adapted_stds = [np.std([r["in_distribution"][model][n][1] for r in all_runs_results]) for n in agent_counts]
            plt.plot(agent_counts, zero_means, 's--', label="MAML (Zero-Shot)", color='green')
            plt.fill_between(agent_counts, np.array(zero_means)-np.array(zero_stds), np.array(zero_means)+np.array(zero_stds), alpha=0.1, color='green')
            plt.plot(agent_counts, adapted_means, 's-', label="MAML (Adapted)", color='red')
            plt.fill_between(agent_counts, np.array(adapted_means)-np.array(adapted_stds), np.array(adapted_means)+np.array(adapted_stds), alpha=0.1, color='red')
        else: 
            color = 'purple' if model == 'MAPPO' else 'saddlebrown'
            means = [np.mean([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            stds = [np.std([r["in_distribution"][model][n][0] for r in all_runs_results]) for n in agent_counts]
            plt.plot(agent_counts, means, '^-', label=model, color=color)
            plt.fill_between(agent_counts, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.1, color=color)
        
    plt.title(f"In-Distribution Performance on {env_name} (Aggregated over {len(SEEDS)} Seeds)", fontsize=16)
    plt.xlabel("Number of Agents", fontsize=14); plt.ylabel("Average Total Reward", fontsize=14)
    plt.xticks(agent_counts); plt.legend(fontsize=12); plt.grid(True); plt.tight_layout(); plt.show()

    # --- Sample Efficiency Plot ---
    plt.figure(figsize=(12, 8))
    for model in ["LightMeta", "MAPPO"]:
        if model in all_runs_results[0]["learning_curves"]:
            curves = [r["learning_curves"][model] for r in all_runs_results]
            all_steps = sorted(list(set(step for curve in curves for step, _ in curve)))
            aligned_curves = []
            for curve in curves:
                if not curve: continue
                steps, rewards = zip(*curve)
                aligned_rewards = np.interp(all_steps, steps, rewards)
                aligned_curves.append(aligned_rewards)

            if aligned_curves:
                mean_curve = np.mean(aligned_curves, axis=0)
                std_curve = np.std(aligned_curves, axis=0)
                plt.plot(all_steps, mean_curve, label=model)
                plt.fill_between(all_steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.title(f"Sample Efficiency on {env_name}", fontsize=16)
    plt.xlabel("Environment Steps", fontsize=14); plt.ylabel("Evaluation Reward", fontsize=14)
    plt.legend(fontsize=12); plt.grid(True); plt.tight_layout(); plt.show()

    # --- OOD, Adaptation Gain, and Compute Tables ---
    if not USE_SIMPLE_SPREAD:
        print("\n" + "="*20 + " OUT-OF-DISTRIBUTION (OOD) RESULTS " + "="*20)
        print(f"{'Model':<20} | {'Mean Reward':<15} | {'Std Dev':<15}")
        print("-"*55)
        for model in model_names:
            if "out_of_distribution" in all_runs_results[0] and model in all_runs_results[0]["out_of_distribution"]:
                rewards = [r["out_of_distribution"][model][0] for r in all_runs_results]
                print(f"{model:<20} | {np.mean(rewards):<15.2f} | {np.std(rewards):<15.2f}")
        
        print("\n" + "="*20 + " ADAPTATION GAIN (N=5 AGENTS) " + "="*20)
        print(f"{'Model':<25} | {'Gain':<15}")
        print("-"*45)
        lm_zero = [r["in_distribution"]["LightMeta"][5][0] for r in all_runs_results]
        lm_lorasa = [r["in_distribution"]["LightMeta"][5][2] for r in all_runs_results]
        maml_zero = [r["in_distribution"]["MAML"][5][0] for r in all_runs_results]
        maml_adapted = [r["in_distribution"]["MAML"][5][1] for r in all_runs_results]
        print(f"{'LightMeta (LoRASA)':<25} | {np.mean(lm_lorasa) - np.mean(lm_zero):<15.2f}")
        print(f"{'MAML':<25} | {np.mean(maml_adapted) - np.mean(maml_zero):<15.2f}")

    print("\n" + "="*20 + " STATISTICAL SIGNIFICANCE (N=5 AGENTS) " + "="*20)
    lm_raw = [item for r in all_runs_results for item in r["in_distribution"]["LightMeta"][5][3]]
    mappo_raw = [item for r in all_runs_results for item in r["in_distribution"]["MAPPO"][5][1]]
    if len(lm_raw) == len(mappo_raw) and len(lm_raw) > 0:
        stat, p = wilcoxon(lm_raw, mappo_raw)
        print(f"LightMeta (Adapted-LoRASA) vs. MAPPO: p-value = {p:.4f}")
    else:
        print("Could not compute significance test (unequal samples or no data).")


    print("\n" + "="*20 + " COMPUTE & TRAINING TIME METRICS " + "="*20)
    print(f"{'Model':<25} | {'Parameters':<20} | {'Training Time (s)':<20}")
    print("-"*70)
    all_model_names = model_names + ["LightMeta_LoRASA"]
    for model in all_model_names:
        if model in all_runs_results[0]["compute_metrics"]:
            params = [r["compute_metrics"][model][0] for r in all_runs_results]
            times = [r["compute_metrics"][model][1] for r in all_runs_results]
            
            if model == "LightMeta_LoRASA":
                full_params = [r["compute_metrics"]["LightMeta"][0] for r in all_runs_results]
                param_str = f"{np.mean(params):.0f} ({np.mean(params)/np.mean(full_params):.2%})"
                time_str = "N/A (part of FT)"
                display_name = "LightMeta (LoRASA Adapters)"
            else:
                param_str = f"{np.mean(params):.0f}"
                time_str = f"{np.mean(times):.2f} ± {np.std(times):.2f}"
                display_name = model

            print(f"{display_name:<25} | {param_str:<20} | {time_str:<20}")

if __name__ == "__main__":
    max_ss_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    ss_action_dim = SimpleSpreadWrapper(num_agents=2).action_dim
    
    ENV_CONFIGS = {
        "SimpleSpread": {"name": "Simple Spread", "fn": SimpleSpreadWrapper, "agent_obs_dim": max_ss_obs_dim, "num_actions": ss_action_dim},
        "CustomEnv": {"name": "Custom Environment", "fn": CustomMultiAgentEnv, "agent_obs_dim": 4, "num_actions": 2}
    }
    UNSEEN_ENV_CONFIG = { "UnseenCustomEnv": {"name": "Unseen Custom", "fn": UnseenCustomMultiAgentEnv} }

    config = ENV_CONFIGS["SimpleSpread"] if USE_SIMPLE_SPREAD else ENV_CONFIGS["CustomEnv"]
    unseen_config = UNSEEN_ENV_CONFIG["UnseenCustomEnv"]

    with mp.Pool(processes=NUM_WORKERS) as pool:
        args_for_pool = [(seed, config, unseen_config) for seed in SEEDS]
        all_runs_results = pool.map(run_comparison, args_for_pool)
    
    process_and_display_results(all_runs_results, config["name"])
