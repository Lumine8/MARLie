import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import matplotlib.pyplot as plt
import time
import collections
import multiprocessing as mp
from pettingzoo.mpe import simple_spread_v3

# --- SCRIPT CONFIGURATION ---
# Define all the different configurations you want to test.
TUNING_CONFIGS = {
    "baseline": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": False,
        "use_relational_bias": True, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": False,
    },
    "with_layernorm": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": True,
        "use_relational_bias": True, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": False,
    },
    "no_relational_bias": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": False,
        "use_relational_bias": False, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": False,
    },
    "with_residual": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": False,
        "use_relational_bias": True, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": True,
    },
    "layernorm_and_residual": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": True,
        "use_relational_bias": True, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": True,
    },
    # --- New Targeted Experiments for Simple Spread ---
    "champion_no_bias": {
        "lr": 0.0003, "entropy_coef": 0.01, "use_layer_norm": True,
        "use_relational_bias": False, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": True,
    },
    "champion_no_bias_high_entropy": {
        "lr": 0.0003, "entropy_coef": 0.05, "use_layer_norm": True,
        "use_relational_bias": False, "training_mode": "meta", "inner_rollouts": 40,
        "gamma": 0.99, "num_heads": 1, "use_residual": True,
    },
}

SEEDS = [42, 123, 888, 2024, 7]
NUM_WORKERS = 6
MAX_AGENTS = 5
EPISODES = 20
META_ITERATIONS = 500

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
        self.agent_positions = local_random_state.randint(20, 580, (self.num_agents, 2))
        self.goal_zones = local_random_state.randint(50, 550, (self.num_agents, 2))
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
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
            self.agent_positions[move_mask] += np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
        for i in range(self.num_agents):
            rewards[i] += max(0, 100 - np.linalg.norm(self.agent_positions[i] - self.goal_zones[i])) / 100
        self.steps += 1; terminated = self.steps >= self.max_steps
        obs = self.agent_states.flatten()
        padded_obs = np.zeros((MAX_AGENTS * 4,), dtype=np.float32); padded_obs[:len(obs)] = obs
        return padded_obs, np.sum(rewards) / self.num_agents, terminated, False, {}

class UnseenCustomMultiAgentEnv(CustomMultiAgentEnv):
    def randomize_environment(self):
        super().randomize_environment()
        local_random_state = np.random.RandomState(self.seed)
        self.move_range = local_random_state.randint(25, 50)
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

# --- Model ---
class LightMetaPolicy(nn.Module):
    def __init__(self, agent_obs_dim, num_actions, use_layer_norm=False, use_relational_bias=True, num_heads=1, use_residual=False):
        super().__init__()
        self.agent_dim, self.num_actions = agent_obs_dim, num_actions
        self.use_layer_norm, self.use_relational_bias = use_layer_norm, use_relational_bias
        self.num_heads, self.use_residual = num_heads, use_residual
        self.d_model = 64
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads
        self.key_transform = nn.Linear(self.agent_dim, self.d_model)
        self.query_transform = nn.Linear(self.agent_dim, self.d_model)
        self.value_transform = nn.Linear(self.agent_dim, self.d_model)
        if self.use_relational_bias: self.agent_relation = nn.Linear(self.d_model, self.d_model)
        if self.use_layer_norm: self.layer_norm = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        output_logit_dim = MAX_AGENTS * self.num_actions
        self.post_attention_fc1 = nn.Linear(self.d_model, self.d_model)
        self.post_attention_act = nn.GELU()
        self.post_attention_fc2 = nn.Linear(self.d_model, output_logit_dim)
        self.value_head = nn.Sequential(nn.Linear(self.d_model, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        agents = x.view(batch_size, MAX_AGENTS, self.agent_dim)
        queries = self.query_transform(agents).view(batch_size, MAX_AGENTS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key_transform(agents).view(batch_size, MAX_AGENTS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value_transform(agents).view(batch_size, MAX_AGENTS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.use_relational_bias:
            rel_queries = queries.permute(0, 2, 1, 3).reshape(batch_size, MAX_AGENTS, self.d_model)
            rel_bias_logits = self.agent_relation(rel_queries).view(batch_size, MAX_AGENTS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attention += torch.matmul(rel_bias_logits, queries.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values).permute(0, 2, 1, 3).reshape(batch_size, MAX_AGENTS, self.d_model)
        context = self.fc_out(context)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.01).float()
        pooled = (context * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        if self.use_layer_norm: pooled = self.layer_norm(pooled)
        ff_out = self.post_attention_fc1(pooled)
        if self.use_residual: ff_out = ff_out + pooled
        ff_out = self.post_attention_act(ff_out)
        action_logits = self.post_attention_fc2(ff_out).view(batch_size, MAX_AGENTS, self.num_actions)
        value = self.value_head(pooled)
        return action_logits, value

# --- Worker Function for Multiprocessing ---
def run_single_seed(args):
    config_name, config, seed, all_env_configs = args
    print(f"  Starting seed {seed} for {config_name}...")
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    
    seed_results = {}

    # --- Train and Evaluate for each domain (Custom and Simple Spread) ---
    for domain in ["Custom", "SimpleSpread"]:
        if domain == "Custom":
            train_env_config = all_env_configs["CustomEnv"]
            eval_env_configs = {
                "Custom Env (ID)": all_env_configs["CustomEnv"],
                "Unseen Env (OOD)": all_env_configs["UnseenEnv"]
            }
        else: # SimpleSpread
            train_env_config = all_env_configs["SimpleSpread"]
            eval_env_configs = {"Simple Spread": all_env_configs["SimpleSpread"]}

        train_env_fn = train_env_config["fn"]
        model = LightMetaPolicy(
            agent_obs_dim=train_env_config["agent_obs_dim"], 
            num_actions=train_env_config["num_actions"],
            use_layer_norm=config["use_layer_norm"],
            use_relational_bias=config.get("use_relational_bias", True),
            num_heads=config.get("num_heads", 1),
            use_residual=config.get("use_residual", False)
        )
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
        for iteration in range(META_ITERATIONS):
            if config.get("training_mode", "meta") == 'fixed_n': num_agents = 3
            else: num_agents = np.random.choice([2, 3, 4, 5])
            
            env = train_env_fn(num_agents=num_agents, seed=seed + iteration)
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            log_probs, rewards, values, entropies = [], [], [], []
            for _ in range(config["inner_rollouts"]):
                action_logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=action_logits[:, :num_agents, :])
                actions = dist.sample()
                next_obs, reward, terminated, _, _ = env.step(actions.numpy().flatten())
                log_probs.append(dist.log_prob(actions).mean()); rewards.append(reward); values.append(value); entropies.append(dist.entropy().mean())
                obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                if terminated: break
            returns = []
            discounted_reward = 0
            for r in reversed(rewards): discounted_reward = r + config["gamma"] * discounted_reward; returns.insert(0, discounted_reward)
            returns, values = torch.tensor(returns, dtype=torch.float32), torch.cat(values).squeeze()
            advantages = returns - values.detach()
            policy_loss = -(torch.stack(log_probs) * advantages).mean()
            value_loss = nn.MSELoss()(values, returns)
            loss = policy_loss + 0.5 * value_loss - config["entropy_coef"] * torch.stack(entropies).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # --- Evaluation for this domain ---
        for eval_name, eval_config in eval_env_configs.items():
            eval_env_fn = lambda seed=None: eval_config["fn"](num_agents=5, seed=seed)
            final_rewards = []
            for ep in range(EPISODES):
                env = eval_env_fn(seed=1000 + ep)
                obs, _ = env.reset()
                total_reward = 0; done = False
                while not done:
                    with torch.no_grad():
                        action_logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        actions = torch.argmax(action_logits[:, :env.num_agents, :], dim=-1).numpy().flatten()
                    obs, reward, terminated, _, _ = env.step(actions)
                    total_reward += reward; done = terminated
                final_rewards.append(total_reward)
            seed_results[eval_name] = np.mean(final_rewards)
    
    print(f"  Finished seed {seed} for {config_name}.")
    return seed_results

# --- Main Orchestrator ---
def train_and_evaluate_config(config_name, config, all_env_configs):
    print(f"\n--- Testing Config: {config_name} ---")
    args_for_pool = [(config_name, config, seed, all_env_configs) for seed in SEEDS]
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results_from_seeds = pool.map(run_single_seed, args_for_pool)
    
    eval_names_from_worker = ["Custom Env (ID)", "Unseen Env (OOD)", "Simple Spread"]
    aggregated_results = {eval_name: [] for eval_name in eval_names_from_worker}
    
    for seed_result in results_from_seeds:
        for eval_name, reward in seed_result.items():
            aggregated_results[eval_name].append(reward)
    
    final_means = {name: np.mean(rewards) for name, rewards in aggregated_results.items()}
    final_stds = {name: np.std(rewards) for name, rewards in aggregated_results.items()}
    return final_means, final_stds

# --- Main Execution ---
if __name__ == "__main__":
    max_ss_obs_dim = SimpleSpreadWrapper.get_max_obs_dim(MAX_AGENTS)
    ss_action_dim = SimpleSpreadWrapper(num_agents=2).action_dim
    
    ALL_ENV_CONFIGS = {
        "CustomEnv": {"fn": CustomMultiAgentEnv, "agent_obs_dim": 4, "num_actions": 2},
        "UnseenEnv": {"fn": UnseenCustomMultiAgentEnv},
        "SimpleSpread": {"fn": SimpleSpreadWrapper, "agent_obs_dim": max_ss_obs_dim, "num_actions": ss_action_dim}
    }

    results = {}
    for config_name, config in TUNING_CONFIGS.items():
        mean_rewards, std_rewards = train_and_evaluate_config(config_name, config, ALL_ENV_CONFIGS)
        results[config_name] = {"means": mean_rewards, "stds": std_rewards}

    # --- Print Final Summary ---
    print("\n\n" + "="*30 + " COMPREHENSIVE TUNING RESULTS " + "="*30)
    
    eval_names = ["Custom Env (ID)", "Unseen Env (OOD)", "Simple Spread"]
    ranks = {name: [] for name in TUNING_CONFIGS.keys()}
    for eval_name in eval_names:
        sorted_by_env = sorted(results.items(), key=lambda item: item[1]["means"][eval_name], reverse=True)
        for rank, (config_name, _) in enumerate(sorted_by_env):
            ranks[config_name].append(rank + 1)
    
    for config_name, config_ranks in ranks.items():
        results[config_name]["avg_rank"] = np.mean(config_ranks)

    sorted_results = sorted(results.items(), key=lambda item: item[1]["avg_rank"])
    
    header = f"{'Configuration':<35} | " + " | ".join([f'{name:<18}' for name in eval_names]) + f" | {'Avg Rank':<10}"
    print(header)
    print("-" * (len(header) + 2))
    for config_name, res in sorted_results:
        mean_scores = " | ".join([f'{res["means"][name]:<18.2f}' for name in eval_names])
        print(f"{config_name:<35} | {mean_scores} | {res['avg_rank']:<10.2f}")
