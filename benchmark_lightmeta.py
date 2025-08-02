import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
import copy
import logging

# Ray and RLlib
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

# PettingZoo and SuperSuit
from pettingzoo.mpe import simple_spread_v3
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

# Stable Baselines3
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- Setup and Configuration ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- Benchmark Constants ---
MAX_AGENTS = 5
TOTAL_TIMESTEPS = 500_000
EVAL_EPISODES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Utility Functions ---
def flatten_obs(obs_dict, all_agents, obs_size):
    flat = np.zeros((MAX_AGENTS, obs_size), dtype=np.float32)
    for i, agent in enumerate(all_agents):
        if agent in obs_dict:
            flat[i, :len(obs_dict[agent])] = obs_dict[agent]
    return flat.flatten()

# --- Custom Meta-Policy Definition ---
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, action_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_layer = nn.Linear(self.agent_dim, 64)
        self.query_layer = nn.Linear(self.agent_dim, 64)
        self.value_layer = nn.Linear(self.agent_dim, 128)
        self.ff_block = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.policy_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, MAX_AGENTS * action_dim))
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.shape[0] if x.dim() > 1 else 1
        x = x.reshape(batch_size, MAX_AGENTS, self.agent_dim)
        keys, queries, values = self.key_layer(x), self.query_layer(x), self.value_layer(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention, values)
        mask = (x.abs().sum(-1, keepdim=True) > 1e-6).float()
        pooled = (context * mask).sum(1) / (mask.sum(1) + 1e-8)
        processed = self.ff_block(pooled)
        action_logits = self.policy_head(processed).view(batch_size, MAX_AGENTS, -1)
        value = self.value_head(processed)
        return action_logits, value

# --- Training and Evaluation Functions ---

def train_lightmeta_from_scratch(policy, env_gen, timesteps=TOTAL_TIMESTEPS, ppo_epochs=4, lr=3e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01):
    print(f"--- Training LightMetaPolicy for {timesteps} steps ---")
    policy.to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_agents = [f"agent_{i}" for i in range(MAX_AGENTS)]
    obs_size = env_gen().observation_space(all_agents[0]).shape[0]
    
    total_steps = 0
    history = {"steps": [], "reward": []}
    ep_reward_buffer = []
    
    start_time = time.time()
    while total_steps < timesteps:
        env = env_gen()
        obs_dict, _ = env.reset()
        ep_obs, ep_actions, ep_log_probs, ep_rewards, ep_values = [], [], [], [], []
        
        while env.agents:
            flat_obs = flatten_obs(obs_dict, all_agents, obs_size)
            obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = policy(obs_tensor)
            
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum()
            acts_np = actions.squeeze().detach().cpu().numpy()
            action_dict = {agent: acts_np[i] for i, agent in enumerate(all_agents) if agent in env.agents}
            
            next_obs_dict, rewards, _, _, _ = env.step(action_dict)
            
            ep_obs.append(flat_obs)
            ep_actions.append(acts_np)
            ep_log_probs.append(log_prob.item())
            ep_rewards.append(sum(rewards.values()))
            ep_values.append(value.item())
            obs_dict = next_obs_dict

        total_steps += len(ep_rewards)
        ep_reward_buffer.append(sum(ep_rewards))
        if len(ep_reward_buffer) > 100: ep_reward_buffer.pop(0)

        # PPO Update Logic
        ep_values.append(0)
        advantages = []
        gae = 0
        for i in reversed(range(len(ep_rewards))):
            delta = ep_rewards[i] + gamma * ep_values[i+1] - ep_values[i]
            gae = delta + gamma * 0.95 * gae
            advantages.insert(0, gae)
        
        returns = np.array(advantages) + np.array(ep_values[:-1])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        obs_t = torch.tensor(np.array(ep_obs), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(np.array(ep_actions), dtype=torch.long, device=DEVICE)
        old_log_probs_t = torch.tensor(ep_log_probs, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        for _ in range(ppo_epochs):
            action_logits, values = policy(obs_t)
            values = values.squeeze()
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions_t).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns_t)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not history["steps"] or total_steps - history["steps"][-1] > timesteps // 50:
            mean_rew = np.mean(ep_reward_buffer) if ep_reward_buffer else float('nan')
            history["steps"].append(total_steps)
            history["reward"].append(mean_rew)
            print(f"LightMeta Steps: {total_steps}/{timesteps} | Avg Reward (100 ep): {mean_rew:.2f}")

    print(f"Finished training in {(time.time() - start_time) / 60:.1f} minutes.")
    return policy, history

def run_rllib(alg_name, env_gen, timesteps=TOTAL_TIMESTEPS):
    print(f"--- Training {alg_name.upper()} for {timesteps} steps ---")
    tune.register_env("spread_env", lambda _: PettingZooEnv(env_gen(aec=True)))
    
    config = (
        PPOConfig()
        .environment("spread_env")
        .framework("torch")
        .env_runners(num_env_runners=1, rollout_fragment_length=200)
        .training(train_batch_size=4000)
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .multi_agent(
            policies={"shared_policy"},
            ## FIXED ## - Updated lambda signature to match modern RLlib API.
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
        )
        .debugging(log_level="ERROR")
    )
    
    algo = config.build()
    history = {"steps": [], "reward": []}
    start_time = time.time()
    
    total_steps = 0
    while total_steps < timesteps:
        result = algo.train()
        total_steps = result.get("timesteps_total", 0)
        mean_reward = result.get("episode_reward_mean", float('nan'))
        
        if not history["steps"] or total_steps - history["steps"][-1] > timesteps // 50:
            history["steps"].append(total_steps)
            history["reward"].append(mean_reward)
            print(f"{alg_name.upper()} Steps: {total_steps}/{timesteps} | Avg Reward: {mean_reward:.2f}")

    print(f"Finished training in {(time.time() - start_time) / 60:.1f} minutes.")
    return algo, history

## FIXED ## - Replaced the entire callback class with a compatible version.
class SB3RewardCallback(BaseCallback):
    def __init__(self, check_freq, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.history = {"steps": [], "reward": []}
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check for episode termination for each vectorized environment
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                # Retrieve episode reward from the info dictionary when an episode ends
                ep_info = self.locals["infos"][idx].get("episode")
                if ep_info:
                    self.episode_rewards.append(ep_info["r"])

        # Log the mean reward at the specified frequency
        if self.n_calls % self.check_freq == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards)
            self.history["steps"].append(self.num_timesteps)
            self.history["reward"].append(mean_reward)
            print(f"IPPO Steps: {self.num_timesteps}/{self.model.num_timesteps} | Avg Reward (last batch): {mean_reward:.2f}")
            self.episode_rewards = [] # Clear the buffer after logging
            
        return True

def run_sb3(env_gen, timesteps=TOTAL_TIMESTEPS):
    print(f"--- Training IPPO (SB3) for {timesteps} steps ---")
    vec_env = pettingzoo_env_to_vec_env_v1(env_gen(parallel=True))
    env = concat_vec_envs_v1(vec_env, 1, base_class='stable_baselines3')

    callback = SB3RewardCallback(check_freq=timesteps // 50)
    model = SB3PPO("MlpPolicy", env, verbose=0, device=DEVICE)
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    print(f"Finished training in {(time.time() - start_time) / 60:.1f} minutes.")
    return model, callback.history

def evaluate_policy(agent, env_gen, episodes=EVAL_EPISODES):
    # This evaluation function remains largely the same, but simplified the done check
    total_rewards = []
    agent_type = "unknown"
    if isinstance(agent, LightMetaPolicy): agent_type = "lightmeta"
    elif isinstance(agent, ray.rllib.algorithms.Algorithm): agent_type = "rllib"
    elif isinstance(agent, SB3PPO): agent_type = "sb3"

    for _ in range(episodes):
        if agent_type == "sb3":
            eval_env = concat_vec_envs_v1(pettingzoo_env_to_vec_env_v1(env_gen(parallel=True)), 1, base_class='stable_baselines3')
            obs = eval_env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                ep_reward += reward.sum()
        else:
            env = env_gen() if agent_type == "lightmeta" else PettingZooEnv(env_gen(aec=True))
            obs, _ = env.reset()
            ep_reward = 0
            all_agents = [f"agent_{i}" for i in range(MAX_AGENTS)]
            obs_size = env_gen().observation_space(all_agents[0]).shape[0]

            while True:
                if not obs: break # Break if the agent dict is empty
                if agent_type == "lightmeta":
                    flat_obs = flatten_obs(obs, all_agents, obs_size)
                    obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = agent(obs_tensor)
                        actions_np = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                    actions = {agent_id: actions_np[i] for i, agent_id in enumerate(all_agents) if agent_id in obs}
                else: # rllib
                    actions = agent.compute_actions(obs, policy_id="shared_policy")
                
                obs, rewards, terminated, truncated, _ = env.step(actions)
                ep_reward += sum(rewards.values())
                
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break
                
        total_rewards.append(ep_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

def plot_learning_curves(histories):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    for name, hist in histories.items():
        steps, rewards = [], []
        for s, r in zip(hist['steps'], hist['reward']):
            if r is not None and not np.isnan(r):
                steps.append(s)
                rewards.append(r)
        plt.plot(steps, rewards, label=name, alpha=0.8)
    
    plt.xlabel("Environment Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.title("Learning Curve Comparison", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    env_gen = lambda parallel=False, aec=False: simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5) if parallel else simple_spread_v3.env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5) if aec else simple_spread_v3.parallel_env(N=MAX_AGENTS, max_cycles=25, local_ratio=0.5)

    # --- Train All Algorithms ---
    lightmeta_policy = LightMetaPolicy(MAX_AGENTS * env_gen().observation_space("agent_0").shape[0])
    trained_lightmeta, lightmeta_hist = train_lightmeta_from_scratch(lightmeta_policy, env_gen)
    
    if ray.is_initialized(): ray.shutdown()
    ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
    mappo_algo, mappo_hist = run_rllib("mappo", env_gen, timesteps=TOTAL_TIMESTEPS)
    ray.shutdown()

    ippo_model, ippo_hist = run_sb3(env_gen, timesteps=TOTAL_TIMESTEPS)

    histories = {
        "LightMeta": lightmeta_hist,
        "MAPPO (RLlib)": mappo_hist,
        "IPPO (SB3)": ippo_hist,
    }

    # --- Evaluate All Algorithms ---
    print("\n--- Final Evaluation ---")
    lightmeta_mean, lightmeta_std = evaluate_policy(trained_lightmeta, env_gen)
    mappo_mean, mappo_std = evaluate_policy(mappo_algo, env_gen)
    ippo_mean, ippo_std = evaluate_policy(ippo_model, env_gen)

    # --- Final Results ---
    results = [
        ("LightMeta", lightmeta_mean, lightmeta_std),
        ("MAPPO (RLlib)", mappo_mean, mappo_std),
        ("IPPO (SB3)", ippo_mean, ippo_std),
    ]

    print("\n" + "="*60)
    print("== Final Performance Comparison ==")
    print("="*60)
    print(f"{'Model':<25} {'Mean Reward':>15} {'Std Dev':>10}")
    print("-"*60)
    for name, mean, std in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<25} {mean:15.2f} {std:10.2f}")
    print("="*60)

    # --- Plotting ---
    plot_learning_curves(histories)

    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color='c', edgecolor='black')
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.title(f"Algorithm Performance Comparison ({TOTAL_TIMESTEPS} steps)", fontsize=16)
    plt.xticks(rotation=15, ha="right")
    min_mean = min(means) if means else 0
    plt.ylim(bottom=min_mean + min_mean * 0.2 if min_mean < 0 else 0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom' if yval >= 0 else 'top', ha='center')
    plt.tight_layout()
    plt.show()