import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3
import time
import copy
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- Constants ---
MAX_AGENTS = 5
MAX_OBS_DIM = 4 + 2 * MAX_AGENTS + 2 * (MAX_AGENTS - 1)  # 22
NUM_MPE_ACTIONS = 5  # simple_spread_v3 has 5 discrete actions per agent


# --- MPE Environment Wrapper ---
class MPEWrapper:
    def __init__(self, num_agents=5, seed=None):
        if num_agents > MAX_AGENTS:
            raise ValueError(f"num_agents ({num_agents}) cannot exceed MAX_AGENTS ({MAX_AGENTS})")

        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=200, continuous_actions=False)
        self.num_agents = num_agents
        self.seed = seed
        self.possible_agents = self.env.possible_agents
        self.agents = []
        self.current_obs_dim_per_agent = 4 + 2 * num_agents + 2 * (num_agents - 1)
        self.obs_dict = None
        self.reset()

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else self.seed
        observations, infos = self.env.reset(seed=effective_seed, options=options)
        self.obs_dict = observations
        self.agents = list(self.obs_dict.keys())
        return self._flatten_obs(), infos

    def step(self, actions):
        if len(actions) != self.num_agents:
            if len(actions) != len(self.agents):
                raise ValueError(f"Expected {len(self.agents)} actions for agents {self.agents}, got {len(actions)}")
            action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        else:
            action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}

        next_obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict = self.env.step(action_dict)
        self.obs_dict = next_obs_dict
        self.agents = list(self.obs_dict.keys())
        num_active_agents_for_reward = len(rewards_dict)
        reward = sum(rewards_dict.values()) / num_active_agents_for_reward if num_active_agents_for_reward > 0 else 0
        terminated = any(terminateds_dict.values())
        truncated = any(truncateds_dict.values())
        return self._flatten_obs(), reward, terminated, truncated, infos_dict

    def _flatten_obs(self):
        padded_obs = np.zeros((MAX_AGENTS * MAX_OBS_DIM,), dtype=np.float32)
        agent_list_for_padding = [f'agent_{i}' for i in range(MAX_AGENTS)]
        for i, agent_key in enumerate(agent_list_for_padding):
            obs = self.obs_dict.get(agent_key, None)
            if obs is not None and isinstance(obs, np.ndarray):
                obs_len = len(obs)
                copy_len = min(MAX_OBS_DIM, obs_len)
                start_idx = i * MAX_OBS_DIM
                end_idx = start_idx + copy_len
                padded_obs[start_idx:end_idx] = obs[:copy_len]
        return padded_obs

    def close(self):
        self.env.close()


def mpe_env_fn(num_agents=5, seed=None):
    return MPEWrapper(num_agents=min(num_agents, MAX_AGENTS), seed=seed)


# --- LightMetaPolicy ---
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if input_dim % MAX_AGENTS != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by MAX_AGENTS ({MAX_AGENTS})")
        self.agent_dim = input_dim // MAX_AGENTS
        self.key_transform = nn.Linear(self.agent_dim, 64)
        self.query_transform = nn.Linear(self.agent_dim, 64)
        self.value_transform = nn.Linear(self.agent_dim, 128)
        self.agent_relation = nn.Linear(64, 64)
        self.post_attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        agents = x.reshape(batch_size, MAX_AGENTS, self.agent_dim)
        keys = self.key_transform(agents)
        queries = self.query_transform(agents)
        values = self.value_transform(agents)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights, values)
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.1).float()
        masked_context = context * agent_mask
        pooled = masked_context.sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8)
        action_logits = self.post_attention(pooled)
        value = self.value_head(pooled)
        if batch_size == 1:
            action_logits = action_logits.squeeze(0)
            value = value.squeeze(0)
        return action_logits, value


# --- Training ---
def train_light_meta_policy(model, env_factory_fn, meta_iterations=300, inner_rollouts=40, gamma=0.99,
                            entropy_coef=0.01, value_loss_coef=0.5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_actions_per_agent = NUM_MPE_ACTIONS
    for iteration in range(meta_iterations):
        if iteration < meta_iterations * 0.25:
            num_agents = np.random.choice([1, 2])
        elif iteration < meta_iterations * 0.33:
            num_agents = np.random.choice([2, 3])
        else:
            num_agents = np.random.choice([4, 5])
        total_loss = 0
        batch_returns = []
        for _ in range(2):
            env = env_factory_fn(num_agents=num_agents, seed=iteration * 10 + _)
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            log_probs_ep, rewards_ep, values_ep, entropies_ep, dones_ep = [], [], [], [], []
            for _ in range(inner_rollouts):
                action_logits, value = model(obs)
                logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)
                dist = torch.distributions.Categorical(logits=logits_reshaped)
                actions_all = dist.sample()
                actions_to_step = actions_all[:env.num_agents].numpy()
                log_prob_active = dist.log_prob(actions_all)[:env.num_agents].mean()
                entropy_active = dist.entropy()[:env.num_agents].mean()
                next_obs, reward, terminated, truncated, _ = env.step(actions_to_step)
                done = terminated or truncated
                log_probs_ep.append(log_prob_active)
                rewards_ep.append(reward)
                values_ep.append(value)
                entropies_ep.append(entropy_active)
                dones_ep.append(done)
                obs = torch.tensor(next_obs, dtype=torch.float32)
                if done:
                    break
            returns = []
            final_value = 0.0
            if not dones_ep[-1]:
                with torch.no_grad():
                    _, final_value_tensor = model(obs)
                    final_value = final_value_tensor.item()
            discounted_reward = final_value
            for r, d in zip(reversed(rewards_ep), reversed(dones_ep)):
                discounted_reward = r + gamma * discounted_reward * (1 - int(d))
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns, dtype=torch.float32)
            batch_returns.append(returns[0].item())
            values_ep = torch.stack(values_ep).squeeze(-1)
            log_probs_ep = torch.stack(log_probs_ep)
            entropies_ep = torch.stack(entropies_ep)
            advantages = returns - values_ep.detach()
            policy_loss = -(log_probs_ep * advantages).mean()
            value_loss = nn.MSELoss()(values_ep, returns)
            entropy_bonus = entropies_ep.mean()
            task_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus
            total_loss += task_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if iteration % 10 == 0 and batch_returns:
            print(f"Light Meta-Iter {iteration} | Avg Start Return: {np.mean(batch_returns):.2f} | Total Loss: {total_loss.item():.4f}")
    return model


# --- Fine-Tuning ---
def fine_tune_light_meta_policy(model, env_factory_fn, max_steps=50, episodes_per_update=5, steps_per_episode=50,
                                batch_size=32, gamma=0.99, value_loss_coef=0.5, entropy_coef=0.01):
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0005)
    num_actions_per_agent = NUM_MPE_ACTIONS
    env = env_factory_fn()
    num_agents = env.num_agents
    experience_buffer = []
    for step in range(max_steps):
        current_ep = 0
        while current_ep < episodes_per_update:
            obs, _ = env.reset()
            done = False
            episode_steps = 0
            while not done and episode_steps < steps_per_episode:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    action_logits, _ = tuned_model(obs_tensor)
                    logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)
                    dist = torch.distributions.Categorical(logits=logits_reshaped)
                    actions_all = dist.sample()
                    actions_indices = actions_all[:num_agents].cpu().numpy()
                next_obs, reward, terminated, truncated, _ = env.step(actions_indices)
                done = terminated or truncated
                experience_buffer.append((obs, actions_indices, reward, next_obs, done))
                obs = next_obs
                episode_steps += 1
            current_ep += 1
        max_buffer_size = 1000
        if len(experience_buffer) > max_buffer_size:
            experience_buffer = experience_buffer[-max_buffer_size:]
        if len(experience_buffer) < batch_size:
            continue
        batch_indices = np.random.choice(len(experience_buffer), batch_size, replace=False)
        batch = [experience_buffer[i] for i in batch_indices]
        batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
        batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
        batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
        batch_next_obs = torch.tensor(np.array([s[3] for s in batch]), dtype=torch.float32)
        batch_dones = torch.tensor([s[4] for s in batch], dtype=torch.float32)
        tuned_model.train()
        action_logits, values = tuned_model(batch_obs)
        logits_reshaped = action_logits.view(batch_size, MAX_AGENTS, num_actions_per_agent)
        logits_active = logits_reshaped[:, :num_agents, :]
        dist_active = torch.distributions.Categorical(logits=logits_active)
        log_probs = dist_active.log_prob(batch_actions)
        mean_log_probs = log_probs.mean(dim=1)
        with torch.no_grad():
            tuned_model.eval()
            _, next_values = tuned_model(batch_next_obs)
            tuned_model.train()
        targets = batch_rewards + gamma * next_values.squeeze(-1) * (1 - batch_dones)
        advantages = targets - values.squeeze(-1).detach()
        policy_loss = -(mean_log_probs * advantages.detach()).mean()
        value_loss = nn.MSELoss()(values.squeeze(-1), targets.detach())
        entropy_bonus = dist_active.entropy().mean(dim=1).mean()
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned_model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 10 == 0:
            print(f"Fine-tune step {step} | Loss: {loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
    env.close()
    return tuned_model


# --- Evaluation ---
def evaluate_meta_policy(model, env_factory_fn, episodes=10):
    total_rewards = []
    num_actions_per_agent = NUM_MPE_ACTIONS
    for ep in range(episodes):
        env = env_factory_fn(seed=42 + ep)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
        done = False
        step_count = 0
        max_eval_steps = 200
        while not done and step_count < max_eval_steps:
            with torch.no_grad():
                action_logits, _ = model(obs)
                logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)
                dist = torch.distributions.Categorical(logits=logits_reshaped)
                actions_all = dist.sample()
                actions = actions_all[:env.num_agents].numpy()
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
            step_count += 1
        total_rewards.append(total_reward)
        env.close()
    return np.mean(total_rewards) if total_rewards else 0


# --- Random Policy ---
def evaluate_random_policy(env_factory_fn, episodes=10):
    total_rewards = []
    for ep in range(episodes):
        env = env_factory_fn(seed=100 + ep)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_eval_steps = 200
        while not done and step_count < max_eval_steps:
            actions = np.random.randint(0, NUM_MPE_ACTIONS, size=env.num_agents)
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        total_rewards.append(total_reward)
        env.close()
    return np.mean(total_rewards) if total_rewards else 0


# --- Benchmark with Table ---
def benchmark_lightmeta():
    input_dim = MAX_OBS_DIM * MAX_AGENTS
    output_dim = NUM_MPE_ACTIONS * MAX_AGENTS
    model = LightMetaPolicy(input_dim, output_dim)
    print(f"Instantiated LightMetaPolicy for MPE:")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  Agent Obs Dim (Internal): {model.agent_dim}")
    print("\nTraining LightMetaPolicy on MPE Simple Spread...")
    start_time = time.time()
    trained_model = train_light_meta_policy(model, mpe_env_fn, meta_iterations=100, inner_rollouts=50)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    agent_counts = [1, 2, 3, 4, 5]
    results = []
    print("\n=== MPE Simple Spread Evaluation ===")
    for num_agents in agent_counts:
        eval_env_factory = lambda seed=None: mpe_env_fn(num_agents=num_agents, seed=seed)
        print(f"\n--- Evaluating for {num_agents} Agents ---")
        zero_shot = evaluate_meta_policy(trained_model, eval_env_factory)
        adapted_model = fine_tune_light_meta_policy(
            trained_model, lambda: mpe_env_fn(num_agents=num_agents, seed=42), max_steps=30
        )
        adapted = evaluate_meta_policy(adapted_model, eval_env_factory)
        random_baseline = evaluate_random_policy(eval_env_factory)
        results.append({
            "Agents": num_agents,
            "LightMeta Zero-Shot": round(zero_shot, 2),
            "LightMeta Adapted": round(adapted, 2),
            "Random Policy": round(random_baseline, 2)
        })
        print(f"Agents: {num_agents} | Zero-Shot: {zero_shot:.2f} | Adapted: {adapted:.2f} | Random: {random_baseline:.2f}")
    df = pd.DataFrame(results)
    print("\n=== Evaluation Results Table ===")
    print(df.to_string(index=False))
    df.to_csv("lightmeta_mpe_results_with_random.csv", index=False)


if __name__ == "__main__":
    benchmark_lightmeta()
