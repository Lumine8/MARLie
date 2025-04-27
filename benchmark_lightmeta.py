import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3
import time
import copy # For fine-tuning deepcopy

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- Constants ---
MAX_AGENTS = 5
# Calculate the observation dimension for the maximum number of agents in simple_spread_v3
# shape = (4 + 2*N + 2*(N - 1)) = 4 + 10 + 8 = 22
MAX_OBS_DIM = 4 + 2*MAX_AGENTS + 2*(MAX_AGENTS-1) # Should be 22
NUM_MPE_ACTIONS = 5 # simple_spread_v3 has 5 discrete actions per agent


# --- MPE Environment Wrapper (Corrected for Padding) ---
# --- MPE Environment Wrapper (Corrected for PettingZoo Parallel API) ---
class MPEWrapper:
    def __init__(self, num_agents=5, seed=None):
        if num_agents > MAX_AGENTS:
            raise ValueError(f"num_agents ({num_agents}) cannot exceed MAX_AGENTS ({MAX_AGENTS})")

        # Create the parallel environment
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=200, continuous_actions=False)
        self.num_agents = num_agents
        self.seed = seed
        # possible_agents is available after env creation in Parallel API
        self.possible_agents = self.env.possible_agents
        self.agents = [] # Will be populated after reset

        self.current_obs_dim_per_agent = 4 + 2*num_agents + 2*(num_agents-1)
        self.obs_dict = None

        # Initial reset called here, but obs assignment happens inside reset now
        self.reset()

    def reset(self, seed=None, options=None):
        # Parallel API reset returns obs and info dictionaries
        # Pass seed and options if the underlying env supports them via the wrapper
        effective_seed = seed if seed is not None else self.seed
        # Note: Seeding behavior can vary across PettingZoo versions. Check docs if needed.
        # Some versions might need options={'seed': effective_seed}
        observations, infos = self.env.reset(seed=effective_seed, options=options)

        self.obs_dict = observations
        # Update the list of active agents after reset
        self.agents = list(self.obs_dict.keys()) # Or self.env.agents if available and reliable after reset

        # Return padded obs and info dict
        return self._flatten_obs(), infos

    def step(self, actions):
        if len(actions) != self.num_agents:
             # Check if the number of actions matches the *current* number of active agents
             if len(actions) != len(self.agents):
                 raise ValueError(f"Expected {len(self.agents)} actions for agents {self.agents}, got {len(actions)}")
             action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        else:
             # Fallback if len(actions) == self.num_agents (initial number) but self.agents list changed
             action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)} # Still map to current agents

        # Parallel API step takes action dict and returns dicts
        next_obs_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict = self.env.step(action_dict)

        self.obs_dict = next_obs_dict
        # Update active agents list AFTER the step
        self.agents = list(self.obs_dict.keys()) # Agents remaining for the next observation

        # Aggregate results
        # Use self.possible_agents for averaging if rewards are sparse (agents might disappear)
        # Or average over currently active agents if appropriate
        num_active_agents_for_reward = len(rewards_dict) # Number of agents that returned a reward
        reward = sum(rewards_dict.values()) / num_active_agents_for_reward if num_active_agents_for_reward > 0 else 0

        # Use terminateds/truncateds dicts directly
        terminated = any(terminateds_dict.values())
        truncated = any(truncateds_dict.values())

        # Return standard 5-tuple
        return self._flatten_obs(), reward, terminated, truncated, infos_dict

    def _flatten_obs(self):
        # (Padding logic remains the same as before)
        padded_obs = np.zeros((MAX_AGENTS * MAX_OBS_DIM,), dtype=np.float32)

        if self.obs_dict is None:
             print("Warning: self.obs_dict is None during _flatten_obs")
             return padded_obs

        agent_list_for_padding = [f'agent_{i}' for i in range(MAX_AGENTS)]

        for i, agent_key in enumerate(agent_list_for_padding):
            # Use self.possible_agents? No, use agent_key constructed index.
            obs = self.obs_dict.get(agent_key, None) # Get obs if agent exists in current step

            if obs is not None and isinstance(obs, np.ndarray):
                obs_len = len(obs)
                copy_len = min(MAX_OBS_DIM, obs_len)
                start_idx = i * MAX_OBS_DIM
                end_idx = start_idx + copy_len
                try:
                    padded_obs[start_idx : end_idx] = obs[:copy_len]
                except Exception as e:
                    print(f"Error during padding assignment:")
                    print(f"  Index(i): {i}, Agent Key: {agent_key}, Obs Type: {type(obs)}, Obs Len: {obs_len}, Copy Len: {copy_len}, Target Slice: [{start_idx}:{end_idx}]")
                    raise e
            # If obs is None, slot remains zero.

        return padded_obs

    def close(self):
        self.env.close()

# --- Environment Factory ---
def mpe_env_fn(num_agents=5, seed=None):
    effective_num_agents = min(num_agents, MAX_AGENTS)
    return MPEWrapper(num_agents=effective_num_agents, seed=seed)


# --- LightMetaPolicy (Adapted for MPE) ---
class LightMetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim): # input_dim=110, output_dim=25
        super().__init__()
        if input_dim % MAX_AGENTS != 0:
             raise ValueError(f"input_dim ({input_dim}) must be divisible by MAX_AGENTS ({MAX_AGENTS})")
        self.agent_dim = input_dim // MAX_AGENTS # Expected observation dim per agent after padding (MAX_OBS_DIM = 22)

        # Layer sizes can be adjusted
        self.key_transform = nn.Linear(self.agent_dim, 64) # Increased size
        self.query_transform = nn.Linear(self.agent_dim, 64) # Increased size
        self.value_transform = nn.Linear(self.agent_dim, 128) # Increased size
        self.agent_relation = nn.Linear(64, 64)

        # Attention mechanism dimension needs to match key/query transform output
        attention_dim = 64

        # Post-attention processing based on context vector size (128)
        self.post_attention = nn.Sequential(
            nn.Linear(128, 128), # Context dim
            nn.GELU(),
            nn.Linear(128, output_dim) # Output logits for Categorical (MAX_AGENTS * NUM_MPE_ACTIONS = 25)
            # Removed Sigmoid - outputting logits now
        )
        # Value head based on context vector size (128)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Handle batch dimension (even if batch size is 1)
        if x.dim() == 1:
            x = x.unsqueeze(0) # Add batch dimension if missing
        batch_size = x.shape[0]

        # Reshape input: (batch_size, input_dim) -> (batch_size, MAX_AGENTS, agent_dim)
        agents = x.reshape(batch_size, MAX_AGENTS, self.agent_dim)

        # Normalize features per agent (optional but can help)
        # agents_mean = agents.mean(dim=-1, keepdim=True)
        # agents_std = agents.std(dim=-1, keepdim=True) + 1e-8
        # agents_norm = (agents - agents_mean) / agents_std

        keys = self.key_transform(agents) # Use original agents or agents_norm
        queries = self.query_transform(agents)
        values = self.value_transform(agents)

        # Scaled Dot-Product Attention
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5)

        # Optional: Add relative positional encoding or relation bias if needed
        # relation_queries = self.agent_relation(queries)
        # relation_bias = torch.bmm(relation_queries, relation_queries.transpose(1, 2)) / (relation_queries.shape[-1] ** 0.5)
        # attention_scores = attention_scores + relation_bias

        attention_weights = torch.softmax(attention_scores, dim=-1)

        context = torch.bmm(attention_weights, values) # (batch_size, MAX_AGENTS, value_dim)

        # Masking and Pooling: Pool features only from "active" agent slots
        # Use sum of absolute values > threshold to detect non-zero padded agents
        agent_mask = (agents.abs().sum(dim=-1, keepdim=True) > 0.1).float()
        # Ensure mask has correct dimensions for broadcasting with context
        masked_context = context * agent_mask # (batch_size, MAX_AGENTS, value_dim)

        # Average pooling over the agent dimension (only considering active agents)
        pooled = masked_context.sum(dim=1) / (agent_mask.sum(dim=1) + 1e-8) # (batch_size, value_dim)

        # Get action logits and value estimate from the pooled representation
        action_logits = self.post_attention(pooled) # (batch_size, MAX_AGENTS * NUM_MPE_ACTIONS)
        value = self.value_head(pooled) # (batch_size, 1)

        # Squeeze batch dim if original input was 1D
        if batch_size == 1 and x.dim() == 1:
             action_logits = action_logits.squeeze(0)
             value = value.squeeze(0)

        return action_logits, value


# --- Training Function (Adapted for MPE & Categorical Actions) ---
def train_light_meta_policy(model, env_factory_fn, meta_iterations=300, inner_rollouts=40, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_actions_per_agent = NUM_MPE_ACTIONS

    for iteration in range(meta_iterations):
        # Sample number of agents for this meta-iteration (can keep this strategy)
        if iteration < meta_iterations * 0.25:
            num_agents = np.random.choice([1, 2])
        elif iteration < meta_iterations * 0.33:
            num_agents = np.random.choice([2, 3])
        else:
            num_agents = np.random.choice([4, 5])

        total_loss = 0
        batch_returns = [] # Track returns for logging

        # Collect experience using current policy
        # Run multiple episodes per meta-iteration for stability
        for _ in range(2): # Number of episodes per meta-iteration update
            env = env_factory_fn(num_agents=num_agents, seed=iteration*10 + _) # Use unique seed
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            log_probs_ep, rewards_ep, values_ep, entropies_ep, dones_ep = [], [], [], [], []

            for _ in range(inner_rollouts): # Max steps per episode
                action_logits, value = model(obs) # Logits shape: (MAX_AGENTS * num_actions_per_agent,)

                # Reshape logits: (MAX_AGENTS * num_actions_per_agent,) -> (MAX_AGENTS, num_actions_per_agent)
                logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)

                # Create distribution for all MAX_AGENTS slots
                dist = torch.distributions.Categorical(logits=logits_reshaped)

                # Sample actions for all MAX_AGENTS slots
                actions_all = dist.sample() # Shape: (MAX_AGENTS,)

                # Select actions for the *current* number of agents in the env
                actions_to_step = actions_all[:env.num_agents].numpy()

                # Calculate log_prob and entropy for the actions taken by *active* agents
                # Only consider the first env.num_agents distributions/actions for loss calculation
                log_prob_active = dist.log_prob(actions_all)[:env.num_agents].mean() # Mean over active agents
                entropy_active = dist.entropy()[:env.num_agents].mean() # Mean over active agents

                next_obs, reward, terminated, truncated, _ = env.step(actions_to_step)
                done = terminated or truncated

                log_probs_ep.append(log_prob_active)
                rewards_ep.append(reward) # Use reward directly from MPE env
                values_ep.append(value)
                entropies_ep.append(entropy_active)
                dones_ep.append(done)

                obs = torch.tensor(next_obs, dtype=torch.float32)
                if done:
                    break

            # Calculate returns (Generalized Advantage Estimation could be better)
            returns = []
            discounted_reward = 0
            # Need value of final state if not done? Use 0 if done, or predict V(s_T)
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
            batch_returns.append(returns[0].item()) # Log initial return of episode

            # Prepare tensors for loss calculation
            values_ep = torch.stack(values_ep).squeeze(-1) # Remove last dim if necessary (depends on value head)
            log_probs_ep = torch.stack(log_probs_ep)
            entropies_ep = torch.stack(entropies_ep)

            # Ensure returns and values have same length
            if len(returns) != len(values_ep):
                 print(f"Warning: Length mismatch! returns: {len(returns)}, values: {len(values_ep)}")
                 # Truncate longer tensor? Or skip episode? Skipping for now.
                 continue # Skip this episode's contribution to loss


            advantages = returns - values_ep.detach() # Advantages using TD error

            # Calculate losses
            policy_loss = -(log_probs_ep * advantages).mean()
            value_loss = nn.MSELoss()(values_ep, returns)
            entropy_bonus = entropies_ep.mean()
            task_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

            total_loss += task_loss # Accumulate loss over episodes in meta-iteration

        # Perform optimization step after collecting experience
        if isinstance(total_loss, torch.Tensor): # Check if any loss was calculated
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
        else:
             print(f"Warning: No loss calculated for meta-iteration {iteration}. Skipping update.")


        if iteration % 10 == 0 and batch_returns:
            print(f"Light Meta-Iter {iteration} | Avg Start Return: {np.mean(batch_returns):.2f} | Total Loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else 0:.4f}")

    return model


# --- Fine-Tuning Function (Adapted for MPE & Categorical Actions) ---
# --- Fine-Tuning Function (Adapted for MPE & Categorical Actions) ---
# Add gamma as an argument with a default value
def fine_tune_light_meta_policy(model, env_factory_fn, max_steps=50, episodes_per_update=5, steps_per_episode=50, batch_size=32, gamma=0.99, value_loss_coef=0.5, entropy_coef=0.01): # Added gamma and loss coefficients
    tuned_model = copy.deepcopy(model)
    optimizer = optim.Adam(tuned_model.parameters(), lr=0.0005) # Slightly lower LR for tuning
    num_actions_per_agent = NUM_MPE_ACTIONS

    env = env_factory_fn() # Create env instance once
    # Check if env has num_agents attribute directly, otherwise infer from possible_agents
    try:
        num_agents = env.num_agents # Get num_agents for this specific tuning task
    except AttributeError:
        # If wrapper doesn't store num_agents directly, infer from env
        num_agents = len(env.env.agents) # Access underlying parallel env agents

    experience_buffer = [] # Store (obs, action_indices, reward, next_obs, done)

    for step in range(max_steps):
        # --- Experience Collection ---
        current_ep = 0
        while current_ep < episodes_per_update:
            obs, _ = env.reset()
            done = False
            episode_steps = 0
            while not done and episode_steps < steps_per_episode:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    action_logits, _ = tuned_model(obs_tensor) # Use current tuned model
                    logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)
                    dist = torch.distributions.Categorical(logits=logits_reshaped)
                    actions_all = dist.sample()
                    # Need to ensure actions are sampled only for the current num_agents
                    # However, the buffer expects actions for the specific num_agents of the env instance
                    actions_indices = actions_all[:num_agents].cpu().numpy() # Store action indices

                next_obs, reward, terminated, truncated, _ = env.step(actions_indices)
                done = terminated or truncated
                experience_buffer.append((obs, actions_indices, reward, next_obs, done))
                obs = next_obs
                episode_steps += 1
            current_ep += 1

        # Keep buffer size manageable (e.g., last N experiences)
        max_buffer_size = 1000
        if len(experience_buffer) > max_buffer_size:
             experience_buffer = experience_buffer[-max_buffer_size:]

        if len(experience_buffer) < batch_size:
             continue # Not enough experience yet

        # --- Policy Update ---
        # Sample a batch
        batch_indices = np.random.choice(len(experience_buffer), batch_size, replace=False)
        batch = [experience_buffer[i] for i in batch_indices]

        batch_obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32)
        # Actions are indices, should be Long type for indexing/loss calculation
        batch_actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.long)
        batch_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
        batch_next_obs = torch.tensor(np.array([s[3] for s in batch]), dtype=torch.float32)
        batch_dones = torch.tensor([s[4] for s in batch], dtype=torch.float32) # 0.0 or 1.0

        # Calculate loss (simplified Actor-Critic style update)
        # Ensure requires_grad is true for model parameters
        tuned_model.train()
        action_logits, values = tuned_model(batch_obs) # (batch, 25), (batch, 1)
        logits_reshaped = action_logits.view(batch_size, MAX_AGENTS, num_actions_per_agent)

        # Get log probs and entropy for the *specific number of agents* this env has (num_agents)
        # We need to select the logits corresponding to the active agents (first num_agents)
        # Handle case where batch_size might be 1
        if batch_size > 1:
             logits_active = logits_reshaped[:, :num_agents, :] # (batch, num_agents, num_actions)
        else:
             # If batch_size is 1, reshape needs care
             logits_active = logits_reshaped.view(1, MAX_AGENTS, num_actions_per_agent)[:, :num_agents, :] #(1, num_agents, num_actions)


        dist_active = torch.distributions.Categorical(logits=logits_active)

        # Ensure batch_actions has the correct shape (batch_size, num_agents)
        if batch_actions.shape != (batch_size, num_agents):
             # This might happen if num_agents changed unexpectedly or buffer stores inconsistent data
             print(f"Warning: Action shape mismatch! Expected {(batch_size, num_agents)}, got {batch_actions.shape}. Skipping update.")
             continue

        log_probs = dist_active.log_prob(batch_actions) # Shape: (batch, num_agents)
        # Average log_prob across agents for policy loss
        mean_log_probs = log_probs.mean(dim=1) # Shape: (batch,)

        # Value prediction for next state
        with torch.no_grad():
             tuned_model.eval() # Set model to eval mode for value prediction
             _, next_values = tuned_model(batch_next_obs) # (batch, 1)
             tuned_model.train() # Set back to train mode

        # TD Target - Use gamma here
        targets = batch_rewards + gamma * next_values.squeeze(-1) * (1 - batch_dones)

        # Calculate losses
        advantages = targets - values.squeeze(-1).detach() # Make sure values requires grad
        # Advantages should not require grad for policy loss term
        policy_loss = -(mean_log_probs * advantages.detach()).mean()
        # Value loss requires values to require grad
        value_loss = nn.MSELoss()(values.squeeze(-1), targets.detach()) # Target should be detached
        # Entropy requires grads through the distribution/logits
        entropy_bonus = dist_active.entropy().mean(dim=1).mean() # Mean over agents, then mean over batch

        # Use coefficients consistent with training function
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned_model.parameters(), max_norm=1.0) # Use same norm as train
        optimizer.step()

        if step % 10 == 0:
             print(f"Fine-tune step {step} | Loss: {loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

    try:
        env.close() # Close the env used for tuning
    except: pass
    return tuned_model


# --- Evaluation Function (Adapted for MPE & Categorical Actions) ---
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
        max_eval_steps = 200 # Limit episode length during evaluation

        while not done and step_count < max_eval_steps:
            with torch.no_grad():
                action_logits, _ = model(obs) # Logits shape: (MAX_AGENTS * num_actions_per_agent,)
                logits_reshaped = action_logits.view(MAX_AGENTS, num_actions_per_agent)
                dist = torch.distributions.Categorical(logits=logits_reshaped)
                actions_all = dist.sample() # Sample for all slots
                actions = actions_all[:env.num_agents].numpy() # Select actions for active agents

            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
            step_count += 1

        total_rewards.append(total_reward)
        try:
             env.close()
        except: pass # Some envs might not have close

    return np.mean(total_rewards) if total_rewards else 0


# --- Main Benchmark Function ---
def benchmark_lightmeta():
    # Dimensions for MPE simple_spread
    input_dim = MAX_OBS_DIM * MAX_AGENTS # 22 * 5 = 110
    output_dim = NUM_MPE_ACTIONS * MAX_AGENTS # 5 * 5 = 25

    # Instantiate the *adapted* policy
    model = LightMetaPolicy(input_dim, output_dim)

    print(f"Instantiated LightMetaPolicy for MPE:")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  Agent Obs Dim (Internal): {model.agent_dim}")

    print("\nTraining LightMetaPolicy on MPE Simple Spread...")
    start_time = time.time()
    # Use the *adapted* training function with the MPE factory
    trained_model = train_light_meta_policy(model, mpe_env_fn, meta_iterations=100, inner_rollouts=50) # Reduced iterations for quicker test
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    agent_counts = [1, 2, 3, 4, 5]
    print("\n=== MPE Simple Spread Evaluation ===")
    for num_agents in agent_counts:
        # Create a specific factory for this number of agents for evaluation
        eval_env_factory = lambda seed=None: mpe_env_fn(num_agents=num_agents, seed=seed)

        print(f"\n--- Evaluating for {num_agents} Agents ---")

        print("Running Zero-Shot Evaluation...")
        # Use the *adapted* evaluation function
        zero_shot = evaluate_meta_policy(trained_model, eval_env_factory)

        print("Running Fine-Tuning and Adaptation Evaluation...")
        # Use the *adapted* fine-tuning function
        adapted_model = fine_tune_light_meta_policy(trained_model, lambda: mpe_env_fn(num_agents=num_agents, seed=42), max_steps=30) # Reduced steps
        # Use the *adapted* evaluation function
        adapted = evaluate_meta_policy(adapted_model, eval_env_factory)

        print(f"Agents: {num_agents} | Zero-Shot Mean Reward: {zero_shot:.2f} | Adapted Mean Reward: {adapted:.2f}")

if __name__ == "__main__":
    benchmark_lightmeta()