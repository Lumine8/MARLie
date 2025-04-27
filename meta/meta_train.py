import torch
import torch.nn as nn
import torch.optim as optim

def meta_train(meta_model, env_fn, meta_iterations=100, inner_rollouts=10, gamma=0.99):
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    for iteration in range(meta_iterations):
        env = env_fn()

        log_probs = []
        rewards = []

        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        for _ in range(inner_rollouts):
            action_probs = meta_model(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)

            next_obs, reward, done, _ = env.step(actions.numpy().astype(int))
            obs = torch.tensor(next_obs, dtype=torch.float32)

            log_probs.append(log_prob.mean())
            rewards.append(reward)

            if done:
                break

        discounted_reward = 0
        returns = []
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)

        loss = -torch.sum(log_probs * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Meta-Iter {iteration} | Return: {returns.sum().item():.2f}")
            
    return meta_model

# Fine-tune meta_model on a fixed target environment before evaluation
def fine_tune_meta_policy(meta_model, env_fn, steps=100):
    optimizer = optim.Adam(meta_model.parameters(), lr=0.0005)
    env = env_fn()
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    for _ in range(steps):
        action_probs = meta_model(obs)
        dist = torch.distributions.Bernoulli(action_probs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)

        next_obs, reward, done, _ = env.step(actions.numpy().astype(int))
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        loss = -log_prob.mean() * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        if done:
            obs = torch.tensor(env.reset(), dtype=torch.float32)

    return meta_model


def evaluate_meta_policy(meta_model, env_fn, episodes=5):
    total_rewards = []
    for _ in range(episodes):
        env = env_fn()
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                probs = meta_model(obs)
                actions = (probs > 0.5).int().numpy()
            obs, reward, done, _ = env.step(actions)
            obs = torch.tensor(obs, dtype=torch.float32)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)
