import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom Multi-Agent Environment
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, render_mode='human'):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.agent_idx = 0  # Track which agent is active
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        self.agent_states = np.zeros((num_agents, 4))  # Placeholder for agent-specific observations
        self.render_mode = render_mode
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))  # Randomized reset
        self.steps = 0
        self.agent_idx = 0
        return self.agent_states[self.agent_idx]  # Return observation for the first agent

    def step(self, action):
        reward = 0
        if action == 1:  # Example reward logic for active agent
            reward = 1

        self.agent_states[self.agent_idx] += np.random.uniform(-0.05, 0.05, size=self.agent_states[self.agent_idx].shape)
        self.agent_idx = (self.agent_idx + 1) % self.num_agents  # Move to the next agent
        self.steps += 1
        done = self.steps >= self.max_steps

        obs = self.agent_states[self.agent_idx]
        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            # Basic gym environment rendering using built-in system (do not block)
            print(f"Agent {self.agent_idx} observation: {self.agent_states[self.agent_idx]} ")
            print(f"Reward: {1 if self.agent_idx == 1 else 0}")
            pass

    def close(self):
        pass


# Create environments for each agent
envs = [DummyVecEnv([lambda: MultiAgentEnv(num_agents=2, render_mode="human")]) for _ in range(2)]

# Define models for each agent
models = [PPO("MlpPolicy", env, verbose=1) for env in envs]

# Training configuration
TIMESTEPS = 10000
EPOCHS = 5

# Train each agent
for i in range(EPOCHS):
    for agent_id, model in enumerate(models):
        print(f"Training Agent {agent_id + 1}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"models/PPO_agent_{agent_id + 1}_{TIMESTEPS * (i + 1)}")

# Testing the trained agents
print("Testing the trained agents...")
env = MultiAgentEnv(num_agents=2, render_mode="human")
obs = env.reset()

episodes = 3
for ep in range(episodes):
    print(f"Starting Episode {ep + 1}")
    obs = env.reset()
    done = False
    while not done:
        actions = []
        for agent_id, model in enumerate(models):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            actions.append(action)

        env.render()

env.close()
