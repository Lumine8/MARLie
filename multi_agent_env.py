import gym
import numpy as np
from stable_baselines3 import PPO

# Custom Multi-Agent Environment with Domain Randomization
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(num_agents * 4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * num_agents)
        self.max_steps = 200
        self.steps = 0
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.agent_positions_history = []
        self.randomize_environment()
    
    def randomize_environment(self):
        """Apply domain randomization by varying agent attributes and environment parameters."""
        self.move_range = np.random.randint(5, 15)  # Random movement step size
        self.start_bounds = np.random.randint(30, 100)  # Random boundary constraints
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.agent_positions = np.random.randint(self.start_bounds, 600 - self.start_bounds, (self.num_agents, 2))
    
    def reset(self):
        """Reset the environment with randomized conditions."""
        self.randomize_environment()
        self.steps = 0
        self.episode_rewards = [[0] for _ in range(self.num_agents)]
        self.agent_positions_history = [self.agent_positions.copy()]
        return self.agent_states.flatten()

    def step(self, actions):
        """Execute one step in the randomized environment."""
        rewards = np.zeros(self.num_agents)
        move_mask = (actions == 1)
        rewards[move_mask] = 1
        
        # Apply random movement magnitude only if agents chose to move
        if np.sum(move_mask) > 0:
            self.agent_positions[move_mask] += np.random.randint(-self.move_range, self.move_range, size=(np.sum(move_mask), 2))
        
        # Clip positions within randomized bounds
        self.agent_positions = np.clip(self.agent_positions, self.start_bounds, 600 - self.start_bounds)
        
        for i in range(self.num_agents):
            self.episode_rewards[i][-1] += rewards[i]
        self.agent_positions_history.append(self.agent_positions.copy())
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self.agent_states.flatten(), np.sum(rewards), done, {"individual_rewards": rewards.tolist()}

# Initialize environment and model
env = MultiAgentEnv(num_agents=2)

# Load pre-trained model if available
try:
    model = PPO.load("ppo_multiagent")
    model.set_env(env)  # Ensure environment is assigned
    print("Loaded pre-trained model.")
except:
    model = PPO("MlpPolicy", env, verbose=1)
    print("No pre-trained model found. Training from scratch.")

# Train or continue training the model
model.learn(total_timesteps=10000)
model.save("ppo_multiagent")  # Save model for future transfer learning
