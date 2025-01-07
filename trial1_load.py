import gym
from stable_baselines3 import PPO
import numpy as np
import warnings
import os

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2", render_mode="human")

env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/170000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, _ = env.reset() 
    done = False
    episode_reward = 0  
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action) 
        
        episode_reward += reward 
        done = terminated or truncated  
    
    print(f"Episode: {ep + 1}, Total Reward: {episode_reward:.2f}")

env.close()
