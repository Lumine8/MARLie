import gym
from stable_baselines3 import PPO, A2C
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore deprecation warnings

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make("LunarLander-v2",render_mode="human")

env.reset() 

# model1 = A2C('MlpPolicy', env, verbose=1) 
# model1.learn(total_timesteps=1000)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir) 


TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") # learning on samples
    model.save(f"{models_dir}/{TIMESTEPS*i}") # saving model

'''episodes = 10

for ep in range(episodes):
    env.render()
    obs = env.reset() 
    done = False
    while not done:
        action, _ = model.predict(obs) 
        obs, reward, terminated, truncated, info = env.step(action) # take action in the environment

    # action = env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action)
    print(f"Episode: {ep+1}, Reward: {reward}")
'''

env.close()