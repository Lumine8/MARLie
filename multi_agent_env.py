import gym
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from stable_baselines3 import PPO

# Custom Multi-Agent Environment
class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(num_agents * 4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * num_agents)
        self.agent_states = np.zeros((num_agents, 4))
        self.agent_positions = np.random.randint(50, 600 - 50, (num_agents, 2))
        self.steps = 0
        self.max_steps = 200
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.losses = []

    def reset(self):
        self.agent_states = np.random.uniform(low=-1, high=1, size=(self.num_agents, 4))
        self.agent_positions = np.random.randint(50, 600 - 50, (self.num_agents, 2))
        self.steps = 0
        for i in range(self.num_agents):
            self.episode_rewards[i].append(0)
        return self.agent_states.flatten()

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        
        for i, action in enumerate(actions):
            if action == 1:
                rewards[i] = 1
                self.agent_positions[i][0] += np.random.randint(-10, 10)
                self.agent_positions[i][1] += np.random.randint(-10, 10)
        
        for i in range(self.num_agents):
            self.episode_rewards[i][-1] += rewards[i]
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self.agent_states.flatten(), np.sum(rewards), done, {"individual_rewards": rewards.tolist()}

# Initialize environment and model
env = MultiAgentEnv(num_agents=2)
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Global variables to maintain state
obs = env.reset()
done = False

# Dash App for Visualization
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='agent-plot'),
    dcc.Interval(id='interval-component', interval=500, n_intervals=0)
])

@app.callback(Output('agent-plot', 'figure'), Input('interval-component', 'n_intervals'))
def update_graph(n):
    global obs, done
    if done:
        obs = env.reset()
        done = False
    
    actions = [model.predict(obs, deterministic=True)[0] for _ in range(env.num_agents)]
    obs, _, done, _ = env.step(actions)
    positions = env.agent_positions
    
    traces = [go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers', marker=dict(size=15)) for pos in positions]
    
    return {'data': traces, 'layout': go.Layout(xaxis=dict(range=[0, 600]), yaxis=dict(range=[0, 400]))}

if __name__ == '__main__':
    app.run_server(debug=True)