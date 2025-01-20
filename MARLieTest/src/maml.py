import torch
import torch.optim as optim
import gym
from pettingzoo.sisl import multiwalker_v9  # Example of a PettingZoo environment import

class MultiAgentMAML:
    def __init__(self, agents, alpha=0.01, beta=0.001):
        self.agents = agents  # List of policy networks
        self.alpha = alpha    # Inner loop learning rate
        self.beta = beta      # Outer loop meta-learning rate
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=self.beta) for agent in agents
        ]

    def create_env(self, env_name):
        """
        Create environment based on the string name in task_configs.json
        """
        try:
            if 'multiwalker' in env_name:  # Adjust for your specific PettingZoo environment names
                env = multiwalker_v9.env()  # Create the PettingZoo multiwalker environment
            else:
                env = gym.make(env_name)  # Gym environment for non-PettingZoo tasks
            return env
        except Exception as e:
            print(f"Error creating environment {env_name}: {e}")
            raise

    def inner_update(self, env, agent, theta):
        """
        Perform task-specific (inner loop) update for a single agent.
        """
        obs = env.reset()  # Reset environment and get initial observation
        if obs is None:  # Ensure that the observation is valid
            print("Observation is invalid or None")
            return theta  # Return unchanged theta if observation is invalid

        # The environment's reset function may return multiple outputs, handle accordingly
        obs = obs[0]  # PettingZoo environments often return a dict or tuple

        theta_prime = agent.state_dict()

        for step in range(10):  # Perform a few gradient steps
            logits = agent(torch.tensor(obs, dtype=torch.float32))  # Forward pass
            action = torch.argmax(logits).item()  # Choose action based on policy
            next_obs, reward, done, truncated, info = env.step(action)  # Take action in environment
            loss = -reward  # Negative reward for maximization

            grad = torch.autograd.grad(loss, agent.parameters(), create_graph=True)
            # Update parameters directly instead of state_dict
            for param, g in zip(agent.parameters(), grad):
                param.data -= self.alpha * g

            if done or truncated:
                break  # Terminate episode if done or truncated

            obs = next_obs  # Update observation

        return agent.parameters()

    def meta_update(self, tasks):
        """
        Perform meta-update for all agents.
        """
        for i, agent in enumerate(self.agents):
            meta_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # Initialize as a tensor with requires_grad

            for task in tasks:
                env_name = task["env"]  # Environment name for the task
                env = self.create_env(env_name)  # Instantiate the environment dynamically
                theta_prime = self.inner_update(env, agent, agent.parameters())  # Inner loop update

                # Evaluate on the same task
                obs = env.reset()  # Reset environment
                if obs is None:  # Ensure observation is valid
                    print("Observation is invalid or None")
                    continue  # Skip this task if there's an issue with the environment reset

                obs = obs[0]  # Handle multiple outputs from PettingZoo reset

                for step in range(10):
                    logits = agent(torch.tensor(obs, dtype=torch.float32))  # Forward pass
                    action = torch.argmax(logits).item()  # Choose action based on policy
                    next_obs, reward, done, truncated, info = env.step(action)  # Step in environment
                    meta_loss += -reward  # Accumulate negative reward for meta-loss

                    if done or truncated:
                        break  # End the evaluation if done or truncated

                    obs = next_obs  # Update observation

            # Perform meta-update on the meta-policy
            self.optimizers[i].zero_grad()
            meta_loss.backward()  # Now meta_loss is a tensor that requires grad
            self.optimizers[i].step()  # Update agent's policy with the meta-gradient
