# MARLie
 Multi-Agent Reinforcement Learning
 
## Overview
MARLie is a multi-agent reinforcement learning framework designed for training and evaluating agents in a gym-based environment. It supports single-agent and multi-agent environments with customizable parameters and rewards. This project leverages the **Stable Baselines 3** library to implement reinforcement learning algorithms.

## Features
- **Single-Agent Training:** Train a single agent using algorithms like PPO and A2C.
- **Multi-Agent Support:** Allows the creation of environments with multiple agents interacting with each other.
- **Model Saving & Loading:** Easily save and load models for future inference.
- **Customizable Environment:** The environment can be adjusted to add new agents or modify existing behaviour.

## Installation
To run this project, ensure you have **Python 3.x** installed. You will also need to install the following dependencies:

```
pip3 install pygame
pip3 install gym
pip3 install stable-baselines3
pip3 install swig
pip3 install box2d box2d-kengz
```

## File and Directory Structure

### `multi_agent_env.py`
A custom environment for multi-agent reinforcement learning. This script defines the environment where multiple agents interact with each other. It is built on the Gym library and uses **Stable Baselines 3** for training multiple agents.

### `trial1.py`
A script for training a single agent in the **LunarLander-v2** environment. It uses the **PPO** algorithm for training and saves models periodically for later use.

### `trial1_load.py`
This script loads a pre-trained model and runs it through multiple test episodes to evaluate its performance in the **LunarLander-v2** environment.

### `models/`
A directory containing saved models during training. Each model is stored with a unique identifier based on the training time steps.

### `logs/`
A directory that holds the TensorBoard logs for tracking the training process, including performance metrics and learning curves.

### `README.md`
Documentation for the project, providing an overview of the repository, installation instructions, usage, and file structure.
