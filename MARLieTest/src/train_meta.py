import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from src.policy import PolicyNetwork
from src.maml import MultiAgentMAML
from pettingzoo.sisl import multiwalker_v7

def main():
    # Folder setup
    os.makedirs("MARLieTest/models", exist_ok=True)
    os.makedirs("MARLieTest/logs/meta_training", exist_ok=True)

    # Load tasks
    with open("MARLieTest/configs/task_configs.json", "r") as f:
        task_configs = json.load(f)

    # Initialize agents
    input_dim = 24  # Observation size
    output_dim = 5  # Action size
    agents = [PolicyNetwork(input_dim, output_dim) for _ in range(3)]
    maml = MultiAgentMAML(agents)

    # Initialize logging
    writers = [SummaryWriter(log_dir=f"MARLieTest/logs/meta_training/agent_{i+1}") for i in range(3)]

    # Meta-training loop
    for epoch in range(100):  # Meta-training epochs
        maml.meta_update(task_configs)

        # Log metrics
        for i, writer in enumerate(writers):
            writer.add_scalar("Meta-Loss", maml.optimizers[i].param_groups[0]["lr"], epoch)

        # Save models
        for i, agent in enumerate(agents):
            agent_save_path = f"MARLieTest/models/agent_{i+1}"
            os.makedirs(agent_save_path, exist_ok=True)  # Ensure the model saving directory exists
            torch.save(agent.state_dict(), f"{agent_save_path}/meta_policy_epoch_{epoch}.pth")
            
        print(f"Epoch {epoch} completed and models saved.")

if __name__ == "__main__":
    main()
