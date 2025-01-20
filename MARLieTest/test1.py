
import argparse
from src.train_meta import main as train_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAML for multi-agent RL.")
    parser.add_argument("--mode", type=str, default="meta_train", help="Mode: meta_train or test")
    args = parser.parse_args()

    if args.mode == "meta_train":
        train_meta()
    else:
        print("Add fine-tuning/testing logic here.")
