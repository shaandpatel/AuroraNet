"""
Hyperparameter tuning script for AuroraNet.
Uses Weights & Biases Sweeps to optimize the KpLSTM model.

This script initializes a sweep controller and provides the command to start
an agent. The agent will run 'src/models/train.py' with different hyperparameter
combinations defined in the configuration below.
"""

import wandb

def main():
    # Ensure we are logged in to W&B. This prevents "HTTP 400: name required for project query"
    wandb.login()

    # Explicitly fetch the default entity (username/team) to ensure the project query is valid
    entity = wandb.Api().default_entity
    print(f"Using W&B Entity: {entity}")

    # 1. Define the sweep configuration
    # This dictionary defines how we want to search for the best model.
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization tries to find the best params efficiently
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        # The command tells the agent how to run your existing train.py
        # ${args} will be replaced by the parameters chosen by the sweep agent
        'command': [
            '${env}',
            '${interpreter}',
            '-m',
            'src.models.train',
            '${args}',
            '--project_name', 'aurora-forecast-tuning',  # Keep tuning runs in a separate project
            '--epochs', '15'  # Limit epochs to speed up the search
        ],
        'parameters': {
            'lr': {
                'min': 0.0001,
                'max': 0.01
            },
            'hidden_size': {
                'values': [32, 64, 128, 256]
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            'dropout': {
                'min': 0.0,
                'max': 0.5
            },
            'batch_size': {
                'values': [32, 64, 128]
            }
        }
    }

    print("Initializing W&B Sweep...")
    
    # 2. Initialize the sweep
    # This sends the config to W&B servers and gets a unique Sweep ID
    sweep_id = wandb.sweep(sweep_config, project="aurora-forecast-tuning", entity=entity)

    print(f"\n Sweep initialized successfully! Sweep ID: {sweep_id}")
    print("="*60)
    print("To start the tuning agent, run the following command in your terminal:")
    print(f"\n    wandb agent {entity}/aurora-forecast-tuning/{sweep_id} --count 20\n")
    print("="*60)
    print("Notes:")
    print(" - The '--count 20' flag limits the agent to 20 runs. Remove it to run indefinitely.")
    print(" - You can run this command in multiple terminals to train in parallel (faster tuning).")
    print(" - Results will be logged to the 'aurora-forecast-tuning' project in W&B.")

if __name__ == "__main__":
    main()
