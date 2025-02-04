# EE641_Learning_to_Survive
Learning to Survive: Organism Evolution in a Simulated Ecosystem Using Reinforcement Learning
![Uploading survivalgameimg.png…]()

## Survival Game Environment

This project implements a custom reinforcement learning (RL) environment called SurvivalGameEnv using the Gymnasium framework. The goal of the agent is to learn optimal survival strategies in a grid-based world by interacting with dynamic food, threats, and shelters. The project includes implementations of Deep Q-Network (DQN) agents using CNN and RNN, evaluation tools, and visualization utilities.

## Project Structure

The project directory includes the following files:
```
├──src/
    ├── environment/
        ├── game.py            # Custom survival game environment
    ├── agents/
        ├── dqn_agent.py       # DQN agent implementation
        ├── random_agent.py    # Random agent for baseline comparison
    ├── utils
        ├──replay_buffer.py    # Replay buffer for DQN training
    ├── train.py               # Training script for the DQN agent
    ├── evaluate.py            # Evaluation script for trained agents
    ├── visualize.py           # Visualization tools for results
├── configs/
    ├── default_config.yaml    # Configuration file for environment and training
├── run.sh                     # Shell script to execute training and evaluation
├── requirements.txt           # Project dependencies
```

## Environment

The environment is a 16x16 (could be customed) grid where the agent learns to survive by consuming food, avoiding threats, and managing internal states like health and hunger. Key features include:
- Dynamic Elements: Food and threats regenerate after interaction.
- Partial Observability: The agent can only see within a defined observation range.
- Threat Behavior: Threats pursue the agent when detected within their perception range.

### Entities
- **Agent**: The player character with attributes health, hunger, and attack.
- **Food**: Consumed to replenish hunger. Food values are randomly generated.
- **Threats**: Enemies that reduce the agent’s health if engaged.
- **Caves**: Shelters where the agent can recover health.

### Action Space

The agent has a discrete action space:
- 0: Stay
- 1: Move Up
- 2: Move Down
- 3: Move Left
- 4: Move Right

### Observation Space

A 4x4 (could be customed) grid centered around the agent:
- Channels include walls, agent position, caves, food values, and threat attack values.

### Rewards
- +2.0 for consuming food (scaled by food value).
- +0.5 for health recovery in caves.
- -0.5 for health loss due to threats or hunger.
- +0.1 for each step survived.
- -10.0 for agent death.
- +10.0 for surviving until the maximum steps.

### Configuration

The default_config.yaml file contains all environment, agent, and training parameters. Example:

environment:
  size: 16
  num_food: 10
  num_threats: 5
  observation_range: 4
  threat_perception_range: 3

agent:
  type: "dqn"
  memory_size: 500000
  batch_size: 128
  gamma: 0.99
  learning_rate: 0.0001
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.998

## How to Run

1. **Setup**

Install the required dependencies:

```
pip install -r requirements.txt
```

2. **Running Experiments**

Modify the default_config.yaml file to set the desired parameters.

Execute the full pipeline using the provided run.sh:
```
chmod +x run.sh
./run.sh
```
3. **If run tasks separately**

(1) **Training**

Run the training script for the DQN agent:
```
python train.py --config default_config.yaml
```

(2) **Evaluation**

Evaluate a trained agent:
```
python evaluate.py --config default_config.yaml --checkpoint results/checkpoints/dqn_latest.pth
```

(3) **Visualization**

Visualize training progress and evaluation results:
```
python visualize.py --log_dir results/logs
```
### Results
- Training Logs: Stored in results/logs.
- Checkpoints: Saved in results/checkpoints every few episodes.
- Evaluation Results: Plotted and saved in results/evaluation.
- visualization: Generated video for visualization

## Code Highlights

#### DQN Agent (dqn_agent.py)
- Implements a Deep Q-Network with experience replay and target network updates.
- Leverage CNN to catch the spatial relationship.
- Supports optional RNN layers (LSTM) for temporal dependencies.

#### Replay Buffer (replay_buffer.py)
- A memory buffer that stores agent experiences for mini-batch sampling.

#### Random Agent (random_agent.py)
- A simple baseline agent that selects actions randomly.

#### Game Environment (game.py)
- Custom Gymnasium environment with dynamic food and threats.
- Partial observability and configurable grid size.

## Future Improvements
- Implement multi-agent interactions.
- Add advanced RL algorithms (e.g., PPO, SAC).
- Introduce new dynamic entities for increased complexity.

## Dependencies
  
This project requires the following dependencies:  

- **numpy** >= 1.21.0  
- **torch** >= 2.0.0  
- **gymnasium** >= 0.29.0  
- **pygame** >= 2.5.0  
- **matplotlib** >= 3.7.0  
- **seaborn** >= 0.12.0  
- **opencv-python** >= 4.8.0  
- **pandas** >= 2.0.0  
- **pyyaml** >= 6.0.0  
- **tqdm** >= 4.65.0  
- **pillow** >= 9.5.0  
- **tensorboard** >= 2.12.0  

## License

This project is released under the MIT License.
