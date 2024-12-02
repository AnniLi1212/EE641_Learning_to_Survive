import os
import yaml
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src import SurvivalGameEnv, DQNAgent, RandomAgent
import argparse

# function to load yaml config
def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# function to setup results logging
def setup_logging(config, run_name=None):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['agent']['type']}_{timestamp}"
    
    log_dir = os.path.join("results/logs", run_name)
    checkpoint_dir = os.path.join("results/checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return log_dir, checkpoint_dir, SummaryWriter(log_dir)

# function to create agent based on config
# TODO: add more agent types
def create_agent(config, env):
    agent_type = config['agent']['type']
    if agent_type == "dqn":
        return DQNAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            config=config['agent']
        )
    elif agent_type == "random":
        return RandomAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# function to evaluate agent performance
def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(total_rewards), np.mean(episode_lengths)

# main training function
def train(config_path, run_name=None):
    config = load_config(config_path)
    # setup environment with config values
    env = gym.make('SurvivalGame-v0',
                  render_mode=config['environment'].get('render_mode', None),
                  max_steps=config['environment'].get('max_steps', 1000),
                  size=config['environment'].get('size', 10))
    
    eval_env = gym.make('SurvivalGame-v0',
                       render_mode=config['environment'].get('render_mode', None),
                       max_steps=config['environment'].get('max_steps', 1000),
                       size=config['environment'].get('size', 10))
    
    # create agent
    agent = create_agent(config, env)
    
    # setup logging
    log_dir, checkpoint_dir, writer = setup_logging(config, run_name)
    
    # training loop
    episode = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    
    while episode < config['training']['max_episodes']:
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            loss = agent.train(state, action, reward, next_state, done)
            state = next_state
            
            if loss is not None:
                writer.add_scalar('Train/Loss', loss, episode)
        
        writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
        
        if episode % config['training']['eval_frequency'] == 0:
            eval_reward, eval_length = evaluate_agent(eval_env, agent, 
                                                    config['training']['eval_episodes'])
            writer.add_scalar('Eval/Average_Reward', eval_reward, episode)
            writer.add_scalar('Eval/Average_Length', eval_length, episode)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                no_improvement_count = 0
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= config['training']['patience']:
                print(f"No improvement for {no_improvement_count} evaluations. Stopping...")
                break
        
        if episode % config['training']['save_frequency'] == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_{episode}.pth'))
        
        episode += 1
    
    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    writer.close()
    env.close()
    eval_env.close()
    
    print("Training completed!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Models saved in: {checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                      help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                      help='Name for this training run')
    args = parser.parse_args()
    
    train(args.config, args.run_name)