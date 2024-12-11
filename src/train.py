import os
import yaml
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src import SurvivalGameEnv, DQNAgent, RandomAgent
import argparse

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

# create agent based on config
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

# evaluate agent performance
def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state, info = env.reset()[0], env.unwrapped._get_info()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action = agent.select_action(state, info=info, training=False)
            next_state, reward, done, truncated, next_info = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
            info = env.unwrapped._get_info()

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(total_rewards), np.mean(episode_lengths)

def train(config_path, run_name=None):
    config = load_config(config_path)
    env = gym.make('SurvivalGame-v0',
                  render_mode=config['environment'].get('render_mode', None),
                  max_steps=config['environment'].get('max_steps', 1000),
                  size=config['environment'].get('size', 16),
                  num_food=config['environment'].get('num_food', 10),
                  num_threats=config['environment'].get('num_threats', 5),
                  food_value_min=config['environment'].get('food_value_min', 10),
                  food_value_max=config['environment'].get('food_value_max', 30),
                  threat_attack_min=config['environment'].get('threat_attack_min', 20),
                  threat_attack_max=config['environment'].get('threat_attack_max', 40),
                  agent_attack_min=config['environment'].get('agent_attack_min', 30),
                  agent_attack_max=config['environment'].get('agent_attack_max', 50),
                  hungry_decay=config['environment'].get('hungry_decay', 2),
                  observation_range=config['environment'].get('observation_range', 4),
                  threat_perception_range=config['environment'].get('threat_perception_range', 2))
    
    eval_env = gym.make('SurvivalGame-v0',
                       render_mode=config['environment'].get('render_mode', None),
                       max_steps=config['environment'].get('max_steps', 1000),
                       size=config['environment'].get('size', 16),
                       num_food=config['environment'].get('num_food', 10),
                       num_threats=config['environment'].get('num_threats', 5),
                       food_value_min=config['environment'].get('food_value_min', 10),
                       food_value_max=config['environment'].get('food_value_max', 30),
                       threat_attack_min=config['environment'].get('threat_attack_min', 20),
                       threat_attack_max=config['environment'].get('threat_attack_max', 40),
                       agent_attack_min=config['environment'].get('agent_attack_min', 30),
                       agent_attack_max=config['environment'].get('agent_attack_max', 50),
                       hungry_decay=config['environment'].get('hungry_decay', 2),
                       observation_range=config['environment'].get('observation_range', 4),
                       threat_perception_range=config['environment'].get('threat_perception_range', 2))
    
    agent = create_agent(config, env)
    log_dir, checkpoint_dir, writer = setup_logging(config, run_name)
    
    # training loop
    episode = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    total_loss = 0
    num_losses = 0
    
    while episode < config['training']['max_episodes']:
        state, info = env.reset()[0], env.unwrapped._get_info()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state, info=info)
            next_state, reward, done, truncated, next_info = env.step(action)
            episode_reward += reward
            
            info = env.unwrapped._get_info()
            next_info = env.unwrapped._get_info()

            loss = agent.train(state, action, reward, next_state, done, info, next_info)
            
            writer.add_scalar('Train/Health', info['health'], episode)

            if loss is not None:
                total_loss += loss
                num_losses += 1
            
            state = next_state
            info = next_info
        
        writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
        if num_losses > 0:
            avg_loss = total_loss / num_losses
            writer.add_scalar('Train/Average_Loss', avg_loss, episode)
            total_loss = 0
            num_losses = 0
        
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