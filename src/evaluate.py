import os
import yaml
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import SurvivalGameEnv, DQNAgent, RandomAgent
import argparse
from src.train import load_config

def create_agent(config, env, model_path=None):
    agent_type = config['agent']['type']
    if agent_type == "dqn":
        agent = DQNAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            config=config['agent']
        )
        if model_path:
            agent.load(model_path)
    elif agent_type == "random":
        agent = RandomAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent

def evaluate_episode(env, agent, render=False):
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    episode_length = 0
    states = [state.tolist()]
    actions = []
    rewards = []
    infos = [info]
    
    while not (done or truncated):
        action = agent.select_action(state, info=info, training=False)
        next_state, reward, done, truncated, next_info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        states.append(next_state.tolist())
        actions.append(int(action))
        rewards.append(float(reward))
        infos.append(next_info)
        
        state = next_state
        info = next_info
        
        if render:
            env.render()
    
    return {
        'reward': total_reward,
        'length': episode_length,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'infos': infos,
        'final_health': info.get('health', 0),
        'action_records': info.get('action_records', {})
    }

def analyze_results(results, save_dir):
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    healths = [np.mean(r['data']['health']) for r in results]
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20)
    plt.title('Distribution of Episode Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'reward_distribution.png'))
    plt.close()
    
    # episode lengths
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20)
    plt.title('Distribution of Episode Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'length_distribution.png'))
    plt.close()
    
    # average health over episodes
    plt.figure(figsize=(10, 6))
    plt.hist(healths, bins=20)
    plt.title('Distribution of Average Health')
    plt.xlabel('Average Health')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'health_distribution.png'))
    plt.close()
    
    # calculate statistics
    stats = {
        'reward': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        },
        'length': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths)
        },
        'health': {
            'mean': np.mean(healths),
            'std': np.std(healths),
            'min': np.min(healths),
            'max': np.max(healths)
        }
    }
    
    return stats

# evaluate a trained agent
def evaluate(config_path, model_path, num_episodes=100, render=False, save_dir=None):
    config = load_config(config_path)
    grid_size = config['environment'].get('size', 16)
    
    env = gym.make('SurvivalGame-v0', 
                  render_mode="human" if render else None,
                  size=grid_size,
                  max_steps=config['environment'].get('max_steps', 1000),
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
                  threat_perception_range=config['environment'].get('threat_perception_range', 2),
                  num_caves=config['environment'].get('num_caves', 2),
                  cave_health_recovery=config['environment'].get('cave_health_recovery', 8))
    
    agent = create_agent(config, env, model_path)
    
    print(f"Evaluating agent for {num_episodes} episodes...")
    results = []
    rewards = []
    lengths = []
    healths = []
    
    for _ in tqdm(range(num_episodes)):
        episode_data = evaluate_episode(env, agent, render)
        results.append(episode_data)
        rewards.append(episode_data['reward'])
        lengths.append(episode_data['length'])
        healths.append(episode_data['final_health'])
    
    stats = {
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        },
        'length': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths))
        },
        'health': {
            'mean': float(np.mean(healths)),
            'std': float(np.std(healths)),
            'min': float(np.min(healths)),
            'max': float(np.max(healths))
        }
    }
    
    results_native = []
    for r in results:
        results_native.append({
            'reward': float(r['reward']),
            'length': int(r['length']),
            'final_health': float(r['final_health']),
            'actions': [int(a) for a in r['actions']],
            'states': r['states'],
            'infos': r['infos'],
            'action_records': r['action_records']
        })
    
    results_file = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'model_path': model_path,
            'num_episodes': num_episodes,
            'statistics': stats,
            'results': results_native
        }, f, indent=4)
    
    print("\nEvaluation Results:")
    print(f"Number of episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {stats['reward']['mean']:.2f} ± {stats['reward']['std']:.2f}")
    print(f"  Range: [{stats['reward']['min']:.2f}, {stats['reward']['max']:.2f}]")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {stats['length']['mean']:.2f} ± {stats['length']['std']:.2f}")
    print(f"  Range: [{stats['length']['min']}, {stats['length']['max']}]")
    print(f"\nHealth Statistics:")
    print(f"  Mean: {stats['health']['mean']:.2f} ± {stats['health']['std']:.2f}")
    print(f"  Range: [{stats['health']['min']:.2f}, {stats['health']['max']:.2f}]")
    print(f"\nEval results saved to: {save_dir}")
    
    env.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model file')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment')
    parser.add_argument('--save-dir', type=str, default=None,
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    evaluate(args.config, args.model, args.episodes, args.render, args.save_dir)