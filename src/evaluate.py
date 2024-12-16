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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
            action_space=env.action_space,
            config=config
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent

def evaluate_episode(env, agent, baseline_agent=None, random_agent=None, render=False):
    initial_seed = env.np_random.bit_generator.state
    episode_data = run_episode(env, agent, render)

    if baseline_agent:
        env.np_random.bit_generator.state = initial_seed
        baseline_data = run_episode(env, baseline_agent, False)
        episode_data['baseline_reward'] = baseline_data['reward']
        episode_data['baseline_action_records'] = baseline_data['action_records']

        initial_state = env.reset()[0]  # Reset to get initial state
        initial_info = env.unwrapped._get_info()
        episode_data['q_value_diff'] = compute_q_value_diff(
            agent, baseline_agent, initial_state, initial_info)

    # random agent
    if random_agent:
        env.np_random.bit_generator.state = initial_seed
        random_data = run_episode(env, random_agent, False)
        episode_data['random_reward'] = random_data['reward']
        episode_data['random_action_records'] = random_data['action_records']
    
    return episode_data

def run_episode(env, agent, render=False):
    state, info = env.reset()
    agent.reset()
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

def compute_q_value_diff(agent, baseline_agent, state, info):
    if agent is None or baseline_agent is None:
        return 0.0
    
    try:
        agent_q_values = agent.get_q_values(state, info)
        baseline_q_values = baseline_agent.get_q_values(state, info)
        
        if agent_q_values is None or baseline_q_values is None:
            return 0.0
            
        q_diff = (agent_q_values - baseline_q_values).mean().item()
        return float(q_diff)
        
    except Exception as e:
        print(f"Warning: Failed to compute Q-value difference: {str(e)}")
        return 0.0
    
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

def compute_statistics(rewards, lengths, healths, baseline_rewards=None, random_rewards=None):
    stats = {
        'num_episodes': len(rewards),
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        },
        'length': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths))
        },
        'health': {
            'mean': float(np.mean(healths)),
            'std': float(np.std(healths)),
            'min': float(np.min(healths)),
            'max': float(np.max(healths))
        }
    }
    
    if baseline_rewards:
        stats['baseline_reward'] = {
            'mean': float(np.mean(baseline_rewards)),
            'std': float(np.std(baseline_rewards)),
            'min': float(np.min(baseline_rewards)),
            'max': float(np.max(baseline_rewards))
        }
    
    if random_rewards:
        stats['random_reward'] = {
            'mean': float(np.mean(random_rewards)),
            'std': float(np.std(random_rewards)),
            'min': float(np.min(random_rewards)),
            'max': float(np.max(random_rewards))
        }
    
    return stats

# evaluate a trained agent
def evaluate(config_path, model_path, baseline_path=None, num_episodes=100, render=False, save_dir=None):
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
                  agent_attack_min=config['environment'].get('agent_attack_min', 25),
                  agent_attack_max=config['environment'].get('agent_attack_max',45),
                  hungry_decay=config['environment'].get('hungry_decay', 2),
                  observation_range=config['environment'].get('observation_range', 4),
                  threat_perception_range=config['environment'].get('threat_perception_range', 2),
                  num_caves=config['environment'].get('num_caves', 5),
                  cave_health_recovery=config['environment'].get('cave_health_recovery', 2),
                  hungry_health_penalty=config['environment'].get('hungry_health_penalty', 2))
    
    agent = create_agent(config, env, model_path)
    baseline_agent = None
    if baseline_path and os.path.exists(baseline_path):
        print(f"Loading baseline model from {baseline_path}")
        baseline_agent = DQNAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            config=config['agent']
        )
        baseline_agent.load(baseline_path)
    else:
        print("No baseline model provided or file not found")

    random_agent = RandomAgent(
        state_shape=env.observation_space.shape,
        action_space=env.action_space,
        config=config
    )
    
    print(f"Evaluating agent for {num_episodes} episodes...")
    results = []
    rewards = []
    lengths = []
    healths = []
    baseline_rewards = []
    random_rewards = []
    q_value_diffs = []
    
    for episode in tqdm(range(num_episodes)):
        episode_data = evaluate_episode(env, agent, baseline_agent, random_agent, render)
        
        # print(f"\nEpisode {episode} summary:")
        # print(f"Trained agent reward: {episode_data['reward']}")
        # print(f"Trained agent actions: {episode_data['action_records']}")
        # if baseline_agent:
        #     print(f"Baseline agent reward: {episode_data['baseline_reward']}")
        #     print(f"Baseline agent actions: {episode_data['baseline_action_records']}")
        # if random_agent:
        #     print(f"Random agent reward: {episode_data['random_reward']}")
        #     print(f"Random agent actions: {episode_data['random_action_records']}")
        
        results.append(episode_data)
        rewards.append(episode_data['reward'])
        lengths.append(episode_data['length'])
        healths.append(episode_data['final_health'])
        if baseline_agent:
            baseline_rewards.append(episode_data['baseline_reward'])
            q_value_diffs.append(episode_data['q_value_diff'])
        if random_agent:
            random_rewards.append(episode_data['random_reward'])
        
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_file = os.path.join(save_dir, 'evaluation_results.json')

        results_native = []
        for r in results:
            result_dict = {
                'reward': float(r['reward']),
                'length': int(r['length']),
                'final_health': float(r['final_health']),
                'actions': [int(a) for a in r['actions']],
                'action_records': r['action_records'],
                'states': r['states'],
                'infos': r['infos']
            }
            if baseline_agent:
                result_dict['baseline_reward'] = float(r['baseline_reward'])
                result_dict['baseline_action_records'] = r['baseline_action_records']
                result_dict['q_value_diff'] = float(r['q_value_diff'])
            if random_agent:
                result_dict['random_reward'] = float(r['random_reward'])
                result_dict['random_action_records'] = r['random_action_records']
            results_native.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': config,
                'model_path': model_path,
                'num_episodes': num_episodes,
                'statistics': compute_statistics(rewards, lengths, healths, baseline_rewards, random_rewards),
                'results': results_native
            }, f, indent=4)
    
    return compute_statistics(rewards, lengths, healths, baseline_rewards, random_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model file')
    parser.add_argument('--baseline', type=str, default=None,
                      help='Path to baseline model file')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment')
    parser.add_argument('--save-dir', type=str, default=None,
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    evaluate(args.config, args.model, args.baseline, args.episodes, args.render, args.save_dir)