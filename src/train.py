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
            action_space=env.action_space,
            config=config
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# evaluate agent performance
def evaluate_agent(env, agent, baseline_agent=None, random_agent=None, num_episodes=10):
    total_rewards = []
    episode_lengths = []
    baseline_rewards = []
    random_rewards = []
    q_value_diffs = []
    
    for _ in range(num_episodes):
        state, info = env.reset()[0], env.unwrapped._get_info()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        if baseline_agent:
            from src.evaluate import compute_q_value_diff
            q_value_diff = compute_q_value_diff(agent, baseline_agent, state, info)
            q_value_diffs.append(q_value_diff)

        while not (done or truncated):
            action = agent.select_action(state, info=info, training=False)
            next_state, reward, done, truncated, next_info = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
            info = env.unwrapped._get_info()

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # eval baseline
        if baseline_agent:
            state, info = env.reset()[0], env.unwrapped._get_info()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = baseline_agent.select_action(state, info=info, training=False)
                next_state, reward, done, truncated, next_info = env.step(action)
                episode_reward += reward
                state = next_state
                info = env.unwrapped._get_info()
            
            baseline_rewards.append(episode_reward) 
        
        # eval random
        if random_agent:
            state, info = env.reset()[0], env.unwrapped._get_info()
            done = False
            truncated = False
            random_reward = 0

            while not (done or truncated):
                action = random_agent.select_action(state, info=info, training=False)
                next_state, reward, done, truncated, next_info = env.step(action)
                random_reward += reward
                state = next_state
                info = env.unwrapped._get_info()

            random_rewards.append(random_reward)
    
    results = {
        'reward': np.mean(total_rewards),
        'length': np.mean(episode_lengths)
    }

    if baseline_agent:
        results.update({
            'baseline_reward': np.mean(baseline_rewards),
            'q_value_diff': np.mean(q_value_diffs),
            'improvement_over_baseline': np.mean(total_rewards) - np.mean(baseline_rewards)
        })
    
    if random_agent:
        results.update({
            'random_reward': np.mean(random_rewards),
            'improvement_over_random': np.mean(total_rewards) - np.mean(random_rewards)
        })
    
    return results

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
                  agent_attack_min=config['environment'].get('agent_attack_min', 25),
                  agent_attack_max=config['environment'].get('agent_attack_max', 45),
                  hungry_decay=config['environment'].get('hungry_decay', 2),
                  observation_range=config['environment'].get('observation_range', 4),
                  threat_perception_range=config['environment'].get('threat_perception_range', 3),
                  num_caves=config['environment'].get('num_caves', 5),
                  cave_health_recovery=config['environment'].get('cave_health_recovery', 2),
                  hungry_health_penalty=config['environment'].get('hungry_health_penalty', 2))
    
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
                       agent_attack_min=config['environment'].get('agent_attack_min', 25),
                       agent_attack_max=config['environment'].get('agent_attack_max', 45),
                       hungry_decay=config['environment'].get('hungry_decay', 2),
                       observation_range=config['environment'].get('observation_range', 4),
                       threat_perception_range=config['environment'].get('threat_perception_range', 3),
                       num_caves=config['environment'].get('num_caves', 5),
                    cave_health_recovery=config['environment'].get('cave_health_recovery', 2),
                       hungry_health_penalty=config['environment'].get('hungry_health_penalty', 2))
    
    agent = create_agent(config, env)
    log_dir, checkpoint_dir, writer = setup_logging(config, run_name)
    
    # training loop
    episode = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    total_loss = 0
    num_losses = 0
    
    # max episode // 10 as baseline storing episode
    baseline_episodes = config['training']['max_episodes'] // 10
    baseline_agent = None
    random_agent = RandomAgent(env.observation_space.shape, env.action_space, config)
    
    while episode < config['training']['max_episodes']:
        state, info = env.reset()[0], env.unwrapped._get_info()
        agent.reset()
        done = False
        truncated = False
        episode_reward = 0

        if episode == baseline_episodes:
            baseline_agent = create_agent(config, env)
            baseline_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
            baseline_path = os.path.join(checkpoint_dir, 'baseline_model.pth')
            baseline_agent.save(baseline_path)
            print(f"Baseline model saved to {baseline_path} at episode {episode}")

        while not (done or truncated):
            action = agent.select_action(state, info=info)
            next_state, reward, done, truncated, next_info = env.step(action)
            episode_reward += reward
            
            info = env.unwrapped._get_info()
            next_info = env.unwrapped._get_info()

            loss = agent.train(state, action, reward, next_state, done, info, next_info)

            if loss is not None and isinstance(agent, DQNAgent):
                total_loss += loss
                num_losses += 1
            
            state = next_state
            info = next_info
        
        writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Train/Health', info['health'], episode)

        if num_losses > 0:
            avg_loss = total_loss / num_losses
            writer.add_scalar('Train/Average_Loss', avg_loss, episode)
            total_loss = 0
            num_losses = 0
        
        if episode % config['training']['eval_frequency'] == 0:
            eval_results = evaluate_agent(eval_env, agent, baseline_agent, random_agent, 
                                            config['training']['eval_episodes'])
            
            writer.add_scalar('Eval/Average_Reward', eval_results['reward'], episode)
            writer.add_scalar('Eval/Average_Length', eval_results['length'], episode)

            if baseline_agent:
                writer.add_scalar('Eval/Baseline_Reward', eval_results['baseline_reward'], episode)
                writer.add_scalar('Eval/Q_Value_Diff', eval_results['q_value_diff'], episode)
                writer.add_scalar('Eval/Improvement_over_Baseline', eval_results['improvement_over_baseline'], episode)
            
            writer.add_scalar('Eval/Random_Reward', eval_results['random_reward'], episode)
            writer.add_scalar('Eval/Improvement_over_Random', eval_results['improvement_over_random'], episode)
            
            if eval_results['reward'] > best_eval_reward:
                best_eval_reward = eval_results['reward']
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