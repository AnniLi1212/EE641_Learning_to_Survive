import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import torch
from datetime import datetime
import gymnasium as gym
from src import SurvivalGameEnv, DQNAgent
import pygame
from PIL import Image
import yaml 

class Visualizer:
    def __init__(self, save_dir=None):
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("results/visualization", f"vis_{timestamp}")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_training_curves(self, tensorboard_log_dir):
        event_acc = EventAccumulator(tensorboard_log_dir)
        event_acc.Reload()

        tags = event_acc.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Progress', fontsize=16)
        
        if 'Train/Episode_Reward' in tags:
            rewards = [(s.step, s.value) for s in event_acc.Scalars('Train/Episode_Reward')]
            steps, values = zip(*rewards)
            axes[0, 0].plot(steps, values)
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
        if 'Eval/Average_Reward' in tags:
            eval_rewards = [(s.step, s.value) for s in event_acc.Scalars('Eval/Average_Reward')]
            steps, values = zip(*eval_rewards)
            axes[0, 1].plot(steps, values)
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            
        if 'Train/Health' in tags:
            health = [(s.step, s.value) for s in event_acc.Scalars('Train/Health')]
            steps, values = zip(*health)
            axes[1, 0].plot(steps, values)
            axes[1, 0].set_title('Agent Health')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Health')
            
        if 'Train/Average_Loss' in tags:
            losses = [(s.step, s.value) for s in event_acc.Scalars('Train/Average_Loss')]
            steps, values = zip(*losses)
            axes[1, 1].plot(steps, values)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()

    def create_episode_video(self, env, agent, save_path):
        try:
            import cv2
            frames = []
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                frame = env.render()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame)
                
                action = agent.select_action(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            print(f"\nEpisode summary:")
            print(f"Total reward: {episode_reward}")
            print(f"Number of frames: {len(frames)}")
            
            if frames:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
                
                for frame in frames:
                    out.write(frame)
                out.release()
                print(f"Episode video saved to: {save_path}")

        except Exception as e:
            print(f"Warning: Failed to create video: {str(e)}")

    def visualize_value_function(self, agent, save_path):
        grid_size = 10
        states = np.zeros((grid_size * grid_size, grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                states[i * grid_size + j, i, j] = 1
        
        states_tensor = torch.FloatTensor(states).to(agent.device)
        
        with torch.no_grad():
            values = agent.policy_net(states_tensor).max(1)[0].cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        value_grid = values.reshape((grid_size, grid_size))
        sns.heatmap(value_grid, cmap='viridis')
        plt.title('State Value Function')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def plot_action_distribution(self, eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        all_actions = []
        for episode in results['results']:
            all_actions.extend(episode['actions'])
        
        plt.figure(figsize=(10, 6))
        action_counts = np.bincount(all_actions)
        action_names = ['Stay', 'Up', 'Down', 'Left', 'Right']
        
        plt.bar(action_names, action_counts)
        plt.title('Distribution of Actions During Evaluation')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'action_distribution.png'))
        plt.close()

    def plot_state_heatmap(self, eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        positions = []
        for episode in results['results']:
            for state in episode['states']:
                # assuming state contains agent position information
                # might need to modify this based on your state representation
                agent_pos = np.array(state).argmax()  # agent position from one-hot encoded state
                x, y = agent_pos // 10, agent_pos % 10  # assuming 10x10 grid
                positions.append((x, y))
        
        plt.figure(figsize=(10, 10))
        position_counts = np.zeros((10, 10))  # assuming 10x10 grid
        for x, y in positions:
            position_counts[x, y] += 1
        
        sns.heatmap(position_counts, cmap='YlOrRd')
        plt.title('Agent Position Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'state_heatmap.png'))
        plt.close()


def main(tensorboard_log_dir=None, eval_results_path=None, model_path=None):
    visualizer = Visualizer()
    
    if tensorboard_log_dir and os.path.exists(tensorboard_log_dir):
        print("Plotting training curves...")
        visualizer.plot_training_curves(tensorboard_log_dir)
    else:
        print("Skipping training curves (no tensorboard logs available)")
    
    print("Plotting action distribution and state heatmap...")
    if eval_results_path and os.path.exists(eval_results_path):
        visualizer.plot_action_distribution(eval_results_path)
        visualizer.plot_state_heatmap(eval_results_path)
    else:
        print(f"Warning: Evaluation results not found at {eval_results_path}")
    
    if model_path and os.path.exists(model_path):
        print("Creating episode video...")
        env = gym.make('SurvivalGame-v0', render_mode='rgb_array')
        
        config_path = os.path.join(os.path.dirname(model_path), '..', '..', 'logs', 
                                 os.path.basename(os.path.dirname(model_path)), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {'agent': {'type': 'dqn'}}
        
        agent = DQNAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            config=config['agent']
        )
        agent.load(model_path)
        
        video_path = os.path.join(visualizer.save_dir, 'episode.mp4')
        visualizer.create_episode_video(env, agent, video_path)
        
        print("Visualizing value function...")
        value_path = os.path.join(visualizer.save_dir, 'value_function.png')
        visualizer.visualize_value_function(agent, value_path)
        
        env.close()
    
    print(f"Visualizations saved to: {visualizer.save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-dir', type=str, required=False, default=None,
                      help='Path to tensorboard log directory')
    parser.add_argument('--eval-results', type=str, required=True,
                      help='Path to evaluation results JSON file')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to model file for episode video creation')
    args = parser.parse_args()
    
    main(args.tensorboard_dir, args.eval_results, args.model)