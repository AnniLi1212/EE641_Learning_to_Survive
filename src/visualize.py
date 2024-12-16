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
from collections import defaultdict

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_comparison_metrics(self, eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        # episode data
        episodes = list(range(len(results['results'])))
        agent_rewards = [r['reward'] for r in results['results']]
        baseline_rewards = [r.get('baseline_reward') for r in results['results'] if 'baseline_reward' in r]
        random_rewards = [r.get('random_reward') for r in results['results'] if 'random_reward' in r]
        q_value_diffs = [r.get('q_value_diff') for r in results['results'] if 'q_value_diff' in r]
    
        # reward comparisons
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, agent_rewards, label='Trained Agent', color='blue', alpha=0.8)
        if baseline_rewards:
            plt.plot(episodes, baseline_rewards, label='Baseline Agent', color='orange', alpha=0.8)
        if random_rewards:
            plt.plot(episodes, random_rewards, label='Random Agent', color='red', alpha=0.8)
        
        plt.title('Reward Comparison Across Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'reward_comparison.png'))
        plt.close()

        # q value diff
        if q_value_diffs:
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, q_value_diffs, color='green', alpha=0.8)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title('Q-Value Differences (Trained - Baseline)')
            plt.xlabel('Episode')
            plt.ylabel('Q-Value Difference')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'q_value_differences.png'))
            plt.close()

        # reward box
        plt.figure(figsize=(10, 6))
        data = []
        labels = []

        if agent_rewards:
            data.append(agent_rewards)
            labels.append('Trained')
        if baseline_rewards:
            data.append(baseline_rewards)
            labels.append('Baseline')
        if random_rewards:
            data.append(random_rewards)
            labels.append('Random')
        
        plt.boxplot(data, tick_labels=labels)
        plt.title('Reward Distribution Comparison')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'reward_distributions.png'))
        plt.close()

        stats = {
            'agent': {
                'mean_reward': np.mean(agent_rewards),
                'std_reward': np.std(agent_rewards),
                'device': results.get('device', 'unknown') 
            }
        }
        
        if baseline_rewards:
            stats['baseline'] = {
                'mean_reward': np.mean(baseline_rewards),
                'std_reward': np.std(baseline_rewards),
                'mean_q_diff': np.mean(q_value_diffs),
                'std_q_diff': np.std(q_value_diffs)
            }
        
        if random_rewards:
            stats['random'] = {
                'mean_reward': np.mean(random_rewards),
                'std_reward': np.std(random_rewards)
            }
        
        with open(os.path.join(self.save_dir, 'comparison_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        return stats
    
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
            axes[1, 0].set_title('Agent Health When Episode Ends')
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
            state, info = env.reset()

            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if hasattr(agent, 'device'):
                state = state.to(agent.device)
            

            done = False
            truncated = False
            total_reward = 0
            
            while not (done or truncated):
                try:
                    frame = env.render()
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, f'Health: {info["health"]:.1f}', 
                                (10, 40), font, 1.0, (0, 0, 0), 2)
                        cv2.putText(frame, f'Hunger: {info["hunger"]:.1f}', 
                                (10, 80), font, 1.0, (0, 0, 0), 2)
                        cv2.putText(frame, f'Attack: {info["attack"]:.1f}', 
                                (10, 120), font, 1.0, (0, 0, 0), 2)
                        cv2.putText(frame, f'Reward: {total_reward:.1f}', 
                                (10, 160), font, 1.0, (0, 0, 0), 2)
                    
                        frames.append(frame)
                    
                    if info is None:
                        info = env.unwrapped._get_info()

                    action = agent.select_action(state, info=info, training=False)
                    next_state, reward, done, truncated, next_info = env.step(action)

                    if isinstance(next_state, np.ndarray):
                        next_state = torch.FloatTensor(next_state)
                    if hasattr(agent, 'device'):
                        next_state = next_state.to(agent.device)
                        

                    total_reward += reward
                    
                    state = next_state
                    info = next_info
                except Exception as e:
                    print(f"Warning: Error during episode recording: {str(e)}")
                    break
            
            if frames:
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 5.0, (width, height))
                
                for frame in frames:
                    out.write(frame)
                
                out.release()
                print(f"Video saved to: {save_path}")
            
            else:
                print("Warning: No frames were captured")

        except Exception as e:
            print(f"Warning: Failed to create video: {str(e)}")
            if 'out' in locals():
                out.release()

    def plot_action_distribution(self, eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)

        agent_counts = {
            'trained': defaultdict(int),
            'baseline': defaultdict(int),
            'random': defaultdict(int)
        }
        action_types = [
            'Stay', 'Move', 
            'Eat Food, Hungry < 50', 'Eat Food, Hungry >= 50',
            'Fight Weaker Threat', 'Fight Stronger Threat',
            'Enter Cave, Health < 100', 'Enter Cave, Health >= 100'
        ]
        for agent_type in agent_counts:
            for action in action_types:
                agent_counts[agent_type][action] = 0
        
        for episode in results['results']:
            # trained agent
            if 'action_records' in episode:
                for action, count in episode['action_records'].items():
                    agent_counts['trained'][action] += count
            
            # basline agent
            if 'baseline_action_records' in episode:
                for action, count in episode['baseline_action_records'].items():
                    agent_counts['baseline'][action] += count

            # random agent
            if 'random_action_records' in episode:
                for action, count in episode['random_action_records'].items():
                    agent_counts['random'][action] += count

        total_actions = {
            'trained': sum(agent_counts['trained'].values()),
            'baseline': sum(agent_counts['baseline'].values()),
            'random': sum(agent_counts['random'].values())
        }
        # print("\nAction counts:")
        # for agent_type in ['trained', 'baseline', 'random']:
        #     print(f"\n{agent_type.capitalize()} Agent:")
        #     print(f"Total actions: {total_actions[agent_type]}")
        #     for action, count in agent_counts[agent_type].items():
        #         print(f"{action}: {count}")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        def plot_agent_actions(ax, counts, total, title):
            if not counts:
                ax.set_title(f'{title}\nTotal Actions: 0')
                return
            
            x_positions = np.arange(5)
            ax2 = ax.twinx()

            basic_values = [counts['Stay'], counts['Move']]
            basic_bars = ax.bar(x_positions[:2], basic_values, color='grey')
            ax.set_ylabel('Basic Action Count', color='grey')
            ax.tick_params(axis='y', labelcolor='grey')

            food_total = counts['Eat Food, Hungry < 50'] + counts['Eat Food, Hungry >= 50']
            food_bar = ax2.bar(x_positions[2], food_total, color=['lightgreen'])
            food_bar.patches[0].set_y(counts['Eat Food, Hungry < 50'])
            food_bar.patches[0].set_height(counts['Eat Food, Hungry >= 50'])
            ax2.bar(x_positions[2], counts['Eat Food, Hungry < 50'], color='green')
            
            fight_total = counts['Fight Stronger Threat'] + counts['Fight Weaker Threat']
            fight_bar = ax2.bar(x_positions[3], fight_total, color=['lightcoral'])
            fight_bar.patches[0].set_y(counts['Fight Weaker Threat'])
            fight_bar.patches[0].set_height(counts['Fight Stronger Threat'])
            ax2.bar(x_positions[3], counts['Fight Weaker Threat'], color='red')
            
            cave_total = counts['Enter Cave, Health < 100'] + counts['Enter Cave, Health >= 100']
            cave_bar = ax2.bar(x_positions[4], cave_total, color=['lightblue'])
            cave_bar.patches[0].set_y(counts['Enter Cave, Health < 100'])
            cave_bar.patches[0].set_height(counts['Enter Cave, Health >= 100'])
            ax2.bar(x_positions[4], counts['Enter Cave, Health < 100'], color='blue')
            
            ax2.set_ylabel('Conditional Action Count', color='darkgreen')
            ax2.tick_params(axis='y', labelcolor='darkgreen')
            
            max_basic = max(basic_values)
            if max_basic == 0:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, max_basic * 1.1)
            
            max_conditional = max(food_total, fight_total, cave_total)
            if max_conditional == 0:
                ax2.set_ylim(0, 1)
            else:
                ax2.set_ylim(0, max_conditional * 1.2)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(['Stay', 'Move', 'Eat Food', 'Fight', 'Enter Cave'], rotation=45)
            ax.set_title(f'{title}\nTotal Actions: {total}')
                    
            def add_value_label(bar, value, ax_to_label):
                if value > 0:
                    ax_to_label.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_y() + bar.get_height()/2.,
                        f'{int(value)}',
                        ha='center', va='center'
                    )

            for bar in basic_bars:
                add_value_label(bar, bar.get_height(), ax)

            if food_total > 0:
                ax2.text(x_positions[2], counts['Eat Food, Hungry < 50']/2,
                    f'{int(counts["Eat Food, Hungry < 50"])}', ha='center', va='center')
                ax2.text(x_positions[2], counts['Eat Food, Hungry < 50'] + counts['Eat Food, Hungry >= 50']/2,
                    f'{int(counts["Eat Food, Hungry >= 50"])}', ha='center', va='center')
            
            if fight_total > 0:
                ax2.text(x_positions[3], counts['Fight Weaker Threat']/2,
                    f'{int(counts["Fight Weaker Threat"])}', ha='center', va='center')
                ax2.text(x_positions[3], counts['Fight Weaker Threat'] + counts['Fight Stronger Threat']/2,
                    f'{int(counts["Fight Stronger Threat"])}', ha='center', va='center')
            
            if cave_total > 0:
                ax2.text(x_positions[4], counts['Enter Cave, Health < 100']/2,
                    f'{int(counts["Enter Cave, Health < 100"])}', ha='center', va='center')
                ax2.text(x_positions[4], counts['Enter Cave, Health < 100'] + counts['Enter Cave, Health >= 100']/2,
                    f'{int(counts["Enter Cave, Health >= 100"])}', ha='center', va='center')
        
        plot_agent_actions(ax1, agent_counts['trained'], total_actions['trained'], 'Trained Agent')
        plot_agent_actions(ax2, agent_counts['baseline'], total_actions['baseline'], 'Baseline Agent')
        plot_agent_actions(ax3, agent_counts['random'], total_actions['random'], 'Random Agent')


        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='grey', label='Basic Action'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Eat Food, Hungry < 50'),
            plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Eat Food, Hungry >= 50'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Fight Weaker Threat'),
            plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='Fight Stronger Threat'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Enter Cave, Health < 100'),
            plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Enter Cave, Health >= 100')
        ]
        fig.legend(handles=legend_elements, 
          loc='center', 
          bbox_to_anchor=(0.5, -0.1),
          ncol=4)
        
        plt.suptitle('Action Distribution Comparison', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'action_distributions_comparison.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_state_heatmap(self, eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        grid_size = results['config']['environment'].get('size', 16)
        position_counts = np.zeros((grid_size, grid_size))
    
        # track agent positions
        for episode in results['results']:
            for info in episode['infos']:
                if 'agent_position' in info:
                    pos = info['agent_position']
                    position_counts[pos[0], pos[1]] += 1
    
        if position_counts.max() > 0:
            position_counts = position_counts / position_counts.max()
    
        plt.figure(figsize=(12, 10))
        hm = sns.heatmap(position_counts, cmap='YlOrRd', 
                         xticklabels=range(grid_size),
                         yticklabels=range(grid_size))
        
        plt.title('Agent Position Heatmap during Evaluation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        hm.collections[0].colorbar.set_label('Normalized Visit Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'state_heatmap.png'))
        plt.close()


def main(tensorboard_log_dir, eval_results_path, model_path, save_dir=None, timestamp=None):
    if eval_results_path and os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_data = json.load(f)
            config = eval_data['config']
    else:
        print("Warning: No evaluation results found, using default config")
        config = {'environment': {}, 'agent': {}}
    if save_dir is None and timestamp is not None:
        save_dir = os.path.join("results/evaluation", f"experiment_{timestamp}")
    
    visualizer = Visualizer(save_dir)
    grid_size = config['environment'].get('size', 16)

    if tensorboard_log_dir and os.path.exists(tensorboard_log_dir):
        print("Plotting training curves...")
        visualizer.plot_training_curves(tensorboard_log_dir)
    else:
        print("Skipping training curves!")

    if eval_results_path and os.path.exists(eval_results_path):
        print("Plotting reward comparison metrics...")
        comparison_stats = visualizer.plot_comparison_metrics(eval_results_path)
        print("\nComparison Statistics:")
        print("Trained Agent:")
        print(f"  Mean Reward: {comparison_stats['agent']['mean_reward']:.2f} ± {comparison_stats['agent']['std_reward']:.2f}")
            
        if 'baseline' in comparison_stats:
            print("\nBaseline Agent:")
            print(f"  Mean Reward: {comparison_stats['baseline']['mean_reward']:.2f} ± {comparison_stats['baseline']['std_reward']:.2f}")
            print(f"  Mean Q-Value Difference: {comparison_stats['baseline']['mean_q_diff']:.2f} ± {comparison_stats['baseline']['std_q_diff']:.2f}")
        
        if 'random' in comparison_stats:
            print("\nRandom Agent:")
            print(f"  Mean Reward: {comparison_stats['random']['mean_reward']:.2f} ± {comparison_stats['random']['std_reward']:.2f}")
            
        print("\nPlotting action distribution and state heatmap...")
        visualizer.plot_action_distribution(eval_results_path)
        visualizer.plot_state_heatmap(eval_results_path)
    else:
        print(f"Warning: Evaluation results not found at {eval_results_path}")
    
    if model_path and os.path.exists(model_path):
        print("Creating episode video...")
        env = gym.make('SurvivalGame-v0', 
                      render_mode='rgb_array',
                      size=grid_size,
                      max_steps=config['environment'].get('max_steps', 1000),
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else 
                              "mps" if torch.backends.mps.is_available() else 
                              "cpu")
        
        agent = DQNAgent(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            config=config['agent']
        )
        agent.load(model_path)

        video_path = os.path.join(visualizer.save_dir, 'episode.mp4')
        visualizer.create_episode_video(env, agent, video_path)
        
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
    parser.add_argument('--save-dir', type=str, default=None,
                      help='Directory to save visualization results')
    parser.add_argument('--timestamp', type=str, default=None,
                      help='Timestamp for consistent directory naming')
    args = parser.parse_args()
    
    main(args.tensorboard_dir, args.eval_results, args.model, args.save_dir, args.timestamp)