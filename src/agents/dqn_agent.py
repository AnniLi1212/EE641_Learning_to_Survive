import torch
import torch.nn as nn
import torch.optim as optim
import random
from ..utils import ReplayBuffer
import os
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_sizes=[256, 128]):
        super().__init__()
        # use conv to process input with 3 channels:
        # grid info (0,1,2,3,4), food value, threat attack
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        
        # calculate conv output size
        # input_shape[1] * input_shape[2]: size of the grid
        conv_out_size = input_shape[1] * input_shape[2] * 32
        additional_state_size = 3  # agent's health, hunger, attack
        flat_size = conv_out_size + additional_state_size
        
        layers = []
        prev_size = flat_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_actions)

    def forward(self, x, additional_state):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.cat([x, additional_state], dim=1)
        x = self.network(x)
        return self.output(x)

class DQNAgent:
    def __init__(self, state_shape, action_space, config):
        self.state_shape = state_shape
        self.action_space = action_space
        self.num_actions = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # hyperparameters from config
        self.batch_size = config.get('batch_size', 128)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.hidden_sizes = config.get('hidden_sizes', [256, 128])
        self.epsilon_decay = config.get('epsilon_decay', 0.997)
        self.target_update = config.get('target_update', 10)
        self.memory_size = config.get('memory_size', 100000)

        # initialize networks
        self.policy_net = DQNNetwork(
            self.state_shape, 
            self.num_actions,
            self.hidden_sizes
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            self.state_shape, 
            self.num_actions,
            self.hidden_sizes
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        # initialize replay memory
        self.memory = ReplayBuffer(self.memory_size)

        # initialize step counter
        self.steps_done = 0
        
    def select_action(self, state, info=None, training=True):
        # random action
        if training and random.random() < self.epsilon:
            action = self.action_space.sample()
            return action
        
        with torch.no_grad():
            # internal state normalized
            additional_state = torch.FloatTensor([
                [info['health'] / 100.0, 
                info['hunger'] / 100.0, 
                info['attack'] / 50.0]
            ]).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # calculate q values and select action
            q_values = self.policy_net(state_tensor, additional_state)
            action = q_values.argmax().item()
            return action
    
    def train(self, state, action, reward, next_state, done, info, next_info):
        # store transition in memory
        additional_state = [
            info['health'] / 100.0,
            info['hunger'] / 100.0,
            info['attack'] / 50.0
        ]
        next_additional_state = [
            next_info['health'] / 100.0,
            next_info['hunger'] / 100.0,
            next_info['attack'] / 50.0
        ]
        self.memory.push(state, action, reward, next_state, done, 
                        additional_state, next_additional_state)
    
        # update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if len(self.memory) < self.batch_size:
            return None
        
        # sample batch, calculate current q
        states, actions, rewards, next_states, dones, add_states, next_add_states = \
            self.memory.sample(self.batch_size, self.device)
        current_q_values = self.policy_net(states, add_states).gather(1, actions.unsqueeze(1))
    
        # calculate target q
        with torch.no_grad():
            next_q_values = self.target_net(next_states, next_add_states).max(1)[0]
            next_q_values[dones.bool()] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': float(self.epsilon),
            'steps_done': int(self.steps_done),
            'state_shape': tuple(int(x) for x in self.state_shape),
            'num_actions': int(self.num_actions)
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = float(checkpoint['epsilon'])
        self.steps_done = int(checkpoint['steps_done'])