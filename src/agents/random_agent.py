# random agent
import torch
class RandomAgent:
    def __init__(self, state_shape, action_space, config=None):
        self.state_shape = state_shape
        self.action_space = action_space
        if config and 'environment' in config:
            self.agent_attack_max = config['environment'].get('agent_attack_max', 45)
        else:
            self.agent_attack_max = 45
    
    def select_action(self, state, info=None, training=True):
        return self.action_space.sample()
    
    def train(self, state, action, reward, next_state, done, info=None, next_info=None):
        return None
    
    def reset(self):
        pass

    def preprocess_state(self, state):
        return state
    
    def policy_net(self, state_tensor, additional_state):
        batch_size = state_tensor.shape[0]
        num_actions = self.action_space.n
        return torch.rand(batch_size, num_actions), None
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass