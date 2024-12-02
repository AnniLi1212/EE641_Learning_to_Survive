from .base_agent import BaseAgent

# random agent
class RandomAgent(BaseAgent):
    def select_action(self, state, training=True):
        return self.action_space.sample()
    
    def train(self, state, action, reward, next_state, done):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass