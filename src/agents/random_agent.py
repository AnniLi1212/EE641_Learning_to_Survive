# random agent
class RandomAgent:
    def __init__(self, state_shape, action_space, config=None):
        self.action_space = action_space
    
    def select_action(self, state, training=True):
        return self.action_space.sample()
    
    def train(self, state, action, reward, next_state, done):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass