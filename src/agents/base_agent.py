from abc import ABC, abstractmethod
import numpy as np

# abstract base class for all agents
class BaseAgent(ABC):
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape # shape of the observation space
        self.action_space = action_space # Gymnasium action space object
        self.num_actions = action_space.n

    @abstractmethod
    # select an action given the current state
    def select_action(self, state, training=True):
        """
        Args:
            state: Current state observation
            training: Boolean indicating whether the agent is training
        
        Returns:
            selected action
        """
        pass

    @abstractmethod
    # train the agent on a single transition
    def train(self, state, action, reward, next_state, done):
        """
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Boolean indicating if episode is done
        """
        pass

    @abstractmethod
    # save the agent's model
    def save(self, path):
        pass

    @abstractmethod
    # load the agent's model
    def load(self, path):
        pass