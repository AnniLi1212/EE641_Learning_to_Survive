import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

# TODO: cave
# TODO: more info & status
# TODO: observation scope
# TODO: better rendering
# TODO: day and night mechanism
# TODO: better reward function
# survival game environment for gymnasium
class SurvivalGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=1000, size=10):
        super().__init__()
        pygame.init()

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        self.window_size = 400 
        
        # 0: stay, 1-4: move in 4 directions
        self.action_space = spaces.Discrete(5)
        
        # 0: empty, 1: wall, 2: food, 3: threat, 4: agent, 5: cave
        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(self.size, self.size),
            dtype=np.int8
        )
        
        # game state
        self.grid = None
        self.agent_position = None
        self.health = None
        self.render_mode = render_mode
        
        # for rendering
        self.window = None
        self.clock = None

    #TODO: add cave observation + obervation scope limited
    def _get_obs(self):
        return self.grid.copy()

    #TODO: more info + more status
    def _get_info(self):
        return {
            "health": self.health,
            "steps": self.current_step,
            "agent_position": self.agent_position
        }
    # reset the environment to initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        
        # boarders
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # initialize agent random position not on wall
        valid_positions = np.where(self.grid == 0)
        idx = self.np_random.integers(0, len(valid_positions[0]))
        self.agent_position = (valid_positions[0][idx], valid_positions[1][idx])
        self.grid[self.agent_position] = 4
        
        # place food and threats
        self._place_items(2, num_items=3)  # food
        self._place_items(3, num_items=2)  # threats
        
        # reset health
        self.health = 100.0
        
        return self._get_obs(), self._get_info()

    # place items on the grid randomly
    def _place_items(self, item_type, num_items):
        for _ in range(num_items):
            empty_positions = np.where(self.grid == 0)
            if len(empty_positions[0]) > 0:
                idx = self.np_random.integers(0, len(empty_positions[0]))
                pos = (empty_positions[0][idx], empty_positions[1][idx])
                self.grid[pos] = item_type

    # execute one time step within the environment
    def step(self, action):
        self.current_step += 1
        
        moves = [
            (0, 0),   # stay
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
        ]
        
        new_pos = (
            self.agent_position[0] + moves[action][0],
            self.agent_position[1] + moves[action][1]
        )
        
        if self.grid[new_pos] != 1:
            self.agent_position = new_pos
            
            # food collection
            if self.grid[new_pos] == 2:
                self.health = min(100, self.health + 20)
                self.grid[new_pos] = 0
                self._place_items(2, 1) # new food
                
            # threat collision
            elif self.grid[new_pos] == 3:
                self.health -= 30
                self.grid[new_pos] = 0
                self._place_items(3, 1)  # new threat
        
        # health decay
        self.health = max(0, self.health - 1)
        
        # reward
        reward = self._calculate_reward()
        
        # if episode is done
        done = self.health <= 0 or self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, self._get_info()

    # calculate reward for the current state
    def _calculate_reward(self):
        reward = 0.0
        
        # reward for maintaining high health
        reward += self.health / 100.0
        
        # penalty for low health
        if self.health < 30:
            reward -= 1.0
        
        # penalty for death
        if self.health <= 0:
            reward -= 10.0
            
        return reward

    # render the current frame for video recording
    # TODO: finer rendering
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.Surface((self.window_size, self.window_size))
        
        # fill background
        self.window.fill((255, 255, 255))
        
        # calculate cell size
        cell_size = self.window_size // self.size
        
        # draw grid lines
        for i in range(self.size + 1):
            pygame.draw.line(self.window, (200, 200, 200), 
                           (0, i * cell_size), 
                           (self.window_size, i * cell_size))
            pygame.draw.line(self.window, (200, 200, 200), 
                           (i * cell_size, 0), 
                           (i * cell_size, self.window_size))
        
        # draw walls
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1: # wall
                    pygame.draw.rect(self.window, (100, 100, 100),
                                   (j * cell_size, i * cell_size, 
                                    cell_size, cell_size))
                elif self.grid[i, j] == 2: # food
                    pygame.draw.circle(self.window, (0, 255, 0),
                                     (j * cell_size + cell_size//2, 
                                      i * cell_size + cell_size//2),
                                     cell_size//4)
                elif self.grid[i, j] == 3: # threat
                    pygame.draw.circle(self.window, (255, 0, 0),
                                     (j * cell_size + cell_size//2, 
                                      i * cell_size + cell_size//2),
                                     cell_size//4)
        
        # draw agent
        agent_x = self.agent_position[1] * cell_size + cell_size//2
        agent_y = self.agent_position[0] * cell_size + cell_size//2
        pygame.draw.circle(self.window, (0, 0, 255),
                         (agent_x, agent_y),
                         cell_size//3)
        
        # draw health bar
        health_width = int(self.window_size * (self.health / 100))
        pygame.draw.rect(self.window, (255, 0, 0),
                        (0, self.window_size - 10, self.window_size, 10))
        pygame.draw.rect(self.window, (0, 255, 0),
                        (0, self.window_size - 10, health_width, 10))
        
        # get numpy array of the surface
        view = pygame.surfarray.array3d(self.window)
        return np.transpose(view, (1, 0, 2))

    # render the environment
    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Survival Game")
            
            # render frame
            frame = self._render_frame()
            
            # convert frame to surface and display
            surf = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            
            # process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            
            return None
        
        elif self.render_mode == "rgb_array":
            return self._render_frame()
                
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None