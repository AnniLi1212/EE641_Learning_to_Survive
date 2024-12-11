import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class SurvivalGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=1000, size=16, 
                 num_food=10, num_threats=5, 
                 food_value_min=10, food_value_max=30,
                 threat_attack_min=20, threat_attack_max=40,
                 agent_attack_min=30, agent_attack_max=50,
                 hungry_decay=2, observation_range=4, threat_perception_range=2):
        super().__init__()
        pygame.init()

        self.size = size
        self.max_steps = max_steps
        self.num_food = num_food
        self.num_threats = num_threats
        self.food_value_min = food_value_min
        self.food_value_max = food_value_max
        self.threat_attack_min = threat_attack_min
        self.threat_attack_max = threat_attack_max
        self.agent_attack_min = agent_attack_min
        self.agent_attack_max = agent_attack_max
        self.hungry_decay = hungry_decay
        self.observation_range = observation_range
        self.threat_perception_range = threat_perception_range
        self.threat_positions = {}
        self.current_step = 0

        # 0: stay, 1-4: move in 4 directions
        self.action_space = spaces.Discrete(5)

        # can only see within the observation range
        obs_size = 2 * self.observation_range + 1
        self.observation_space = spaces.Box(
            low=np.array([[[0 for _ in range(obs_size)] for _ in range(obs_size)], # grid channel
                        [[0 for _ in range(obs_size)] for _ in range(obs_size)], # food values channel
                        [[0 for _ in range(obs_size)] for _ in range(obs_size)]], # threat attacks channel
                dtype=np.float32),
            high=np.array([[[5 for _ in range(obs_size)] for _ in range(obs_size)],
                        [[self.food_value_max for _ in range(obs_size)] for _ in range(obs_size)],
                        [[self.threat_attack_max for _ in range(obs_size)] for _ in range(obs_size)]],
                dtype=np.float32),
            dtype=np.float32
        )
        
        # game state
        self.grid = None
        self.agent_position = None
        self.health = None
        self.hunger = None
        self.attack = None
        self.threat_attacks = {}
        self.food_values = {}
        self._previous_position = None
        self._previous_health = None
        self._previous_hunger = None 
        
        # for rendering
        self.window = None
        self.clock = None
        self.window_size = 800 
        self.render_mode = render_mode

    def _get_obs(self):
        # pad environment on edges as walls
        padded_grid = np.ones((self.size + 2*self.observation_range, self.size + 2*self.observation_range), 
                               dtype=np.int8)
        padded_food_values = np.zeros_like(padded_grid, dtype=np.float32)
        padded_threat_attacks = np.zeros_like(padded_grid, dtype=np.float32)
    
        # fill real grid area
        padded_grid[self.observation_range:self.observation_range+self.size,
                    self.observation_range:self.observation_range+self.size] = self.grid
        for pos, value in self.food_values.items():
            padded_food_values[pos[0] + self.observation_range, 
                            pos[1] + self.observation_range] = value
        
        for pos, attack in self.threat_attacks.items():
            padded_threat_attacks[pos[0] + self.observation_range, 
                                pos[1] + self.observation_range] = attack
        
        # observation window coordinate
        center_x = self.agent_position[0] + self.observation_range
        center_y = self.agent_position[1] + self.observation_range
        
        # extract observation window
        observation = np.stack([
            padded_grid[
                center_x - self.observation_range : center_x + self.observation_range + 1,
                center_y - self.observation_range : center_y + self.observation_range + 1
            ],
            padded_food_values[
                center_x - self.observation_range : center_x + self.observation_range + 1,
                center_y - self.observation_range : center_y + self.observation_range + 1
            ],
            padded_threat_attacks[
                center_x - self.observation_range : center_x + self.observation_range + 1,
                center_y - self.observation_range : center_y + self.observation_range + 1
            ]
        ])
        
        return observation

    def _get_info(self):
        return {
            "health": self.health,
            "hunger": self.hunger,
            "attack": self.attack,
            "steps": self.current_step,
            "agent_position": self.agent_position
        }
    
    # reset the environment to initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.food_values = {}
        self.threat_positions = {}
        self.threat_attacks = {}
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        
        # boarders
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # initialize agent random position not on wall
        valid_positions = np.where(self.grid == 0)
        idx = self.np_random.integers(0, len(valid_positions[0]))
        self.agent_position = (int(valid_positions[0][idx]), int(valid_positions[1][idx]))
        self.grid[self.agent_position] = 4

        # place initial food
        for _ in range(self.num_food):
            empty_positions = np.where(self.grid == 0)
            if len(empty_positions[0]) > 0:
                idx = self.np_random.integers(0, len(empty_positions[0]))
                pos = (int(empty_positions[0][idx]), int(empty_positions[1][idx]))
                self.grid[pos] = 2
                self.food_values[pos] = self.np_random.uniform(
                    self.food_value_min, self.food_value_max)
                
        # place initial threats
        for _ in range(self.num_threats):
            empty_positions = np.where(self.grid == 0)
            if len(empty_positions[0]) > 0:
                idx = self.np_random.integers(0, len(empty_positions[0]))
                pos = (int(empty_positions[0][idx]), int(empty_positions[1][idx]))
                self.grid[pos] = 3
                attack_value = self.np_random.uniform(
                    self.threat_attack_min, self.threat_attack_max)
                self.threat_positions[pos] = {
                    'attack': attack_value,
                    'last_move': 0
                }
                self.threat_attacks[pos] = attack_value
            
        # initialize status
        self.health = 100.0
        self.hunger = 100.0
        self.attack = self.np_random.uniform(self.agent_attack_min, self.agent_attack_max)

        self.current_step = 0
        self._previous_position = self.agent_position
        self._previous_health = self.health
        self._previous_hunger = self.hunger
        
        return self._get_obs(), self._get_info()

    # place items on the grid
    def _place_items(self, item_type, num_items):
        if item_type == 2:  # food
            for _ in range(num_items):
                empty_positions = np.where(self.grid == 0)
                if len(empty_positions[0]) > 0:
                    idx = self.np_random.integers(0, len(empty_positions[0]))
                    pos = (int(empty_positions[0][idx]), int(empty_positions[1][idx]))
                    self.grid[pos] = item_type
                    self.food_values[pos] = self.np_random.uniform(
                        self.food_value_min, self.food_value_max)

        elif item_type == 3:  # threat
            for _ in range(num_items):
                empty_positions = np.where(self.grid == 0)
                if len(empty_positions[0]) > 0:
                    idx = self.np_random.integers(0, len(empty_positions[0]))
                    pos = (int(empty_positions[0][idx]), int(empty_positions[1][idx]))
                    self.grid[pos] = item_type
                    attack_value = self.np_random.uniform(
                        self.threat_attack_min, self.threat_attack_max)
                    self.threat_positions[pos] = {
                        'attack': attack_value,
                        'last_move': 0
                    }
                    self.threat_attacks[pos] = attack_value

    def _move_threats(self):
        new_threat_positions = {}
        new_threat_attacks = {}
        
        for pos, threat_data in self.threat_positions.items():
            # clear old pos
            self.grid[pos] = 0
            
            # check if agent is in perception range
            agent_in_range = (
                abs(pos[0] - self.agent_position[0]) <= self.threat_perception_range and
                abs(pos[1] - self.agent_position[1]) <= self.threat_perception_range
            )
            
            if agent_in_range and threat_data['attack'] > self.attack:
                # move towards agent
                dx = np.clip(self.agent_position[0] - pos[0], -1, 1)
                dy = np.clip(self.agent_position[1] - pos[1], -1, 1)
                new_pos = (pos[0] + dx, pos[1] + dy)
            else:
                # random move
                directions = [(0,1), (0,-1), (1,0), (-1,0)]
                dx, dy = self.np_random.choice(directions)
                new_pos = (pos[0] + dx, pos[1] + dy)
            
            # new position is valid
            if (0 < new_pos[0] < self.size-1 and 
                0 < new_pos[1] < self.size-1 and 
                self.grid[new_pos] == 0):
                # update threat pos
                self.grid[new_pos] = 3
                new_threat_positions[new_pos] = threat_data
                new_threat_attacks[new_pos] = threat_data['attack']
            else:
                # if cannot move then stay
                self.grid[pos] = 3
                new_threat_positions[pos] = threat_data
                new_threat_attacks[pos] = threat_data['attack']
        
        self.threat_positions = new_threat_positions
        self.threat_attacks = new_threat_attacks
        
    def step(self, action):
        self.current_step += 1
        self._previous_position = self.agent_position 
        self._previous_health = self.health
        self._previous_hunger = self.hunger
        old_pos = (self.agent_position[0], self.agent_position[1])
        
        # execute action
        if action == 1: # up
            new_pos = (max(0, self.agent_position[0] - 1), self.agent_position[1])
        elif action == 2: # down
            new_pos = (min(self.size - 1, self.agent_position[0] + 1), self.agent_position[1])
        elif action == 3: # left
            new_pos = (self.agent_position[0], max(0, self.agent_position[1] - 1))
        elif action == 4: # right
            new_pos = (self.agent_position[0], min(self.size - 1, self.agent_position[1] + 1))
        else: # stay
            new_pos = (self.agent_position[0], self.agent_position[1])
        
        # update hunger
        self.hunger = max(0, self.hunger - self.hungry_decay)
        if self.hunger < 20: # reduce health
            self.health = max(0, self.health - 1)

        # movement and interactions
        moved = False
        # 0: empty, 1: wall, 2: food, 3: threat, 4: agent
        if self.grid[new_pos] != 1:
            # if move into a threat, attack
            if self.grid[new_pos] == 3:
                threat_attack = self.threat_attacks.get(new_pos, self.threat_attack_min)
                if self.attack >= threat_attack: # agent eats threat
                    self.hunger = min(100, self.hunger + threat_attack)
                    self.grid[old_pos] = 0
                    self.grid[new_pos] = 4
                    self.agent_position = new_pos
                    if new_pos in self.threat_positions:
                        del self.threat_positions[new_pos]
                    if new_pos in self.threat_attacks:
                        del self.threat_attacks[new_pos]
                    self._place_items(3, num_items=1)
                    moved = True
                else:
                    # agent lose health
                    self.health = max(0, self.health - threat_attack // 2)
                    moved = False
            else:
                # normal movement
                self.grid[old_pos] = 0
                self.agent_position = new_pos
                moved = True
                
                # food collection
                if self.grid[new_pos] == 2:
                    food_value = self.food_values.get(new_pos, self.food_value_min)
                    self.hunger = min(100, self.hunger + food_value)
                    if new_pos in self.food_values:
                        del self.food_values[new_pos]
                    #self._place_items(2, num_items=1)
            
        if moved:
            self.grid[self.agent_position] = 4
            self._move_threats()

        # move threats
        new_threat_positions = {}
        new_threat_attacks = {}
        
        for pos, threat_data in list(self.threat_positions.items()):
            self.grid[pos] = 0
            agent_in_range = (
                abs(pos[0] - self.agent_position[0]) <= self.threat_perception_range and
                abs(pos[1] - self.agent_position[1]) <= self.threat_perception_range
            )
            
            attack_value = threat_data['attack']
            # if agent in range and attack is greater than agent's
            if agent_in_range and attack_value > self.attack:
                dx = np.clip(self.agent_position[0] - pos[0], -1, 1)
                dy = np.clip(self.agent_position[1] - pos[1], -1, 1)
                new_pos = (int(pos[0] + dx), int(pos[1] + dy))
            else: # random move
                directions = [(0,1), (0,-1), (1,0), (-1,0)]
                dx, dy = self.np_random.choice(directions)
                new_pos = (int(pos[0] + dx), int(pos[1] + dy))
            
            # new position is valid
            if (0 <= new_pos[0] < self.size and 
                0 <= new_pos[1] < self.size and 
                self.grid[new_pos] not in [1, 2, 3, 4]):
                self.grid[new_pos] = 3
                new_threat_positions[new_pos] = threat_data
                new_threat_attacks[new_pos] = attack_value
            else: # cannot move, stay
                self.grid[pos] = 3
                new_threat_positions[pos] = threat_data
                new_threat_attacks[pos] = attack_value
        
        self.threat_positions = new_threat_positions
        self.threat_attacks = new_threat_attacks

        # after movement
        threats_to_remove = []
        for threat_pos in list(self.threat_positions.keys()):
            if threat_pos == self.agent_position: 
                threat_attack = self.threat_attacks[threat_pos]
                if self.attack >= threat_attack:
                    self.hunger = min(100, self.hunger + threat_attack)
                    threats_to_remove.append(threat_pos)
                else:
                    self.health = max(0, self.health - threat_attack // 2)
        
        # remove threats, place new
        for pos in threats_to_remove:
            if pos in self.threat_positions:
                self.grid[pos] = 0
                del self.threat_positions[pos]
                del self.threat_attacks[pos]
                self._place_items(3, num_items=1)

        reward = self._calculate_reward()
        terminated = self.health <= 0 or self.current_step >= self.max_steps
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self):
        reward = 0.0
        
        # health
        reward += self.health / 100.0
        if self.health < 30:
            reward -= 2.0
        if self.health <= 0:
            reward -= 10.0

        # # hunger
        # if self.hunger > self._previous_hunger:
        #     reward += 1.0
        # elif self.hunger < 20:
        #     reward -= 1.0
        
        # # beat threats
        # if self.grid[self._previous_position] == 3 and self.attack >= self.threat_attacks.get(self._previous_position, 35):
        #     reward += 2.0
        # elif self.health < self._previous_health:
        #     reward -= 1.0
        
        # # staying near walls
        # if (self.agent_position[0] in [0, self.size-1] or 
        #     self.agent_position[1] in [0, self.size-1]):
        #     reward -= 0.3

        return reward

    def _render_frame(self):
        if self.window is None:
            self.window = pygame.Surface((self.window_size, self.window_size))
        
        # fill background
        self.window.fill((255, 255, 255))
        cell_size = self.window_size // self.size

        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # draw grid lines
        for i in range(self.size + 1):
            pygame.draw.line(self.window, (200, 200, 200), 
                           (0, i * cell_size), 
                           (self.window_size, i * cell_size))
            pygame.draw.line(self.window, (200, 200, 200), 
                           (i * cell_size, 0), 
                           (i * cell_size, self.window_size))
        
        # draw walls and entities
        font = pygame.font.Font(None, cell_size // 2)
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
                    value = self.food_values.get((i, j))
                    if value is not None:
                        text = font.render(f"{value:.0f}", True, (0, 0, 0))
                        text_rect = text.get_rect(center=(j * cell_size + cell_size//2, 
                                                        i * cell_size + cell_size//2))
                        self.window.blit(text, text_rect)

        # draw agent
        agent_x = self.agent_position[1] * cell_size + cell_size//2
        agent_y = self.agent_position[0] * cell_size + cell_size//2
        pygame.draw.circle(self.window, (0, 0, 255),
                        (agent_x, agent_y),
                        cell_size//3)
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 3:  # threat
                    pygame.draw.circle(self.window, (255, 0, 0),
                                    (j * cell_size + cell_size//2, 
                                    i * cell_size + cell_size//2),
                                    cell_size//4)
                    attack = self.threat_attacks.get((i, j), 0)
                    text = font.render(f"{attack:.0f}", True, (0, 0, 0))
                    text_rect = text.get_rect(center=(j * cell_size + cell_size//2, 
                                            i * cell_size + cell_size//2))
                    self.window.blit(text, text_rect)
        
        # observation window
        fog_surface = pygame.Surface((self.window_size, self.window_size))
        fog_surface.fill((100, 100, 100))
        fog_surface.set_alpha(64)

        visible_area = pygame.Surface((self.window_size, self.window_size))
        visible_area.fill((100, 100, 100))

        visible_rect_size = (2 * self.observation_range + 1) * cell_size
        visible_rect_x = self.agent_position[1] * cell_size - self.observation_range * cell_size
        visible_rect_y = self.agent_position[0] * cell_size - self.observation_range * cell_size

        pygame.draw.rect(fog_surface, (0, 0, 0, 0),
                        (visible_rect_x, visible_rect_y, 
                         visible_rect_size, visible_rect_size))
        
        self.window.blit(fog_surface, (0, 0))
        
        # status bars
        bar_height = 10
        health_width = int(self.window_size * (self.health / 100))
        pygame.draw.rect(self.window, (255, 0, 0),
                        (0, self.window_size - 2*bar_height, self.window_size, bar_height))
        pygame.draw.rect(self.window, (0, 255, 0),
                        (0, self.window_size - 2*bar_height, health_width, bar_height))
        
        hunger_width = int(self.window_size * (self.hunger / 100))
        pygame.draw.rect(self.window, (100, 100, 100),
                        (0, self.window_size - bar_height, self.window_size, bar_height))
        pygame.draw.rect(self.window, (255, 255, 0),
                        (0, self.window_size - bar_height, hunger_width, bar_height))
        
        view = pygame.surfarray.array3d(self.window)
        return np.transpose(view, (1, 0, 2))

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Survival Game")
            
            frame = self._render_frame()
            surf = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            
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