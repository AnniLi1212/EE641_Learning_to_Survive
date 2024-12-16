import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class SurvivalGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=1000, size=16, 
                 num_food=10, num_threats=5, 
                 food_value_min=10, food_value_max=30,
                 threat_attack_min=20, threat_attack_max=40,
                 agent_attack_min=25, agent_attack_max=45,
                 hungry_decay=2, observation_range=4, threat_perception_range=2,
                 num_caves=5, cave_health_recovery=2, hungry_health_penalty=3):
        super().__init__()
        pygame.init()

        self.agent_img = pygame.image.load('src/utils/imgs/agent.png')
        self.threat_img = pygame.image.load('src/utils/imgs/threat.png')
        self.food_img = pygame.image.load('src/utils/imgs/food.png')

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
        self.num_caves = num_caves
        self.cave_health_recovery = cave_health_recovery
        self.threat_positions = {}
        self.current_step = 0
        self.cave_positions = set()
        self.action_records = {}
        self.hungry_health_penalty = hungry_health_penalty

        # 0: stay, 1-4: move in 4 directions
        self.action_space = spaces.Discrete(5)

        # can only see within the observation range
        obs_size = 2 * self.observation_range + 1
        self.observation_space = spaces.Box(
            low=np.array([
                # one-hot grid category channels
                [[0 for _ in range(obs_size)] for _ in range(obs_size)],  # wall
                [[0 for _ in range(obs_size)] for _ in range(obs_size)],  # agent
                [[0 for _ in range(obs_size)] for _ in range(obs_size)],  # cave
                # food values channel
                [[0 for _ in range(obs_size)] for _ in range(obs_size)],
                # threat values channel
                [[0 for _ in range(obs_size)] for _ in range(obs_size)]
            ], dtype=np.float32),
            high=np.array([
                # one-hot grid category channels
                [[1 for _ in range(obs_size)] for _ in range(obs_size)],  # wall
                [[1 for _ in range(obs_size)] for _ in range(obs_size)],  # agent
                [[1 for _ in range(obs_size)] for _ in range(obs_size)],  # cave
                # food values channel
                [[self.food_value_max for _ in range(obs_size)] for _ in range(obs_size)],
                # threat values channel
                [[self.threat_attack_max for _ in range(obs_size)] for _ in range(obs_size)]
            ], dtype=np.float32),
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
        self.window_size = 1200 
        self.render_mode = render_mode

    def _get_obs(self):
        # create padded environment
        obs_range = self.observation_range
        padded_size = self.size + 2 * obs_range
        
        padded_grid = np.ones((padded_size, padded_size), dtype=np.int8)
        padded_food = np.zeros((padded_size, padded_size), dtype=np.float32)
        padded_threats = np.zeros((padded_size, padded_size), dtype=np.float32)
        
        # actual environment
        for i in range(self.size):
            for j in range(self.size):
                if 1 in self.grid[i,j]:  # wall
                    padded_grid[i + obs_range, j + obs_range] = 1
                elif 4 in self.grid[i,j]:  # agent
                    padded_grid[i + obs_range, j + obs_range] = 2
                elif 5 in self.grid[i,j]:  # cave
                    padded_grid[i + obs_range, j + obs_range] = 3
        
        # fill food and threat values
        for pos, value in self.food_values.items():
            padded_food[int(pos[0]) + obs_range, int(pos[1]) + obs_range] = value
            
        for pos, attack in self.threat_attacks.items():
            padded_threats[int(pos[0]) + obs_range, int(pos[1]) + obs_range] = attack
        
        # extract observation window
        center_x = self.agent_position[0] + obs_range
        center_y = self.agent_position[1] + obs_range
        
        grid_window = padded_grid[
            center_x - obs_range : center_x + obs_range + 1,
            center_y - obs_range : center_y + obs_range + 1
        ]
        
        food_window = padded_food[
            center_x - obs_range : center_x + obs_range + 1,
            center_y - obs_range : center_y + obs_range + 1
        ]
        
        threat_window = padded_threats[
            center_x - obs_range : center_x + obs_range + 1,
            center_y - obs_range : center_y + obs_range + 1
        ]
        
        # create one-hot encoded grid
        obs_size = 2 * obs_range + 1
        one_hot_grid = np.zeros((3, obs_size, obs_size), dtype=np.float32)
        for i in range(3):  # 3 categories: wall,agent,cave
            one_hot_grid[i] = (grid_window == i+1).astype(np.float32)
        
        # stack all channels
        observation = np.concatenate([
            one_hot_grid.astype(np.float32),
            food_window[np.newaxis, :, :].astype(np.float32),
            threat_window[np.newaxis, :, :].astype(np.float32)
        ], axis=0)
        
        return observation

    def _get_info(self):
        return {
            "health": self.health,
            "hunger": self.hunger,
            "attack": self.attack,
            "steps": self.current_step,
            "agent_position": self.agent_position,
            "action_records": self.action_records
        }
    
    # reset the environment to initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.food_values = {}
        self.threat_positions = {}
        self.threat_attacks = {}
        self.cave_positions = set()
        self.action_records = {
            "Stay": 0, 
            "Move": 0, 
            "Eat Food, Hungry < 50": 0, 
            "Eat Food, Hungry >= 50": 0, 
            "Fight Stronger Threat": 0, 
            "Fight Weaker Threat": 0,
            "Enter Cave, Health < 100": 0,
            "Enter Cave, Health >= 100": 0
        }

        self.grid = np.empty((self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i, j] = set()
        
        # boarders
        for i in range(self.size):
            self.grid[0,i].add(1)
            self.grid[-1,i].add(1)
            self.grid[i,0].add(1)
            self.grid[i,-1].add(1)
        
        # initialize agent random position not on wall
        valid_positions = [(i,j) for i in range(1, self.size-1) 
                           for j in range(1, self.size-1) 
                           if 1 not in self.grid[i,j]]
        idx = self.np_random.integers(0, len(valid_positions))
        self.agent_position = valid_positions[idx]
        self.grid[self.agent_position].add(4)

        # place initial food
        for _ in range(self.num_food):
            empty_positions = [(i,j) for i in range(1, self.size-1) 
                              for j in range(1, self.size-1) 
                              if 1 not in self.grid[i,j]]
            if len(empty_positions) > 0:
                idx = self.np_random.integers(0, len(empty_positions))
                pos = empty_positions[idx]
                self.grid[pos].add(2)
                self.food_values[pos] = self.np_random.uniform(
                    self.food_value_min, self.food_value_max)
                
        # place initial threats
        for _ in range(self.num_threats):
            empty_positions = [(i,j) for i in range(1, self.size-1) 
                              for j in range(1, self.size-1) 
                              if 1 not in self.grid[i,j]]
            if len(empty_positions) > 0:
                idx = self.np_random.integers(0, len(empty_positions))
                pos = empty_positions[idx]
                self.grid[pos].add(3)
                attack_value = self.np_random.uniform(
                    self.threat_attack_min, self.threat_attack_max)
                self.threat_positions[pos] = {
                    'attack': attack_value,
                    'last_move': 0
                }
                self.threat_attacks[pos] = attack_value
            
        # place caves
        for _ in range(self.num_caves):
            empty_positions = [(i,j) for i in range(1, self.size-1) 
                              for j in range(1, self.size-1) 
                              if 1 not in self.grid[i,j]]
            idx = self.np_random.integers(0, len(empty_positions))
            pos = empty_positions[idx]
            self.grid[pos].add(5)
            self.cave_positions.add(pos)
            
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
                valid_positions = [(i, j) for i in range(1, self.size-1) 
                                   for j in range(1, self.size-1) 
                                   if 1 not in self.grid[i,j] and 4 not in self.grid[i,j]]
                if valid_positions:
                    idx = self.np_random.integers(0, len(valid_positions))
                    pos = valid_positions[idx]
                    self.grid[pos].add(item_type)
                    self.food_values[pos] = self.np_random.uniform(
                        self.food_value_min, self.food_value_max)

        elif item_type == 3:  # threat
            for _ in range(num_items):
                valid_positions = [(i, j) for i in range(1, self.size-1) 
                                   for j in range(1, self.size-1) 
                                   if 1 not in self.grid[i,j] and 4 not in self.grid[i,j]]
                if valid_positions:
                    idx = self.np_random.integers(0, len(valid_positions))
                    pos = valid_positions[idx]
                    self.grid[pos].add(item_type)
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
        
        for pos in sorted(self.threat_positions.keys()):
            threat_data = self.threat_positions[pos]

            grid_pos = (int(pos[0]), int(pos[1]))
            
            # check if agent is in perception range
            agent_in_range = (
                abs(grid_pos[0] - self.agent_position[0]) <= self.threat_perception_range and
                abs(grid_pos[1] - self.agent_position[1]) <= self.threat_perception_range
            )

            if agent_in_range and threat_data['attack'] > self.attack:
                # move towards agent
                dx = np.clip(self.agent_position[0] - grid_pos[0], -1, 1)
                dy = np.clip(self.agent_position[1] - grid_pos[1], -1, 1)
                new_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
            else:
                # random move
                directions = [(0,1), (0,-1), (1,0), (-1,0)]
                dx, dy = self.np_random.choice(directions)
                new_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
            
            # new position is valid
            if (0 < new_pos[0] < self.size-1 and 
                0 < new_pos[1] < self.size-1 and 
                1 not in self.grid[new_pos]):

                # update threat pos
                self.grid[grid_pos].discard(3)
                self.grid[new_pos].add(3)
                base_pos = (new_pos[0], new_pos[1])
                offset = 0

                while base_pos in new_threat_positions:
                    offset += 0.1
                    base_pos = (new_pos[0] + offset, new_pos[1])

                new_threat_positions[base_pos] = threat_data
                new_threat_attacks[base_pos] = threat_data['attack']
            else:
                # if cannot move then stay
                new_threat_positions[pos] = threat_data
                new_threat_attacks[pos] = threat_data['attack']
                self.grid[grid_pos].add(3)
        
        self.threat_positions = new_threat_positions
        self.threat_attacks = new_threat_attacks
        
    def step(self, action):
        self.current_step += 1
        self._previous_position = self.agent_position 
        self._previous_health = self.health
        self._previous_hunger = self.hunger
        
        # update hunger
        self.hunger = max(0, self.hunger - self.hungry_decay)
        if self.hunger < 20: # reduce health
            self.health = max(0, self.health - self.hungry_health_penalty)

        # Move agent based on action
        if action == 1: # up
            new_pos = (max(0, self.agent_position[0] - 1), self.agent_position[1])
            self.action_records["Move"] += 1
        elif action == 2: # down
            new_pos = (min(self.size - 1, self.agent_position[0] + 1), self.agent_position[1])
            self.action_records["Move"] += 1
        elif action == 3: # left
            new_pos = (self.agent_position[0], max(0, self.agent_position[1] - 1))
            self.action_records["Move"] += 1
        elif action == 4: # right
            new_pos = (self.agent_position[0], min(self.size - 1, self.agent_position[1] + 1))
            self.action_records["Move"] += 1
        else: # stay
            new_pos = self.agent_position
            self.action_records["Stay"] += 1

        # movement and interactions
        moved = False
        if 1 not in self.grid[new_pos]:
            # if move into a threat, attack
            if 3 in self.grid[new_pos]:
                threat_attack = self.threat_attacks.get(new_pos, self.threat_attack_min)
                if self.attack >= threat_attack: # agent wins
                    self.action_records["Fight Weaker Threat"] += 1
                    self.hunger = min(100, self.hunger + threat_attack)
                    self.grid[new_pos].discard(3)
                    self.grid[new_pos].add(4)
                    self.agent_position = new_pos
                    if new_pos in self.threat_positions:
                        del self.threat_positions[new_pos]
                    if new_pos in self.threat_attacks:
                        del self.threat_attacks[new_pos]
                    self._place_items(3, num_items=1)
                    moved = True
                else:
                    # agent loses health
                    self.action_records["Fight Stronger Threat"] += 1
                    attack_diff = threat_attack - self.attack
                    self.health = max(0, self.health - attack_diff)
                    moved = False
            else:
                moved = True
                if self._previous_position in self.cave_positions:
                    if self.health < 100:
                        self.action_records["Enter Cave, Health < 100"] += 1
                    else:
                        self.action_records["Enter Cave, Health >= 100"] += 1
                    self.grid[self._previous_position].discard(4)
                    self.grid[self._previous_position].add(5)
                else:
                    self.grid[self._previous_position].discard(4)
                self.agent_position = new_pos
                
                # food collection
                if 2 in self.grid[new_pos]:
                    if self.hunger < 50:
                        self.action_records["Eat Food, Hungry < 50"] += 1
                    else:
                        self.action_records["Eat Food, Hungry >= 50"] += 1
                    food_value = self.food_values.get(new_pos, self.food_value_min)
                    self.hunger = min(100, self.hunger + food_value)
                    self.grid[new_pos].discard(2)
                    if new_pos in self.food_values:
                        del self.food_values[new_pos]
            
        if moved:
            if self._previous_position in self.cave_positions:
                self.grid[self._previous_position].discard(4)
                self.grid[self._previous_position].add(5)
            else:
                self.grid[self._previous_position].discard(4)

            self.grid[self.agent_position].add(4)

        self._move_threats()

        if self.agent_position in self.cave_positions:
            self.health = min(100, self.health + self.cave_health_recovery)

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
                self.grid[pos].discard(3)
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
        reward += 0.1 # staying alive
        
        # reach max steps
        if self.current_step >= self.max_steps:
            reward += 10.0

        # health
        if self.health - self._previous_health > 0:
            reward += 1
        elif self.health - self._previous_health < 0:
            reward -= 0.5
        if self.health < 30:
            reward -= 0.5
        if self.health >= 70:
            reward += 0.5
        if self.health <= 0:
            reward -= 10.0

        # eat food, value related
        if self.hunger > self._previous_hunger:
            reward += 2.0 * (self.hunger - self._previous_hunger) / 100.0
        
        # beat threats
        if self.grid[self._previous_position] == 3 and self.health > self._previous_health:
            reward += 2.0
        # elif self.health < self._previous_health:
        #     reward -= 2.0

        # cave
        if self._previous_position in self.cave_positions and self.health < 100:
            reward += 0.5
        
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

        scaled_agent = pygame.transform.scale(self.agent_img, (cell_size, cell_size))
        scaled_food = pygame.transform.scale(self.food_img, (cell_size, cell_size))
        scaled_threat = pygame.transform.scale(self.threat_img, (cell_size, cell_size))
    

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
        
        font = pygame.font.Font(None, cell_size // 2)
        # draw base objects
        for i in range(self.size):
            for j in range(self.size):
                if 1 in self.grid[i,j]: # wall
                    pygame.draw.rect(self.window, (100, 100, 100),
                                   (j * cell_size, i * cell_size, 
                                    cell_size, cell_size))
                elif 2 in self.grid[i,j]: # food
                    food_rect = scaled_food.get_rect(center=(j * cell_size + cell_size//2, 
                                                            i * cell_size + cell_size//2))
                    self.window.blit(scaled_food, food_rect)
                elif 5 in self.grid[i,j]:  # cave
                    pygame.draw.rect(self.window, (140, 70, 20),
                                   (j * cell_size, i * cell_size, 
                                    cell_size, cell_size))
        
        # draw values
        for i in range(self.size):
            for j in range(self.size):
                if 2 in self.grid[i,j]:  # food value
                    value = self.food_values.get((i, j))
                    if value is not None:
                        text = font.render(f"{value:.0f}", True, (255, 0, 0))
                        text_rect = text.get_rect(center=(j * cell_size + cell_size//2, 
                                                        i * cell_size - cell_size//4))
                        self.window.blit(text, text_rect)
                elif 3 in self.grid[i,j]:  # threat
                    threat_rect = scaled_threat.get_rect(center=(j * cell_size + cell_size//2, 
                                                            i * cell_size + cell_size//2))
                    self.window.blit(scaled_threat, threat_rect)
                    attack = self.threat_attacks.get((i, j), 0)
                    text = font.render(f"{attack:.0f}", True, (255, 0, 0))
                    text_rect = text.get_rect(center=(j * cell_size + cell_size//2, 
                                                    i * cell_size - cell_size//4))
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

        # draw agent
        agent_x = self.agent_position[1] * cell_size + cell_size//2
        agent_y = self.agent_position[0] * cell_size + cell_size//2
        agent_rect = scaled_agent.get_rect(center=(agent_x, agent_y))
        self.window.blit(scaled_agent, agent_rect)

        # status bars
        bar_height = 15
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
                self.clock = pygame.time.Clock()
            
            frame = self._render_frame()
            surf = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
            self.window.blit(surf, (0, 0))
            pygame.display.flip()

            self.clock.tick(self.metadata["render_fps"])
            
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