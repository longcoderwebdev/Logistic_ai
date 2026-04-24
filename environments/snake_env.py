import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80 # Tốc độ cao để train nhanh

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Cuộc chiến AI: A* Search vs Deep Q-Network')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Khởi tạo Rắn 1 (A*) - Xuất phát bên trái
        self.head_astar = Point(self.w/4, self.h/2)
        self.snake_astar = [self.head_astar, 
                           Point(self.head_astar.x-BLOCK_SIZE, self.head_astar.y)]
        
        # Khởi tạo Rắn 2 (DQN) - Xuất phát bên phải
        self.head_dqn = Point(3*self.w/4, self.h/2)
        self.snake_dqn = [self.head_dqn, 
                         Point(self.head_dqn.x+BLOCK_SIZE, self.head_dqn.y)]
        
        self.direction_astar = Direction.RIGHT
        self.direction_dqn = Direction.LEFT
        
        self.score_astar = 0
        self.score_dqn = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake_astar or self.food in self.snake_dqn:
            self._place_food()

    def _move(self, action, agent_name):
        # Xác định đang xử lý cho rắn nào
        if agent_name == "ASTAR":
            current_dir = self.direction_astar
            head = self.head_astar
        else:
            current_dir = self.direction_dqn
            head = self.head_dqn

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(current_dir)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Đi thẳng
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Rẽ phải
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Rẽ trái

        x = head.x
        y = head.y
        if new_dir == Direction.RIGHT: x += BLOCK_SIZE
        elif new_dir == Direction.LEFT: x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN: y += BLOCK_SIZE
        elif new_dir == Direction.UP: y -= BLOCK_SIZE
        
        new_head = Point(x, y)

        # Cập nhật lại hướng và đầu
        if agent_name == "ASTAR":
            self.direction_astar = new_dir
            self.head_astar = new_head
        else:
            self.direction_dqn = new_dir
            self.head_dqn = new_head

    def step(self, action_astar, action_dqn):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0, 0, True, self.score_astar, self.score_dqn, "QUIT"

        # 1. Di chuyển cả 2 con
        self._move(action_astar, "ASTAR")
        self._move(action_dqn, "DQN")
        
        self.snake_astar.insert(0, self.head_astar)
        self.snake_dqn.insert(0, self.head_dqn)
        
        # 2. Kiểm tra va chạm (Logic đối kháng quan trọng)
        reward_astar = 0
        reward_dqn = 0
        game_over = False
        winner = None

        # Kiểm tra Rắn A* tử vong
        if self.is_collision(self.head_astar, "ASTAR"):
            game_over = True
            reward_astar = -10
            reward_dqn = 10 
            winner = "DQN (Deep Learning)"
            
        # Kiểm tra Rắn DQN tử vong
        if self.is_collision(self.head_dqn, "DQN"):
            game_over = True
            reward_dqn = -10
            reward_astar = 10
            winner = "A* Search (Classic AI)"

        if game_over:
            return reward_astar, reward_dqn, game_over, self.score_astar, self.score_dqn, winner

        # 3. Tranh giành mồi
        if self.head_astar == self.food:
            self.score_astar += 1
            reward_astar = 10
            self._place_food()
            self.snake_dqn.pop()
        elif self.head_dqn == self.food:
            self.score_dqn += 1
            reward_dqn = 10
            self._place_food()
            self.snake_astar.pop()
        else:
            self.snake_astar.pop()
            self.snake_dqn.pop()
            
        return reward_astar, reward_dqn, game_over, self.score_astar, self.score_dqn, winner

    def is_collision(self, pt, agent_name):
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if agent_name == "ASTAR" and pt in self.snake_astar[1:]: return True
        if agent_name == "DQN" and pt in self.snake_dqn[1:]: return True
        if agent_name == "ASTAR" and pt in self.snake_dqn: return True
        if agent_name == "DQN" and pt in self.snake_astar: return True
        return False

    def render(self):
        self.display.fill(BLACK)
        # Rắn A* (Màu Xanh Dương)
        for pt in self.snake_astar:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Rắn DQN (Màu Xanh Lá)
        for pt in self.snake_dqn:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        # Mồi
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
        self.clock.tick(SPEED)