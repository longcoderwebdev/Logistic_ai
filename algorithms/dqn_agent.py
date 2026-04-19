import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ==========================================
# 1. MẠNG NEURAL NETWORK (MODEL)
# ==========================================
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Lưu model sau khi train xong
        import os
        model_folder_path = './logs'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# ==========================================
# 2. BỘ HUẤN LUYỆN (TRAINER)
# ==========================================
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # reshape thành (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Dự đoán Q-values cho trạng thái hiện tại
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Cập nhật Q-value theo phương trình Bellman
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Tính toán Loss và cập nhật trọng số (Backpropagation)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# ==========================================
# 3. TÁC TỬ (AGENT)
# ==========================================
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class DQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Tham số khám phá (Exploration rate)
        self.gamma = 0.9 # Hệ số chiết khấu (Discount rate)
        self.memory = deque(maxlen=MAX_MEMORY) # Replay Buffer
        
        # Input: 11 trạng thái. Hidden: 256 nơ-ron. Output: 3 hành động
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Đây là hàm quan trọng nhất để viết báo cáo.
        Định nghĩa vector 11 trạng thái: 
        [Nguy hiểm thẳng, Nguy hiểm phải, Nguy hiểm trái, 
         Hướng Trái, Hướng Phải, Hướng Lên, Hướng Xuống,
         Mồi Trái, Mồi Phải, Mồi Lên, Mồi Xuống]
        """
        head = game.head_dqn
        
        # Tính toán các điểm xung quanh đầu rắn
        point_l = [head.x - 20, head.y]
        point_r = [head.x + 20, head.y]
        point_u = [head.x, head.y - 20]
        point_d = [head.x, head.y + 20]
        
        # Hàm hỗ trợ kiểm tra va chạm bằng class Point
        from environments.snake_env import Point
        dir_l = game.direction_dqn.name == 'LEFT'
        dir_r = game.direction_dqn.name == 'RIGHT'
        dir_u = game.direction_dqn.name == 'UP'
        dir_d = game.direction_dqn.name == 'DOWN'

        state = [
            # Nguy hiểm phía trước
            (dir_r and game.is_collision(Point(*point_r), "DQN")) or 
            (dir_l and game.is_collision(Point(*point_l), "DQN")) or 
            (dir_u and game.is_collision(Point(*point_u), "DQN")) or 
            (dir_d and game.is_collision(Point(*point_d), "DQN")),

            # Nguy hiểm bên phải
            (dir_u and game.is_collision(Point(*point_r), "DQN")) or 
            (dir_d and game.is_collision(Point(*point_l), "DQN")) or 
            (dir_l and game.is_collision(Point(*point_u), "DQN")) or 
            (dir_r and game.is_collision(Point(*point_d), "DQN")),

            # Nguy hiểm bên trái
            (dir_d and game.is_collision(Point(*point_r), "DQN")) or 
            (dir_u and game.is_collision(Point(*point_l), "DQN")) or 
            (dir_r and game.is_collision(Point(*point_u), "DQN")) or 
            (dir_l and game.is_collision(Point(*point_d), "DQN")),
            
            # Hướng di chuyển hiện tại
            dir_l, dir_r, dir_u, dir_d,
            
            # Vị trí thức ăn so với đầu rắn
            game.food.x < game.head_dqn.x,  # Thức ăn bên trái
            game.food.x > game.head_dqn.x,  # Thức ăn bên phải
            game.food.y < game.head_dqn.y,  # Thức ăn bên trên
            game.food.y > game.head_dqn.y   # Thức ăn bên dưới
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Tradeoff giữa Exploration (khám phá ngẫu nhiên) và Exploitation (Dùng model)
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Chọn hành động ngẫu nhiên lúc mới học
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Suy luận từ Mạng Neural
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move