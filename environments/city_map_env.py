import pygame
import random
import math
from algorithms.astar_graph import astar_graph_search_generator, euclidean_distance

# Bảng màu
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 200, 0)
ORANGE = (255, 165, 0)
MAGENTA = (200, 0, 255)
CYAN = (0, 255, 255) # Màu của "Chiếc xe" di chuyển

class CityMapEnv:
    def __init__(self, w=800, h=600, num_nodes=50):
        self.w = w
        self.h = h
        self.num_nodes = num_nodes
        self.nodes = []
        self.edges = {}
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI GPS Navigation - A* Animation')
        self.font = pygame.font.SysFont('segoeui', 18)
        
        self._generate_map()
        
        self.start_node = 0
        self.end_node = self.num_nodes - 1
        
        self.path = None
        self.closed_set = set()
        self.open_set = set()
        self.current_node = None
        self.search_done = False
        
        # --- CÁC BIẾN MỚI CHO CHUYỂN ĐỘNG CƠ ---
        self.agent_pos = None      
        self.current_target_idx = 1  # Bắt đầu từ 1 thay vì 0, và đổi tên biến
        self.agent_speed = 5       
        self.move_dir = 1            # Thêm biến này để xác định hướng (1 là tiến, -1 là lùi)
        
        self.search_generator = astar_graph_search_generator(self.nodes, self.edges, self.start_node, self.end_node)

    def _generate_map(self):
        for _ in range(self.num_nodes):
            x = random.randint(50, self.w - 50)
            y = random.randint(50, self.h - 50)
            self.nodes.append((x, y))
            self.edges[len(self.nodes)-1] = []

        connection_radius = 150 
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dist = euclidean_distance(self.nodes[i], self.nodes[j])
                if dist < connection_radius:
                    self.edges[i].append((j, dist))
                    self.edges[j].append((i, dist))

    def step_search(self):
        """Xử lý logic 2 giai đoạn: Tìm đường và Di chuyển khứ hồi"""
        # GIAI ĐOẠN 1: AI đang lan tỏa tìm đường
        if not self.search_done:
            try:
                self.current_node, self.closed_set, self.open_set, self.path = next(self.search_generator)
                if self.path is not None:
                    self.search_done = True
                    # Khởi tạo vị trí xe bắt đầu từ Điểm Xuất Phát
                    self.agent_pos = list(self.nodes[self.path[0]]) 
                    self.current_target_idx = 1
                    self.move_dir = 1
            except StopIteration:
                self.search_done = True
        
        # GIAI ĐOẠN 2: Di chuyển khứ hồi (đi đi lại lại)
        else:
            if self.path:
                # Lấy tọa độ của nút mục tiêu tiếp theo
                target_node = self.nodes[self.path[self.current_target_idx]]
                
                # Tính toán Vector hướng đi
                dx = target_node[0] - self.agent_pos[0]
                dy = target_node[1] - self.agent_pos[1]
                dist = math.hypot(dx, dy) 
                
                # Nếu đã cán mốc nút mục tiêu
                if dist < self.agent_speed:
                    self.agent_pos = list(target_node) # Khớp chính xác vào tâm nút
                    
                    # Chuyển sang nút tiếp theo
                    self.current_target_idx += self.move_dir
                    
                    # LOGIC ĐẢO CHIỀU: Nếu vượt qua đích hoặc lùi qua vạch xuất phát
                    if self.current_target_idx >= len(self.path):
                        self.move_dir = -1 # Quay đầu đi lùi
                        self.current_target_idx = len(self.path) - 2 # Nút kề cuối
                    elif self.current_target_idx < 0:
                        self.move_dir = 1  # Quay đầu đi tới
                        self.current_target_idx = 1 # Nút thứ hai
                else:
                    # Di chuyển xe từng chút một dọc theo vector
                    self.agent_pos[0] += (dx / dist) * self.agent_speed
                    self.agent_pos[1] += (dy / dist) * self.agent_speed

    def render(self):
        self.display.fill(WHITE)

        # 1. Vẽ đường phố
        for i in self.edges:
            for neighbor, _ in self.edges[i]:
                pygame.draw.line(self.display, GRAY, self.nodes[i], self.nodes[neighbor], 2)

        # 2. Vẽ tuyến đường màu đỏ
        if self.path:
            for i in range(len(self.path) - 1):
                p1 = self.nodes[self.path[i]]
                p2 = self.nodes[self.path[i+1]]
                pygame.draw.line(self.display, RED, p1, p2, 6)

        # 3. Vẽ các nút
        for idx, pos in enumerate(self.nodes):
            color = BLUE
            radius = 6
            if idx in self.closed_set: color = ORANGE 
            elif idx in self.open_set: color = YELLOW 
            if idx == self.current_node and not self.search_done:
                color = MAGENTA
                radius = 10
            if idx == self.start_node:
                color = GREEN
                radius = 12
            elif idx == self.end_node:
                color = RED
                radius = 12
            pygame.draw.circle(self.display, color, pos, radius)

        # --- PHẦN MỚI: VẼ CHIẾC XE DI CHUYỂN ---
        if self.agent_pos:
            # Vẽ viền ngoài màu đen
            pygame.draw.circle(self.display, BLACK, (int(self.agent_pos[0]), int(self.agent_pos[1])), 12)
            # Vẽ lõi trong màu lục lam (Cyan)
            pygame.draw.circle(self.display, CYAN, (int(self.agent_pos[0]), int(self.agent_pos[1])), 8)

        info_text = "Orange: Explored | Yellow: Open | Magenta: Current Node | Cyan: Vehicle Moving"
        text_surface = self.font.render(info_text, True, BLACK)
        self.display.blit(text_surface, (10, 10))

        pygame.display.flip()