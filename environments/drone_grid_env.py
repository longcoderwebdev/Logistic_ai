import pygame
import random
from algorithms.tsp_heuristic import tsp_2opt_generator
from algorithms.astar import AStarSolver

# Bảng màu
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (200, 200, 200)
BLUE = (0, 100, 255)       # Điểm giao hàng
RED = (255, 50, 50)        # Đường bay đang thử nghiệm
GREEN = (0, 200, 0)        # Trạm gốc (Depot) / Đường bay tối ưu
TEXT_COLOR = (50, 50, 50)

class DroneEnv:
    def __init__(self, w=1000, h=750, num_points=12):
        self.w = w
        self.h = h
        self.num_points = num_points
        self.block_size = 25
        self.points = []
        self.obstacles = set()
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI Drone Delivery - TSP 2-Opt với Vùng Cấm Bay (Obstacles)')
        self.font = pygame.font.SysFont('segoeui', 20)
        self.large_font = pygame.font.SysFont('segoeui', 28, bold=True)
        
        self._generate_obstacles()
        self._generate_points()
        
        # Vẽ trước giao diện text thông báo Đang tính toán ma trận
        self.display.fill(WHITE)
        loading_text = self.large_font.render("Đang tính toán ma trận tìm đường A* (Khoảng 2-3 giây)...", True, RED)
        self.display.blit(loading_text, (20, 20))
        pygame.display.flip()
        
        self._build_distance_matrix()
        
        # Biến trạng thái
        self.current_route = []
        self.current_distance = float('inf')
        self.is_done = False
        
        # Khởi tạo Generator
        self.optimizer = tsp_2opt_generator(self.points, getattr(self, 'dist_matrix', None))

    def _generate_obstacles(self):
        """Tạo ngẫu nhiên các vùng cấm bay là hình chữ nhật"""
        for _ in range(10):
            w = random.randint(3, 7) * self.block_size
            h = random.randint(3, 7) * self.block_size
            x = random.randint(1, (self.w - w) // self.block_size - 1) * self.block_size
            y = random.randint(5, (self.h - h) // self.block_size - 1) * self.block_size # Né UI ở trên (y >= 5)
            for bx in range(x, x + w, self.block_size):
                for by in range(y, y + h, self.block_size):
                    self.obstacles.add((bx, by))

    def _generate_points(self):
        """Tạo tọa độ giao hàng nằm gọn trong grid và không đè lên vật cản"""
        def get_valid_point():
            while True:
                x = random.randint(1, self.w // self.block_size - 2) * self.block_size
                y = random.randint(5, self.h // self.block_size - 2) * self.block_size
                pt = (x, y)
                if pt not in self.obstacles and pt not in self.points:
                    return pt

        self.points.append(get_valid_point()) # Trạm gốc
        for _ in range(self.num_points - 1):
            self.points.append(get_valid_point())

    def _build_distance_matrix(self):
        """Dùng A* lập bảng khoảng cách thực 100% lách tường cho mọi cặp đỉnh"""
        self.dist_matrix = [[0] * self.num_points for _ in range(self.num_points)]
        self.path_matrix = [[None] * self.num_points for _ in range(self.num_points)]
        
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                path = AStarSolver.get_path(self.obstacles, self.points[i], self.points[j], self.w, self.h, self.block_size)
                if path:
                    dist = len(path) * self.block_size
                else:
                    dist = 999999 # Không thể tới
                    path = [self.points[i], self.points[j]]
                    
                self.dist_matrix[i][j] = self.dist_matrix[j][i] = dist
                self.path_matrix[i][j] = path
                self.path_matrix[j][i] = path[::-1] # Đảo ngược đường khi đi ngược

    def step_optimize(self):
        """Thực thi 1 bước gỡ rối của AI"""
        if not self.is_done:
            try:
                self.current_route, self.current_distance, self.is_done = next(self.optimizer)
            except StopIteration:
                self.is_done = True

    def render(self):
        self.display.fill(WHITE)

        # Vẽ vùng cấm bay
        for obs in self.obstacles:
            pygame.draw.rect(self.display, BLACK, pygame.Rect(obs[0], obs[1], self.block_size, self.block_size))

        if self.current_route:
            line_color = GREEN if self.is_done else RED
            line_width = 3 if self.is_done else 1

            # Vẽ quỹ đạo thực tế tránh dính tường
            for i in range(len(self.current_route)):
                u = self.current_route[i]
                v = self.current_route[(i + 1) % len(self.current_route)]
                
                path = self.path_matrix[u][v]
                if path and len(path) > 1:
                    # Offset trung tâm ô để đường nối đi qua tâm
                    offset = self.block_size // 2
                    points = [(p[0] + offset, p[1] + offset) for p in path]
                    pygame.draw.lines(self.display, line_color, False, points, line_width)

        # Vẽ các điểm giao hàng
        for idx, pos in enumerate(self.points):
            center_pos = (pos[0] + self.block_size // 2, pos[1] + self.block_size // 2)
            if idx == 0:
                pygame.draw.circle(self.display, GREEN, center_pos, 12)
                pygame.draw.circle(self.display, WHITE, center_pos, 12, 2)
            else:
                pygame.draw.circle(self.display, BLUE, center_pos, 6)

        # Thông số
        info_text = f"Trạng thái: {'HOÀN THÀNH TỐI ƯU' if self.is_done else 'AI ĐANG GỠ RỐI LỘ TRÌNH...'}"
        dist_text = f"Chiều dài bay A* thực tế: {int(self.current_distance)} (Né vật cản)"
        
        text_surface1 = self.large_font.render(info_text, True, GREEN if self.is_done else RED)
        text_surface2 = self.font.render(dist_text, True, TEXT_COLOR)
        
        self.display.blit(text_surface1, (20, 20))
        self.display.blit(text_surface2, (20, 60))

        pygame.display.flip()