import pygame
import random
from algorithms.tsp_heuristic import tsp_2opt_generator

# Bảng màu
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 100, 255)       # Điểm giao hàng
RED = (255, 50, 50)        # Đường bay đang thử nghiệm
GREEN = (0, 200, 0)        # Trạm gốc (Depot) / Đường bay tối ưu cuối
TEXT_COLOR = (50, 50, 50)

class DroneEnv:
    def __init__(self, w=900, h=700, num_points=35):
        self.w = w
        self.h = h
        self.num_points = num_points
        self.points = []
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI Drone Delivery - Tối ưu hóa TSP bằng 2-Opt')
        self.font = pygame.font.SysFont('segoeui', 20)
        self.large_font = pygame.font.SysFont('segoeui', 28, bold=True)
        
        self._generate_points()
        
        # Biến trạng thái
        self.current_route = []
        self.current_distance = float('inf')
        self.is_done = False
        
        # Khởi tạo Generator
        self.optimizer = tsp_2opt_generator(self.points)

    def _generate_points(self):
        """Tạo ngẫu nhiên tọa độ các điểm giao hàng"""
        # Điểm đầu tiên (index 0) luôn là Trạm Gốc (Depot)
        self.points.append((self.w // 2, self.h // 2)) 
        
        for _ in range(self.num_points - 1):
            x = random.randint(50, self.w - 50)
            y = random.randint(100, self.h - 50) # Né khoảng trống ở trên cho chữ
            self.points.append((x, y))

    def step_optimize(self):
        """Thực thi 1 bước gỡ rối của AI"""
        if not self.is_done:
            try:
                self.current_route, self.current_distance, self.is_done = next(self.optimizer)
            except StopIteration:
                self.is_done = True

    def render(self):
        self.display.fill(WHITE)

        if self.current_route:
            # Màu đường bay: Đỏ mờ khi đang tính, Xanh lá đậm khi hoàn thành
            line_color = GREEN if self.is_done else RED
            line_width = 4 if self.is_done else 1

            # Vẽ các đường nối giữa các điểm giao hàng
            for i in range(len(self.current_route)):
                p1 = self.points[self.current_route[i]]
                # Nối điểm cuối về lại điểm đầu để tạo vòng khép kín
                p2 = self.points[self.current_route[(i + 1) % len(self.current_route)]]
                pygame.draw.line(self.display, line_color, p1, p2, line_width)

        # Vẽ các điểm giao hàng
        for idx, pos in enumerate(self.points):
            if idx == 0:
                # Trạm Gốc (To hơn, hình vuông/tròn màu Xanh)
                pygame.draw.circle(self.display, GREEN, pos, 10)
                pygame.draw.circle(self.display, BLACK, pos, 10, 2)
            else:
                # Điểm khách hàng (Màu xanh dương)
                pygame.draw.circle(self.display, BLUE, pos, 6)

        # Hiển thị thông số (Độ dài tuyến đường đại diện cho Năng lượng tiêu thụ)
        info_text = f"Trạng thái: {'HOÀN THÀNH TỐI ƯU' if self.is_done else 'AI ĐANG GỠ RỐI...'}"
        dist_text = f"Tổng quãng đường (Năng lượng): {int(self.current_distance)} đơn vị"
        
        text_surface1 = self.large_font.render(info_text, True, GREEN if self.is_done else RED)
        text_surface2 = self.font.render(dist_text, True, TEXT_COLOR)
        
        self.display.blit(text_surface1, (20, 20))
        self.display.blit(text_surface2, (20, 60))

        pygame.display.flip()