import sys
import random
import pygame
from environments.snake_env import SnakeGameAI , Direction
from algorithms.dqn_agent import DQNAgent
from algorithms.astar import AStarSolver
from helper import plot_training_progress
from environments.city_map_env import CityMapEnv
from environments.drone_grid_env import DroneEnv

# def get_dummy_astar_action():
#     """
#     Hàm tạo hành động ngẫu nhiên cho Rắn A*.
#     Tạm thời sử dụng hàm này làm "bao cát" để Rắn DQN tập luyện trước.
#     Sau khi DQN khôn lên, ta sẽ lắp logic của thuật toán A* thực thụ vào đây.
#     """
#     action = [0, 0, 0]
#     action[random.randint(0, 2)] = 1
#     return action




def get_astar_action(game):
    """
    Sử dụng thuật toán A* để tính toán đường đi ngắn nhất đến mồi.
    """
    start = (game.head_astar.x, game.head_astar.y)
    end = (game.food.x, game.food.y)

    # 1. Gom tất cả thân rắn (cả 2 con) thành vật cản
    obstacles = []
    for pt in game.snake_astar[1:]:
        obstacles.append((pt.x, pt.y))
    for pt in game.snake_dqn:
        obstacles.append((pt.x, pt.y))

    # 2. Gọi thuật toán A*
    path = AStarSolver.get_path(obstacles, start, end, game.w, game.h, 20)

    # 3. Chuyển đổi đường đi (tọa độ) thành hành động (Trái/Phải/Thẳng)
    action = [1, 0, 0] # Mặc định đi thẳng
    
    if path and len(path) > 1:
        next_pt = path[1] # Lấy điểm tiếp theo cần bước tới
        
        # Xác định hướng tuyệt đối cần đi
        if next_pt[0] > start[0]: next_dir = Direction.RIGHT
        elif next_pt[0] < start[0]: next_dir = Direction.LEFT
        elif next_pt[1] > start[1]: next_dir = Direction.DOWN
        elif next_pt[1] < start[1]: next_dir = Direction.UP
        else: next_dir = game.direction_astar

        # So sánh với hướng hiện tại để ra lệnh Rẽ
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction_astar)

        if next_dir == clock_wise[idx]:
            action = [1, 0, 0] # Đi thẳng
        elif next_dir == clock_wise[(idx + 1) % 4]:
            action = [0, 1, 0] # Rẽ phải
        else:
            action = [0, 0, 1] # Rẽ trái

    return action




def train_dual_ai():
    print("="*50)
    print(" BẮT ĐẦU HUẤN LUYỆN DEEP Q-NETWORK")
    print("="*50)
    
    # Khởi tạo các mảng lưu dữ liệu để vẽ biểu đồ
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_dqn = 0
    
    agent = DQNAgent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move_dqn = agent.get_action(state_old)
        
        # Rắn A* sử dụng thuật toán tìm đường thực thụ
        action_astar = get_astar_action(game)

        reward_astar, reward_dqn, done, score_astar, score_dqn, winner = game.step(action_astar, final_move_dqn)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move_dqn, reward_dqn, state_new, done)
        agent.remember(state_old, final_move_dqn, reward_dqn, state_new, done)
        
        game.render()

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score_dqn > record_dqn:
                record_dqn = score_dqn
                agent.model.save()

            print(f"Trận: {agent.n_games} | Điểm DQN: {score_dqn} | Kỷ lục: {record_dqn} | Thắng: {winner}")
            
            # --- PHẦN MỚI: CẬP NHẬT BIỂU ĐỒ ---
            plot_scores.append(score_dqn)
            total_score += score_dqn
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Gọi hàm vẽ
            plot_training_progress(plot_scores, plot_mean_scores)


def run_gps_navigation():
    print("="*50)
    print(" BẮT ĐẦU MÔ PHỎNG GPS NAVIGATION (TRỰC QUAN HÓA)")
    print("="*50)
    
    # Tạo bản đồ với 120 điểm (càng nhiều điểm vẽ càng đẹp)
    env = CityMapEnv(w=1000, h=700, num_nodes=120) 
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Cho AI suy luận 1 bước mỗi khung hình
        env.step_search()
        
        # Vẽ lên màn hình
        env.render()
        
        # Tốc độ tính toán: 20 bước/giây (Tăng số này lên để AI lan tỏa nhanh hơn)
        clock.tick(20) 
    
    pygame.quit()


def run_drone_delivery():
    print("="*50)
    print(" BẮT ĐẦU MÔ PHỎNG DRONE DELIVERY (TSP)")
    print("="*50)
    
    # 35 điểm là con số đẹp để biểu diễn sự chằng chịt và thời gian gỡ rối vừa phải
    env = DroneEnv(w=1000, h=750, num_points=35) 
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # AI thực hiện tối ưu
        # Chạy vòng lặp nhỏ này nhiều lần trong 1 frame để đẩy nhanh tốc độ gỡ rối
        for _ in range(5): 
            env.step_optimize()
            
        env.render()
        clock.tick(60) # 60 FPS
    
    pygame.quit()




def main_menu():
    while True:
        print("\n" + "="*50)
        print("  AI LOGISTICS SUITE - MAIN DASHBOARD")
        print("="*50)
        print("1. Huấn luyện AI DQN (Đối đầu với A* dummy)")
        print("2. GPS Navigation (A* Heuristic) - Đang phát triển")
        print("3. Drone Delivery (TSP Optimization) - Đang phát triển")
        print("0. Thoát")
        print("="*50)
        
        choice = input("Nhập lựa chọn của bạn (0-3): ")
        
        if choice == '1':
            train_dual_ai()
        elif choice == '2':
            run_gps_navigation() # Gọi hàm mới
        elif choice == '3':
            run_drone_delivery()
        elif choice == '0':
            sys.exit()
        else:
            print("Chức năng đang được tích hợp. Vui lòng thử lại sau.")

if __name__ == '__main__':
    main_menu()