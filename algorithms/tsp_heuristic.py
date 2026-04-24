import math

def euclidean_distance(p1, p2):
    """Khoảng cách đường chim bay giữa 2 điểm"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_total_distance(route, points, dist_matrix=None):
    """Tính tổng chiều dài của một lộ trình bay"""
    dist = 0
    for i in range(len(route)):
        # Cộng khoảng cách từ điểm i đến điểm i+1 (và quay về điểm 0 ở cuối)
        u = route[i]
        v = route[(i + 1) % len(route)]
        if dist_matrix:
            dist += dist_matrix[u][v]
        else:
            dist += euclidean_distance(points[u], points[v])
    return dist

def tsp_2opt_generator(points, dist_matrix=None):
    """
    Thuật toán Heuristic 2-Opt cho bài toán Drone Delivery.
    Thuật toán này liên tục tìm và gỡ các đoạn đường bay bị bắt chéo nhau.
    """
    num_points = len(points)
    
    # Khởi tạo một lộ trình ngẫu nhiên ban đầu (đi tuần tự từ 0 đến n-1)
    best_route = list(range(num_points))
    best_distance = calculate_total_distance(best_route, points, dist_matrix)
    
    yield best_route, best_distance, False # Trạng thái xuất phát

    improved = True
    while improved:
        improved = False
        for i in range(1, num_points - 2):
            for j in range(i + 1, num_points):
                if j - i == 1: 
                    continue # Bỏ qua 2 đỉnh sát nhau
                
                # Tạo lộ trình mới bằng cách đảo ngược thứ tự thăm của đoạn [i:j]
                # Đây chính là thao tác "gỡ chéo"
                new_route = best_route[:]
                new_route[i:j] = best_route[i:j][::-1]
                
                new_distance = calculate_total_distance(new_route, points, dist_matrix)
                
                # Trả về trạng thái đang thử nghiệm để làm hoạt ảnh
                yield new_route, new_distance, False
                
                # Nếu đường gỡ chéo này ngắn hơn đường cũ, ta cập nhật lại kỷ lục
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improved = True
                    # Trả về ngay khi có sự cải thiện rõ rệt
                    yield best_route, best_distance, False
                    
    # Khi không thể gỡ thêm được nữa -> Hoàn thành
    yield best_route, best_distance, True