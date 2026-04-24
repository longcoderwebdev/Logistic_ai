import math
import heapq

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def astar_graph_search_generator(nodes, edges, start_idx, end_idx, w1=0.5, w2=0.5):
    """
    Sử dụng yield để trả về trạng thái từng bước cho việc vẽ hoạt ảnh.
    w1: Trọng số ưu tiên Thời gian
    w2: Trọng số ưu tiên Năng lượng (Tiết kiệm)
    """
    open_list = []
    heapq.heappush(open_list, (0, start_idx, [start_idx]))
    
    g_scores = {node: float('inf') for node in range(len(nodes))}
    g_scores[start_idx] = 0
    
    closed_set = set()
    open_set_tracker = {start_idx}

    while open_list:
        current_f, current_node, path = heapq.heappop(open_list)
        
        if current_node in open_set_tracker:
            open_set_tracker.remove(current_node)

        if current_node in closed_set:
            continue
            
        closed_set.add(current_node)

        # TRẢ VỀ TRẠNG THÁI HIỆN TẠI (Đang tính toán)
        yield current_node, closed_set, open_set_tracker, None

        # Nếu đã đến đích
        if current_node == end_idx:
            # TRẢ VỀ ĐƯỜNG ĐI CUỐI CÙNG
            yield current_node, closed_set, open_set_tracker, path
            return

        # Duyệt các ngã rẽ
        if current_node in edges:
            for edge_info in edges[current_node]:
                # Tương thích ngược hoặc sử dụng time_cost/energy_cost mới
                if len(edge_info) == 2:
                    neighbor, dist = edge_info
                    cost = dist
                else:
                    neighbor, dist, time_cost, energy_cost, terrain = edge_info
                    cost = w1 * time_cost + w2 * energy_cost

                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current_node] + cost

                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    
                    # Heuristic cần được scale hợp lý dựa trên w1, w2
                    h_dist = euclidean_distance(nodes[neighbor], nodes[end_idx])
                    # Ta coi thời gian bé nhất là h_dist * 0.5, năng lượng bé nhất là h_dist * 0.5
                    h_cost = w1 * (h_dist * 0.5) + w2 * (h_dist * 0.5) 
                    f = tentative_g + h_cost
                    
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(open_list, (f, neighbor, new_path))
                    open_set_tracker.add(neighbor)
    
    # Không tìm thấy đường
    yield None, closed_set, open_set_tracker, []