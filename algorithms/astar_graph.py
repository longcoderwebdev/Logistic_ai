import math
import heapq

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def astar_graph_search_generator(nodes, edges, start_idx, end_idx):
    """
    Sử dụng yield để trả về trạng thái từng bước cho việc vẽ hoạt ảnh.
    """
    open_list = []
    heapq.heappush(open_list, (0, start_idx, [start_idx]))
    
    g_scores = {node: float('inf') for node in range(len(nodes))}
    g_scores[start_idx] = 0
    
    closed_set = set()
    open_set_tracker = {start_idx} # Giúp UI biết nút nào đang trong hàng đợi

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
            for neighbor, cost in edges[current_node]:
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current_node] + cost

                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h = euclidean_distance(nodes[neighbor], nodes[end_idx])
                    f = tentative_g + h
                    
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(open_list, (f, neighbor, new_path))
                    open_set_tracker.add(neighbor)
    
    # Không tìm thấy đường
    yield None, closed_set, open_set_tracker, []