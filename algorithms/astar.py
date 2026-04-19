import heapq

class AStarNode:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

class AStarSolver:
    @staticmethod
    def heuristic(p1, p2):
        # Khoảng cách Manhattan
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def get_path(grid, start, end, grid_width, grid_height, block_size):
        """
        Tìm đường đi an toàn nhất.
        Dữ liệu nhận vào là tọa độ dạng khối (block) chứ không phải pixel.
        """
        start_node = AStarNode(start)
        end_node = AStarNode(end)
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, start_node)

        # 4 hướng di chuyển
        directions = [(0, -block_size), (0, block_size), (-block_size, 0), (block_size, 0)]

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_set.add(current_node.position)

            if current_node.position == end_node.position:
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            for direction in directions:
                node_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

                # Check biên
                if node_position[0] > grid_width - block_size or node_position[0] < 0 or \
                   node_position[1] > grid_height - block_size or node_position[1] < 0:
                    continue

                # Check đụng vật cản (thân rắn)
                if node_position in grid:
                    continue

                if node_position in closed_set:
                    continue

                new_node = AStarNode(node_position, current_node)
                new_node.g = current_node.g + block_size
                new_node.h = AStarSolver.heuristic(new_node.position, end_node.position)
                new_node.f = new_node.g + new_node.h

                heapq.heappush(open_list, new_node)

        return None # Rắn bị kẹt