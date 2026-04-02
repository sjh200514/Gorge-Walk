import json
import collections
import numpy as np

def solve_best_order():
    # 1. 加载地图数据
    with open('conf/map_data/F_level_1.json', 'r') as f:
        map_data = json.load(f)

    # 定义关键点坐标ID
    start_node = 1865  # [29, 9]
    end_node = 759     # [11, 55]
    # 宝箱 0-9 的坐标 ID
    treasures = [1230, 604, 620, 2733, 2071, 3192, 2298, 1527, 2657, 3497]
    
    # 建立所有关键点列表: [Start, T0, T1, ..., T9, End]
    nodes = [start_node] + treasures + [end_node]
    num_nodes = len(nodes)
    num_treasures = len(treasures)

    print(f"正在计算 {num_nodes} 个关键点之间的最短路径（基于地图拓扑）...")

    # 2. BFS 计算两两点之间的最短步数 (拓扑距离)
    def get_dist_matrix(target_nodes, graph):
        dist_matrix = np.full((num_nodes, num_nodes), 999999)
        for i, src in enumerate(target_nodes):
            # 对每个关键点跑一次 BFS
            queue = collections.deque([(src, 0)])
            visited = {src}
            found_count = 0
            while queue:
                curr, d = queue.popleft()
                # 检查是否到达了其他关键点
                if curr in target_nodes:
                    idx = target_nodes.index(curr)
                    dist_matrix[i][idx] = d
                    found_count += 1
                
                if found_count == num_nodes: break

                # 探索四个方向的移动
                if str(curr) in graph:
                    for move in graph[str(curr)].values():
                        next_node = move[0]
                        if next_node not in visited:
                            visited.add(next_node)
                            queue.append((next_node, d + 1))
        return dist_matrix

    dist_matrix = get_dist_matrix(nodes, map_data)

    # 3. 状态压缩 DP (TSP)
    # dp[mask][i] 表示在访问了 mask (二进制表示) 集合的宝箱后，当前停留在第 i 个宝箱的最短路径
    # mask 仅针对 10 个宝箱 (1 << 10)
    dp = np.full((1 << num_treasures, num_treasures), 999999.0)
    parent = np.full((1 << num_treasures, num_treasures), -1)

    # 初始化：从起点到第一个宝箱
    for i in range(num_treasures):
        dp[1 << i][i] = dist_matrix[0][i + 1] # 0是起点, i+1是第i个宝箱

    # 遍历所有状态
    for mask in range(1, 1 << num_treasures):
        for u in range(num_treasures):
            if not (mask & (1 << u)): continue
            if dp[mask][u] == 999999: continue
            
            # 尝试去下一个宝箱 v
            for v in range(num_treasures):
                if mask & (1 << v): continue # 已经去过了
                new_mask = mask | (1 << v)
                new_dist = dp[mask][u] + dist_matrix[u + 1][v + 1]
                if new_dist < dp[new_mask][v]:
                    dp[new_mask][v] = new_dist
                    parent[new_mask][v] = u

    # 4. 找到回终点最短的路径
    full_mask = (1 << num_treasures) - 1
    min_total_dist = 999999
    last_treasure = -1

    for i in range(num_treasures):
        total = dp[full_mask][i] + dist_matrix[i + 1][num_nodes - 1]
        if total < min_total_dist:
            min_total_dist = total
            last_treasure = i

    # 5. 回溯路径顺序
    path = []
    curr_mask = full_mask
    curr_node = last_treasure
    while curr_node != -1:
        path.append(curr_node)
        prev_node = parent[curr_mask][curr_node]
        curr_mask = curr_mask ^ (1 << curr_node)
        curr_node = prev_node
    
    path.reverse() # 得到宝箱索引顺序

    # 6. 输出结果
    print("\n--- 优化结果 ---")
    print(f"最短总步数: {min_total_dist}")
    print(f"最佳宝箱访问顺序 (索引): {path}")
    
    final_seq = [treasures[i] for i in path]
    print(f"最佳坐标ID序列: {final_seq}")
    print(f"包含终点的完整序列: {final_seq + [end_node]}")
    
    return final_seq + [end_node]

if __name__ == "__main__":
    solve_best_order()