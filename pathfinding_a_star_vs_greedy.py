import heapq
import numpy as np
import matplotlib.pyplot as plt
import time

# Определение направлений (вверх, вниз, влево, вправо)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def generate_maze(size, obstacle_prob=0.2):
    maze = np.random.choice([0, 1], size=size, p=[1 - obstacle_prob, obstacle_prob])
    maze[0, 0] = 0  # Стартовая точка
    maze[size[0] - 1, size[1] - 1] = 0  # Финишная точка
    return maze


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Манхэттенское расстояние


def greedy_best_first_search(maze, start, goal):
    pq = [(heuristic(start, goal), start)]  # Очередь с приоритетами
    visited = set()
    came_from = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(maze, neighbor) and neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    return None


def a_star_search(maze, start, goal):
    pq = [(0 + heuristic(start, goal), 0, start)]  # (f, g, node)
    visited = set()
    came_from = {}
    g_score = {start: 0}

    while pq:
        _, g, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(maze, neighbor) and neighbor not in visited:
                tentative_g = g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(pq, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current

    return None


def is_valid(maze, pos):
    x, y = pos
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] == 0


def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()
    return path


def visualize_maze(maze, path=None, title="Maze"):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap="gray_r")

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker="o", color="red", markersize=5, linewidth=2)

    plt.title(title)
    plt.show()


# Основной код
maze_size = (20, 20)
maze = generate_maze(maze_size)
start, goal = (0, 0), (maze_size[0] - 1, maze_size[1] - 1)

start_time = time.time()
greedy_path = greedy_best_first_search(maze, start, goal)
greedy_time = time.time() - start_time

start_time = time.time()
a_star_path = a_star_search(maze, start, goal)
a_star_time = time.time() - start_time

print(f"Greedy Search Time: {greedy_time:.6f}s, Path Length: {len(greedy_path) if greedy_path else 'No Path'}")
print(f"A* Search Time: {a_star_time:.6f}s, Path Length: {len(a_star_path) if a_star_path else 'No Path'}")

visualize_maze(maze, greedy_path, "Greedy Best-First Search")
visualize_maze(maze, a_star_path, "A* Search")