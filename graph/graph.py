from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


class Graph:
    def __init__(self, directed=False, weighted=True):
        self.graph = defaultdict(list)
        self.directed = directed
        self.weighted = weighted
        self.lock = threading.Lock()
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        with self.lock:
            self.vertices.add(u)
            self.vertices.add(v)
            self.graph[u].append((v, weight if self.weighted else 1))
            if not self.directed:
                self.graph[v].append((u, weight if self.weighted else 1))

    def remove_edge(self, u, v):
        with self.lock:
            self.graph[u] = [(vertex, weight) for vertex, weight in self.graph[u] if vertex != v]
            if not self.directed:
                self.graph[v] = [(vertex, weight) for vertex, weight in self.graph[v] if vertex != u]

    def has_edge(self, u, v):
        return any(vertex == v for vertex, _ in self.graph[u])

    def get_vertices(self):
        return list(self.vertices)

    def get_edges(self):
        edges = []
        for u in self.graph:
            for v, weight in self.graph[u]:
                if self.directed or u <= v:
                    edges.append((u, v, weight))
        return edges

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()
            yield vertex

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()

        visited.add(start)
        yield start

        for neighbor, _ in self.graph[start]:
            if neighbor not in visited:
                yield from self.dfs(neighbor, visited)

    def shortest_path(self, start, end):
        heap = [(0, start, [])]
        visited = set()

        while heap:
            (cost, node, path) = heapq.heappop(heap)
            if node not in visited:
                visited.add(node)
                path = path + [node]

                if node == end:
                    return cost, path

                for neighbor, weight in self.graph[node]:
                    if neighbor not in visited:
                        heapq.heappush(heap, (cost + weight, neighbor, path))

        return float('inf'), []

    def is_cyclic(self):
        visited = set()
        rec_stack = set()

        def dfs_cycle(v):
            visited.add(v)
            rec_stack.add(v)

            for neighbor, _ in self.graph[v]:
                if neighbor not in visited:
                    if dfs_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(v)
            return False

        for node in self.graph:
            if node not in visited:
                if dfs_cycle(node):
                    return True
        return False

    def topological_sort(self):
        if not self.directed:
            raise ValueError("Topological sort is only applicable to directed graphs")

        in_degree = {v: 0 for v in self.vertices}
        for u in self.graph:
            for v, _ in self.graph[u]:
                in_degree[v] += 1

        queue = deque([v for v in in_degree if in_degree[v] == 0])
        result = []

        while queue:
            v = queue.popleft()
            result.append(v)
            for neighbor, _ in self.graph[v]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.vertices):
            raise ValueError("Graph has a cycle")

        return result

    def strongly_connected_components(self):
        def dfs(v, stack):
            visited.add(v)
            for neighbor, _ in self.graph[v]:
                if neighbor not in visited:
                    dfs(neighbor, stack)
            stack.append(v)

        def reverse_graph():
            reversed_graph = Graph(directed=True, weighted=self.weighted)
            for u in self.graph:
                for v, weight in self.graph[u]:
                    reversed_graph.add_edge(v, u, weight)
            return reversed_graph

        visited = set()
        stack = []
        for v in self.vertices:
            if v not in visited:
                dfs(v, stack)

        reversed_graph = reverse_graph()
        visited.clear()
        result = []

        while stack:
            v = stack.pop()
            if v not in visited:
                component = list(reversed_graph.dfs(v))
                result.append(component)

        return result

    def parallel_bfs(self, start, num_workers=4):
        visited = set([start])
        queue = deque([start])
        result = []

        def worker():
            while True:
                with self.lock:
                    if not queue:
                        return
                    vertex = queue.popleft()

                result.append(vertex)

                neighbors = [n for n, _ in self.graph[vertex] if n not in visited]
                with self.lock:
                    visited.update(neighbors)
                    queue.extend(neighbors)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(lambda _: worker(), range(num_workers))

        return result

    def minimum_spanning_tree(self):
        if self.directed:
            raise ValueError("Minimum spanning tree is only applicable to undirected graphs")

        edges = [(weight, u, v) for u in self.graph for v, weight in self.graph[u]]
        edges.sort()

        parent = {v: v for v in self.vertices}
        rank = {v: 0 for v in self.vertices}

        def find(item):
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(x, y):
            xroot = find(x)
            yroot = find(y)

            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1

        mst = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, weight))

        return mst

    def to_adjacency_matrix(self):
        vertices = sorted(self.vertices)
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}

        matrix = np.full((n, n), np.inf)
        np.fill_diagonal(matrix, 0)

        for u in self.graph:
            for v, weight in self.graph[u]:
                matrix[vertex_to_index[u], vertex_to_index[v]] = weight
                if not self.directed:
                    matrix[vertex_to_index[v], vertex_to_index[u]] = weight

        return matrix, vertices

    def all_pairs_shortest_paths(self):
        matrix, vertices = self.to_adjacency_matrix()
        dist_matrix = floyd_warshall(csr_matrix(matrix))

        return {(vertices[i], vertices[j]): dist_matrix[i, j]
                for i in range(len(vertices))
                for j in range(len(vertices))
                if i != j}

    def articulation_points(self):
        def dfs(u, parent):
            nonlocal time
            children = 0
            low[u] = disc[u] = time
            time += 1
            is_articulation = False

            for v, _ in self.graph[u]:
                if v not in disc:
                    children += 1
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if parent is not None and low[v] >= disc[u]:
                        is_articulation = True
                elif v != parent:
                    low[u] = min(low[u], disc[v])

            if (parent is None and children > 1) or (parent is not None and is_articulation):
                articulation_points.add(u)

        articulation_points = set()
        disc = {}
        low = {}
        time = 0

        for v in self.vertices:
            if v not in disc:
                dfs(v, None)

        return list(articulation_points)

    def max_flow(self, source, sink):
        def bfs(s, t, parent):
            visited = set()
            queue = deque([s])
            visited.add(s)
            parent[s] = None

            while queue:
                u = queue.popleft()
                for v, capacity in self.graph[u]:
                    if v not in visited and capacity > 0:
                        queue.append(v)
                        visited.add(v)
                        parent[v] = u
                        if v == t:
                            return True
            return False

        parent = {}
        max_flow = 0

        while bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, next(capacity for v, capacity in self.graph[parent[s]] if v == s))
                s = parent[s]

            max_flow += path_flow

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u] = [(v, capacity - path_flow if v == v else capacity) for v, capacity in self.graph[u]]
                self.graph[v].append((u, path_flow))
                v = parent[v]

        return max_flow

    def pagerank(self, damping_factor=0.85, epsilon=1e-8, max_iterations=100):
        n = len(self.vertices)
        pagerank = {v: 1 / n for v in self.vertices}

        for _ in range(max_iterations):
            prev_pagerank = pagerank.copy()
            for v in self.vertices:
                incoming = sum(
                    prev_pagerank[u] / len(self.graph[u]) for u in self.vertices if v in [x for x, _ in self.graph[u]])
                pagerank[v] = (1 - damping_factor) / n + damping_factor * incoming

            if all(abs(pagerank[v] - prev_pagerank[v]) < epsilon for v in self.vertices):
                break

        return pagerank


# Usage example
if __name__ == "__main__":
    g = Graph()
    g.add_edge(0, 1, 4)
    g.add_edge(0, 7, 8)
    g.add_edge(1, 2, 8)
    g.add_edge(1, 7, 11)
    g.add_edge(2, 3, 7)
    g.add_edge(2, 8, 2)
    g.add_edge(2, 5, 4)
    g.add_edge(3, 4, 9)
    g.add_edge(3, 5, 14)
    g.add_edge(4, 5, 10)
    g.add_edge(5, 6, 2)
    g.add_edge(6, 7, 1)
    g.add_edge(6, 8, 6)
    g.add_edge(7, 8, 7)

    print("BFS starting from vertex 0:", list(g.bfs(0)))
    print("DFS starting from vertex 0:", list(g.dfs(0)))
    print("Shortest path from 0 to 4:", g.shortest_path(0, 4))
    print("Is the graph cyclic?", g.is_cyclic())
    print("Parallel BFS starting from vertex 0:", g.parallel_bfs(0))
    print("Minimum Spanning Tree:", g.minimum_spanning_tree())
    print("All Pairs Shortest Paths:", g.all_pairs_shortest_paths())
    print("Articulation Points:", g.articulation_points())
    print("PageRank:", g.pagerank())

    # For directed graph operations
    dg = Graph(directed=True)
    dg.add_edge(0, 1)
    dg.add_edge(0, 2)
    dg.add_edge(1, 2)
    dg.add_edge(2, 0)
    dg.add_edge(2, 3)
    dg.add_edge(3, 3)

    print("Topological Sort:", dg.topological_sort())
    print("Strongly Connected Components:", dg.strongly_connected_components())
    print("Max Flow (0 -> 3):", dg.max_flow(0, 3))