import heapq
import sys

def dijkstra(adj, src):

    V = len(adj)

    pq = []

    dist = [sys.maxsize] * V

    dist[src] = 0
    heapq.heappush(pq, (0, src))

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, w in adj[u]:

            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist


if __name__ == "__main__":
    src = 0
    
    adj = [
        [(1, 4), (2, 8)],
        [(0, 4), (4, 6), (2, 3)],
        [(0, 8), (3, 2), (1, 3)],
        [(2, 2), (4, 10)],
        [(1, 6), (3, 10)]
    ]
    
    result = dijkstra(adj, src)
    print(*result)
