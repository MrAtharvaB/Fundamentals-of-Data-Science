function dfs(node, currentTime, adj, vis, V, shortest, ways) {
	if (currentTime > shortest[0]) return;

	if (node === V - 1) {
		if (currentTime < shortest[0]) {
			shortest[0] = currentTime;
			ways[0] = 1;
		} else if (currentTime === shortest[0]) {
			ways[0]++;
		}
		return;
	}

	vis[node] = 1;

	for (let p of adj[node]) {
		let next = p[0];
		let wt = p[1];
		if (vis[next] === 0) {
			dfs(next, currentTime + wt, adj, vis, V, shortest, ways);
		}
	}

	vis[node] = 0;
}

function countPaths(adj) {
	let V = adj.length;
	let vis = Array(V).fill(0);
	let shortest = [Number.MAX_SAFE_INTEGER];
	let ways = [0];

	dfs(0, 0, adj, vis, V, shortest, ways);
	return ways[0];
}

function addEdge(adj, u, v, wt) {
	adj[u].push([v, wt]);
	adj[v].push([u, wt]);
}

let V = 4;
let adj = Array.from({ length: V }, () => []);

addEdge(adj, 0, 1, 2);
addEdge(adj, 0, 3, 5);
addEdge(adj, 1, 2, 3);
addEdge(adj, 1, 3, 3);
addEdge(adj, 2, 3, 4);

console.log(countPaths(adj));
