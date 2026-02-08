function comparator(a, b) {
	if (a[0] > b[0]) return -1;
	if (a[0] < b[0]) return 1;
	return 0;
}

class PriorityQueue {
	constructor(compare) {
		this.heap = [];
		this.compare = compare;
	}

	enqueue(value) {
		this.heap.push(value);
		this.bubbleUp();
	}

	bubbleUp() {
		let index = this.heap.length - 1;
		while (index > 0) {
			let element = this.heap[index],
			    parentIndex = Math.floor((index - 1) / 2),
			    parent = this.heap[parentIndex];
			if (this.compare(element, parent) < 0) break;
			this.heap[index] = parent;
			this.heap[parentIndex] = element;
			index = parentIndex;
		}
	}

	dequeue() {
		let max = this.heap[0];
		let end = this.heap.pop();
		if (this.heap.length > 0) {
			this.heap[0] = end;
			this.sinkDown(0);
		}
		return max;
	}

	sinkDown(index) {
		let left = 2 * index + 1,
		    right = 2 * index + 2,
		    largest = index;

		if (
		    left < this.heap.length &&
		    this.compare(this.heap[left], this.heap[largest]) > 0
		) {
			largest = left;
		}

		if (
		    right < this.heap.length &&
		    this.compare(this.heap[right], this.heap[largest]) > 0
		) {
			largest = right;
		}

		if (largest !== index) {
			[this.heap[largest], this.heap[index]] = [
				this.heap[index],
				this.heap[largest],
			];
			this.sinkDown(largest);
		}
	}

	isEmpty() {
		return this.heap.length === 0;
	}
}

function isValidCell(i, j, n) {
	return i >= 0 && i < n && j >= 0 && j < n;
}

function minimumCostPath(grid) {
	let n = grid.length;
	const pq = new PriorityQueue(comparator);
	let cost = Array.from({ length: n }, () => Array(n).fill(Infinity));
	cost[0][0] = grid[0][0];
	let dir = [[-1, 0], [1, 0], [0, -1], [0, 1]];

	pq.enqueue([grid[0][0], 0, 0]);

	while (!pq.isEmpty()) {
		let [c, i, j] = pq.dequeue();

		for (let d of dir) {
			let x = i + d[0];
			let y = j + d[1];

			if (isValidCell(x, y, n) && cost[i][j] + grid[x][y] < cost[x][y]) {
				cost[x][y] = cost[i][j] + grid[x][y];
				pq.enqueue([cost[x][y], x, y]);
			}
		}
	}

	return cost[n - 1][n - 1];
}

let grid = [
	[9, 4, 9, 9],
	[6, 7, 6, 4],
	[8, 3, 3, 7],
	[7, 4, 9, 10]
];

console.log(minimumCostPath(grid));
