# Hot100 图论与 Trie：节点、边、搜索和依赖关系

[返回总索引](../README.md)

## 基础一：图是什么？

图由顶点集合 V 和边集合 E 构成。

- 无向图：关系是双向的，例如互相连接的道路。
- 有向图：边有方向，例如“先修课 → 后续课程”。
- 有权图：边上有代价。本文网格一步的代价相同，不涉及带权最短路算法。
- 环：沿边走一圈回到原点。
- 连通分量：无向图中互相可以到达的一组顶点。

树可以看作无环的连通无向图，但一般图可能有环，因此遍历通常需要访问标记。

### 三种表示方法

| 表示 | 示例 | 空间与用途 |
|---|---|---|
| 邻接表 | `graph[u] = [v1, v2]` | `O(V+E)`，适合稀疏图 |
| 邻接矩阵 | `connected[u][v]` | `O(V²)`，查边方便 |
| 隐式图 | 网格 `(r,c)` 的四邻居 | 不必真的建立所有边 |

无向图边 `(u,v)` 通常要在两个邻接表中都添加；有向图只添加对应方向。

## 基础二：DFS 与 BFS

- DFS 深度优先：沿一个分支走到底，可用递归或栈，适合连通块、路径搜索。
- BFS 广度优先：按距离一层层扩展，使用 `deque`，适合无权图最短步数、多源扩散。
- 全图可能不连通，要从每个尚未访问的顶点启动遍历。
- 一般在**入栈或入队时就标记访问**，避免同一个点被多个邻居重复加入。

### 和回溯的访问标记有什么不同？

- 岛屿数量：已经属于某个连通块的格子以后不必再搜索，标记不撤销。
- 单词搜索：一个格子只是在当前路径不能重复，换路径可以使用，因此标记必须撤销。

不要把所有 DFS 都写成“标记后一定撤销”。是否恢复由问题含义决定。

## 基础三：入度与拓扑排序

入度是指向某个顶点的边数。拓扑排序适用于有向无环图 DAG，使所有前驱出现在后继之前。

流程：把入度为 0 的点入队 → 移除它们的出边 → 新出现的入度 0 点入队。若最后没有处理完所有点，剩余依赖中存在环。

## 200. 岛屿数量

题目：[岛屿数量](https://leetcode.cn/problems/number-of-islands/)。`"1"` 是陆地，`"0"` 是水，四方向相连的陆地组成岛屿。

### 状态与思路

遇到尚未访问的陆地，答案加 1，并用 DFS 标记整块连通陆地。这里用显式栈避免大网格导致递归过深。

```python
class Solution:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        ans = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != "1":
                    continue
                ans += 1
                grid[r][c] = "0"
                stack = [(r, c)]
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < rows and 0 <= ny < cols
                                and grid[nx][ny] == "1"):
                            grid[nx][ny] = "0"
                            stack.append((nx, ny))
        return ans
```

### 推演与正确性

`[["1","1","0"],["0","0","1"]]`：左上两格是一个连通块，右下单独一个，结果 2。每块只在第一次遇到时计数，此次搜索会标记它的全部成员，因此既不重复也不遗漏。

**易错点**：不能把对角线当邻居；题目使用字符串 `"1"`，不是整数 1；本实现把陆地改成水，会修改输入。若需保留输入，改用 `visited` 集合。

时间 `O(RC)`，栈最坏空间 `O(RC)`。

## 994. 腐烂的橘子

题目：[腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)。0 空地、1 新鲜、2 腐烂；每分钟腐烂橘子同时感染四邻居，求全部腐烂所需最短时间。

### 状态与思路

把所有初始腐烂橘子同时放进队列，它们都是距离 0 的起点。每一轮只处理当前层，代表过去一分钟。

```python
from collections import deque


class Solution:
    def orangesRotting(self, grid):
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        minutes = 0
        while queue and fresh > 0:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and grid[nr][nc] == 1):
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

### 推演、易错点与复杂度

`[[2,1,1],[1,1,0],[0,1,1]]` 需要 4 分钟。所有源同时扩展，首次感染某格的时间就是它到最近源的无权最短距离。

没有新鲜橘子返回 0；有新鲜橘子却没有腐烂源，返回 -1。不能先从一个源完整搜索，再处理另一个源，否则不再表示同时传播。此实现会修改网格。

时间 `O(RC)`，队列空间 `O(RC)`。

## 207. 课程表

题目：[课程表](https://leetcode.cn/problems/course-schedule/)。`[course, prerequisite]` 表示必须先上 prerequisite，判断能否完成全部课程。

### 状态与思路

建立边 `prerequisite → course`。`indegree[course]` 表示还未完成的先修依赖数。不断学习入度为 0 的课程并释放后续依赖。

```python
from collections import deque


class Solution:
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for course, pre in prerequisites:
            graph[pre].append(course)
            indegree[course] += 1
        queue = deque(i for i in range(numCourses) if indegree[i] == 0)
        finished = 0
        while queue:
            course = queue.popleft()
            finished += 1
            for nxt in graph[course]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return finished == numCourses
```

### 推演与正确性

`numCourses=2`，`[[1,0]]`：0 入度为 0，学完后 1 入度变为 0，两门都完成。

如果再加 `[0,1]`，形成互相依赖，开始没有可学课程，返回 False。DAG 总存在入度 0 的点；有环时环内依赖无法全部释放。

**易错点**：读清边方向；没有边的独立课程也要计入；需要比较处理数量，不能仅判断队列最后是否为空，因为成功失败最后都会空。

时间与空间都是 `O(V+E)`。

## 208. 实现 Trie（前缀树）

题目：[实现 Trie](https://leetcode.cn/problems/implement-trie-prefix-tree/)。支持插入单词、查完整单词、查是否存在某个前缀。

### 基础与状态

Trie 的边标字符，从根到节点的路径表示一个前缀。例如插入 `apple`、`app`，两者共享 `a→p→p`。

节点需要两类信息：

- `children`：下一个字符到子节点的映射。
- `is_end`：当前前缀是否也恰好是一个已插入的完整单词。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _walk(self, text):
        node = self.root
        for ch in text:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix):
        return self._walk(prefix) is not None
```

### 推演、易错点与复杂度

只插入 `apple` 后，`startsWith("app")` 为 True，`search("app")` 为 False；再插入 `app` 后，后者才变 True。

**易错点**：路径存在不代表完整单词存在；重复插入只需重复设置终止标记，不需要新建整条路径。

长度为 L 的插入或查询平均时间 `O(L)`；查询辅助空间 `O(1)`；整棵 Trie 的空间与不同前缀总数成正比，上界 `O(S)`，S 为全部插入字符串长度之和。

## 入门路线

先把网格想象成没有显式建边的图，学习 200 的连通块，再学 994 的多源层序扩散；用 207 理解有向依赖和环。Trie 虽归在该分类，但重点是字符串共享前缀的数据结构，与一般最短路问题不同。
