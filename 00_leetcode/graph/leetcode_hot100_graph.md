# Hot100 图论与 Trie：节点、边、搜索和依赖关系

[返回总索引](../README.md)

本文先讲图与遍历的基础，再讲 Hot100 的 4 道题。**伪代码使用 `text` 代码块，只表达流程；Python 代码块是可以运行的实现。** 同一道题的不同版本分别使用，避免多个 `Solution` 类互相覆盖。

## 阅读路线

1. [图的基本概念](#基础一图是什么)：节点、边、路径、环、连通性。
2. [图的表示与构造](#基础二怎样在代码中表示图)：边列表、邻接表、邻接矩阵、网格。
3. [栈、队列与递归](#基础三栈队列与递归调用栈)：理解程序如何保存“下一步做什么”。
4. [DFS](#基础四dfs深度优先搜索)：递归、显式栈及准确模拟调用栈。
5. [BFS](#基础五bfs广度优先搜索)：队列、分层递归、最短距离。
6. [访问标记](#基础六visited-究竟记录什么)：何时保留、何时撤销、如何判断有向环。
7. [拓扑排序](#基础七入度与拓扑排序)：从先修依赖理解算法。
8. [岛屿数量](#200-岛屿数量) → [腐烂的橘子](#994-腐烂的橘子) → [课程表](#207-课程表) → [Trie](#208-实现-trie前缀树)。

## 基础一：图是什么？

### 1. 顶点与边

图通常写成 `G=(V,E)`：V 是顶点集合，E 是边集合。顶点也叫节点，边表示两个节点之间的关系。

```text
        A          F
       / \
      B   C
      |   |
      D   E
```

这个无向图有 6 个顶点、4 条边：A—B、A—C、B—D、C—E。F 没有相连的边，称为孤立点。

后面的遍历例子主要使用这张图，从 A 出发，按照邻接表中的顺序考虑邻居。

### 2. 无向图、有向图、有权图

| 概念 | 含义 | 例子 |
|---|---|---|
| 无向图 | 边没有方向，两端可以互相到达 | 双向道路、四方向相邻网格 |
| 有向图 | 只能沿箭头方向走 | `先修课 → 后续课程` |
| 无权图 | 不区分边的代价，通常每步视为 1 | 走过多少条边 |
| 有权图 | 每条边有自己的代价 | 道路长度、费用、时间 |

有向/无向与有权/无权是不同维度，可以组合。例如道路既可能单向，也可能有不同长度。

**普通 BFS 保证的是无权图的最少边数**。若边权不同，边数最少不一定总代价最小，不能直接用普通 BFS 代替带权最短路算法。

### 3. 邻居、路径、距离和环

- **邻居**：与当前节点直接相连的节点。有向图中，本文遍历的是当前节点的出边邻居。
- **路径**：顺着边依次经过一组节点。例如 A→B→D。
- **路径长度**：无权图中按边数计。A→B→D 长度为 2，不是 3。
- **可达**：从一个节点沿边能够走到另一个节点。
- **最短距离**：所有可行路径中最少的边数；起点到自身为 0。
- **环**：存在一条闭合路线，沿边能回到起点。对无向简单图，刚走 A—B 再沿同一条边退回 A，不作为一个简单环。

有向图里 A 能到 B，不代表 B 能到 A。添加 `A→B` 时，不要自动添加 `B→A`。

### 4. 度、入度与出度

- 无向简单图中，一个节点的度是与它相连的边数。例如图中 A 的度为 2。
- 有向图中，**入度**是指向它的边数，**出度**是从它指出的边数。
- 例如 `0→1、2→1`，节点 1 的入度是 2，表示它有两个直接前驱。

在课程表题里，入度不是课程编号，也不是距离，而是“还有多少条先修依赖没满足”。

### 5. 连通分量与树

无向图中，一组互相可达、且不能再加入其他顶点的节点构成一个连通分量。上图有两个：`{A,B,C,D,E}` 与 `{F}`。

从 A 遍历一次只能访问第一个分量，无法访问 F。想遍历整张图，就必须在外层检查所有顶点，对未访问顶点重新启动搜索。

有向图的连通性要区分强连通、弱连通，不能直接把“从任意未访问点出发的次数”当成强连通分量数。本专题的岛屿计数对应的是无向连通块。

树是无环的连通无向图。树中两点之间只有一条简单路径；一般图可能有多条路径，甚至有环，因此遍历通常需要访问标记。

## 基础二：怎样在代码中表示图？

### 1. 边列表：记录每条关系

```text
vertices = [A, B, C, D, E, F]
edges = [(A,B), (A,C), (B,D), (C,E)]
```

边列表直观，适合接收输入。但如果每次找一个节点的邻居都扫描所有边，会重复做很多工作。

注意 F 不在任何边里，因此**不能仅从边列表推断全部顶点**；课程表用 `numCourses`，一般图可以单独提供 vertices。

### 2. 邻接表：给每个节点保存邻居

```text
A: [B, C]
B: [A, D]
C: [A, E]
D: [B]
E: [C]
F: []
```

遍历 A 的邻居，只需读取 `graph[A]`。无向边保存两次，是同一条边在两个端点的记录，不代表原图有两条不同边。

下面的构造函数支持任意可哈希的顶点名称，如整数、字符串。约定每条边的端点都在 vertices 中。

```python
def build_graph(vertices, edges, directed=False):
    graph = {vertex: [] for vertex in vertices}
    for start, end in edges:
        graph[start].append(end)
        if not directed:
            graph[end].append(start)
    return graph
```

例如 `build_graph(["A","B","C","D","E","F"], [("A","B"),("A","C"),("B","D"),("C","E")])` 就得到上面的邻接表。

若顶点是 `0..n-1`，也可以用列表：`graph = [[] for _ in range(n)]`。不要写 `[[]] * n`，后者会让所有位置引用同一个邻居列表。

### 3. 邻接矩阵：用二维表表示是否有边

`matrix[u][v]` 为 True 表示有边 u→v。无向图的矩阵关于主对角线对称。

对于带权图，可以在矩阵里存权重，但要另用 `None` 或其他明确标记表示“没有边”，不要让它与权重为 0 的合法边混淆。

| 表示 | 存储空间 | 枚举一个节点的邻居 | 查询是否有一条边 |
|---|---|---|---|
| 边列表 | `O(V+E)`，含顶点列表 | 通常扫描 `O(E)` | 通常 `O(E)` |
| 邻接表（邻居为列表） | `O(V+E)` | 与该点邻居数成正比 | 通常扫描该点邻居列表 |
| 邻接矩阵 | `O(V²)` | 扫描一整行 `O(V)` | `O(1)` |

这里 V、E 在复杂度中表示顶点数和边数。邻接表中无向边虽然存两份，但常数因子 2 不改变 `O(V+E)`。

### 4. 网格是“隐式图”

在网格中，可以把坐标 `(r,c)` 当节点。上下左右就是候选邻居：

```text
             (r-1,c)
                ↑
(r,c-1)  ←   (r,c)   →  (r,c+1)
                ↓
             (r+1,c)
```

并不是四个方向都能走，还要检查：是否越界、是否为可通行格子、是否已经访问。

每个格子最多 4 个邻居，可以现场计算，无需真的建立一张邻接表。岛屿数量和腐烂的橘子都是这种表示。

## 基础三：栈、队列与递归调用栈

### 1. 栈：最后进入的任务先处理

Python 列表尾部可以作为栈：`append(x)` 入栈，`pop()` 出栈。

```text
先放 A，再放 B，再放 C
栈：[A, B, C]，最右边是栈顶
取出顺序：C、B、A
```

DFS 用栈保存尚待探索的任务，因此容易沿最近发现的分支继续向下。

### 2. 队列：最早进入的任务先处理

Python 使用 `collections.deque`：`append(x)` 入队，`popleft()` 出队。

```text
先放 A，再放 B，再放 C
队列：[A, B, C]，最左边是队首
取出顺序：A、B、C
```

BFS 用队列保证近的一层先处理，远的一层后处理。列表的 `pop(0)` 会移动后面的元素，不适合频繁作为队列出队。

### 3. 递归其实也在用栈

执行 `dfs(A)`，里面调用 `dfs(B)`，再调用 `dfs(D)`：

```text
调用栈：[dfs(A)]
调用栈：[dfs(A), dfs(B)]
调用栈：[dfs(A), dfs(B), dfs(D)]
D 处理结束 → 回到 B 调用 D 后的下一行
B 处理结束 → 回到 A 调用 B 后的下一行
```

每一层保存自己的局部参数和执行位置。这就是为什么递归天然适合“先处理一个邻居及其全部后续，再回来处理下一个邻居”。

**递归是程序的实现方式；DFS/BFS 是决定搜索顺序的策略。** 使用递归不自动等于 BFS 或 DFS，要看下一次调用究竟处理哪个范围。

Python 不会自动把尾递归优化成循环。路径很深的递归 DFS，或层数很多的递归 BFS，可能触及递归深度限制；显式栈、队列适合处理这类输入。

## 基础四：DFS——深度优先搜索

### 1. 一句话理解

> 从当前节点出发，先沿一个未访问邻居深入探索；这一支结束后，回来继续探索其他邻居。

在示例图中，一种递归 DFS 访问顺序为：`A → B → D → C → E`。F 不可达，需另外启动。

这个顺序由邻接表顺序决定，并不唯一。若 A 的邻接表先列 C，就可能先走 C、E。

### 2. 递归 DFS 的函数契约

`DFS(u)` 的含义：

> 如果 u 尚未访问，就标记并处理它，然后沿出边继续探索所有尚未访问的邻居，直到这一轮能够发现的节点都被处理。

“标记”表示以后不再重复启动该节点；“处理”可以是收集值、统计数量或更新答案，两者不是同一个动作。

### 3. 递归 DFS 伪代码

```text
DFS(u):
    如果 u 已访问:
        返回

    标记 u 已访问
    处理 u

    对于 u 的每个邻居 v:
        如果 v 未访问:
            DFS(v)

从某个起点搜索:
    visited = 空集合
    DFS(start)

遍历整张图:
    visited = 空集合
    对于图中的每个顶点 u:
        如果 u 未访问:
            DFS(u)
```

若图为无向图，外层每次对一个新起点调用 DFS，就完整访问一个新连通分量，可以在此处将分量数加 1。

### 4. 递归 DFS 的 Python 实现

以下通用遍历函数都约定：起点存在于邻接表中，每个邻居也都是邻接表的键。

```python
def dfs_recursive(graph, start):
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return order
```

`visited` 和 `order` 被所有递归层共享，但每层的 node 是自己的局部参数。`dfs(neighbor)` 返回后，当前层的循环继续尝试下一个邻居，不会从起点重新开始。

### 5. 非递归 DFS：用显式栈保存待处理节点

```text
visited = {start}
stack = [start]

当 stack 非空:
    u = 弹出栈顶
    处理 u

    对于 u 的每个邻居 v:
        如果 v 未访问:
            标记 v 已访问
            将 v 压入栈
```

```python
def dfs_iterative(graph, start):
    visited = {start}
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        order.append(node)

        # 后入先出，所以反向压栈，让列表靠前的邻居优先弹出。
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return order
```

**示例推演**，栈最右边是栈顶：

| 弹出的节点 | 处理邻居后的栈 | 已输出 |
|---|---|---|
| A | `[C,B]` | A |
| B | `[C,D]` | A,B |
| D | `[C]` | A,B,D |
| C | `[E]` | A,B,D,C |
| E | `[]` | A,B,D,C,E |

为什么入栈就标记？一个节点可能被多个邻居发现。先标记能保证它只入栈一次，避免重复任务。

这个简洁版本适合**可达性、连通块搜索**。它一次将多个邻居标为已发现，在含交叉连接的一般图中，不保证与递归版产生完全相同的访问顺序和 DFS 树。上表的顺序一致来自该例子的结构，不能据此认为反向压栈总能精确模拟递归。

### 6. 怎样精确模拟递归 DFS？

如果需要与递归完全一致的进入/退出顺序，例如处理完成状态或后序信息，栈里要保存**调用帧**，不仅是一个节点，还包括“下一个要检查的邻居位置”。

```text
标记 start 已访问
执行 start 的进入操作
stack = [(start, 0)]

当 stack 非空:
    (u, next_index) = 查看栈顶调用帧

    如果 next_index 等于 u 的邻居数量:
        执行 u 的退出操作
        弹出栈顶调用帧
        继续下一轮

    v = u 的第 next_index 个邻居
    将栈顶调用帧的 next_index 加 1

    如果 v 未访问:
        标记 v 已访问
        执行 v 的进入操作
        压入调用帧 (v, 0)
```

这次每次只进入一个新邻居，父节点的帧保留在栈中，等子节点完成后再检查下一个邻居，与函数调用和返回一一对应。

### 7. 复杂度

邻接表遍历中，每个节点处理一次，每条邻接记录检查一次，时间为 `O(V+E)`，辅助空间最坏 `O(V)`。如果只从一个起点搜索，可按实际可达子图的规模计算。

如果使用邻接矩阵，需要为每个访问节点扫描整行，完整遍历时间通常为 `O(V²)`。因此不能脱离存储方式一概说 DFS 为线性时间。

## 基础五：BFS——广度优先搜索

### 1. 一句话理解

> 先处理距离起点为 0 的点，再处理距离为 1 的点，再处理距离为 2 的点……

示例图从 A 出发：

```text
第 0 层：A
第 1 层：B、C
第 2 层：D、E
不可达：F
```

一种 BFS 顺序是 `A → B → C → D → E`。同层内部顺序可以不同，但较近的层不会排到更远的层之后。

### 2. 非递归 BFS 伪代码：队列版本

```text
distance[start] = 0
标记 start 已访问
queue = [start]

当 queue 非空:
    u = 从队首取出
    处理 u

    对于 u 的每个邻居 v:
        如果 v 未访问:
            标记 v 已访问
            distance[v] = distance[u] + 1
            parent[v] = u
            将 v 加入队尾
```

新发现的节点排到队尾，所以当前层已有的节点会先处理。**必须在入队时标记**，否则多个同层节点可能反复把同一个邻居加入队列。

### 3. Python：同时求访问顺序、距离与前驱

```python
from collections import deque


def bfs_iterative(graph, start):
    queue = deque([start])
    distance = {start: 0}
    parent = {start: None}
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            # distance 中已经有这个节点，就表示已经发现并入队过。
            if neighbor in distance:
                continue
            distance[neighbor] = distance[node] + 1
            parent[neighbor] = node
            queue.append(neighbor)

    return order, distance, parent
```

这里 `distance` 的键兼任 visited，不需要再存一个重复的集合。字典中没有 F，表示 F 从 A 不可达。

示例中 `parent[D]=B`、`parent[B]=A`。如果想恢复 A 到 D 的一条最短路径，可以从 D 反复查 parent，直到 A，再把结果反转，得到 `[A,B,D]`。起点的 parent 为 None 只是标记；恢复时以“到达起点”作为结束条件更通用。

### 4. 队列手算过程

| 出队节点 | 新发现节点 | 处理后的队列 | 新距离 |
|---|---|---|---|
| A | B、C | `[B,C]` | B=1，C=1 |
| B | D | `[C,D]` | D=2 |
| C | E | `[D,E]` | E=2 |
| D | 无 | `[E]` | — |
| E | 无 | `[]` | — |

### 5. 为什么第一次发现就是最短距离？

假设 u 的距离为 d，那么沿一条边发现 v，可得到长度 d+1 的路径。若 v 存在更短路径，那么它一定能从更早的某一层被发现，而那些层已经处理过了。

所以 v 尚未被访问时，这次记录的 d+1 就是最短距离，之后无需重复更新。这个理由依赖**所有边每步代价相同**。

DFS 则可能先沿很长的分支找到目标，不能把第一次到达时的深度直接当成最短距离。

### 6. BFS 可以写成递归吗？

可以，但递归调用的单位必须保持“按层”，而不是见到一个邻居就立即递归深入。

正确的思路是：**一次函数调用处理整个当前层，收集好整个下一层，再递归处理下一层。** 这只是用递归组织分层循环，实际学习和解题中通常优先使用队列版。

#### 递归分层 BFS 伪代码

```text
visited = {start}

BFS_LAYER(current_layer, depth):
    如果 current_layer 为空:
        返回

    next_layer = 空列表
    对于 current_layer 中每个节点 u:
        处理 u，它的距离为 depth
        对于 u 的每个邻居 v:
            如果 v 未访问:
                标记 v 已访问
                将 v 加入 next_layer

    BFS_LAYER(next_layer, depth + 1)

BFS_LAYER([start], 0)
```

#### 对应 Python 实现

```python
def bfs_recursive_by_level(graph, start):
    visited = {start}
    order = []
    distance = {}

    def visit_layer(current_layer, depth):
        if not current_layer:
            return

        next_layer = []
        for node in current_layer:
            order.append(node)
            distance[node] = depth
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_layer.append(neighbor)

        visit_layer(next_layer, depth + 1)

    visit_layer([start], 0)
    return order, distance
```

**错误思路**：在 `for neighbor` 内直接调用一个只处理该邻居的递归函数，并让它继续深入。这样会先走完一条分支，变成 DFS，不再是 BFS。

#### 递归版与队列版的区别

| 对比项 | 队列 BFS | 分层递归 BFS |
|---|---|---|
| 下一步工作如何保存 | 队列 | 当前层、下一层列表 |
| 怎样进入下一层 | 队列顺序自然保证，或固定层大小 | 当前层全部处理完后递归 |
| 递归栈 | 无 | 与可达层数成正比 |
| 邻接表时间 | `O(V+E)` | `O(V+E)` |
| 辅助空间上界 | `O(V)` | `O(V)`，包含保留的层列表与调用栈 |
| 使用建议 | 通常首选 | 用来理解递归与搜索策略的区别 |

递归版即使只在函数末尾调用自己，也不会被 Python 自动优化为循环。一条很长的链会产生很多层，队列版更适合这类输入。

### 7. 多源 BFS 与显式分层

如果有多个起点，把它们**全部以距离 0 同时放进初始队列**。后续第一次到达某点的距离，就是它到最近一个起点的最短距离。

如果题目按分钟扩散，可以固定每轮开始的队列长度：

```text
当队列非空:
    level_size = 当前队列长度
    重复 level_size 次:
        弹出一个当前层节点
        把新发现的下一层节点加入队尾
    一整层处理结束，再进入下一分钟
```

不能在这轮里把刚入队的节点也一直处理到底，否则一分钟内就可能传播多步。腐烂的橘子正是这个模板。

## 基础六：visited 究竟记录什么？

### 1. 搜索可达点：发现一次就够

图中存在 `A→B→A`，若不标记，就可能无限来回搜索。即使无环，某点也可能由多条路径到达，标记能避免重复工作。

连通块搜索关注“是否能到达”，一旦到达过，就不必因为换了一条路径而再展开它，因此标记不撤销。

### 2. 回溯路径：仅禁止当前路径内重复

单词搜索关注“这条路径能否拼出单词”。一个格子在当前路径中用过，不能再次使用；但换条路径，它仍然可能有用，所以返回时要撤销标记。

| 任务 | 标记含义 | 返回时是否撤销 |
|---|---|---|
| 岛屿数量、普通可达性 | 整轮搜索已经发现过 | 不撤销 |
| BFS 最短距离 | 已经首次发现并确定距离 | 不撤销 |
| 单词搜索、枚举简单路径 | 当前路径已经使用 | 撤销 |
| 有向图 DFS 判环 | 区分未进入、正在处理、已经完成 | 从“正在处理”转为“完成” |

### 3. 为什么有向图判环不能只看 visited？

考虑 `A→B、A→C、C→B`。先从 A 搜完 B，再经过 C 指向 B，这时 B 已访问，但图中没有环。

要判断是不是返回了当前递归路径上的祖先，需要三种状态：

- 0：从未访问。
- 1：正在处理，还在当前调用栈中。
- 2：已经处理完成，其后续检查已经结束。

沿边遇到状态 1，才说明当前路径绕回祖先，形成有向环；遇到状态 2，只是到达了已经检查过的部分。

无向图判环还要排除返回父节点的同一条边，不能直接照搬有向图的规则。

## 基础七：入度与拓扑排序

### 1. 拓扑顺序是什么？

对有向图，把所有节点排成一个顺序，使每条边 `u→v` 都满足 u 在 v 前面。这个顺序称为拓扑顺序，只有有向无环图 DAG 才存在。

它通常不唯一。例如 `0→2、1→2`，`[0,1,2]` 和 `[1,0,2]` 都合法。

### 2. 从“现在能做哪些任务”理解入度

一个节点入度为 0，表示当前没有尚未满足的前驱依赖，可以立即处理。处理 u 后，u 的每个后继都少了一条未满足依赖，所以将入度减 1。

```text
根据有向边计算每个节点的入度
将所有入度为 0 的节点加入队列
processed = 0

当队列非空:
    取出 u
    processed += 1
    对于每条边 u → v:
        indegree[v] -= 1
        如果 indegree[v] == 0:
            将 v 入队

如果 processed == 顶点总数:
    没有环，可以完成全部任务
否则:
    剩余依赖中存在环
```

这就是 Kahn 拓扑排序。它使用队列，与 BFS 形式相似，但入队条件是“依赖清零”，不是“第一次被某个邻居发现”，不能把这里的队列顺序当作普通图最短距离。

### 3. 为什么处理不完意味着有环？

如果一张非空有向图无环，一定存在入度为 0 的节点。否则不断沿入边向前追，在有限节点中必然重复遇到某个节点，反而形成环。

因此无环图可以不断删除入度为 0 的点，直到全部处理。若卡住且还有节点，剩余图中必然有环；不过剩余的每一个点未必都在环上，有些只是被环阻塞的后续任务。

## 200. 岛屿数量

题目：[岛屿数量](https://leetcode.cn/problems/number-of-islands/)。`"1"` 是陆地，`"0"` 是水，四方向相连的陆地组成岛屿。

### 1. 把题目翻译成图问题

- 每个陆地格子是一个节点。
- 上下左右相邻的陆地之间有无向边。
- 一个岛屿就是一个连通分量。

因此问题变成：**有多少块彼此不连通的陆地集合？** 我们不需要枚举到某个格子的所有路径，只要确定它属于哪一块。

### 2. 总体伪代码

```text
islands = 0
逐行逐列扫描格子:
    如果格子是尚未访问的陆地:
        islands += 1
        从这里开始搜索，标记整个相连的岛屿
返回 islands
```

答案只在“发现新岛屿起点”时加 1，不是在访问每一格时加 1。

下面三个版本统一用 `grid[r][c]="0"` 表示已访问：原本的水和已经访问过的陆地，后续都不需要再次进入，所以可以复用同一标记。这些版本都会修改输入；比较不同版本时，应分别传入网格的副本。

### 3. 非递归 DFS：显式栈版本

`mark_island(start_row,start_col)` 的含义：给定一个尚未访问的陆地起点，把与它相连的全部陆地标记掉。

将“扫描新岛屿”与“处理一个岛屿”分成两层函数，便于看清各自职责。

```python
class Solution:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def mark_island(start_row, start_col):
            grid[start_row][start_col] = "0"
            stack = [(start_row, start_col)]

            while stack:
                row, col = stack.pop()
                for dr, dc in directions:
                    next_row = row + dr
                    next_col = col + dc

                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if grid[next_row][next_col] != "1":
                        continue

                    # 发现时立即标记，避免同一个格子重复入栈。
                    grid[next_row][next_col] = "0"
                    stack.append((next_row, next_col))

        islands = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    islands += 1
                    mark_island(r, c)
        return islands
```

### 4. 递归 DFS：用函数调用完成同一件事

`dfs(row,col)` 的定义：如果这里是未访问陆地，就标记它，再继续处理四个方向；如果越界、是水或已访问，则返回。

不同版本使用不同类名便于区分；单独提交该版本时，将类名改为 `Solution`。

```python
class SolutionRecursive:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])

        def dfs(row, col):
            if not (0 <= row < rows and 0 <= col < cols):
                return
            if grid[row][col] != "1":
                return

            grid[row][col] = "0"
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)

        islands = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    islands += 1
                    dfs(row, col)
        return islands
```

四个调用按代码顺序执行。第一个 `dfs(row+1,col)` 会先完成它的整段探索，返回后才调用上方邻居。这正是递归 DFS 的深入再返回。

此版本直观，但大岛屿可能形成很深的调用链。实际遇到大网格时，优先使用上面的显式栈或下面的队列版本。

### 5. BFS 版本：同一个连通块按层搜索

本题只关心一块陆地能否到达另一块，DFS 和 BFS 都能找齐整座岛，访问顺序不影响计数。

```python
from collections import deque


class SolutionBFS:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        islands = 0

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != "1":
                    continue

                islands += 1
                grid[row][col] = "0"
                queue = deque([(row, col)])
                while queue:
                    current_row, current_col = queue.popleft()
                    for dr, dc in directions:
                        next_row = current_row + dr
                        next_col = current_col + dc
                        if not (0 <= next_row < rows and 0 <= next_col < cols):
                            continue
                        if grid[next_row][next_col] != "1":
                            continue
                        grid[next_row][next_col] = "0"
                        queue.append((next_row, next_col))

        return islands
```

这里没有计算距离，也不需要知道每一层有多少格子，所以直接用队列处理即可，不必写 `level_size`。

### 6. 手算过程

```text
原网格：      找到第一座岛后：  找到第二座岛后：
1 1 0        0 0 0           0 0 0
0 0 1        0 0 1           0 0 0
```

1. 扫到 `(0,0)`，是新陆地，`islands=1`。
2. 搜索会访问 `(0,0)` 和 `(0,1)`，将它们标记掉。
3. 外层后来扫到 `(0,1)` 时，它已经不是未访问陆地，不会重复加 1。
4. 扫到 `(1,2)`，开始第二次搜索，`islands=2`。

`(0,1)` 与 `(1,2)` 只在对角线上接触，没有上下左右的边，因此不属于同一座岛。

### 7. 正确性、易错点与复杂度

每次搜索只沿相邻陆地前进，不会跨海或跨对角线，所以不会混合两座岛；沿所有合法方向继续探索，又能标记整个连通块，所以不会漏掉同一座岛的成员。

- 先检查坐标范围，再访问网格。
- 本题是字符串 `"1"`、`"0"`，不是橘子题的整数 1、0。
- 不要在递归返回时改回 `"1"`，这里不是枚举不同路径，不需要撤销访问标记。
- 若要保留输入，可以单独用 visited 集合记录坐标，或调用前复制各行。

三个版本时间均为 `O(RC)`，最坏辅助空间均为 `O(RC)`。把标记写在输入里只是省掉独立 visited，显式栈、队列或递归调用栈仍然占空间。

## 994. 腐烂的橘子

题目：[腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)。0 空地、1 新鲜、2 腐烂；每分钟腐烂橘子同时感染四邻居，求全部腐烂所需最短时间。

### 1. 为什么是多源 BFS？

每个有橘子的格子可以视为节点，上下左右相邻的橘子之间有边，一条边代表传播需要 1 分钟。

每个初始腐烂橘子都是传播源。某个新鲜橘子最早腐烂的时刻，就是它到**最近传播源**的距离，因此应使用多源 BFS。

只从一个腐烂橘子出发，无法正确模拟其他源同时传播；普通 DFS 即使能找到可达格子，也不保证首次到达的路径最短。

### 2. 变量含义与总体流程

| 变量 | 含义 |
|---|---|
| `queue` | 当前等待向周围传播的腐烂橘子 |
| `fresh` | 还剩多少新鲜橘子 |
| `minutes` | 已经完成了多少轮传播 |
| `level_size` | 本轮开始时应处理的橘子数量 |

```text
扫描网格:
    所有腐烂橘子入队
    统计新鲜橘子数量

只要还有新鲜橘子，并且队列还有传播源:
    固定当前层大小
    处理当前层每个腐烂橘子:
        检查四个邻居
        邻居是新鲜橘子 → 标为腐烂，fresh 减 1，加入下一层
    minutes 加 1

如果 fresh 为 0，返回 minutes
否则返回 -1
```

### 3. 易读 Python 实现

```python
from collections import deque


class Solution:
    def orangesRotting(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue = deque()
        fresh = 0

        # 第一步：所有传播源同时入队，并统计待感染的橘子。
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1

        # 第二步：按分钟进行整层传播。
        minutes = 0
        while queue and fresh > 0:
            level_size = len(queue)
            for _ in range(level_size):
                row, col = queue.popleft()
                for dr, dc in directions:
                    next_row = row + dr
                    next_col = col + dc
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if grid[next_row][next_col] != 1:
                        continue

                    grid[next_row][next_col] = 2
                    fresh -= 1
                    queue.append((next_row, next_col))
            minutes += 1

        if fresh == 0:
            return minutes
        return -1
```

### 4. 逐分钟推演

输入：

```text
2 1 1
1 1 0
0 1 1
```

初始有 6 个新鲜橘子，唯一传播源为 `(0,0)`。

| 时刻 | 这一时刻新腐烂的格子 | 剩余 fresh |
|---:|---|---:|
| 0 | 初始源 `(0,0)` | 6 |
| 1 | `(1,0)`、`(0,1)` | 4 |
| 2 | `(1,1)`、`(0,2)` | 2 |
| 3 | `(2,1)` | 1 |
| 4 | `(2,2)` | 0 |

因此答案为 4。第二分钟 `(1,1)` 同时可能被两个邻居发现，但首次发现就改成 2，后一个邻居看到它已经腐烂，就不会再次入队或重复减少 fresh。

### 5. 为什么原地改成 2 不会导致同一分钟连续传播？

“标记腐烂”和“向邻居传播”分开进行。新感染的节点虽然立即改成 2，却只被放入队尾；本轮只执行原来的 `level_size` 次出队，因此它要到下一轮才会传播。

如果直接逐格扫描，一旦发现旁边有 2 就立即感染，还让这个新 2 在同轮继续传播，就可能让一整条链在一分钟内腐烂，违背题意。

### 6. 边界、正确性与复杂度

- 开始没有新鲜橘子：循环不执行，返回 0。
- 有新鲜橘子但没有腐烂源：队列为空，返回 -1。
- 新鲜橘子被空地隔开、不可达：传播结束后 fresh 仍大于 0，返回 -1。
- 最后一批橘子被感染后，不需要让它们再传播一次，所以循环还判断 `fresh > 0`，避免多计算一轮。

多源 BFS 首次感染给出每个格子的最早时间，所有橘子完成的最早时刻就是这些时间的最大值。该实现通过逐层计数得到这个最大值。

时间 `O(RC)`，队列空间最坏 `O(RC)`。实现会修改网格中的橘子状态。

## 207. 课程表

题目：[课程表](https://leetcode.cn/problems/course-schedule/)。`[course, prerequisite]` 表示必须先上 prerequisite，判断能否完成全部课程。

### 1. 先把输入方向读对

`[1,0]` 的含义是“想学 1，必须先学 0”，所以建立有向边 `0→1`，不是 `1→0`。

如果出现 `0→1→2→0`，每门课都在等待环中的另一门先完成，没有起点，无法学完。

因此本题等价于：**这个有向依赖图中是否没有环？**

### 2. 方案一：拓扑排序，按可学习顺序处理

`graph[pre]` 保存“完成 pre 后，可以减少一条依赖的后续课程”。`indegree[course]` 表示这门课还剩多少直接先修依赖。

队列保存目前就可以学习的课程，`finished` 统计真正处理完的课程数。

#### 易读 Python 实现

```python
from collections import deque


class Solution:
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses

        # 建图：先修课指向后续课程。
        for course, pre in prerequisites:
            graph[pre].append(course)
            indegree[course] += 1

        # 找出一开始就能学习的所有课程，包括没有边的独立课程。
        queue = deque()
        for course in range(numCourses):
            if indegree[course] == 0:
                queue.append(course)

        finished = 0
        while queue:
            course = queue.popleft()
            finished += 1

            # 已经完成 course，解除它对每个后续课程的一条依赖。
            for next_course in graph[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    queue.append(next_course)

        return finished == numCourses
```

#### 手算一个菱形依赖图

```text
输入：numCourses=4
prerequisites=[[1,0],[2,0],[3,1],[3,2]]

     0
    / \
   1   2
    \ /
     3
```

初始入度为 `[0,1,1,2]`，队列为 `[0]`。

| 刚学完的课程 | 更新后的入度 | 队列 | finished |
|---:|---|---|---:|
| 0 | `[0,0,0,2]` | `[1,2]` | 1 |
| 1 | `[0,0,0,1]` | `[2]` | 2 |
| 2 | `[0,0,0,0]` | `[3]` | 3 |
| 3 | `[0,0,0,0]` | `[]` | 4 |

学完 1 后，3 还不能入队，因为它还依赖 2。只有所有直接先修依赖都清零，才说明 3 已准备好。

### 3. 方案二：递归 DFS，用三色状态判断有向环

这个版本不模拟实际学习顺序，而是直接检查有没有循环依赖。

`dfs(course)` 的定义：检查从 course 沿出边能到达的部分是否无环，无环返回 True，发现环返回 False。

```text
DFS(u):
    如果 state[u] == 1:
        返回 False    # 回到当前递归路径，形成环
    如果 state[u] == 2:
        返回 True     # 这部分已经完整检查过

    state[u] = 1
    对于每个后继 v:
        如果 DFS(v) 返回 False:
            返回 False
    state[u] = 2
    返回 True

对所有课程分别检查 DFS，只要一次发现环就失败
```

```python
class SolutionDFS:
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        for course, pre in prerequisites:
            graph[pre].append(course)

        # 0：未访问；1：当前调用栈中；2：已完成检查。
        state = [0] * numCourses

        def dfs(course):
            if state[course] == 1:
                return False
            if state[course] == 2:
                return True

            state[course] = 1
            for next_course in graph[course]:
                if not dfs(next_course):
                    return False
            state[course] = 2
            return True

        for course in range(numCourses):
            if not dfs(course):
                return False
        return True
```

#### 为什么状态 1 是环，状态 2 不是？

例如 `0→1→2→0`，调用过程是：

```text
进入 0，state[0]=1
    进入 1，state[1]=1
        进入 2，state[2]=1
            再次遇到 0，state[0] 仍为 1 → 存在环
```

而在上面的菱形图里，可能先搜索 `0→1→3`，3 完成后状态为 2；再搜索 `0→2→3`，到达已完成的 3 只表示两个分支共享后续课程，不代表有环。

完成后设置为 2，不是恢复成 0：从该节点出发无环这个结论可以复用，避免后续再次展开整个子图。如果发现环，整题会立即返回 False，此时无需为继续搜索恢复状态。

### 4. 如何选择版本？

| 版本 | 核心状态 | 成功依据 | 更适合的理解角度 |
|---|---|---|---|
| 拓扑排序 | 剩余入度、可学习队列 | 实际处理数量等于全部课程数 | 模拟依赖解除 |
| 递归 DFS | 未访问、处理中、已完成 | 没有指向当前调用栈节点的边 | 检查循环依赖 |

两者都是 `O(V+E)` 时间、`O(V+E)` 辅助空间，包含建图。依赖链很深时，队列版本不受递归深度影响；如果用显式栈改写三色 DFS，应该使用前文“调用帧”结构，在真正退出节点时才设置为 2。

### 5. 易错点

- 没有依赖关系的独立课程也需要计入总数。
- 只从课程 0 出发不够，另一个不连通部分也可能有环。
- 不能只判断队列最后是否为空，成功和失败最后都会空；应比较 finished。
- 不能把“指向任意 visited 节点”都当成环，DFS 要区分状态 1 和 2。
- 课程依赖图不是必须有唯一拓扑顺序，本题只问是否存在至少一种可行顺序。

## 208. 实现 Trie（前缀树）

题目：[实现 Trie](https://leetcode.cn/problems/implement-trie-prefix-tree/)。支持插入单词、查完整单词、查是否存在某个前缀。

### 1. Trie 与普通图、二叉树有什么关系？

Trie 是一种树形结构，可以看作边上带字符标签的有向树。它不是二叉树，一个节点可以有多个孩子，例如分别对应 a、b、c。

根表示空前缀；从根沿字符依次向下，走出的整条路径表示一个前缀。节点自身不一定需要再存字符，因为字符已经保存在父节点的 children 映射中。

插入 `app、apple、bat` 后：

```text
root
├── a
│   └── p
│       └── p  *        表示单词 app 在这里结束
│           └── l
│               └── e  *   表示 apple 在这里结束
└── b
    └── a
        └── t  *        表示 bat 在这里结束
```

星号是结束标记，不是额外字符。注意 app 的结束节点还能有孩子，所以“一个单词结束”不等于“必须到达叶子”。

### 2. 每个节点保存什么？

| 属性 | 含义 | 例子 |
|---|---|---|
| `children` | 字符 → 下一个节点 | `node.children["p"]` |
| `is_end` | 当前路径是否为一个完整单词 | app 插入后，第三个字符对应节点为 True |

`node = node.children[ch]` 只是把当前引用移动到已有子节点，不会复制节点；`node.children[ch] = TrieNode()` 才是在树中新增结构。

与一般图搜索不同，查询某个单词时，下一个字符已经给定，只需沿唯一对应的边走，不必 DFS/BFS 枚举全部分支，也不需要 visited。

### 3. 三个操作的伪代码

```text
INSERT(word):
    node = root
    对于 word 的每个字符 ch:
        如果 node 没有 ch 对应的孩子:
            创建这个孩子
        node = ch 对应的孩子
    将 node.is_end 设为 True

WALK(text):
    node = root
    对于 text 的每个字符 ch:
        如果没有对应孩子:
            返回不存在
        走到对应孩子
    返回最后到达的节点

SEARCH(word):
    node = WALK(word)
    返回 node 存在，并且 node.is_end 为 True

STARTS_WITH(prefix):
    返回 WALK(prefix) 能走到某个节点
```

### 4. 易读 Python 实现

```python
class TrieNode:
    def __init__(self):
        # 每个节点有自己的孩子字典和结束标记。
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
        # 必须在整个单词处理完后设置。
        node.is_end = True

    def _walk(self, text):
        # 返回匹配 text 后到达的节点；中途缺少边则返回 None。
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

### 5. 手算插入与查询

先插入 apple：

1. 根没有 a，创建 a 节点并走过去。
2. 依次创建 p、p、l、e。
3. 只将 e 对应节点的 `is_end` 设为 True，中间的 app 节点此时仍为 False。

此时的查询结果：

| 操作 | 结果 | 原因 |
|---|---|---|
| `search("apple")` | True | 整条路径存在，终点有结束标记 |
| `search("app")` | False | 路径存在，但 app 尚未作为完整单词插入 |
| `startsWith("app")` | True | 前缀路径存在即可 |
| `search("apply")` | False | 走到 appl 后，没有 y 这条边 |

再插入 app：只走已有的 a→p→p，把最后节点的 `is_end` 设为 True，**不需要新建任何节点**。此后 `search("app")` 为 True，而 apple 仍然存在。

### 6. 易错点与复杂度

- 路径存在与完整单词存在是两个条件，必须保留结束标记。
- 不要在每个字符之后都设置 `is_end=True`，否则 apple 会错误地把 a、ap、app、appl 都当成已插入单词。
- 每个节点的 children 要独立，不能让所有节点共享同一个字典。
- 重复插入相同单词只是再次设置同一个结束标记，不会重复创建整条路径。

长度为 L 的插入或查询平均时间为 `O(L)`，查询辅助空间为 `O(1)`；一次插入最多创建 L 个新节点。整棵树的空间上界为 `O(S+1)`，S 是全部插入字符串长度之和，额外的 1 是根节点；共享前缀通常会减少实际节点数。

## 入门路线

### 先练基础，再回到题目

1. 用 A、B、C、D、E、F 的例子构造邻接表，手动写出 DFS 与 BFS 的访问顺序。
2. 给图加一条连接，再确认不同分支共享节点时，visited 如何防止重复处理。
3. 比较递归 DFS 与显式栈版；如果要模拟返回过程，再阅读调用帧伪代码。
4. 比较队列 BFS 与分层递归 BFS，确认每个节点的距离一致。
5. 做岛屿数量，理解“外层数连通块，内层搜索一整块”。
6. 做腐烂的橘子，理解“多源同时出发，整层对应一分钟”。
7. 做课程表，用同一组输入分别画入度变化和 DFS 三色状态。
8. 做 Trie，理解前缀共享和完整单词结束标记。

### 四道题的状态对比

| 题目 | 节点是什么 | 主要状态 | 什么时候产生答案 |
|---|---|---|---|
| 岛屿数量 | 陆地格子 | 是否已访问、待处理栈/队列 | 每启动一次新连通块搜索，计数加 1 |
| 腐烂的橘子 | 橘子格子 | 当前传播层、fresh、minutes | 所有新鲜橘子感染后返回轮数 |
| 课程表 | 课程 | 剩余入度，或 DFS 三色 | 全部课程可处理，或搜索发现有向环 |
| Trie | 字符前缀 | children、is_end、当前节点 | 走完整个查询，再检查所需条件 |

### 写完代码后的自检问题

- 图是否有方向？每条输入关系对应哪条边？
- 起点只有一个，还是要从多个起点同时搜索，或者遍历所有连通部分？
- 标记的是“发现过”“当前路径中用过”，还是“已经处理完成”？
- 在入队/入栈时标记了吗？有没有重复统计同一个节点？
- 是否真正需要按层计数？新加入的节点会不会被错误地在同一轮处理？
- 返回上一层后，要撤销路径状态，还是保留已经访问/完成的结论？
- 复杂度有没有计入邻接表、队列、显式栈或递归栈？

**记住：搜索函数的名字不重要，重要的是它负责哪一部分状态、按什么顺序处理节点，以及哪些信息能够在后续搜索中复用。**
