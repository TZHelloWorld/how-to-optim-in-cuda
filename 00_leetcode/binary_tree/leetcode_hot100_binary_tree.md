# Hot100 二叉树：递归含义、遍历顺序与子树信息

[返回总索引](../README.md)

题面中的 `root=[...]` 使用层序表示：从上到下、从左到右列出节点，None 表示缺少孩子；`root=[]` 表示空树。函数真正接收的是 TreeNode 根引用，下文提供本地构造函数。返回“节点”时指原对象，返回“值”或“列表”时会另外说明。

## 基础一：树与二叉树

树由节点和边组成。根没有父节点，叶子没有孩子。二叉树的每个节点最多有左、右两个孩子。

```text
        4          根
       / \
      2   6
     / \
    1   3          1、3、6 是叶子
```

- 子树：某个节点以及它的全部后代。
- 深度：从根向下到当前节点的距离；题目可能按节点或按边计数，先确认定义。
- 高度：从当前节点向下到最深叶子的距离。
- 本文最大深度按节点数，空树为 0；直径按边数。
- 平衡树高度通常为 `O(log n)`；退化成链的树高度为 `O(n)`。

### 二叉搜索树 BST

每个节点的整个左子树值都小于它，整个右子树值都大于它。本专题第 98 题要求严格大小，不允许重复值。

BST 的中序遍历严格递增。普通二叉树没有这个性质，不能随意比较节点值来决定向哪边搜索。

## 基础二：节点与本地构造

LeetCode 提供 `TreeNode`，本地运行题解前先执行本节。层序列表中的 `None` 表示缺少孩子，不是数值为零的节点。

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values):
            if values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
    return root
```

## 基础三：DFS、BFS 与递归契约

| 遍历 | 顺序 | 常见用途 |
|---|---|---|
| 前序 | 根、左、右 | 复制结构、展开、构建 |
| 中序 | 左、根、右 | BST 有序访问 |
| 后序 | 左、右、根 | 汇总子树高度、路径贡献 |
| 层序 BFS | 一层一层 | 最短层数、层次视图 |

递归树题先回答：**函数返回给父节点什么信息？**

- 高度函数返回一棵子树的高度。
- 最大路径和函数返回一条可以向父节点延伸的单边路径贡献。
- 返回值不一定等于整题答案，整题答案可能在遍历过程中更新。

DFS 递归栈空间通常为 `O(h)`，BFS 队列空间为 `O(w)`，`h` 是高度，`w` 是最大层宽。Python 对极深递归有限制；本文递归实现便于理解，面对很深的退化树，可改为显式栈遍历，不能把递归空间写成 `O(1)`。

## 94. 二叉树的中序遍历

题目：[二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)。按左、根、右输出节点值。

### 题目描述

给定二叉树根节点 root，按中序顺序返回全部节点值：先遍历左子树，再访问当前节点，最后遍历右子树，每棵子树都遵守相同规则。

**输入与约束**：树可以为空，不保证是二叉搜索树，节点值可以重复。输出每个真实节点的值一次，空位置不加入结果；进阶可使用迭代方式实现。

```text
输入：root=[1,None,2,3]
输出：[1,3,2]
解释：1 没有左孩子；右子树根为 2，其左孩子为 3。

输入：root=[]
输出：[]
```

### 从题意到算法

当前节点必须等待左子树处理完才能输出。递归会自动保存等待的祖先，迭代版则用栈保存它们：一路向左压栈，遇到空后弹出访问，再转向右子树。栈和当前引用都为空时，才表示整棵树完成。

**状态**：栈保存尚未访问自身、但正在处理其左子树的祖先。

```python
class Solution:
    def inorderTraversal(self, root):
        ans = []
        stack = []
        cur = root
        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            ans.append(cur.val)
            cur = cur.right
        return ans
```

**推演**：根 2、左 1、右 3，先把 2、1 入栈；弹 1，再弹 2，最后处理右子树 3，得到 `[1,2,3]`。

**易错点**：弹栈访问后转向右孩子，不是再转左；栈空但 `cur` 非空时还需继续。

时间 `O(n)`；辅助空间 `O(h)`，另计输出 `O(n)`。

## 104. 二叉树的最大深度

题目：[二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)。求最多经过多少个节点到达叶子。

### 题目描述

给定二叉树 root，返回从根到最远叶子的一条路径上包含的节点数量。叶子是没有任何孩子的节点；空树深度为 0。

**输入与约束**：树可能高度不平衡，也可能退化成链。深度按节点数计算，单节点树深度为 1，不是按边数计算。

```text
输入：root=[3,9,20,None,None,15,7]
输出：3
解释：路径 3→20→15 或 3→20→7 都经过 3 个节点。

输入：root=[]
输出：0
```

### 从题意到算法

递归可用“当前高度等于左右子树最大高度加一”推导。下面使用层序遍历：队列先放根，每次只处理本轮已有的一层，再把孩子留给下一轮；完整处理了几层，最大深度就是几。这个实现也适合很深、可能触及递归限制的树。

**思路**：逐层 BFS，每处理完整一层，深度加 1。递归关系也可以理解为 `height(node)=1+max(height(left),height(right))`。

```python
from collections import deque


class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        queue = deque([root])
        depth = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            depth += 1
        return depth
```

**推演**：`[3,9,20,None,None,15,7]` 分为 `[3]`、`[9,20]`、`[15,7]`，深度 3。

**易错点**：每轮开始固定当前队列长度，后面入队的属于下一层。

时间 `O(n)`，辅助空间 `O(w)`。

## 226. 翻转二叉树

题目：[翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)。每个节点交换左右子树。

### 题目描述

给定二叉树 root，将它左右镜像翻转：对每个节点都交换其左孩子与右孩子，返回翻转后的根节点。

**输入与约束**：树可以为空，节点值无须有序；需要交换整棵子树的连接，不只是交换两个孩子的值。本节实现修改原树。

```text
输入：root=[4,2,7,1,3,6,9]
输出的树：[4,7,2,9,6,3,1]
解释：根的两个子树交换，每棵子树内部也继续交换。

输入：root=[]
输出：None（空树）
```

### 从题意到算法

每个节点的任务相同：交换两个孩子，再让两个子调用分别翻转交换后的子树。空节点返回空，不再深入。只有根交换一次还不够，递归确保所有层都完成相同变换；返回的是当前根引用，不需要创建新节点。

**函数定义**：`invertTree(node)` 翻转以 node 为根的整棵子树，并返回根引用。

```python
class Solution:
    def invertTree(self, root):
        if root is None:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

**推演**：`[2,1,3]` 变成 `[2,3,1]`。每个节点局部交换，递归保证所有后代也交换，因此整体镜像。

**易错点**：只交换根的两个孩子不够；修改的是原树结构。

时间 `O(n)`，递归辅助空间 `O(h)`。

## 101. 对称二叉树

题目：[对称二叉树](https://leetcode.cn/problems/symmetric-tree/)。判断树是否关于根左右镜像。

### 题目描述

给定二叉树 root，判断它是否关于根的竖直中轴线对称。对称既要求对应位置的节点值相同，也要求对应位置同时存在或同时为空。

**输入与约束**：允许重复节点值；不能仅比较每层的数值集合或只看左右孩子值。左右子树必须在所有层都互为镜像。

```text
输入：root=[1,2,2,3,4,4,3]
输出：True

输入：root=[1,2,2,None,3,None,3]
输出：False
解释：两个 3 都是右孩子，结构不镜像。
```

### 从题意到算法

比较两个子树 a、b：一个为空一个非空立即失败，都空则成立；非空时要求值相等，并交叉比较 a.left 与 b.right、a.right 与 b.left。用交叉递归完整表达镜像关系，比两棵子树各自遍历后比较数值更可靠。

**函数定义**：`mirror(a,b)` 判断两棵子树是否互为镜像。需要比较 a.left 与 b.right、a.right 与 b.left。

```python
class Solution:
    def isSymmetric(self, root):
        def mirror(a, b):
            if a is None or b is None:
                return a is b
            return (a.val == b.val
                    and mirror(a.left, b.right)
                    and mirror(a.right, b.left))

        return root is None or mirror(root.left, root.right)
```

**推演**：`[1,2,2,3,4,4,3]` 中外侧 3 对 3、内侧 4 对 4，成立。

**易错点**：左右子树各自中序相同不代表结构对称；必须同时比较值、空节点位置和镜像结构。

时间 `O(n)`，辅助空间 `O(h)`。

## 543. 二叉树的直径

题目：[二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)。任意两点间最长路径的边数，路径不一定经过根。

### 题目描述

给定二叉树 root，返回树的直径，即任意两个节点之间最长简单路径所包含的边数。路径不能重复节点，也不一定经过根或从根出发。

**输入与约束**：原题至少一个节点；单节点树直径为 0。节点值对答案没有影响，重要的是树的连接结构。

```text
输入：root=[1,2,3,4,5]
输出：3
解释：4→2→1→3 或 5→2→1→3 都有 3 条边。

输入：root=[1]
输出：0
```

### 从题意到算法

每条路径都有一个最高节点，可分成它的左侧支路与右侧支路。因此遍历每个节点时，用左右子树高度之和更新直径；向父节点返回的却只能是较长一侧高度加一。把“用于全局答案的双边路径”和“可继续向上的单边高度”分开，才能同时正确计算。

**函数定义**：`height(node)` 返回以 node 为根的最大节点深度。若左右高度为 `left、right`，经过当前节点的最长路径边数就是 `left+right`。

```python
class Solution:
    def diameterOfBinaryTree(self, root):
        ans = 0

        def height(node):
            nonlocal ans
            if node is None:
                return 0
            left = height(node.left)
            right = height(node.right)
            ans = max(ans, left + right)
            return 1 + max(left, right)

        height(root)
        return ans
```

**推演**：`[1,2,3,4,5]`，节点 1 的左右高度分别 2 和 1，路径 `4→2→1→3` 有 3 条边。

**易错点**：返回给父节点的是单边高度，不是左右之和；全局答案要在每个节点更新，不只看根。

时间 `O(n)`，辅助空间 `O(h)`。

## 102. 二叉树的层序遍历

题目：[二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)。按层输出二维列表。

### 题目描述

给定二叉树 root，按从根到叶的层级顺序返回节点值。每一层单独放进一个列表，层内从左到右排列。

**输入与约束**：树可以为空，空树返回空列表；不要输出代表缺失孩子的 None，也不要将不同层混成一个列表。

```text
输入：root=[3,9,20,None,None,15,7]
输出：[[3],[9,20],[15,7]]

输入：root=[1]
输出：[[1]]
```

### 从题意到算法

用队列保存当前层，先记录队列长度，再恰好弹出这个数量的节点。它们的孩子按左、右顺序入队，留给下一层；当前层的值收集完毕后一次加入答案。固定层大小是保证分层正确的关键。

**状态**：每轮开始队列中恰好是当前层的节点，固定长度后处理。

```python
from collections import deque


class Solution:
    def levelOrder(self, root):
        if root is None:
            return []
        queue = deque([root])
        ans = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            ans.append(level)
        return ans
```

**推演**：`[3,9,20,None,None,15,7]` 输出 `[[3],[9,20],[15,7]]`。

**易错点**：不能在同一轮把新入队的下一层也处理掉；使用 `deque.popleft()`。

时间 `O(n)`；辅助空间 `O(w)`，输出 `O(n)`。

## 108. 将有序数组转换为二叉搜索树

题目：[将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)。建立高度平衡的 BST。

### 题目描述

给定严格升序整数数组 nums，将它转换为高度平衡的二叉搜索树，返回根节点。高度平衡要求每个节点左右子树高度差不超过 1；二叉搜索树要求左侧所有值小于根，右侧所有值大于根。

**输入与约束**：原题 nums 非空，元素互异；必须使用所有元素，允许返回任意合法平衡 BST，不要求唯一树形。

```text
输入：nums=[-10,-3,0,5,9]
一种合法输出：root=[0,-3,9,-10,None,5]
解释：中序遍历仍得到原升序数组，每个节点左右高度差不超过 1。

输入：nums=[1]
输出：root=[1]
```

### 从题意到算法

根取中间元素，左右两半分别递归建左右子树。排序性质保证左值小右值大，近似等分保证高度平衡。传递区间边界而不是反复切片，可以避免额外数组复制；区间为空时返回 None。

**函数定义**：`build(left,right)` 用闭区间 `nums[left:right+1]` 构树。取中点为根，左右区间分别构左右子树。

```python
class Solution:
    def sortedArrayToBST(self, nums):
        def build(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            node = TreeNode(nums[mid])
            node.left = build(left, mid - 1)
            node.right = build(mid + 1, right)
            return node

        return build(0, len(nums) - 1)
```

**推演**：`[-10,-3,0,5,9]` 选 0 为根，两边各 2 个元素；每次近似平分，得到平衡结构。

**易错点**：不同合法中点选择可能产生不同树，都算正确；避免反复切片带来额外复制。

时间 `O(n)`；辅助栈 `O(log n)`，新树空间 `O(n)`。

## 98. 验证二叉搜索树

题目：[验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)。判断整棵树是否满足严格 BST 性质。

### 题目描述

给定二叉树 root，判断是否为有效二叉搜索树：对每个节点，它的整个左子树值都必须严格小于它，整个右子树值都必须严格大于它，且两个子树自身也满足相同要求。

**输入与约束**：原题树非空；值可为负数，重复值不符合严格 BST 规则。只比较父节点和直接孩子是不够的，还要满足所有祖先施加的范围限制。

```text
输入：root=[2,1,3]
输出：True

输入：root=[5,4,6,None,None,3,7]
输出：False
解释：3 虽小于它的父节点 6，却位于根 5 的右子树中。
```

### 从题意到算法

递归携带允许的开区间 `(low,high)`。左子树把上界收紧为当前值，右子树把下界收紧为当前值，其他祖先边界继续保留。某个节点不在范围内就失败；这样一次遍历检查整个子树关系，而非只有相邻父子关系。

**函数定义**：`valid(node,low,high)` 要求当前子树每个值满足祖先传下来的范围限制。

```python
class Solution:
    def isValidBST(self, root):
        def valid(node, low, high):
            if node is None:
                return True
            if not low < node.val < high:
                return False
            return (valid(node.left, low, node.val)
                    and valid(node.right, node.val, high))

        return valid(root, float("-inf"), float("inf"))
```

**推演**：`[5,1,4,None,None,3,6]` 中，右子树根 4 不大于 5，立即失败。更深层节点也必须遵守祖先范围。

**易错点**：只比较父子不够；右子树中的每一个节点都必须大于根，而不是只检查右孩子。

时间 `O(n)`，辅助空间 `O(h)`。

## 230. 二叉搜索树中第 K 小的元素

题目：[二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)。原题保证 k 合法。

### 题目描述

给定二叉搜索树根 root 和正整数 k，返回树中按从小到大排序后的第 k 个节点值。k 从 1 开始计数。

**输入与约束**：root 非空且保证是 BST，`1<=k<=节点总数`；返回值不是节点对象，也不是下标。只需找到第 k 小，不要求修改树。

```text
输入：root=[3,1,4,None,2], k=1
输出：1

相同树，k=3
输出：3，中序值序列为 [1,2,3,4]。
```

### 从题意到算法

BST 中序访问天然按升序排列，无需再排序。用栈完成中序遍历，每真正访问一个节点就令 k 减一；减到 0 时返回。计数要发生在左子树处理完、弹栈访问根时，而不是刚把根压入栈时。

**思路**：BST 中序遍历就是升序，第 k 次访问的节点就是答案，可以提前停止。

```python
class Solution:
    def kthSmallest(self, root, k):
        stack = []
        cur = root
        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right
```

**推演**：BST `[3,1,4,None,2]` 中序为 `1,2,3,4`，`k=2` 返回 2。

**易错点**：计数发生在弹栈访问节点时，不是在压栈时。

时间 `O(h+k)`、最坏 `O(n)`；辅助空间 `O(h)`。

## 199. 二叉树的右视图

题目：[二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)。返回从右侧可看到的每层最后一个节点。

### 题目描述

给定二叉树 root，想象站在树的右侧向左看，按从上到下顺序返回每一层能看到的最右节点值。

**输入与约束**：树可为空；每层最多输出一个值。它不是一直沿 right 指针走：若右子树较浅，更深处的左子树节点也可能可见。

```text
输入：root=[1,2,3,None,5,None,4]
输出：[1,3,4]

输入：root=[1,2]
输出：[1,2]，第二层只有左孩子，它仍然可见。
```

### 从题意到算法

对每层从左到右进行 BFS，最后一个出队节点就是这一层最右的节点。固定本层大小后，只记录下标为 size-1 的值，再把左右孩子加入下一层。与普通层序遍历相比，区别只是每层保留的内容不同。

**思路**：从左到右层序遍历，记录每层最后出队的节点。

```python
from collections import deque


class Solution:
    def rightSideView(self, root):
        if root is None:
            return []
        queue = deque([root])
        ans = []
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if i == size - 1:
                    ans.append(node.val)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
        return ans
```

**推演**：`[1,2,3,None,5,None,4]` 输出 `[1,3,4]`。

**易错点**：右视图不是不断走 `root.right`。右边子树较浅时，左子树更深的节点也能被看到。

时间 `O(n)`；辅助空间 `O(w)`，输出 `O(h)`。

## 114. 二叉树展开为链表

题目：[二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)。原地按前序顺序展开，所有 left 为 `None`，right 指向下一个节点。

### 题目描述

给定二叉树 root，将它原地展开成一条使用 TreeNode 的单链结构。展开顺序必须等于原树前序遍历，即根、左子树、右子树；每个节点 left 都为 None，right 指向下一个节点。

**输入与约束**：树可为空；修改原节点连接，不是创建 ListNode 链，也不是只返回一个值列表。进阶要求 `O(1)` 辅助空间。

```text
输入：root=[1,2,5,3,4,None,6]
修改后：1 → 2 → 3 → 4 → 5 → 6（箭头均为 right）
解释：所有节点的 left 都清空，顺序为原树的前序顺序。

输入：root=[]
修改后：仍为空树
```

### 从题意到算法

当前根后面应该先接完整左子树，再接原右子树。找到左子树沿 right 走到的最右节点，把原右子树挂到它的 right；再把左子树搬到当前 right，并清空 left。沿新的 right 继续处理，会逐步完成前序展开，无需辅助栈或结果数组。

**思路**：若当前有左子树，把原右子树接到左子树最右节点的右边，然后将整棵左子树搬到右边。

```python
class Solution:
    def flatten(self, root):
        cur = root
        while cur is not None:
            if cur.left is not None:
                predecessor = cur.left
                while predecessor.right is not None:
                    predecessor = predecessor.right
                predecessor.right = cur.right
                cur.right = cur.left
                cur.left = None
            cur = cur.right
```

**推演**：根 1、左子树 2 带孩子 3、4，右子树 5 带孩子 6，依次重接后得到 `1→2→3→4→5→6`。

**正确性要点**：前序要求“当前节点 → 整个左子树 → 整个右子树”，重接保持这个相对顺序，后续再展开子树内部。

**易错点**：一定清空 `left`；无需创建新节点。寻找前驱的右链扫描累计为线性，原有右边只会参与有限次遍历。

时间 `O(n)`，辅助空间 `O(1)`。

## 105. 从前序与中序遍历序列构造二叉树

题目：[从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)。节点值互不相同。

### 题目描述

给定同一棵二叉树的前序遍历数组 preorder 和中序遍历数组 inorder，重建这棵树并返回根节点。前序顺序为根、左、右；中序顺序为左、根、右。

**输入与约束**：两个数组非空、长度相同、包含相同的一组互异元素，保证来自一棵合法二叉树。无重复值使根在中序中的位置唯一，因此可以唯一重建。

```text
输入：preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]
输出的树：root=[3,9,20,None,None,15,7]

输入：preorder=[-1], inorder=[-1]
输出的树：root=[-1]
```

### 从题意到算法

前序片段的第一个值是根，在中序里找到它，就把当前节点集合分成左右两段。左段长度又确定前序中左子树的范围，剩下的是右子树。用字典快速查中序位置、用起点与长度描述区间，递归构造子树，避免反复搜索和切片。

**状态与思路**：前序的第一个元素是根；在中序中找到根的位置，就知道左右子树各有多少节点。`build(pre_left,in_left,size)` 返回指定片段的子树。

```python
class Solution:
    def buildTree(self, preorder, inorder):
        position = {value: i for i, value in enumerate(inorder)}

        def build(pre_left, in_left, size):
            if size == 0:
                return None
            value = preorder[pre_left]
            mid = position[value]
            left_size = mid - in_left
            node = TreeNode(value)
            node.left = build(pre_left + 1, in_left, left_size)
            node.right = build(pre_left + 1 + left_size, mid + 1,
                               size - left_size - 1)
            return node

        return build(0, 0, len(preorder))
```

**推演**：前序 `[3,9,20,15,7]`、中序 `[9,3,15,20,7]`，根 3 在中序位置 1，左子树只有 9，右子树有 3 个节点。

**易错点**：右子树前序起点要跨过整个左子树；用下标字典避免每层调用 `index()` 扫描。

时间平均 `O(n)`；字典与栈辅助空间 `O(n)`，另有新树 `O(n)`。

## 437. 路径总和 III

题目：[路径总和 III](https://leetcode.cn/problems/path-sum-iii/)。统计从任意节点开始、只能向下走、和为目标的路径数量。

### 题目描述

给定二叉树 root 和整数 targetSum，统计节点值之和等于 targetSum 的路径数量。路径可以从任意节点开始、在任意后代结束，但只能沿父到子的方向向下走。

**输入与约束**：树可为空，节点值和目标可以为负数或零；路径至少包含一个节点。路径不必经过根或叶子，不能从左子树经过父节点转到右子树。

```text
输入：root=[10,5,-3,3,2,None,11,3,-2,None,1], targetSum=8
输出：3
解释：路径分别为 5→3、5→2→1、-3→11。

输入：root=[], targetSum=0
输出：0，空路径不计入答案。
```

### 从题意到算法

沿当前根到节点的路径维护前缀和。当前和为 total，历史前缀 `total-targetSum` 的每次出现都对应一条以当前节点结尾的合法路径。进入节点时登记前缀、离开时撤销，保证哈希表只含当前祖先链，避免错误拼接两个兄弟分支。

**状态**：`count` 只保存当前根到节点路径上的历史前缀和次数。当前和为 total 时，历史 `total-targetSum` 的数量就是以当前节点结尾的答案数。

```python
from collections import defaultdict


class Solution:
    def pathSum(self, root, targetSum):
        count = defaultdict(int)
        count[0] = 1

        def dfs(node, total):
            if node is None:
                return 0
            total += node.val
            ans = count.get(total - targetSum, 0)
            count[total] += 1
            ans += dfs(node.left, total)
            ans += dfs(node.right, total)
            count[total] -= 1
            if count[total] == 0:
                del count[total]
            return ans

        return dfs(root, 0)
```

**推演**：沿路径 `10→5→3`，当前和 18，目标 8，需要历史前缀 10，得到路径 `5→3`。

**易错点**：回到父节点时必须撤销当前前缀，否则会把左、右兄弟分支错误拼成一条路径；这里只允许向下，不允许转弯穿过父节点。

时间平均 `O(n)`，辅助空间 `O(h)`。代码删除归零的键，查询用 `get`，避免为无关前缀保留越来越多的键。

## 236. 二叉树的最近公共祖先

题目：[二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)。原题保证 p、q 存在且不同。

### 题目描述

给定二叉树 root 和其中两个节点 p、q，返回它们的最近公共祖先：同时是两者祖先、且在树中深度最大的节点。一个节点也被视为自己的祖先。

**输入与约束**：节点值互异，p、q 是不同的真实节点且保证都在树中。输入是普通二叉树，不保证 BST；返回原树中的节点引用。

```text
输入：root=[3,5,1,6,2,0,8,None,None,7,4]，p 为节点 5，q 为节点 1
输出：节点 3

相同树，p 为节点 5，q 为节点 4
输出：节点 5，因为 5 自身也是 4 的祖先。
```

### 从题意到算法

递归在每棵子树中寻找 p、q：没找到返回 None，找到目标节点可直接返回它。若当前节点左右子树都返回非空，说明两个目标分居两边，当前就是最近交汇点；若只一侧非空，继续把该侧结果上传。题目保证两目标存在，使这种返回规则足够判断答案。

**函数含义**：若子树没有目标，返回 `None`；只找到一个，返回它；两个都找到，返回它们的最近公共祖先。

```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if root is None or root is p or root is q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None and right is not None:
            return root
        return left if left is not None else right
```

**推演**：p、q 分属根的两侧，左右都返回非空，根就是答案；若 p 本身是 q 的祖先，返回 p。

**易错点**：普通二叉树不能根据大小走某一边；使用节点身份比较。若改题为目标可能不存在，需要额外统计是否都找到。

时间 `O(n)`，辅助空间 `O(h)`。

## 124. 二叉树中的最大路径和

题目：[二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)。路径可以从任意节点到任意节点，但不能重复节点，路径非空。

### 题目描述

给定非空二叉树 root，路径由一串沿父子边相邻的节点组成，每个节点最多出现一次，至少包含一个节点。返回所有路径中节点值之和的最大值。

**输入与约束**：节点值可以为负数；路径不必经过根，可以从左子树经过某个祖先再走向右子树。与路径总和 III 不同，本题允许这样转弯，但不能分叉成三条支路。

```text
输入：root=[-10,9,20,None,None,15,7]
输出：42
解释：路径 15→20→7 的和为 42。

输入：root=[-3]
输出：-3，必须选择非空路径。
```

### 从题意到算法

每个节点收集左右子树能向上提供的最大单边贡献，负贡献可以舍弃。把“左贡献+当前值+右贡献”作为全局候选；返回父节点时只能带上左右中较大的一边，否则再接父节点会产生非法分叉。全局答案初始化为负无穷，才能正确处理全负树。

**函数定义**：`gain(node)` 返回“从 node 开始向下走一条支路”的最大贡献。向父节点返回时只能选左或右一边；更新整题答案时可以把左右两边经过当前节点连接起来。

```python
class Solution:
    def maxPathSum(self, root):
        ans = float("-inf")

        def gain(node):
            nonlocal ans
            if node is None:
                return 0
            left = max(0, gain(node.left))
            right = max(0, gain(node.right))
            ans = max(ans, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return ans
```

**推演**：`[-10,9,20,None,None,15,7]`，在 20 处连接左右，得到 `15+20+7=42`；向 -10 返回时只能返回 `20+15=35`。

**易错点**：负贡献可以不接，所以截断为 0；全局答案不能初始化为 0，否则全负树会出错；返回左右之和给父节点会产生分叉，不再是一条路径。原题根非空。

时间 `O(n)`，辅助空间 `O(h)`。

## 快速入门路线

先掌握 94、102 两种遍历；再用 104、226、101 训练“递归函数到底返回什么”；之后学 BST 的 108、98、230；最后做 543、437、236、124。

最关键的对比：**直径/最大路径和的全局候选可以连接左右两边，但返回父节点的信息只能沿一边延伸。**
