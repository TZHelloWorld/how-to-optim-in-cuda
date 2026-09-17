# Hot100 二叉树：递归含义、遍历顺序与子树信息

[返回总索引](../README.md)

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
