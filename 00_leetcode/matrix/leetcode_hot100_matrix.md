# Hot100 矩阵：坐标、边界与原地标记

[返回总索引](../README.md)

## 基础

- `matrix[r][c]`：先行后列，行号向下增加，列号向右增加。
- `rows = len(matrix)`，`cols = len(matrix[0])`，二者不要混淆。
- 建立独立二维列表用 `[[0] * cols for _ in range(rows)]`；`[[0] * cols] * rows` 会让多行引用同一个列表。
- 顺时针旋转 90 度，旧位置 `(r,c)` 对应新位置 `(c,n-1-r)`。
- 矩阵是否有序、是否正方形，决定可用哪些性质。

## 73. 矩阵置零

题目：[矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)。原矩阵中的零，会使其整行整列变为零，要求原地处理。

### 状态与思路

用第一列记录某一行是否需要清零，用第一行记录某一列是否需要清零。但第一行和第一列本来是否有零要单独保存，避免标记互相干扰。

```python
class Solution:
    def setZeroes(self, matrix):
        if not matrix or not matrix[0]:
            return
        rows, cols = len(matrix), len(matrix[0])
        first_row = any(x == 0 for x in matrix[0])
        first_col = any(matrix[r][0] == 0 for r in range(rows))

        for r in range(1, rows):
            for c in range(1, cols):
                if matrix[r][c] == 0:
                    matrix[r][0] = matrix[0][c] = 0
        for r in range(1, rows):
            for c in range(1, cols):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0
        if first_row:
            for c in range(cols):
                matrix[0][c] = 0
        if first_col:
            for r in range(rows):
                matrix[r][0] = 0
```

`[[1,1,1],[1,0,1],[1,1,1]]`：用 `(1,0)` 和 `(0,1)` 标记，随后清掉中间行列，最后保持原先不需要全清的首行首列。

**易错点**：发现零就立刻清整行整列，会把新产生的零当成原始零继续传播；先标记、再清内部、最后处理首行首列。

时间 `O(RC)`，辅助空间 `O(1)`。

## 54. 螺旋矩阵

题目：[螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)。按顺时针螺旋顺序返回元素。

### 状态与思路

`top、bottom、left、right` 是尚未访问区域的边界。依次走上、右、下、左边，每走完一边就收缩对应边界。

```python
class Solution:
    def spiralOrder(self, matrix):
        if not matrix or not matrix[0]:
            return []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        ans = []
        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                ans.append(matrix[top][c])
            top += 1
            for r in range(top, bottom + 1):
                ans.append(matrix[r][right])
            right -= 1
            if top <= bottom:
                for c in range(right, left - 1, -1):
                    ans.append(matrix[bottom][c])
                bottom -= 1
            if left <= right:
                for r in range(bottom, top - 1, -1):
                    ans.append(matrix[r][left])
                left += 1
        return ans
```

`[[1,2,3],[4,5,6],[7,8,9]]`：第一圈 `1,2,3,6,9,8,7,4`，剩余中心 `5`。

**易错点**：走下边和左边前重新检查边界，避免只剩一行或一列时重复访问。

时间 `O(RC)`；辅助空间 `O(1)`，输出 `O(RC)`。

## 48. 旋转图像

题目：[旋转图像](https://leetcode.cn/problems/rotate-image/)。原地将正方形矩阵顺时针旋转 90 度。

### 思路

先转置，再反转每一行：`(r,c) → (c,r) → (c,n-1-r)`，恰好等于顺时针旋转。

```python
class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        for r in range(n):
            for c in range(r + 1, n):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        for row in matrix:
            row.reverse()
```

`[[1,2],[3,4]]` 转置为 `[[1,3],[2,4]]`，每行反转得到 `[[3,1],[4,2]]`。

**易错点**：转置只交换对角线一侧，如果把 `(r,c)` 和 `(c,r)` 都遍历，就会交换两次恢复原样；本题矩阵是正方形。

时间 `O(n²)`，辅助空间 `O(1)`。

## 240. 搜索二维矩阵 II

题目：[搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)。每行、每列分别升序，判断目标是否存在。

### 状态与思路

从右上角出发：当前值太大，当前列从这里往下都更大，删除这一列；当前值太小，当前行向左都更小，删除这一行。

```python
class Solution:
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        r, c = 0, cols - 1
        while r < rows and c >= 0:
            x = matrix[r][c]
            if x == target:
                return True
            if x > target:
                c -= 1
            else:
                r += 1
        return False
```

`[[1,4,7],[2,5,8],[3,6,9]]` 查找 5：7 太大向左到 4，4 太小向下到 5。

**易错点**：不能直接展平后二分，因为下一行第一个数未必大于上一行最后一个数；那是第 74 题的额外条件。

时间 `O(R+C)`，辅助空间 `O(1)`。

## 小结

画一个 `2×3` 的矩阵检查行列含义，再用单行、单列、`1×1` 检查边界。矩阵题的主要错误往往是重复访问、访问越界或过早覆盖标记。
