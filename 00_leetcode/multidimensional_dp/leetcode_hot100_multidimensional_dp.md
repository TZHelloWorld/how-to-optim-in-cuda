# Hot100 多维动态规划：网格、区间与两个字符串

[返回总索引](../README.md)

## 基础：二维状态并不一定代表二维坐标

- 网格 DP：`dp[r][c]` 表示到达某个格子的结果。
- 区间 DP：`dp[left][right]` 表示一个区间的性质。
- 双字符串 DP：`dp[i][j]` 表示两个字符串前缀之间的关系。

先确认每一维的含义，再根据依赖决定遍历顺序。例如区间回文依赖内部 `[left+1,right-1]`，不能随意从左向右填写。

二维列表要写 `[[0] * cols for _ in range(rows)]`，不要让多行共享同一对象。这里优先保留完整 DP 表，便于观察转移；部分题可以之后压缩成一行。

## 62. 不同路径

题目：[不同路径](https://leetcode.cn/problems/unique-paths/)。机器人从左上到右下，每次只能向右或向下，求路径数。

### 状态与转移

`dp[r][c]` 是到达格子 `(r,c)` 的方案数。最后一步来自上方或左方，两类路径互不重叠，所以相加。

第一行只能一直向右，第一列只能一直向下，方案数均为 1。

```python
class Solution:
    def uniquePaths(self, m, n):
        dp = [[1] * n for _ in range(m)]
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        return dp[m - 1][n - 1]
```

**推演**：`m=3,n=3` 的表为：

```text
1 1 1
1 2 3
1 3 6
```

**易错点**：这里没有障碍；每条路径按最后一步唯一分类，不需要额外去重。

时间、辅助空间均为 `O(mn)`。压缩为一行后可降为 `O(n)` 空间。

## 64. 最小路径和

题目：[最小路径和](https://leetcode.cn/problems/minimum-path-sum/)。只向右或向下走，使路径上的数字和最小。

### 状态与转移

`dp[r][c]` 是到达当前格子的最小总和，包含当前格子。一般位置取上方和左方较小者，再加当前值。边缘只有一种来源。

```python
class Solution:
    def minPathSum(self, grid):
        rows, cols = len(grid), len(grid[0])
        dp = [[0] * cols for _ in range(rows)]
        dp[0][0] = grid[0][0]
        for r in range(1, rows):
            dp[r][0] = dp[r - 1][0] + grid[r][0]
        for c in range(1, cols):
            dp[0][c] = dp[0][c - 1] + grid[0][c]
        for r in range(1, rows):
            for c in range(1, cols):
                dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
        return dp[-1][-1]
```

**推演**：`[[1,3,1],[1,5,1],[4,2,1]]` 最优路径沿上边再向下，和为 7。

**易错点**：局部选相邻较小格子不一定全局最优；第一行第一列要累计，不是简单复制输入。实现保留输入不变。

时间、辅助空间均为 `O(RC)`；可以滚动优化到 `O(C)`。

## 5. 最长回文子串

题目：[最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)。返回最长的连续回文子串，多个答案可返回任意一个。

### 状态与转移

`dp[left][right]` 表示闭区间 `s[left:right+1]` 是否回文。

成立条件：两端字符相同，且长度不超过 2，或者内部区间也是回文。

遍历时 left 从大到小，确保使用 `dp[left+1][right-1]` 前已经算过。

```python
class Solution:
    def longestPalindrome(self, s):
        if not s:
            return ""
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        start, best = 0, 1
        for left in range(n - 1, -1, -1):
            for right in range(left, n):
                if s[left] == s[right] and (right - left <= 1 or dp[left + 1][right - 1]):
                    dp[left][right] = True
                    length = right - left + 1
                    if length > best:
                        start, best = left, length
        return s[start:start + best]
```

**推演**：`"babad"` 中，单字符先成立，内部 `a` 使 `bab` 成立，内部 `b` 使 `aba` 成立，返回其中一个长度 3 的结果即可。

**易错点**：子串必须连续；如果只比较首尾、不验证内部，`abca` 会误判；保存下标和长度，最后切片一次。

时间 `O(n²)`，辅助空间 `O(n²)`。中心扩展可保持 `O(n²)` 时间、降低为 `O(1)` 辅助空间；这里展示区间 DP 以理解依赖。

## 1143. 最长公共子序列

题目：[最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)。求两个字符串中共同出现、顺序一致、可以不连续的最长序列长度。

### 状态与转移

`dp[i][j]` 表示 `text1[:i]` 和 `text2[:j]` 的 LCS 长度，因此需要额外一行一列表示空前缀。

- 当前末字符相同：`dp[i-1][j-1]+1`。
- 不同：至少舍弃某一边的末字符，取 `max(dp[i-1][j],dp[i][j-1])`。

```python
class Solution:
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

**推演**：`"abcde"` 与 `"ace"`，依次匹配 a、c、e，答案 3，中间 b、d 可以跳过。

**易错点**：末字符不同不能直接置零，那是“最长公共连续子串”类型的转移；前缀长度 i 对应字符下标 i-1。

时间、辅助空间均为 `O(mn)`；可用滚动数组减少空间。

## 72. 编辑距离

题目：[编辑距离](https://leetcode.cn/problems/edit-distance/)。用插入、删除、替换，把 word1 变成 word2，求最少操作数。

### 状态与转移

`dp[i][j]` 表示将 `word1[:i]` 变成 `word2[:j]` 的最少操作数。

- 末字符相同，无需新增操作：`dp[i-1][j-1]`。
- 删除 word1 末字符：`dp[i-1][j]+1`。
- 插入 word2 末字符：`dp[i][j-1]+1`。
- 替换末字符：`dp[i-1][j-1]+1`。

空串变成长 j 的字符串，需要 j 次插入；长 i 的字符串变成空串，需要 i 次删除。

```python
class Solution:
    def minDistance(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
```

**推演**：`horse → rorse → rose → ros`，一次替换、两次删除，共 3 次。

**易错点**：每个转移都要根据“前缀变换”的含义理解，不能只记左、上、左上取最小；空字符串输入依靠首行首列自然处理。

时间、辅助空间均为 `O(mn)`。

## 总结：遍历顺序由依赖决定

| 状态 | 依赖 | 合理顺序 |
|---|---|---|
| 网格坐标 | 上方、左方 | 从上到下、从左到右 |
| 回文区间 | 左端更大、右端更小的内部区间 | 左端倒序，或按长度递增 |
| 两个前缀 | 上、左、左上 | 两维前缀长度递增 |

调试时用短字符串画出完整 DP 表。若出现“引用的位置还没算”，先修正遍历顺序，不要用默认零值掩盖依赖错误。
