# Hot100 动态规划：先说清状态，再写转移

[返回总索引](../README.md)

## 基础一：动态规划到底在做什么？

回溯会枚举选择路径。如果多条路径走到同一个“剩余问题”，每次重新计算就会重复。动态规划保存已经求出的子问题结果，用它们推导更大的问题。

例如爬到第 n 阶的方案数，需要第 n-1 阶和第 n-2 阶的方案数，而它们又会重复需要更早的阶数。把每一阶的结果记录下来，就不必反复递归。

- **记忆化搜索**：自顶向下递归，计算过的状态直接返回缓存。
- **递推 DP**：自底向上，按依赖顺序填写状态表。
- 两者常常是同一组状态和转移的两种执行方式。

不是所有递归都需要 DP。只有存在可复用的重复状态，缓存才有价值。若需要输出全部组合，保存一个数值也不能代替枚举全部答案。

## 基础二：写 DP 的五个问题

1. **状态是什么？** 用完整中文定义 `dp[i]`。它是“前 i 个”，还是“以 i 结尾”？
2. **最后一步是什么？** 分类讨论最后一个选择，从较小状态推导。
3. **初始状态是什么？** 空前缀、零金额、第一项分别如何定义？
4. **按什么顺序计算？** 使用某个状态前，它必须已经算完。
5. **答案在哪里？** 可能是 `dp[n]`，也可能是 `max(dp)`。

不能先背转移公式、再猜状态。相同的 `dp[i]` 写法，在不同题里含义完全不同。

## 基础三：从回溯走向 DP

组合总和回溯状态包含“当前候选下标、剩余金额”，因为要区分完整组合。零钱兑换只求最少硬币数，可以把状态简化为“凑出金额 a 的最少硬币数”。

核心检查：**状态是否包含决定未来结果所需的全部信息？** 例如乘积最大子数组只保存最大乘积不够，负数可能把最小乘积翻成最大，所以要保留两种状态。

复杂度通常为“状态数量 × 每个状态的转移代价”。滚动数组只减少存储，不改变原来的状态含义。

## 70. 爬楼梯

题目：[爬楼梯](https://leetcode.cn/problems/climbing-stairs/)。每次走 1 或 2 阶，求到第 n 阶的不同走法。

### 状态、转移与初始化

- `dp[i]`：恰好走到第 i 阶的方法数。
- 最后一步来自 i-1 或 i-2，两类互不重复：`dp[i]=dp[i-1]+dp[i-2]`。
- `dp[0]=1` 表示不走也是一种空方案，`dp[1]=1`。
- 只依赖前两项，所以用两个变量滚动保存。

```python
class Solution:
    def climbStairs(self, n):
        prev, cur = 1, 1
        for _ in range(2, n + 1):
            prev, cur = cur, prev + cur
        return cur
```

**推演**：n=4 时，方案数依次为 `1、1、2、3、5`，答案 5。

**易错点**：这是有顺序的走法，`1+2` 和 `2+1` 不同；同时赋值会先计算右侧，不会被更新后的 prev 干扰。

时间 `O(n)`，辅助空间 `O(1)`（按常用整数运算模型）。

## 118. 杨辉三角

题目：[杨辉三角](https://leetcode.cn/problems/pascals-triangle/)。生成前 numRows 行。

### 状态与转移

`rows[r][c]` 是第 r 行第 c 列。两边为 1，中间来自上一行相邻两个数之和。

```python
class Solution:
    def generate(self, numRows):
        rows = []
        for r in range(numRows):
            row = [1] * (r + 1)
            for c in range(1, r):
                row[c] = rows[r - 1][c - 1] + rows[r - 1][c]
            rows.append(row)
        return rows
```

**推演**：上一行 `[1,2,1]`，新行两端为 1，中间是 `1+2`、`2+1`，得到 `[1,3,3,1]`。

**易错点**：每一行长度不同；每轮新建一个列表，不能反复追加同一个可变列表引用。

时间 `O(r²)`，总输出空间 `O(r²)`；除输出外辅助空间 `O(1)`，构造中的行最终也是输出的一部分。

## 198. 打家劫舍

题目：[打家劫舍](https://leetcode.cn/problems/house-robber/)。不能选择相邻房屋，求最大金额。

### 状态与转移

令 `dp[i]` 是前 i 间房的最大收益。面对第 i 间房：

- 不偷它：`dp[i-1]`。
- 偷它：前一间不能偷，收益 `dp[i-2]+nums[i-1]`。

两者取最大。代码的 `prev2、prev1` 是处理当前房屋前的这两个历史状态。

```python
class Solution:
    def rob(self, nums):
        prev2 = prev1 = 0
        for money in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + money)
        return prev1
```

**推演**：`[2,7,9,3,1]` 的前缀最优为 `2、7、11、11、12`，选择 2、9、1。

**易错点**：不是简单挑奇数位或偶数位；需要在每一步比较偷与不偷。

时间 `O(n)`，辅助空间 `O(1)`。

## 279. 完全平方数

题目：[完全平方数](https://leetcode.cn/problems/perfect-squares/)。用最少个完全平方数凑出 n，同一个平方数可重复使用。

### 状态与转移

`dp[x]` 表示凑出 x 的最少数量。最后选 `square`，前面就需要 `dp[x-square]`，因此取最小的 `dp[x-square]+1`。

```python
from math import isqrt


class Solution:
    def numSquares(self, n):
        squares = [i * i for i in range(1, isqrt(n) + 1)]
        dp = [0] + [n + 1] * n
        for total in range(1, n + 1):
            for square in squares:
                if square > total:
                    break
                dp[total] = min(dp[total], dp[total - square] + 1)
        return dp[n]
```

**推演**：n=12，`dp[4]=1`，`dp[8]=2`，`dp[12]=3`，对应 `4+4+4`。

**易错点**：不能总选不超过剩余值的最大平方数。12 贪心会选 `9+1+1+1`，需要 4 个，而最优只需 3 个。

时间 `O(n√n)`，辅助空间 `O(n)`。

## 322. 零钱兑换

题目：[零钱兑换](https://leetcode.cn/problems/coin-change/)。每种正面额可无限使用，求凑出 amount 的最少硬币数，不可能则 -1。

### 状态、边界与转移

- `dp[a]`：恰好组成金额 a 的最少硬币数。
- `dp[0]=0`，不需要硬币。
- 其余初始化为大于任何可行数量的值 `amount+1`。
- 枚举最后一枚硬币 coin：`dp[a]=min(dp[a],dp[a-coin]+1)`。

```python
class Solution:
    def coinChange(self, coins, amount):
        inf = amount + 1
        dp = [0] + [inf] * amount
        for total in range(1, amount + 1):
            for coin in coins:
                if coin <= total:
                    dp[total] = min(dp[total], dp[total - coin] + 1)
        return dp[amount] if dp[amount] != inf else -1
```

**推演**：`coins=[1,2,5]`，amount=11，最后选 5 时考察 `dp[6]+1`，最终得到 `5+5+1`，答案 3。

**易错点**：`dp[0]` 是 0，不是 1；不可达状态不能初始化为 0；这里求数量最少，不是统计组合总数，不能把 min 改成累加。

时间 `O(amount × m)`，m 为面额数；辅助空间 `O(amount)`。

## 139. 单词拆分

题目：[单词拆分](https://leetcode.cn/problems/word-break/)。判断字符串能否被字典单词完全拼接，单词可重复使用。

### 状态与转移

`dp[end]` 表示前 end 个字符 `s[:end]` 能否拆分。如果某个字典单词 word 恰好是这段前缀的结尾，而且之前的前缀可拆分，就得到 True。

```python
class Solution:
    def wordBreak(self, s, wordDict):
        words = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for end in range(1, len(s) + 1):
            for word in words:
                start = end - len(word)
                if start >= 0 and dp[start] and s.startswith(word, start, end):
                    dp[end] = True
                    break
        return dp[-1]
```

**推演**：`"leetcode"`、字典 `["leet","code"]`，`dp[4]=True`，后面匹配 `code` 得到 `dp[8]=True`。

**易错点**：`dp[0]=True` 是第一次匹配的基础；不能找到一个单词就不考虑其他切法，例如较长的首次匹配可能导致后面失败。`break` 只表示当前 end 已找到一种可行拆法。

设字符串长度 n，字典去重后 m 个单词、最大长度 L，时间上界 `O(nmL)`，包括字符串比较；辅助状态 `O(n+m)`，字典字符串本身复用输入。

## 300. 最长递增子序列

题目：[最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)。求严格递增子序列长度，可以跳过元素。

### 状态与转移

`dp[i]` 是**必须以 nums[i] 结尾**的最长递增子序列长度。枚举前一个位置 j：如果 `j<i` 且 `nums[j]<nums[i]`，可以接在它后面。

```python
class Solution:
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

**推演**：`[10,9,2,5,3,7,101,18]`，可以选 `2,3,7,18`，长度 4。

**易错点**：答案是 `max(dp)`，不一定是最后一项；严格递增使用 `<`；不能只与相邻元素比较，因为子序列允许跳跃。

时间 `O(n²)`，辅助空间 `O(n)`。进阶可维护“长度为 L 的递增子序列的最小结尾”，结合二分优化到 `O(n log n)`；先理解这里的状态，再学优化。

## 152. 乘积最大子数组

题目：[乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)。求非空连续子数组的最大乘积。

### 为什么要两个状态？

负数乘以最小负积可能成为最大正积。因此同时维护以当前位置结尾的 `high` 最大积和 `low` 最小积。

新元素 x 有三种来源：自己重新开始、旧 high 乘 x、旧 low 乘 x。

```python
class Solution:
    def maxProduct(self, nums):
        high = low = ans = nums[0]
        for i in range(1, len(nums)):
            x = nums[i]
            old_high, old_low = high, low
            high = max(x, old_high * x, old_low * x)
            low = min(x, old_high * x, old_low * x)
            ans = max(ans, high)
        return ans
```

**推演**：`[-2,3,-4]`：在 3 处 high=3、low=-6；到 -4，旧 low 乘 -4 得到 24。

**易错点**：更新 low 时必须使用旧 high；零会自然切断之前的乘积链；不能照搬最大子数组和只保留一个最大状态。

时间 `O(n)`，辅助空间 `O(1)`。

## 416. 分割等和子集

题目：[分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)。正整数数组能否分成两个和相等的子集，每个元素只能用一次。

### 转成 0/1 背包

总和为奇数一定失败。否则只需判断能否选出和为 `total/2` 的子集，剩余元素自然是另一半。

`dp[s]` 表示已经处理的数字中，是否能选出和 s。考虑 x 时，更新 `dp[s] = dp[s] or dp[s-x]`。

```python
class Solution:
    def canPartition(self, nums):
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for x in nums:
            for value in range(target, x - 1, -1):
                dp[value] = dp[value] or dp[value - x]
        return dp[target]
```

### 为什么必须倒序？

倒序保证读取的 `dp[value-x]` 还是“没使用当前 x”之前的状态。如果正序，刚用 x 更新出来的状态会再次使用 x，变成无限重复选择。

**推演**：`[1,5,11,5]` 总和 22，只需凑出 11，选择单个 11 或 `1+5+5` 都成立。

**易错点**：这里每个位置只能用一次，不能套用零钱兑换的无限使用规则。

时间 `O(n × target)`，辅助空间 `O(target)`。

## 32. 最长有效括号

题目：[最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)。求最长连续合法括号子串长度。

### 状态与转移

`dp[i]` 表示**恰好以 i 结尾**的最长有效括号长度。只有 `s[i]=')'` 时可能非零。

1. 前一位是 `(`：形成 `()`，再接上 `dp[i-2]`。
2. 前一位是 `)`：先跨过前一段合法括号，找到 `j=i-dp[i-1]-1`。如果 j 是 `(`，它可以与当前 `)` 配对，再接上 j 前面的有效段。

```python
class Solution:
    def longestValidParentheses(self, s):
        dp = [0] * len(s)
        ans = 0
        for i in range(1, len(s)):
            if s[i] != ")":
                continue
            if s[i - 1] == "(":
                dp[i] = 2 + (dp[i - 2] if i >= 2 else 0)
            else:
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == "(":
                    dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
            ans = max(ans, dp[i])
        return ans
```

### 推演

`"()(())"`：在最后一位，前一段 `()` 长度为 2，跨过它后找到外层 `(`；形成 `(())` 长度 4，再接上开头 `()`，总长 6。

**易错点**：Python 负下标不会自动报越界，而是访问末尾，所以 `i-2、j、j-1` 的边界要显式判断；求的是连续子串，不能把隔开的合法部分随意相加。

时间 `O(n)`，辅助空间 `O(n)`。

## 状态对比：最值得记住的差别

| 题目 | 状态类型 | 答案 |
|---|---|---|
| 爬楼梯、平方数、零钱兑换 | 恰好到达某个阶数/金额 | 目标状态 |
| 打家劫舍、单词拆分 | 前缀整体的最优/可行性 | 最后一个前缀 |
| LIS、乘积最大子数组、有效括号 | 必须以某个位置结尾 | 所有结尾状态的最优 |
| 等和子集 | 处理前若干元素后的可达和 | target 是否可达 |

建议练习时先用数组保存完整状态并手算，再做滚动优化。二维问题继续阅读[多维动态规划](../multidimensional_dp/leetcode_hot100_multidimensional_dp.md)。
