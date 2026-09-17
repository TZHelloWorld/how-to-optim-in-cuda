# Hot100 子串与连续区间：前缀和、单调队列、覆盖窗口

[返回总索引](../README.md)

## 基础：同为连续区间，解法不一定相同

- **前缀和**：把一段区间的和转成两个历史状态之差。
- **滑动窗口**：维护连续区间内的频次、长度或合法性。
- **单调队列**：删除永远不可能成为最优值的窗口候选。

子串、子数组必须连续；子序列可以跳过元素。不要只看到“子串”二字就固定使用某个模板。

## 560. 和为 K 的子数组

题目：[和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)。统计和恰好为 `k` 的连续非空子数组数量，数组可能包含负数。

### 状态与推导

设 `prefix[j]` 是前 `j` 个元素之和，则区间 `[i,j)` 的和为 `prefix[j]-prefix[i]`。

当当前前缀和为 `total`，需要查找历史中出现过多少个 `total-k`。`count` 保存历史前缀和的出现次数。

```python
from collections import defaultdict


class Solution:
    def subarraySum(self, nums, k):
        count = defaultdict(int)
        count[0] = 1
        total = ans = 0
        for x in nums:
            total += x
            ans += count[total - k]
            count[total] += 1
        return ans
```

### 推演、易错点与复杂度

`[1,1,1]`、`k=2`：前缀和依次是 `1、2、3`。到 `2` 时找到一次历史前缀 `0`；到 `3` 时找到一次历史前缀 `1`，总数 2。

`count[0]=1` 代表空前缀，才能统计从数组开头开始的答案。先查询再记录当前前缀，避免 `k=0` 时把空区间算进去。

有负数时，扩大窗口可能让和减小，缩小可能让和增大，所以不能简单用“和过大就收缩”。

时间平均 `O(n)`，辅助空间 `O(n)`。

## 239. 滑动窗口最大值

题目：[滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)。返回每个长度为 `k` 的窗口的最大值。

### 状态与思路

双端队列保存下标，满足：

1. 下标从前到后递增，方便淘汰过期元素。
2. 对应数值从前到后严格递减，队首就是最大值。

新元素如果不小于队尾，旧队尾可以删除：新元素更大或相等，而且更晚过期，旧元素以后不可能更有用。

```python
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums, k):
        queue = deque()
        ans = []
        for i, x in enumerate(nums):
            while queue and queue[0] <= i - k:
                queue.popleft()
            while queue and nums[queue[-1]] <= x:
                queue.pop()
            queue.append(i)
            if i >= k - 1:
                ans.append(nums[queue[0]])
        return ans
```

### 推演

`[1,3,-1,-3,5]`，`k=3`：第一窗口最大值为 3；第二窗口仍为 3；5 进入时淘汰队尾所有较小值，第三窗口最大值为 5。

**易错点**：保存数值而不保存下标，会难以判断重复值何时过期；Python 列表 `pop(0)` 是线性操作，队列应使用 `deque`。

每个下标最多入队一次、出队一次，时间 `O(n)`；辅助空间 `O(k)`，另计输出。

## 76. 最小覆盖子串

题目：[最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)。找到包含 `t` 全部字符及其次数的最短 `s` 子串，没有则返回空字符串。

### 状态与思路

`need[ch]` 表示还缺几个字符，允许为负数，负数表示多出来的数量。`missing` 表示一共还缺多少个字符，不是还缺多少种。

右端扩张直到 `missing == 0`，然后不断收缩左端，寻找更短的合法窗口。

```python
from collections import Counter


class Solution:
    def minWindow(self, s, t):
        if not t:
            return ""
        need = Counter(t)
        missing = len(t)
        left = 0
        best_start = 0
        best_len = len(s) + 1

        for right, ch in enumerate(s):
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1

            while missing == 0:
                length = right - left + 1
                if length < best_len:
                    best_start, best_len = left, length
                old = s[left]
                need[old] += 1
                if need[old] > 0:
                    missing += 1
                left += 1

        if best_len > len(s):
            return ""
        return s[best_start:best_start + best_len]
```

### 推演与正确性

`s="ADOBECODEBANC"`、`t="ABC"`：第一次覆盖得到 `ADOBEC`；继续移动右端并收缩，最终得到 `BANC`。

对每个右端点，只要覆盖成立就继续删除左侧冗余字符，因此不会遗漏更短候选。某个左端点离开前已经考虑了当时的最短机会，以后右端再增加，不会让同一左端点得到更短结果。

**易错点**：`t="AABC"` 需要两个 A，不能只使用集合；先记录合法答案再删除左侧字符；最好只保存起点和长度，最后切片一次，避免搜索过程中反复复制长字符串。

**复杂度**：时间平均 `O(len(s)+len(t))`；辅助空间 `O(Σ)`，`Σ` 为输入出现的字符种类数，另计返回字符串。

## 小结

| 需求 | 关键状态 |
|---|---|
| 区间和等于目标的数量 | 历史前缀和频次 |
| 固定窗口最大值 | 尚未过期、没有被更优值淘汰的下标 |
| 最短覆盖 | 每个字符的缺口、总缺口 |
