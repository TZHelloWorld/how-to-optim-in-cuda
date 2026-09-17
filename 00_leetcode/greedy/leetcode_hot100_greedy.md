# Hot100 贪心：局部选择必须有理由

[返回总索引](../README.md)

## 基础：贪心与动态规划的差别

动态规划通常保留多个子问题的最佳结果；贪心在当前阶段选择一种决策并继续，不回头枚举全部方案。

贪心正确性不能依赖“看起来最好”。常见证明方式：

- **支配关系**：状态 A 的未来选择不比 B 少，那么 B 可以丢弃。
- **交换论证**：把某个最优解的第一步换成贪心选择，答案不会变差。
- **区间覆盖**：维护当前步数能达到的完整范围，每次最大化下一轮覆盖。

本专题每道题都说明为什么当前选择不会损失全局答案。

## 121. 买卖股票的最佳时机

题目：[买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)。最多买卖一次，买入必须在卖出之前。

**状态**：`lowest` 是此前最低买入价。固定今天卖出时，买得越便宜收益越高，因此只需保留此前最小值。

```python
class Solution:
    def maxProfit(self, prices):
        lowest = float("inf")
        ans = 0
        for price in prices:
            ans = max(ans, price - lowest)
            lowest = min(lowest, price)
        return ans
```

**推演**：`[7,1,5,3,6,4]`，买入价更新到 1，在 6 卖出收益 5。

**易错点**：不能把全局最高和最低直接相减，最低可能出现在最高之后；下降数组允许不交易，答案为 0。

时间 `O(n)`，辅助空间 `O(1)`。

## 55. 跳跃游戏

题目：[跳跃游戏](https://leetcode.cn/problems/jump-game/)。`nums[i]` 是从 i 最多能跳多远，判断能否到最后一个位置。

**状态与理由**：`farthest` 是从已处理且可达位置出发能覆盖的最远下标。只要 `i <= farthest`，i 就可以到达，从 i 能进一步扩展范围。

```python
class Solution:
    def canJump(self, nums):
        farthest = 0
        for i, jump in enumerate(nums):
            if i > farthest:
                return False
            farthest = max(farthest, i + jump)
            if farthest >= len(nums) - 1:
                return True
        return False
```

**推演**：`[2,3,1,1,4]`，位置 0 覆盖到 2，位置 1 可以把覆盖扩展到 4，于是成功；`[3,2,1,0,4]` 最远停在 3，无法到 4。

**易错点**：不要先使用不可达位置的跳跃能力再判断能否到它；每次是“最多”跳多少，不是必须跳固定距离。原题数组非空。

时间 `O(n)`，辅助空间 `O(1)`。

## 45. 跳跃游戏 II

题目：[跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)。求到末尾的最少跳数，原题保证可达。

### 状态与思路

- `end`：当前跳数可以覆盖的最右边界。
- `farthest`：扫描当前覆盖范围时，下一跳能覆盖的最远位置。
- 扫描到 end，必须使用新的一跳，于是更新边界。

相当于把隐式图 BFS 的每一层压缩成一个连续下标区间。

```python
class Solution:
    def jump(self, nums):
        steps = 0
        end = farthest = 0
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            if i == end:
                steps += 1
                end = farthest
        return steps
```

**推演**：`[2,3,1,1,4]`，第一跳覆盖 `[1,2]`；扫描这个范围，位置 1 能把下一轮扩展到 4，第二跳到达终点。

**易错点**：不用真的指定每次跳到哪个点，扫描全部当前可达点更稳妥；循环不处理最后一格，否则可能已经到终点却又多计一次跳跃。

时间 `O(n)`，辅助空间 `O(1)`。

## 763. 划分字母区间

题目：[划分字母区间](https://leetcode.cn/problems/partition-labels/)。将字符串切成尽可能多段，同一个字母不能出现在不同段中。

**状态与理由**：记录每个字母最后出现的位置。当前段出现过的所有字母，都必须在这一段中结束，所以段末至少要达到这些最后位置的最大值 end。

```python
class Solution:
    def partitionLabels(self, s):
        last = {ch: i for i, ch in enumerate(s)}
        start = end = 0
        ans = []
        for i, ch in enumerate(s):
            end = max(end, last[ch])
            if i == end:
                ans.append(end - start + 1)
                start = i + 1
        return ans
```

**推演**：`"ababcbacadefegdehijhklij"`，从 a 开始必须延伸到最后一个 a 的位置 8，期间出现的 b、c 都在此之前结束，所以第一段长度 9，最终 `[9,7,8]`。

**正确性**：到达 end 前不能切，否则某字母跨段；恰好到 end 时立即切是最早合法切法，能留下最多后续划分机会。

**易错点**：end 要取最大值，不能简单改成当前字符最后下标，否则可能缩回去。

时间平均 `O(n)`，辅助空间 `O(Σ)`，另计输出；小写字母表下 Σ 至多 26。

## 小结

“维护当前最优信息”比“每次取最大的元素”更接近贪心本质。能解释为何其他历史状态被当前状态支配，才能放心丢弃它们。
