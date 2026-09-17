# Hot100 普通数组：区间、前后缀与原地操作

[返回总索引](../README.md)

## 基础

Python 列表可以按下标平均常数时间访问；尾部 `append/pop` 摊还常数时间，中间插入或删除通常需要移动元素。

- 排序可以把杂乱问题变成相邻关系问题。
- 前缀、后缀可以把“排除自己”的计算拆成左右两部分。
- 原地算法需要特别注意赋值顺序和被覆盖的信息。
- `nums[:]`、`nums[::-1]` 都会创建新列表；`nums.sort()` 修改原列表。

## 53. 最大子数组和

题目：[最大子数组和](https://leetcode.cn/problems/maximum-subarray/)。返回非空连续子数组的最大和。

### 状态与转移

`ending` 表示“必须以当前元素结尾”的最大子数组和。对新元素 `x`，只有两种选择：单独从 `x` 开始，或接上前面的最佳结尾。

```python
class Solution:
    def maxSubArray(self, nums):
        ending = ans = nums[0]
        for i in range(1, len(nums)):
            x = nums[i]
            ending = max(x, ending + x)
            ans = max(ans, ending)
        return ans
```

`[-2,1,-3,4,-1,2,1,-5,4]` 中，走到 `4` 时重新开始，接上 `-1,2,1` 得到 6。

**易错点**：不能把答案初始化为 0，全负数时答案应该是最大的负数；“以当前位置结尾的最优”和“全局最优”不是同一个状态。

时间 `O(n)`，辅助空间 `O(1)`。这也是动态规划的滚动状态写法。

## 56. 合并区间

题目：[合并区间](https://leetcode.cn/problems/merge-intervals/)。合并所有重叠的闭区间。

### 状态与思路

按左端点排序。`ans` 保存已经合并好的区间；新区间只需要和最后一个比较，因为更早的区间已经与后面分离。

```python
class Solution:
    def merge(self, intervals):
        intervals.sort(key=lambda pair: pair[0])
        ans = []
        for start, end in intervals:
            if not ans or start > ans[-1][1]:
                ans.append([start, end])
            else:
                ans[-1][1] = max(ans[-1][1], end)
        return ans
```

`[[1,3],[2,6],[8,10]]`：第二个区间的起点 2 不大于当前终点 3，合并成 `[1,6]`；8 超过 6，另起一段。

**易错点**：`[1,4]` 和 `[4,5]` 也需要合并；合并时终点取最大值，不能直接覆盖，例如 `[1,10]` 包含 `[2,3]`。

时间 `O(n log n)`；排序辅助空间最坏 `O(n)`，输出 `O(n)`。此实现会重排输入区间的顺序。

## 189. 轮转数组

题目：[轮转数组](https://leetcode.cn/problems/rotate-array/)。原地将数组向右移动 `k` 步。

### 思路：三次反转

把数组分为 `A + B`，其中 B 是末尾 k 个元素，目标是 `B + A`。整体反转得到 `reverse(B) + reverse(A)`，再分别反转两段即可。

```python
class Solution:
    def rotate(self, nums, k):
        n = len(nums)
        if n == 0:
            return
        k %= n

        def reverse(left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```

`[1,2,3,4,5,6,7]`，`k=3`：整体反转 `[7,6,5,4,3,2,1]`；反转前 3 个 `[5,6,7,4,3,2,1]`；反转后半段得到 `[5,6,7,1,2,3,4]`。

**易错点**：先取模；`k=0` 时第一个局部区间为空，代码自然处理；用切片拼接虽然简洁，但不是常数额外空间。

时间 `O(n)`，辅助空间 `O(1)`。

## 238. 除自身以外数组的乘积

题目：[除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)。不能使用除法，返回每个位置之外所有元素的乘积。

### 状态与思路

答案等于“左边所有数的乘积 × 右边所有数的乘积”。先把左侧乘积写入结果，再从右向左补上右侧乘积。

```python
class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)
        ans = [1] * n
        prefix = 1
        for i in range(n):
            ans[i] = prefix
            prefix *= nums[i]
        suffix = 1
        for i in range(n - 1, -1, -1):
            ans[i] *= suffix
            suffix *= nums[i]
        return ans
```

`[1,2,3,4]`：第一遍得到 `[1,1,2,6]`；右侧乘积依次补入，得到 `[24,12,8,6]`。

**易错点**：先写当前结果，再把当前元素乘进前缀或后缀，否则会包含自身。空乘积是 1。这种写法天然支持零和负数。

时间 `O(n)`；除输出数组外辅助空间 `O(1)`，输出空间 `O(n)`。

## 41. 缺失的第一个正数

题目：[缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)。在线性时间、常数辅助空间内找最小未出现正整数。

### 状态与思路

长度为 `n` 的数组，答案必在 `[1,n+1]`。把值 `x ∈ [1,n]` 尽量放到下标 `x-1`，相当于用输入数组充当“存在性表”。

```python
class Solution:
    def firstMissingPositive(self, nums):
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n:
                target = nums[i] - 1
                if nums[target] == nums[i]:
                    break
                nums[i], nums[target] = nums[target], nums[i]
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1
```

### 推演与正确性

`[3,4,-1,1]`：不断将 3 放到下标 2、4 放到下标 3、1 放到下标 0，最终可得到 `[1,-1,3,4]`。下标 1 没有 2，答案是 2。

每次有效交换都会让一个合法正整数到达正确位置，已经正确的值不会再被破坏，因此总交换次数为 `O(n)`。

**易错点**：一定要检查目标位置是否已经有同样的值，否则重复数字会导致无限交换；交换后的当前数字可能还没放好，所以用 `while`，不是 `if`。此算法会修改输入。

时间 `O(n)`，辅助空间 `O(1)`。

## 小结

看到原地要求，优先考虑交换、反转、读写指针或把下标本身作为信息；看到左右贡献，考虑前后缀；看到区间，先考虑排序后是否只需比较邻居。
