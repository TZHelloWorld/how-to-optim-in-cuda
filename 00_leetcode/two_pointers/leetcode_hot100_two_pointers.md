# Hot100 双指针：用移动规则排除不可能

[返回总索引](../README.md)

## 基础：双指针不只是两个下标

双指针的核心是证明“移动这一个指针，不会漏掉答案”。常见形式：

- 同向读写：一个指针读取输入，一个指针写入有效结果。
- 相向收缩：左右从两端靠近，利用大小关系排除候选。
- 快慢指针：在链表中找中点或环，见链表专题。
- 滑动窗口：维护连续区间的左右端点，见滑动窗口专题。

区分空间要求：原地修改通常意味着不能创建一个与输入等长的新数组。Python 的切片常常会隐式创建新列表。

## 283. 移动零

题目：[移动零](https://leetcode.cn/problems/move-zeroes/)。把零移到末尾，保持非零元素相对顺序，原地修改。

### 状态与思路

`write` 是下一个非零元素应该放的位置。处理到 `read` 前，`nums[:write]` 已按原顺序保存所有读过的非零元素。

```python
class Solution:
    def moveZeroes(self, nums):
        write = 0
        for read in range(len(nums)):
            if nums[read] != 0:
                nums[write], nums[read] = nums[read], nums[write]
                write += 1
```

### 推演、易错点与复杂度

`[0,1,0,3,12]`：遇到 `1` 交换到位置 0，遇到 `3` 交换到位置 1，遇到 `12` 交换到位置 2，得到 `[1,3,12,0,0]`。

每个非零元素依次进入写入区，因此顺序不变。`write` 只在读到非零时增加；题目要求修改输入，无需返回新数组。

时间 `O(n)`，辅助空间 `O(1)`。

## 11. 盛最多水的容器

题目：[盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)。选两条竖线，面积为宽度乘以较短高度。

### 状态与思路

从最外侧开始。面积为 `(right-left) * min(height[left], height[right])`。每次移动较短边。

为什么？若左边更短，保留左边并把右边向内移，宽度变小，而有效高度不可能超过原来的左边高度，所以不可能比当前更好。左边可以安全丢弃。

```python
class Solution:
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            ans = max(ans, (right - left) * min(height[left], height[right]))
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return ans
```

### 推演、易错点与复杂度

`[1,8,6,2,5,4,8,3,7]`：先算两端面积 `8`，移动高度 `1` 的左端；新面积是 `7 × 7 = 49`。

不要移动较高边来“保留更高的墙”，真正限制面积的是短边。等高时移动任意一侧都可以。

时间 `O(n)`，辅助空间 `O(1)`。

## 15. 三数之和

题目：[三数之和](https://leetcode.cn/problems/3sum/)。返回和为零的全部不重复三元组。

### 状态与思路

先排序，固定第一个数 `nums[i]`，再用相向双指针找另外两个数：

- 和太小，增大左指针，尝试更大的数字。
- 和太大，减小右指针，尝试更小的数字。
- 和为零，记录答案并跳过重复值。

```python
class Solution:
    def threeSum(self, nums):
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
        return ans
```

### 推演与正确性

`[-1,0,1,2,-1,-4]` 排序为 `[-4,-1,-1,0,1,2]`。固定第一个 `-1`，可以找到 `[-1,-1,2]` 和 `[-1,0,1]`。固定第二个 `-1` 会重复，因此跳过。

排序保证了和变化的单调性，所以指针每次跳过的候选都不可能成功。

**易错点**：去重的是“值相同带来的重复答案”，不能禁止三元组中有两个相同值，例如 `[-1,-1,2]` 合法；`[0,0,0]` 只能输出一次。

**复杂度**：时间 `O(n²)`；Python 排序最坏需要 `O(n)` 辅助空间，不应把整份实现写成严格 `O(1)` 空间；另计答案空间。此实现会排序输入。

## 42. 接雨水

题目：[接雨水](https://leetcode.cn/problems/trapping-rain-water/)。求所有柱子之间能存储的总水量。

### 从公式到双指针

位置 `i` 的水量为：`min(左侧最高柱, 右侧最高柱) - height[i]`，最高柱包含自身，所以结果不会为负。

双指针维护 `left_max`、`right_max`。如果 `left_max <= right_max`，说明左侧当前位置右边已经存在足够高的墙，它的水位可以确定为 `left_max`，无需知道更远处的右侧最大值。

```python
class Solution:
    def trap(self, height):
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        ans = 0
        while left <= right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if left_max <= right_max:
                ans += left_max - height[left]
                left += 1
            else:
                ans += right_max - height[right]
                right -= 1
        return ans
```

### 推演、易错点与复杂度

`[2,0,2]`：两端高度都是 2，中间左侧最高与右侧最高都为 2，中间贡献 `2-0=2`。

与容器题不同，这里是逐个位置累计水量，不是选择两堵墙求一个矩形。先更新最大高度，再计算当前水量。

时间 `O(n)`，辅助空间 `O(1)`。

## 小结

每次移动指针前，先说明：哪个位置的结果已经确定？或者哪一批候选已经证明不可能更优？无法解释这一点时，不要仅凭直觉移动指针。
