# Hot100 二分查找：区间定义与单调性

[返回总索引](../README.md)

## 基础：不是“看到排序就随便折半”

二分要求可以根据一次比较排除一整半候选。最常见的是单调真假边界：左边都不满足，右边都满足，寻找第一个满足的位置。

本文优先使用半开区间 `[left,right)` 的 lower_bound 模板：

- 目标：第一个值 `>= target` 的下标。
- 条件：`left < right`。
- 中间值偏小：`left = mid + 1`。
- 中间值满足：`right = mid`，保留它作为答案候选。
- 结束：`left == right`，就是边界，可以等于数组长度。

其他题可能使用闭区间 `[left,right]`，必须配套调整条件与更新，不能混用模板。

## 35. 搜索插入位置

题目：[搜索插入位置](https://leetcode.cn/problems/search-insert-position/)。有序数组中找目标下标，不存在则返回保持有序的插入位置。

**状态与思路**：寻找第一个不小于目标的位置。

```python
class Solution:
    def searchInsert(self, nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
```

**推演**：`[1,3,5,6]` 查 2，先排除 5 及右边，再找到第一个 `>=2` 的位置 1。

**易错点**：返回值可能是 n，表示插到末尾；不需要找到相等值才能返回。

时间 `O(log n)`，辅助空间 `O(1)`。

## 74. 搜索二维矩阵

题目：[搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)。每行有序，而且每行首元素大于上一行末元素。

**思路**：逻辑展平成一维有序数组。下标 mid 映射为 `row=mid//cols`、`col=mid%cols`，无需真的复制展平。

```python
class Solution:
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols
        while left < right:
            mid = (left + right) // 2
            if matrix[mid // cols][mid % cols] < target:
                left = mid + 1
            else:
                right = mid
        return left < rows * cols and matrix[left // cols][left % cols] == target
```

**推演**：`[[1,3,5],[7,9,11]]` 的逻辑数组是 `[1,3,5,7,9,11]`，9 位于一维下标 4，即 `(1,1)`。

**易错点**：除数和余数使用列数；第 240 题只有行列分别有序，不能使用这套展平二分。

时间 `O(log(RC))`，辅助空间 `O(1)`。

## 34. 在排序数组中查找元素的第一个和最后一个位置

题目：[查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)。不存在返回 `[-1,-1]`。

**思路**：先找第一个 `>= target`，再找第一个 `>= target+1`，后者减一就是最后一个 target。本题元素为整数。

```python
class Solution:
    def searchRange(self, nums, target):
        def lower_bound(value):
            left, right = 0, len(nums)
            while left < right:
                mid = (left + right) // 2
                if nums[mid] < value:
                    left = mid + 1
                else:
                    right = mid
            return left

        start = lower_bound(target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        return [start, lower_bound(target + 1) - 1]
```

**推演**：`[5,7,7,8,8,10]` 查 8，两个边界是 3 和 5，结果 `[3,4]`。

**易错点**：先确认目标存在，不能把插入位置当成真实下标；Python 整数不会因 `target+1` 溢出。

时间 `O(log n)`，辅助空间 `O(1)`。

## 33. 搜索旋转排序数组

题目：[搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)。互异升序数组旋转后查找目标，不存在返回 -1。

**状态与思路**：使用闭区间。每次左右两半至少有一半保持有序，先识别有序半边，再判断目标是否落在它的数值范围内。

```python
class Solution:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

**推演**：`[4,5,6,7,0,1,2]` 查 0，mid 是 7，左半有序但不包含 0，所以搜索右半。

**易错点**：使用 `nums[left] <= nums[mid]`，长度很小时两者可能是同一个位置；本题互异元素是判断单调半边的重要前提。

时间 `O(log n)`，辅助空间 `O(1)`。

## 153. 寻找旋转排序数组中的最小值

题目：[寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)。元素互异，返回最小值。

**状态与思路**：最小值始终在闭区间 `[left,right]` 内。比较 mid 与右端：

- mid 更大，断点在右边，最小值在 `mid+1..right`。
- mid 更小，`mid..right` 有序，最小值可能是 mid 或在它左边，保留 mid。

```python
class Solution:
    def findMin(self, nums):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

**推演**：`[3,4,5,1,2]`，5 大于右端 2，最小值在右半；继续缩小到 1。

**易错点**：不能在第二个分支写 `right=mid-1`，因为 mid 自己可能就是最小值；未旋转数组也能自然处理。

时间 `O(log n)`，辅助空间 `O(1)`。

## 4. 寻找两个正序数组的中位数

题目：[寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)。要求对数时间，两个数组不会同时为空。

### 状态与分割条件

在短数组中二分切点 i，在另一个数组切 j，使左侧一共拥有 `(m+n+1)//2` 个元素。目标是两边所有左侧元素不大于右侧元素。

由于各数组内部有序，只需检查两个交叉边界：`a_left <= b_right` 且 `b_left <= a_right`。

```python
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        a, b = nums1, nums2
        if len(a) > len(b):
            a, b = b, a
        m, n = len(a), len(b)
        half = (m + n + 1) // 2
        left, right = 0, m
        while left <= right:
            i = (left + right) // 2
            j = half - i
            a_left = a[i - 1] if i else float("-inf")
            a_right = a[i] if i < m else float("inf")
            b_left = b[j - 1] if j else float("-inf")
            b_right = b[j] if j < n else float("inf")
            if a_left > b_right:
                right = i - 1
            elif b_left > a_right:
                left = i + 1
            else:
                if (m + n) % 2:
                    return max(a_left, b_left)
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
```

### 推演与移动理由

`a=[1,3]`、`b=[2,4]`：各取一个到左侧，左边 `[1,2]`、右边 `[3,4]`，中位数 `(2+3)/2=2.5`。

若 a_left 太大，说明 a 左侧取多了，要减小 i；若 b_left 太大，说明 a 左侧取少了，要增加 i。空分区用无穷哨兵处理。

**易错点**：切点表示左侧元素个数，不是某个被选中元素的下标；在短数组上二分保证 j 合法；奇数长度左侧多一个，答案是左侧最大值。

时间 `O(log(min(m,n)+1))`，辅助空间 `O(1)`。

## 边界自检

每次写完用“目标小于全部、大于全部、刚好等于首尾、只有一个元素、有重复值”检查。先说清区间是闭区间还是半开区间，再决定 `mid` 是否要保留。
