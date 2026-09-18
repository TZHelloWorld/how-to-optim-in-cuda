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

### 题目描述

给定严格升序、无重复元素的整数数组 nums 和目标 target。若 target 已存在，返回其下标；否则返回应插入的位置，使插入后数组仍保持升序。

**输入与约束**：数组非空，要求 `O(log n)` 时间。只返回位置，不实际插入；结果可能为 0，也可能为数组长度 n。

```text
输入：nums=[1,3,5,6], target=5
输出：2

输入：nums=[1,3,5,6], target=2
输出：1，插入后为 [1,2,3,5,6]。
```

### 从题意到算法

“存在则定位，不存在则插入”统一为找第一个不小于 target 的位置。维护半开区间，遇到中间值偏小就舍弃它及左边，遇到足够大则保留它并继续找更早位置。区间为空时边界就是答案，无需把相等单独分支处理。

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

### 题目描述

给定 m×n 整数矩阵 matrix 和 target。每行从左到右非递减，且除第一行外，每行第一个数大于上一行最后一个数。判断矩阵是否包含 target。

**输入与约束**：原题矩阵非空，可不是正方形；要求 `O(log(mn))` 时间。行与行之间的大小关系使整个矩阵按行展开后仍有序。

```text
输入：matrix=[[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3
输出：True

相同矩阵，target=13
输出：False
```

### 从题意到算法

无需真的把矩阵复制成数组，只把它视为长度 m×n 的有序序列。一维下标 mid 对应行 `mid//n`、列 `mid%n`。对这个虚拟数组找第一个不小于目标的位置，最后确认位置未越界且值相等即可。

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

### 题目描述

给定非递减排序整数数组 nums 和 target，返回 target 在数组中第一次、最后一次出现的下标 `[first,last]`。若不存在，返回 `[-1,-1]`。

**输入与约束**：允许空数组和重复值，要求 `O(log n)` 时间。找到某个 target 后线性向两侧扫描，最坏不满足时间要求。

```text
输入：nums=[5,7,7,8,8,10], target=8
输出：[3,4]

输入：nums=[], target=0
输出：[-1,-1]
```

### 从题意到算法

重复值在排序数组里连续出现，所以找两个边界即可：第一个 `>=target`，以及第一个 `>target`。对于整数，第二个可写成第一个 `>=target+1`，再减一得到末位置。先确认第一个边界真的等于 target，防止将插入位置误判为答案。

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

### 题目描述

一个严格升序、元素互异的整数数组，在某个位置旋转后得到 nums，例如 `[0,1,2,4,5,6,7]` 可变成 `[4,5,6,7,0,1,2]`。给定 nums 和 target，返回 target 在当前 nums 中的下标，不存在返回 -1。

**输入与约束**：数组非空，也可能等效于未旋转；要求 `O(log n)` 时间。旋转保持两段内部顺序，没有重复值。

```text
输入：nums=[4,5,6,7,0,1,2], target=0
输出：4

相同数组，target=3
输出：-1
```

### 从题意到算法

整体不再有序，但用 mid 切开后至少有一半有序。先根据端点比较识别有序半边，再判断 target 是否落在该半边的值域内；若不在，就搜索另一半。每轮都能安全排除一半，关键是不要对整个旋转数组直接套普通大小二分。

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

### 题目描述

给定由严格升序数组旋转得到的非空数组 nums，数组元素互不相同。返回数组中的最小元素，要求 `O(log n)` 时间。

**输入与约束**：数组可以只有一个元素，也可能旋转整圈而仍保持原升序。返回最小值本身，不是下标；不需要知道旋转次数。

```text
输入：nums=[3,4,5,1,2]
输出：1

输入：nums=[11,13,15,17]
输出：11
```

### 从题意到算法

比较 mid 与当前右端：若中间值更大，说明最小值在右侧断点之后，mid 可排除；否则 mid 到右端有序，最小值可能就在 mid，或在它左边，因此保留 mid。维护最小值始终在闭区间内，直到只剩一个位置。

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

### 题目描述

给定两个非递减排序数组 nums1、nums2，长度分别为 m、n，返回把所有元素合在一起后的中位数。总长度为奇数时取中间元素，偶数时取中间两个数的平均值。

**输入与约束**：可以有重复值和负数，其中一个数组可以为空，但不会同时为空。要求 `O(log(m+n))` 时间，因此完整合并后再取中位数不满足目标复杂度。

```text
输入：nums1=[1,3], nums2=[2]
输出：2.0，合并后为 [1,2,3]。

输入：nums1=[1,2], nums2=[3,4]
输出：2.5，等于 (2+3)/2。
```

### 从题意到算法

中位数只取决于“较小一半的最大值”和“较大一半的最小值”，不需要排出整个合并序列。在短数组里二分左侧取多少个元素，另一数组的数量由总左半长度确定。只检查两个交叉边界是否有序，就能判断切分是否正确及应向哪边移动；下面详细展开切点和哨兵处理。

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
