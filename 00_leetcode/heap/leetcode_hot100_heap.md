# Hot100 堆：只维护当前最需要的极值

[返回总索引](../README.md)

## 基础：堆不是完整排序

堆是一棵满足父子大小关系的完全二叉树，通常用数组表示。下标 i 的孩子是 `2i+1`、`2i+2`。

Python `heapq` 默认最小堆：

- `heap[0]` 是最小值，`O(1)`。
- `heappush/heappop/heapreplace` 为 `O(log k)`，k 是堆大小。
- `heapify` 原地建堆为 `O(k)`。
- 堆数组整体不一定有序，只有堆顶极值有保证。
- 对数字取负，可以用最小堆模拟最大堆。

常见反直觉点：保留最大的 k 个元素，使用容量 k 的**最小堆**，这样最容易淘汰这 k 个里面最小的。

## 215. 数组中的第 K 个最大元素

题目：[数组中的第 K 个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)。重复值也算名次。

### 题目描述

给定整数数组 nums 和整数 k，返回将数组按非递增顺序排序后第 k 个元素的值。这里第 k 大包含重复项，不是第 k 个不同的值。

**输入与约束**：数组非空，`1<=k<=len(nums)`，可有负数和重复值。题目希望直接选择元素，而非将整个数组完整排序；下面使用固定容量堆实现。

```text
输入：nums=[3,2,1,5,6,4], k=2
输出：5

输入：nums=[3,2,3,1,2,4,5,5,6], k=4
输出：4，降序前四项为 [6,5,5,4]。
```

### 从题意到算法

只需要最大的 k 个数，无需维护剩余元素的排序。用大小 k 的最小堆保存当前入选者，堆顶是入选者中最小的，也就是淘汰边界。新值更大就替换堆顶，否则忽略。扫描完毕，堆顶恰好是全数组第 k 大，重复值按各自出现次数保留。

**状态**：堆始终保存已处理元素中最大的 k 个；堆顶是其中最小的，也就是当前第 k 大。

```python
import heapq


class Solution:
    def findKthLargest(self, nums, k):
        heap = nums[:k]
        heapq.heapify(heap)
        for i in range(k, len(nums)):
            if nums[i] > heap[0]:
                heapq.heapreplace(heap, nums[i])
        return heap[0]
```

**推演**：`[3,2,1,5,6,4]`、k=2，保留的最大两数最终为 5、6，堆顶 5。

**易错点**：不是第 k 个不同的值；不要用集合去重；`heapreplace` 会直接替换堆顶，因此先确认新值更大。

时间 `O(n log(k+1))` 上界，辅助空间 `O(k)`。快速选择可取得平均线性时间，但堆版本更容易保证稳定的上界。

## 347. 前 K 个高频元素

题目：[前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)。返回出现次数最高的 k 个数，答案顺序任意。

### 题目描述

给定整数数组 nums 和正整数 k，返回出现频率最高的 k 个不同元素。答案中返回元素值，不返回次数，可以按任意顺序输出。

**输入与约束**：数组非空，`1<=k<=不同元素数量`；保证答案集合唯一。题目进阶要求时间优于 `O(n log n)`。本节先介绍容量 k 的堆，再给出满足线性时间目标的分桶实现。

```text
输入：nums=[1,1,1,2,2,3], k=2
输出：[1,2]
解释：1 出现 3 次，2 出现 2 次，频率最高。

输入：nums=[1], k=1
输出：[1]
```

### 从题意到算法

先用哈希表把“找频率”转成 `(元素,次数)` 列表，再按频次而不是数值选 Top K。最小堆顶表示目前入选的最低频率，新候选更高时替换它。频率只可能在 1 到 n 之间，还可以直接把频率当桶下标，跳过比较排序，得到后面的线性时间实现。

**状态与思路**：先计数，再用容量 k 的最小堆保留频次最高的元素，堆中存 `(频次, 数值)`。

```python
import heapq
from collections import Counter


class Solution:
    def topKFrequent(self, nums, k):
        counts = Counter(nums)
        heap = []
        for value, frequency in counts.items():
            if len(heap) < k:
                heapq.heappush(heap, (frequency, value))
            elif frequency > heap[0][0]:
                heapq.heapreplace(heap, (frequency, value))
        return [value for _, value in heap]
```

**推演**：`[1,1,1,2,2,3]`、k=2，频次为 3、2、1，保留 1 和 2。

**易错点**：比较的是频次，不是数值大小；堆内输出不保证按频次降序，但题目允许任意顺序。

设不同元素数为 u，时间平均 `O(n+u log(k+1))`，辅助空间 `O(u+k)`。

### 按频次分桶：补齐线性时间进阶

当 u、k 都与 n 同阶时，上面的堆版本最坏仍可能达到 `O(n log n)`。要对所有合法 k 都达到线性时间，可以利用频率的有限范围：

1. 统计每个值的频率 f。
2. 将该值放入 `buckets[f]`；一个桶保存所有恰好出现 f 次的不同值。
3. 从频率 n 向 1 扫描，依次收集元素，满 k 个就返回。

桶号已经表示频率高低，因此不必对不同元素比较排序。每个输入元素只参与一次计数，每个不同值只入桶一次，再扫描 n 个桶，总时间平均 `O(n)`，辅助空间 `O(n)`。

```python
from collections import Counter


class SolutionBucket:
    def topKFrequent(self, nums, k):
        counts = Counter(nums)
        buckets = [[] for _ in range(len(nums) + 1)]
        for value, frequency in counts.items():
            buckets[frequency].append(value)

        answer = []
        for frequency in range(len(nums), 0, -1):
            for value in buckets[frequency]:
                answer.append(value)
                if len(answer) == k:
                    return answer
```

对于 `[1,1,1,2,2,3]`，非空桶为 `buckets[3]=[1]`、`buckets[2]=[2]`、`buckets[1]=[3]`。取前两个元素就得到 `[1,2]`。单独提交此版本时，将类名改为 `Solution`。

## 295. 数据流的中位数

题目：[数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)。不断加入数字，随时查询中位数。

### 题目描述

设计 MedianFinder 类，初始没有数据。`addNum(num)` 加入一个整数，`findMedian()` 返回目前全部数据的中位数：数量为奇数时取排序后中间值，为偶数时取中间两值平均数。

**输入与约束**：数据可以重复或为负数；保证查询中位数前至少加入过一个数。添加操作无返回值，查询不移除数据。同一个对象需要支持交错进行的添加和查询。

```text
操作：MedianFinder(), addNum(1), addNum(2), findMedian(), addNum(3), findMedian()
对应结果：[None,None,None,1.5,None,2.0]
解释：前两项中位数为 (1+2)/2，三项时中位数为 2。
```

### 从题意到算法

中位数只需要两半的边界，不需要每次完整排序。用最大堆保存较小一半，最小堆保存较大一半；保证前者数量等于后者或多一个，同时左边所有值不大于右边。插入后通过移动堆顶恢复这两个条件，查询只读一个或两个堆顶即可。

### 状态与不变量

- `low` 最大堆，保存较小的一半，用负数实现。
- `high` 最小堆，保存较大的一半。
- low 的任何值都不大于 high 的任何值。
- low 的数量等于 high，或恰好多一个。

因此奇数时中位数是 low 的堆顶，偶数时是两个堆顶的平均值。

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.low = []
        self.high = []

    def addNum(self, num):
        heapq.heappush(self.low, -num)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def findMedian(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2
```

### 推演与正确性

加入 1、2 后，low 保存 1，high 保存 2，中位数 1.5；再加入 3，平衡后 low 保存 1、2，high 保存 3，中位数 2。

先把新数放入 low，再把 low 最大值移到 high，保证两边数值有序；必要时把 high 最小值移回 low，恢复数量关系。

**易错点**：low 里的符号要在移动和读取时转换；只保持数量平衡却不保持两边大小关系，不能正确求中位数。原题保证查询前至少有一个数。

插入 `O(log n)`，查询 `O(1)`，空间 `O(n)`。

## 小结

求固定容量的 Top K，用一只堆维护淘汰边界；求动态中位数，用两只堆维护左右分界。堆不需要把所有元素排好，只维护题目需要的极值即可。
