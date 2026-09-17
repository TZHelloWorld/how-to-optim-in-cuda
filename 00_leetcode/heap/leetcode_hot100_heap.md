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

## 295. 数据流的中位数

题目：[数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)。不断加入数字，随时查询中位数。

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
