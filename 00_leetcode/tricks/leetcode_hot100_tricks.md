# Hot100 技巧：异或、投票、分区与隐式环

[返回总索引](../README.md)

## 基础：技巧依赖题目保证

这些题的短代码往往建立在很强的条件上：其他元素恰好出现两次、多数元素保证存在、取值限定为 0/1/2，或者数值可当作下标。

先写出条件，再用对应性质推导；条件改变时，算法未必仍然正确。

## 136. 只出现一次的数字

题目：[只出现一次的数字](https://leetcode.cn/problems/single-number/)。除一个元素出现一次外，其余均出现两次。

**性质与状态**：异或满足 `x^x=0`、`x^0=x`，并满足交换律、结合律。全部异或后，成对元素抵消，只剩单独元素。

```python
class Solution:
    def singleNumber(self, nums):
        ans = 0
        for x in nums:
            ans ^= x
        return ans
```

**推演**：`[4,1,2,1,2]` 可以重排异或运算为 `4^(1^1)^(2^2)=4`。

**易错点**：Python `^` 是异或，不是乘方；如果其他元素出现三次，就不能直接套用。

时间 `O(n)`，辅助空间 `O(1)`，按固定字长整数模型计算。

## 169. 多数元素

题目：[多数元素](https://leetcode.cn/problems/majority-element/)。返回出现次数严格超过 n/2 的元素，保证存在。

**状态与思路**：Boyer–Moore 投票法。`candidate` 是当前候选，`votes` 表示和其他值配对抵消后剩余的净票数。

```python
class Solution:
    def majorityElement(self, nums):
        candidate = None
        votes = 0
        for x in nums:
            if votes == 0:
                candidate = x
            votes += 1 if x == candidate else -1
        return candidate
```

**推演**：`[2,2,1,1,1,2,2]` 中不断抵消不同元素，最终候选为 2。多数元素比其他所有元素加起来还多，删除一对不同值后，它仍能在剩余非空部分保持多数。

**易错点**：votes 不是候选在原数组的真实出现次数；若题目不保证多数元素存在，需要再遍历一次验证候选次数。

时间 `O(n)`，辅助空间 `O(1)`。

## 75. 颜色分类

题目：[颜色分类](https://leetcode.cn/problems/sort-colors/)。原地把只含 0、1、2 的数组排序。

### 状态：四个区间

```text
[0, left)       全部为 0
[left, i)       全部为 1
[i, right]      尚未处理
(right, n)      全部为 2
```

```python
class Solution:
    def sortColors(self, nums):
        left = i = 0
        right = len(nums) - 1
        while i <= right:
            if nums[i] == 0:
                nums[left], nums[i] = nums[i], nums[left]
                left += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
            else:
                i += 1
```

**推演**：`[2,0,2,1,1,0]` 遇到 2，把它换到右侧；新换来的 0 还未检查，所以 i 不动，再把 0 交换到左侧，最终 `[0,0,1,1,2,2]`。

**易错点**：交换 2 后不能立刻 i++，因为从右侧换来的是未知元素；交换 0 后可以前进，因为左侧换回来的位置属于已处理区域或就是自身。

时间 `O(n)`，辅助空间 `O(1)`；修改输入。

## 31. 下一个排列

题目：[下一个排列](https://leetcode.cn/problems/next-permutation/)。原地变成字典序刚好更大的排列；若已最大，则变成最小排列。

### 思路：尽量晚地变大，后面尽量小

1. 从右向左找第一个 `nums[i] < nums[i+1]`，i 是可以增大的最右位置。
2. 右侧后缀非递增，从右向左找第一个大于 nums[i] 的元素，它是最小的可用更大值。
3. 交换后反转后缀，使后缀升序，得到尽可能小的增加。

```python
class Solution:
    def nextPermutation(self, nums):
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

**推演**：`[1,3,2]`，可增大位置是 1，后缀中最小的更大值是 2；交换成 `[2,3,1]`，反转后缀得到 `[2,1,3]`。

**易错点**：比较必须处理相等元素，寻找严格增大；完全降序时 i=-1，直接反转全部变为最小排列。

时间 `O(n)`，辅助空间 `O(1)`。

## 287. 寻找重复数

题目：[寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)。长度 n+1，所有值在 `[1,n]`，只有一个不同的重复值但可重复多次。不能修改数组，要求常数辅助空间。

### 把数组看作链表

将“下标 i 的后继”定义为 `nums[i]`。从下标 0 开始反复跳转，有限节点中必然进入环。由于没有值 0，起点 0 不在环里；进入环的节点同时有路径前驱和环内前驱，因此对应的值出现多次。

问题转成第 142 题：找环入口。

```python
class Solution:
    def findDuplicate(self, nums):
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        start = 0
        while start != slow:
            start = nums[start]
            slow = nums[slow]
        return start
```

**推演**：`[1,3,4,2,2]` 的下标跳转是 `0→1→3→2→4→2…`，入口下标 2 对应重复值 2。

**易错点**：这里比较的是下标状态，前进操作是 `nums[index]`；普通任意整数数组不能这样做，因为值可能不在合法下标范围。整个过程不会修改输入。

时间 `O(n)`，辅助空间 `O(1)`。

## 小结

先识别题目保证，再想它允许哪些结构化操作：出现两次可抵消，过半可投票，少数固定值可分区，值域可作下标时可能隐藏图结构。
