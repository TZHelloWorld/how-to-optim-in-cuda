# Hot100 哈希：把“查找”变成一次定位

[返回总索引](../README.md)

## 基础：字典、集合与键

- `dict` 保存“键 → 值”，例如“数字 → 下标”“前缀和 → 出现次数”。
- `set` 只记录某个元素是否出现，天然去重。
- 在通常的哈希分布假设下，查询、插入、删除平均为 `O(1)`，不是任何情况下都保证最坏 `O(1)`。
- 字典键必须可哈希：整数、字符串、由可哈希元素组成的元组可以；列表不能直接当键。
- “遇到一个数字，就去找它需要的伙伴”，是哈希题的常见思路。

本专题：1 两数之和、49 字母异位词分组、128 最长连续序列。

## 1. 两数之和

题目：[两数之和](https://leetcode.cn/problems/two-sum/)。找到两个不同下标，使对应数字之和等于目标。题目保证有且只有一组解。

### 状态与思路

遍历到 `nums[i] = x` 时，`seen` 只保存下标小于 `i` 的元素。需要的另一个数字是 `target - x`，直接查询即可。

```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, x in enumerate(nums):
            other = target - x
            if other in seen:
                return [seen[other], i]
            seen[x] = i
        return []
```

### 推演与正确性

`[2,7,11,15]`，目标 `9`：先遇到 `2`，没有 `7`，记下 `{2:0}`；再遇到 `7`，查询到 `2`，返回 `[0,1]`。

任何一对答案都会有先出现、后出现的元素。处理后者时，前者已经在字典中，因此不会漏掉答案。

**易错点**：先查询再插入，否则 `[3]`、目标 `6` 会错误地把同一个位置用两次；`[3,3]` 则应该返回两个不同下标。

**复杂度**：时间平均 `O(n)`；辅助空间 `O(n)`。

## 49. 字母异位词分组

题目：[字母异位词分组](https://leetcode.cn/problems/group-anagrams/)。把字符种类和数量相同、顺序可能不同的字符串分组。

### 状态与思路

将排序后的字符作为“统一标识”。例如 `eat`、`tea`、`ate` 都对应 `aet`。`groups[key]` 保存具有这个标识的全部原字符串。

```python
from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs):
        groups = defaultdict(list)
        for word in strs:
            key = "".join(sorted(word))
            groups[key].append(word)
        return list(groups.values())
```

### 推演与正确性

`["eat","tea","tan","ate"]`：

```text
aet → [eat, tea, ate]
ant → [tan]
```

排序后相同，当且仅当每种字符出现次数相同，因此不会把不同组混在一起，也不会把同组拆开。

**易错点**：不能只用 `set(word)`，因为它会丢失字符次数，例如 `"aab"` 和 `"abb"` 会被误分一组；空字符串也需要正常分组。

**复杂度**：有 `m` 个字符串，最大长度为 `k`，时间 `O(mk log k)`；包含分组与键的存储为 `O(mk)` 上界。题目只含小写字母时，也可用长度 26 的计数元组把时间优化到 `O(mk)`。

## 128. 最长连续序列

题目：[最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)。找数值连续的最长序列长度，不要求在原数组位置连续。

### 状态与思路

使用集合快速查询下一个整数是否存在。只从序列起点开始扩展：如果 `x - 1` 存在，`x` 就不是起点，不用重复搜索。

```python
class Solution:
    def longestConsecutive(self, nums):
        values = set(nums)
        ans = 0
        for x in values:
            if x - 1 in values:
                continue
            end = x
            while end in values:
                end += 1
            ans = max(ans, end - x)
        return ans
```

### 推演与正确性

`[100,4,200,1,3,2]` 中，`1` 是起点，扩展得到 `1,2,3,4`，长度 `4`。遇到 `2、3、4` 都会跳过扩展，因为它们存在前驱。

每个最大连续序列恰好有一个起点，起点搜索会覆盖整段。虽然有嵌套循环，每个数只属于一次扩展，平均总时间仍为线性。

**易错点**：应该遍历去重后的 `values`，避免重复起点导致重复扩展；不要混淆“数值连续”和“原数组下标连续”。

**复杂度**：时间平均 `O(n)`；辅助空间 `O(n)`。

## 小结

| 需求 | 字典或集合中保存什么 |
|---|---|
| 找目标伙伴 | 已出现数字 → 下标 |
| 按共同属性分组 | 规范化标识 → 原元素列表 |
| 判断相邻数是否存在 | 去重后的数值集合 |

做题前先问：什么信息值得作为键？值需要是下标、次数，还是一组元素？
