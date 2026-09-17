# Hot100 滑动窗口：维护一个连续区间

[返回总索引](../README.md)

## 基础：窗口含义、合法性与更新顺序

窗口通常写成闭区间 `[left, right]`，长度是 `right-left+1`。

常见框架：右端加入新元素 → 更新统计 → 必要时移动左端并移除旧元素 → 更新答案。

- 最长合法窗口：不合法时不断收缩，恢复合法后更新最大长度。
- 最短满足窗口：满足条件时更新答案，再尝试收缩到更短，见子串专题的最小覆盖子串。
- 固定长度窗口：长度超过要求，就删除左边多出来的元素。

左右指针都只前进时，每个元素最多进入、离开一次，因此嵌套 `while` 也可以是 `O(n)`。

窗口能否使用取决于性质。例如含负数的“和等于 K”不能直接套用“和过大就缩小”的规则，见子串专题。

## 3. 无重复字符的最长子串

题目：[无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)。求没有重复字符的最长连续子串长度。

### 状态与思路

`window` 保存当前窗口内字符。处理新字符 `ch` 时，如果它已存在，就从左侧移除，直到旧的 `ch` 被移出去，再把新字符加入。

```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        window = set()
        left = 0
        ans = 0
        for right, ch in enumerate(s):
            while ch in window:
                window.remove(s[left])
                left += 1
            window.add(ch)
            ans = max(ans, right - left + 1)
        return ans
```

### 推演

以 `"abba"` 为例：

| 新字符 | 调整后窗口 | 长度 |
|---|---|---:|
| 第一个 a | a | 1 |
| 第一个 b | ab | 2 |
| 第二个 b | b，先移出 a，再移出旧 b | 1 |
| 第二个 a | ba | 2 |

结果为 2。每个右端点下，窗口是以它结尾的最长无重复子串，因此比较所有右端点即可。

**易错点**：必须用 `while`，重复字符可能不在最左边，删一次不一定够；不能只求去重后的字符集合大小，子串要求连续。

**复杂度**：时间平均 `O(n)`；辅助空间 `O(min(n, Σ))`，`Σ` 是字符种类数。

## 438. 找到字符串中所有字母异位词

题目：[找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)。返回 `s` 中与 `p` 字符数量完全一致的子串起点，字符均为小写英文字母。

### 状态与思路

窗口长度固定为 `len(p)`。用两个长度为 26 的数组保存目标和窗口的字母次数。

```python
class Solution:
    def findAnagrams(self, s, p):
        m = len(p)
        if m > len(s):
            return []
        need = [0] * 26
        window = [0] * 26
        for ch in p:
            need[ord(ch) - ord("a")] += 1
        ans = []
        for right, ch in enumerate(s):
            window[ord(ch) - ord("a")] += 1
            if right >= m:
                old = s[right - m]
                window[ord(old) - ord("a")] -= 1
            if right >= m - 1 and window == need:
                ans.append(right - m + 1)
        return ans
```

### 推演与正确性

`s="cbaebabacd"`，`p="abc"`：长度 3 的窗口依次为 `cba、bae、aeb……bac、acd`。只有 `cba` 和 `bac` 的计数等于目标，返回 `[0,6]`。

每个长度为 `m` 的窗口都被检查一次，计数相同等价于异位词，因此不重不漏。

**易错点**：长度没达到 `m` 时不能记录答案；移出的是 `s[right-m]`；只看字符种类、不看次数会把 `aab` 和 `abb` 混淆。原题 `p` 非空。

**复杂度**：时间 `O(26n + m)`，固定字母表下为 `O(n+m)`；辅助空间 `O(26)`，另计输出。

## 与其他专题的联系

- [子串专题](../substring/leetcode_hot100_substring.md)：最小覆盖子串、单调队列窗口、前缀和。
- 判断是否能用窗口时，问“左端移动后，状态变化是否能被高效维护？移动规则是否能保证不漏解？”
