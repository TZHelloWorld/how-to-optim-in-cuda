# Hot100 栈：最近的未完成任务与单调性

[返回总索引](../README.md)

## 基础：后进先出

栈是后进先出 LIFO：最后放进去的元素最先取出。Python 用列表尾部实现：`append` 入栈，`pop` 出栈，`stack[-1]` 看栈顶。

适用情形：括号配对、嵌套表达式、递归调用模拟，以及寻找“下一个更大/更小元素”。

**单调栈**是栈内元素按某种大小关系排列。新元素出现时，弹出的元素可以立即确定答案；没有弹出的继续等待。通常保存下标，便于计算距离。

## 20. 有效的括号

题目：[有效的括号](https://leetcode.cn/problems/valid-parentheses/)。圆括号、方括号、花括号必须类型和嵌套顺序都匹配。

### 题目描述

给定只包含 `(`、`)`、`[`、`]`、`{`、`}` 的字符串 s，判断是否为有效括号串。每个右括号必须与之前最近一个尚未匹配的同类型左括号配对，最后不能有未配对括号。

**输入与约束**：原题字符串非空，只含这六种字符；括号数量相等不一定有效，嵌套顺序也必须正确。返回布尔值。

```text
输入：s="()[]{}"
输出：True

输入：s="([)]"
输出：False，圆括号和方括号交叉，不能正确嵌套。
```

### 从题意到算法

“最近的未匹配左括号”正好对应栈顶。遇到左括号就入栈，右括号则检查栈是否非空且类型匹配，再弹出。任何一步失败就返回 False；全部字符处理后还要确认栈为空，排除剩余左括号。顺序匹配无法只靠每种括号的计数完成。

**状态**：栈保存尚未匹配的左括号。右括号只能匹配最近的未匹配左括号，即栈顶。

```python
class Solution:
    def isValid(self, s):
        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []
        for ch in s:
            if ch in "([{":
                stack.append(ch)
            elif not stack or stack.pop() != pairs[ch]:
                return False
        return not stack
```

**推演**：`"([])"` 依次压入 `(`、`[`，`]` 弹出 `[`，`)` 弹出 `(`，最后为空；`"([)]"` 在 `)` 处类型不匹配。

**易错点**：左右数量相等不代表有效；读完后栈也必须为空。原题字符仅为这六种括号。

时间 `O(n)`，辅助空间 `O(n)`。

## 155. 最小栈

题目：[最小栈](https://leetcode.cn/problems/min-stack/)。支持 push、pop、top、getMin，操作需要常数时间。

### 题目描述

设计 MinStack 类，实现普通栈的 `push(val)` 入栈、`pop()` 弹出栈顶、`top()` 读取栈顶，并额外提供 `getMin()` 返回当前栈内最小值。所有操作都要求常数时间。

**输入与约束**：元素为整数，可为负数并允许重复；题目保证调用 pop、top、getMin 时栈非空。构造、push、pop 无返回值，top 和 getMin 返回整数；读取不删除元素。

```text
操作：MinStack(), push(-2), push(0), push(-3), getMin(), pop(), top(), getMin()
对应结果：[None,None,None,None,-3,None,0,-2]
解释：弹出 -3 后，最小值恢复为 -2。
```

### 从题意到算法

只维护一个全局最小值，弹出它之后就不知道旧最小值是多少。让每层保存“自身值”和“截至本层的最小值”，入栈时与旧最小值比较，弹栈时连同本层最小信息一起移除。于是上一层的最小值自然恢复，重复最小值也能正确处理。

**状态**：每层保存 `(当前值, 截止这一层的最小值)`。弹栈后，上一层的最小值会自动恢复。

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        minimum = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, minimum))

    def pop(self):
        self.stack.pop()

    def top(self):
        return self.stack[-1][0]

    def getMin(self):
        return self.stack[-1][1]
```

**推演**：压入 `-2、0、-3`，保存的最小值依次是 `-2、-2、-3`；弹出 -3 后，最小值恢复为 -2。

**易错点**：只用一个全局 minimum，弹出最小值后无法常数时间找回旧最小值。原题保证查询和弹出时非空。

每次操作摊还 `O(1)`，空间 `O(n)`。

## 394. 字符串解码

题目：[字符串解码](https://leetcode.cn/problems/decode-string/)。如 `3[a2[c]]` 解码为 `accaccacc`，输入格式合法。

### 题目描述

给定编码字符串 s，其中 `k[encoded_string]` 表示把方括号内解码得到的字符串重复 k 次。编码可以嵌套，也可以与普通字母拼接，返回完整解码字符串。

**输入与约束**：输入格式保证合法，k 为正整数且可能有多位数字；字母是实际内容，数字只用于表示重复次数，不是待保留的普通字符。括号总能正确配对。

```text
输入：s="3[a]2[bc]"
输出："aaabcbc"

输入：s="3[a2[c]]"
输出："accaccacc"，先解出 a2[c] 为 acc，再重复三次。
```

### 从题意到算法

进入方括号相当于开始一个子问题，要暂存外层已读内容和重复次数；遇到右括号时，内层已完成，将它重复后接回外层。栈保存嵌套上下文，当前层用片段列表收集内容并在结束时 join，避免每个字符都与长前缀反复拼接。

**状态**：当前层维护字符片段列表 parts 和重复次数 number；遇到 `[` 保存外层上下文，遇到 `]` 完成内层并返回外层。

```python
class Solution:
    def decodeString(self, s):
        stack = []
        parts = []
        number = 0
        for ch in s:
            if ch.isdigit():
                number = number * 10 + int(ch)
            elif ch == "[":
                stack.append((parts, number))
                parts = []
                number = 0
            elif ch == "]":
                inside = "".join(parts)
                parts, repeat = stack.pop()
                parts.append(inside * repeat)
            else:
                parts.append(ch)
        return "".join(parts)
```

**推演**：`3[a2[c]]`：先完成 `2[c] → cc`，回到内层得到 `acc`，再重复 3 次。

**易错点**：次数可能是多位数，例如 `12[a]`；进入新层需要清空 number；返回外层时要保留 `[` 前已有的内容。

设输入长度 n，输出长度 L，嵌套深度 d。时间可给出安全上界 `O(n+(d+1)L)`，因为中间字符串可能在多个嵌套层被复制，不能只看输入长度。存储空间 `O(n+L)` 上界，包含构造输出。

## 739. 每日温度

题目：[每日温度](https://leetcode.cn/problems/daily-temperatures/)。每一天还要等多少天才出现更高温度，没有则 0。

### 题目描述

给定整数数组 temperatures，表示每天的温度。返回等长数组 answer，其中 `answer[i]` 是从第 i 天起还需等待几天，才会首次遇到温度严格更高的一天；后面没有更高温度则为 0。

**输入与约束**：数组非空，温度可重复；相等不算更高，只考虑之后的日期。返回等待天数，不是更高温度的值或绝对下标。

```text
输入：temperatures=[73,74,75,71,69,72,76,73]
输出：[1,1,4,2,1,1,0,0]
解释：下标 2 的 75 需要等到下标 6 的 76，所以等待 4 天。

输入：temperatures=[30,30]
输出：[0,0]
```

### 从题意到算法

让每个尚未找到答案的日期在单调栈中等待。新温度高于栈顶温度时，就能确定栈顶日期首次遇到更暖日，弹出并记录下标差；继续处理更早仍能被当前日解决的日期。剩余温度从栈底到栈顶非递增，每个日期只入栈、出栈一次。

**状态**：栈保存还没找到更暖一天的下标，对应温度从底到顶非递增。新温度更高时，栈顶的等待结束。

```python
class Solution:
    def dailyTemperatures(self, temperatures):
        ans = [0] * len(temperatures)
        stack = []
        for i, temperature in enumerate(temperatures):
            while stack and temperature > temperatures[stack[-1]]:
                old = stack.pop()
                ans[old] = i - old
            stack.append(i)
        return ans
```

**推演**：`[73,74,75,71,69,72,76,73]`，72 出现时，让 69 等待 1 天、71 等待 2 天；76 会处理仍在栈中的 72、75 等。

**正确性**：一个旧下标在栈中说明此前没有更高温度。第一次被弹出时，当前日就是它最近的更暖一天。

**易错点**：相同温度不算更暖，必须用 `>`；答案是下标差，不是温度差。

时间 `O(n)`，因为每个下标最多进出栈各一次；辅助空间 `O(n)`，另计输出。

## 84. 柱状图中最大的矩形

题目：[柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)。选择连续柱子组成矩形，求最大面积。

### 题目描述

给定非负整数数组 heights，每个元素是一根宽度为 1 的柱子的高度，柱子彼此紧邻。求完全位于柱状图覆盖范围内的最大矩形面积，矩形需要横跨一段连续柱子，高度不能超过这段中的最矮柱子。

**输入与约束**：数组非空，柱高可为 0。不是计算积水，也不能跳过较矮的中间柱子；只返回面积。

```text
输入：heights=[2,1,5,6,2,3]
输出：10
解释：选高度 5、6 的两根柱子，矩形宽 2、高 5。

输入：heights=[2,4]
输出：4，可以选高 4 宽 1，或高 2 宽 2。
```

### 从题意到算法

把每根柱子视为矩形的限制高度，就需要知道它向左右能扩展多远。递增栈保存右边界尚未确定的柱子；遇到更低新柱时，弹出的柱子获得右边界，弹出后的栈顶提供左侧界限。末尾补 0 让剩余正高度全部结算，重复高度的完整宽度会由较靠左的等高柱最终覆盖。

### 状态与思路

把每根柱子当作矩形的最低高度，寻找它左右第一个更低的边界。递增栈中的柱子还没确定右边界，新柱子更低时，栈顶就能结算面积。

代码允许等高柱子留在栈中；重复高度的最宽区间会由其中更靠左的一根最终计算。

```python
class Solution:
    def largestRectangleArea(self, heights):
        values = heights + [0]
        stack = []
        ans = 0
        for i, height in enumerate(values):
            while stack and values[stack[-1]] > height:
                h = values[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                ans = max(ans, h * width)
            stack.append(i)
        return ans
```

### 推演与边界解释

`[2,1,5,6,2,3]` 中遇到第二个 2，弹出 6，面积 `6×1=6`；再弹出 5，它可以覆盖 5、6 两根柱子，面积 `5×2=10`。

弹出后，左边可用范围从 `left+1` 开始，右边到 `i-1`，宽度为 `i-left-1`。如果左侧仍有等高柱子，此次算的范围可能更窄，但后续弹出更靠左等高柱子时会覆盖完整宽度。

**易错点**：要在弹出后读取新的栈顶作为左边界；末尾追加高度 0，触发剩余正高度柱子结算；这里复制数组，不修改输入。

时间 `O(n)`，辅助空间 `O(n)`。

## 小结

括号和解码的栈保存“尚未完成的上下文”；单调栈保存“还在等待更大或更小元素的候选”。问清楚一次弹栈意味着什么，代码就更容易推导。
