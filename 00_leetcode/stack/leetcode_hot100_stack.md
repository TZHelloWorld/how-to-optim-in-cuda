# Hot100 栈：最近的未完成任务与单调性

[返回总索引](../README.md)

## 基础：后进先出

栈是后进先出 LIFO：最后放进去的元素最先取出。Python 用列表尾部实现：`append` 入栈，`pop` 出栈，`stack[-1]` 看栈顶。

适用情形：括号配对、嵌套表达式、递归调用模拟，以及寻找“下一个更大/更小元素”。

**单调栈**是栈内元素按某种大小关系排列。新元素出现时，弹出的元素可以立即确定答案；没有弹出的继续等待。通常保存下标，便于计算距离。

## 20. 有效的括号

题目：[有效的括号](https://leetcode.cn/problems/valid-parentheses/)。圆括号、方括号、花括号必须类型和嵌套顺序都匹配。

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
