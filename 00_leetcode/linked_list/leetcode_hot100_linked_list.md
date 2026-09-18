# Hot100 链表：先理解节点与引用，再练指针操作

[返回总索引](../README.md)

本文把链表基础、Hot100 的 14 道链表题，以及 LRU 的完整设计放在一起。阅读时按“状态含义 → 操作流程 → Python 实现 → 指针推演 → 正确性与边界”理解。

## 阅读导航

- 基础：[链表与复杂度](#基础一链表是什么)、[节点定义与构造](#基础二节点定义与本地构造)、[指针与常用操作](#基础三四个高频技巧)。
- 遍历与判断：[160 相交链表](#160-相交链表)、[206 反转链表](#206-反转链表)、[234 回文链表](#234-回文链表)、[141 判断环](#141-环形链表)、[142 找环入口](#142-环形链表-ii)。
- 基本改链：[21 合并两条有序链表](#21-合并两个有序链表)、[2 两数相加](#2-两数相加)、[19 删除倒数节点](#19-删除链表的倒数第-n-个结点)、[24 两两交换](#24-两两交换链表中的节点)。
- 分组与复杂操作：[25 K 组反转](#25-k-个一组翻转链表)、[138 随机链表复制](#138-随机链表的复制)、[148 排序链表](#148-排序链表)、[23 合并 K 条链表](#23-合并-k-个升序链表)。
- 设计题：[146 LRU 缓存完整设计](#146-lru-缓存)。
- 复习：[方法对比与练习路线](#入门练习路线与检查清单)。

## 基础一：链表是什么？

数组把元素放在可以按下标访问的位置；链表通过节点之间的引用连接数据。

```text
head
 ↓
[1 | next] → [2 | next] → [3 | None]
```

一个节点包含值 `val` 和后继引用 `next`。`head` 只是第一个节点的引用，不是整条链表的拷贝。

| 操作 | 单链表代价 |
|---|---|
| 查找第 k 个节点 | `O(k)`，需要从头走 |
| 已知前驱，插入或删除后继 | `O(1)` |
| 查找尾节点 | 没有尾指针时 `O(n)` |
| 反向访问 | 单链表不直接支持；双链表有 `prev` |

“链表删除是常数时间”指已经知道必要节点的情况，不包括查找位置的时间。

### 单链表、双向链表和环形链表

```text
单链表：    A → B → C → None

双向链表：  None ← A ↔ B ↔ C → None

带环链表：  A → B → C → D
                ↑       |
                └───────┘
```

- 单链表只有 next，从某个节点通常不能直接知道它的前驱。
- 双向链表还有 prev，可以在知道节点的情况下直接调整两侧连接，LRU 会用到。
- 带环链表沿 next 会重复到达某个节点，没有正常的尾部 None，不能用普通遍历一直走到空。

图里的 A、B、C 是**节点身份**，节点内部的值可以相同。例如 A.val 与 B.val 都为 1，A 和 B 仍是两个不同节点。

## 基础二：节点定义与本地构造

LeetCode 提供节点定义。本地运行下面的题解时，先执行本节的定义；之后每道题的 `Solution` 单独使用。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Node:
    # 第 138 题的随机链表节点
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def build_list(values):
    dummy = ListNode()
    tail = dummy
    for value in values:
        tail.next = ListNode(value)
        tail = tail.next
    return dummy.next


def to_list(head):
    # 仅用于确定无环的链表
    values = []
    while head is not None:
        values.append(head.val)
        head = head.next
    return values
```

### 引用赋值与修改节点的区别

```text
p = head       p 和 head 指向同一个节点
p = p.next     只移动 p，不改变链表连接
p.next = q     修改 p 所指节点的后继，链表结构发生变化
p is q         判断是不是同一个节点
p.val == q.val 只判断值是否相同
```

链表相交、检测环，判断的是节点身份，不是数值。

### 为什么移动参数不会让调用者丢失 head？

调用 `to_list(head)` 时，函数内的 head 与外部变量开始都指向同一个节点。执行 `head=head.next` 只是让函数内的局部变量重新指向后继，不会修改外部变量。

但执行 `head.next=None` 是修改双方引用的同一个节点，所以外部看到的连接也会变化。

这也解释了为什么反转、删除头节点等题通常返回一个新头：**链表中真实节点的连接可以原地改变，但调用者还需要把自己的头引用更新到正确位置。**

### 本地构造时发生了什么？

`build_list([1,2,3])` 的构造过程是：

```text
dummy → None                  tail=dummy
dummy → 1 → None              tail=节点1
dummy → 1 → 2 → None          tail=节点2
dummy → 1 → 2 → 3 → None      tail=节点3
```

dummy 留在最前面方便找到结果，tail 一直移动方便追加。最终返回 `dummy.next`，不会把虚拟节点当成数据返回。

只有明确无环时才调用 `to_list`。打印环形链表可以限制步数，或额外记录访问过的节点身份，避免无限循环。

## 基础三：四个高频技巧

1. **虚拟头节点 dummy**：让删除头节点和删除其他节点使用同一套逻辑，最后返回 `dummy.next`。
2. **保存后继**：修改 `cur.next` 前，先保存原来的 `nxt`，否则可能找不到后半段。
3. **快慢指针**：快指针走两步，慢指针走一步，可以找中点或判断环。
4. **画箭头**：每次修改的不是值，而是边。先画 `prev → cur → nxt`，再写赋值。

本专题按“遍历判断 → 基本改链 → 分段与排序 → 复杂结构”组织，涵盖 14 题。

### 1. dummy 为什么能统一删除头节点？

没有 dummy 时，普通节点有前驱，但 head 没有。如果要删 head，就需要单独修改头引用。

加上 dummy 后：

```text
dummy → head → ...
```

每个真实节点都有前驱。删除第一个节点也变成 `dummy.next=dummy.next.next`，返回 `dummy.next` 就能得到更新后的头。

### 2. 插入和删除：先画出需要改的边

以下是操作示意，使用时需要先确认相关节点存在。

```text
在 prev 后插入 node：
原来：prev → after
目标：prev → node → after
顺序：node.next = prev.next
      prev.next = node

删除 prev 后的节点：
原来：prev → removed → after
目标：prev → after
操作：prev.next = prev.next.next
```

删除意味着从这条链上绕过该节点，不是把它的值改成 0，也不是保证这个对象立刻销毁。其他变量如果还引用它，仍然能访问这个对象。

### 3. 先保存后继，再修改指针

```text
prev     cur → nxt → rest

nxt = cur.next     保存原后继
cur.next = prev    修改当前箭头
prev = cur        更新已处理部分的头
cur = nxt         继续处理原来的后继
```

`nxt` 只是保存引用，不是复制剩余链表。改变 `cur.next` 后，原来的后半段仍可以通过 nxt 找到。

### 4. 快慢指针的终止条件决定 slow 停在哪里

不同题的“中点”要求不完全相同。本篇两种常用写法：

| 用途 | 初始值 | 循环条件 | 长度 4 时 slow | 长度 5 时 slow |
|---|---|---|---|---|
| 回文：找到后半段的前驱 | slow、fast 都为 head | `fast.next` 与 `fast.next.next` 都非空 | 第 2 个节点 | 第 3 个节点 |
| 归并排序：拆成两段 | slow=head，fast=head.next | fast 与 fast.next 都非空 | 第 2 个节点 | 第 3 个节点 |

不要看到快慢指针就固定背同一段代码，要先问：希望 slow 停在中间节点、前半段末尾，还是某个目标的前驱？

### 5. 每次循环维护一个清晰的状态

例如反转链表：已反转部分与未处理部分；合并链表：已合并前缀与两条剩余链；分组反转：已完成组、当前组、后续组。

先明确各部分的入口引用，修改连接时就不会把某段链表弄丢。常见难点不是数字计算，而是**还有哪个引用能够找到未处理的数据**。

## 160. 相交链表

题目：[相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)。返回两条无环链表的第一个公共节点，无交点返回 `None`。

### 题目描述

给定两个单链表头节点 headA、headB，找出它们从哪个节点开始共享同一段链表，返回这个第一个公共节点；不存在交点时返回 None。

**输入与约束**：原题两条链均非空、整个结构无环；相交按节点对象身份判断，不是值相同。函数结束后必须保持原有链表结构。期望 `O(m+n)` 时间、`O(1)` 空间。

```text
A：A1(4) → A2(1) → C1(8) → C2(4) → C3(5)
B：B1(5) → B2(6) → B3(1) → C1(8) → C2(4) → C3(5)
输出：节点 C1（值为 8），不是整数 8，也不是另建一个值为 8 的节点。

若 A=[1,2]、B=[1,2] 的节点分别独立创建，输出 None。
```

平台示例中的交点值、跳过节点数只是构造测试数据的信息，实际函数只接收两个头引用。

**状态与思路**：指针 `a` 走完 A 后走 B，`b` 走完 B 后走 A。两者走过的总路径长度相同，抵消了不同长度的前缀。

### “相交”到底指什么？

```text
A1 → A2 ─┐
         ↓
         C1 → C2 → None
         ↑
B1 → B2 → B3
```

两条路径从 C1 开始引用完全相同的节点。单链表节点只有一个 next，所以一旦真正相交，后面的尾部必然共享，不会再分叉。

两条各自独立的 `[1,2,3]` 链表只是值相同，不算相交。

### 为什么交换起点可以消除长度差？

设 A 的独有前缀长度为 x，B 的为 y，公共尾长为 z。先走 A 再走 B 的独有前缀，会经过 `x+z+y` 个节点；反过来是 `y+z+x`，两者相等。

代码从尾部 None 切换到另一条头时也占一次循环更新，但两边各有一次这种切换，因此不会破坏对齐。长度相同的两条链可能在第一次经过公共尾部时就已经相遇，无需切换。

```text
每轮同时移动两个指针：
    a 非空 → 沿 A 或 B 当前链继续；a 为空 → 换到 B 的头
    b 非空 → 沿当前链继续；b 为空 → 换到 A 的头
直到 a 和 b 指向同一个对象（或者一起为 None）
```

```python
class Solution:
    def getIntersectionNode(self, headA, headB):
        a, b = headA, headB
        while a is not b:
            a = a.next if a is not None else headB
            b = b.next if b is not None else headA
        return a
```

**推演**：A 独有前缀长 2，B 独有前缀长 3，共用尾部长 2。切换头节点后，两者都会经历“一条完整链 + 另一条独有前缀”，在公共尾部入口相遇；无交点时最终一起到 `None`。

对上图逐轮执行，位置如下：

| 同时更新次数 | a | b |
|---:|---|---|
| 0 | A1 | B1 |
| 1 | A2 | B2 |
| 2 | C1 | B3 |
| 3 | C2 | C1 |
| 4 | None | C2 |
| 5 | B1 | None |
| 6 | B2 | A1 |
| 7 | B3 | A2 |
| 8 | C1 | C1，停止 |

如果没有公共节点，走完两条链后会同时到 None，返回 None。全程只移动局部引用，不会修改输入链表。

**易错点**：两条链都含值 8 不代表相交，必须引用同一个节点。

时间 `O(m+n)`，辅助空间 `O(1)`。

## 206. 反转链表

题目：[反转链表](https://leetcode.cn/problems/reverse-linked-list/)。将链表箭头全部反向。

### 题目描述

给定单链表头节点 head，将整条链表反转，返回反转后的头节点。原来的尾成为新头，原来的头成为新尾。

**输入与约束**：链表无环，可以为空；值可以重复。下面通过重新连接原节点完成反转，结果尾节点的 next 应为 None。题目也允许用递归或迭代实现。

```text
输入：head=1→2→3→4→5→None
输出：5→4→3→2→1→None 的头节点

输入：head=None
输出：None
```

**状态定义**：`prev` 是已经反转部分的头；`cur` 是尚未处理部分的头。

### 把一次循环理解成“搬一个节点”

原来：

```text
已反转：prev → ... → None
未处理：cur → nxt → ... → None
```

拿出 cur，让它指向 prev，再让 prev 指向它。原后继 nxt 成为下一次待处理节点。

```text
while cur 不是空:
    保存 cur 原来的 next
    把 cur 的 next 指向已反转部分
    已反转部分的头改为 cur
    待处理部分的头改为原 next
返回已反转部分的头
```

```python
class Solution:
    def reverseList(self, head):
        prev = None
        cur = head
        while cur is not None:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev
```

**推演**：`1→2→3`，先得到 `1→None`，再得到 `2→1→None`，最后 `3→2→1→None`。

| 时刻 | prev 指向的已反转部分 | cur 指向的未处理部分 |
|---|---|---|
| 开始 | None | `1→2→3→None` |
| 处理 1 后 | `1→None` | `2→3→None` |
| 处理 2 后 | `2→1→None` | `3→None` |
| 处理 3 后 | `3→2→1→None` | None |

第一轮 `prev=None`，恰好把原来的头变成新尾，并使新尾指向 None。循环结束时所有节点恰好搬了一次，返回 prev 就是反转后的头。

不能只写 `cur.next=prev` 后再执行 `cur=cur.next`，那样会走回刚反转的部分，而不是原来的后半段。

**易错点**：先保存 `nxt`，再修改箭头；循环结束时 `cur` 为 `None`，新头是 `prev`。该实现修改原链表。

时间 `O(n)`，辅助空间 `O(1)`。

## 234. 回文链表

题目：[回文链表](https://leetcode.cn/problems/palindrome-linked-list/)。判断节点值序列是否正反相同。

### 题目描述

给定单链表头节点 head，判断其节点值按从头到尾读取后是否为回文：正向和反向读取必须得到完全相同的值序列。返回 True 或 False。

**输入与约束**：原题链表非空且无环，节点值为 0 到 9；进阶要求 `O(n)` 时间、`O(1)` 辅助空间。比较的是值，不要求首尾是同一个节点。下面实现还会恢复临时反转的部分。

```text
输入：head=1→2→2→1
输出：True

输入：head=1→2
输出：False
```

**思路**：快慢指针找到前半段末尾，反转后半段，再逐个比较；最后恢复后半段，避免仅做判断却改变输入结构。

### 为什么需要反转后半段？

数组可以直接比较首尾，但单链表只能向后走。把后半段反转后，“从头向后走”和“从原尾向前比较”就都变成沿 next 的普通遍历。

```text
寻找中点
将 slow.next 开始的后半段反转，保存新头 second
a 从原 head 出发，b 从 second 出发
逐个比较 a.val 与 b.val
无论比较结果如何，都反转回后半段并接好
返回比较结果
```

### 奇数和偶数长度怎样处理？

- 偶数 `1→2→2→1`：slow 停在第 2 个节点，后半段是后面的 `2→1`。
- 奇数 `1→2→3→2→1`：slow 停在中间的 3，后半段是 `2→1`，中间值无需参与比较。

因此统一从 `slow.next` 开始反转，并且只比较到 b 为空即可。

```python
class Solution:
    def isPalindrome(self, head):
        if head is None or head.next is None:
            return True

        def reverse(node):
            prev = None
            while node is not None:
                nxt = node.next
                node.next = prev
                prev, node = node, nxt
            return prev

        slow = fast = head
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next
        second = reverse(slow.next)
        a, b = head, second
        answer = True
        while b is not None:
            if a.val != b.val:
                answer = False
                break
            a, b = a.next, b.next
        slow.next = reverse(second)
        return answer
```

**推演**：`1→2→2→1`，后半段反转为 `1→2`，与前半段依次比较。奇数长度时中间节点不需要比较。

用节点身份区分重复值：

```text
原链：A(1) → B(2) → C(2) → D(1) → None
slow=B

后半段反转后：second 指向 D(1) → C(2) → None
比较：A 与 D；B 与 C
```

反转过程中 `slow.next` 仍引用原后半段头 C，不代表它自动变成 second；比较使用单独保存的 second 引用。恢复时 `reverse(second)` 返回 C，再执行 `slow.next=C`，重新得到原结构。

`answer=False` 后先 break 而不直接 return，就是为了保证恢复代码仍会执行。恢复只调整指针，不需要保存一份节点数组，所以辅助空间仍为常数。

**易错点**：发现不相等也要先恢复链表再返回；比较到短的后半段结束即可。

时间 `O(n)`，辅助空间 `O(1)`。

## 141. 环形链表

题目：[环形链表](https://leetcode.cn/problems/linked-list-cycle/)。判断沿 `next` 是否会重复到达同一个节点。

### 题目描述

给定单链表头节点 head，判断沿 next 不断前进时，是否会再次到达之前访问过的同一个节点。如果会，说明链表中有环，返回 True；否则返回 False。

**输入与约束**：head 可以为空；重复的节点值不代表有环。平台可能用 pos 表示尾节点连接到的下标，pos=-1 表示无环，但 pos 不会传给你的函数。进阶要求 `O(1)` 辅助空间。

```text
输入结构：A(3) → B(2) → C(0) → D(-4)，D.next 指向 B
输出：True

输入结构：1→2→None
输出：False
```

**思路**：快慢指针。无环时快指针先到末尾；有环时，快指针在环上每轮相对慢指针多走一步，最终追上。

### 为什么一定能追上？

设环长为 c。两者都进入环后，每轮 fast 走 2 步，slow 走 1 步，相对位移每轮增加 1。用模 c 看它们的位置差，至多经过 c 轮就会变成 0，即相遇。

这里不要求 fast 与 slow 在走动过程中停留的每个中间位置都比较，只需每轮移动完后比较即可。

### 两个边界条件的含义

```text
fast 非空：还有节点可读
fast.next 非空：还能安全执行 fast.next.next
```

`fast.next.next` 自身可以为 None，它会让下次循环自然结束，不需要额外要求它非空。

```python
class Solution:
    def hasCycle(self, head):
        slow = fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
```

**推演**：`1→2→3→2…` 中，两个指针进入 `2、3` 构成的环后会相遇。

| 轮数 | slow | fast |
|---:|---|---|
| 初始 | 节点 1 | 节点 1 |
| 1 | 节点 2 | 节点 3 |
| 2 | 节点 3 | 节点 3，相遇 |

初始两个引用相同不代表存在环，所以必须先移动再判断。单节点 `node.next=node` 则会在第一次移动后相遇，正确返回 True；单节点指向 None 返回 False。

题目示例里的 `pos` 只是平台用于构造环的说明，并不是函数参数，不能依赖它求解。

**易错点**：在移动之后判断相遇，否则两者初始都在头部会误判；读取 `fast.next.next` 前检查前两级引用。

时间 `O(n)`，辅助空间 `O(1)`。

## 142. 环形链表 II

题目：[环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)。返回入环的第一个节点。

### 题目描述

给定单链表 head，如果有环，返回从头开始走时第一次进入环的节点；如果没有环，返回 None。不能修改链表来标记或切断环。

**输入与约束**：链表可为空；结果是原节点引用，不是它的数值。pos 是平台构造环的说明，不是函数参数。进阶要求常数辅助空间。

```text
输入结构：A(3) → B(2) → C(0) → D(-4)，D.next 指向 B
输出：节点 B
解释：B 是从链表头出发进入环的第一个节点。

输入：1→2→None
输出：None
```

**思路与推导**：设头到入口距离 `a`，入口沿环到相遇点距离 `b`，环长 `c`。快慢指针相遇时，慢指针路程可写为 `a+b+q·c`，而它也等于若干圈长，因此 `a+b` 是 `c` 的倍数。从头和相遇点各走一步，再走 `a` 步就会在入口相遇。

### 第一阶段：找到环中的一个相遇点

先使用第 141 题的快慢指针。如果 fast 先到 None，就没有环，直接返回 None。如果相遇，只能说明找到了环内的某个节点，还不能直接返回它。

### 第二阶段：为什么“一个回头，两者同速”可以找到入口？

令慢指针走过的边数为 d，快指针走过 2d。相遇时，快指针比慢指针多走若干整圈：

$$
2d-d=kc,\qquad d=kc.
$$

慢指针路径也可以写成：

$$
d=a+b+qc.
$$

因此 `a+b` 是 c 的倍数，等价于：从相遇点沿环再走 a 步，就会回到环入口。

这时让 start 从 head 出发、slow 从相遇点出发，每次都走 1 步：

- start 走 a 步到达入口。
- slow 同样走 a 步，也到达入口，中途可能绕过整圈。

在 start 进入环之前，它位于环外，不可能与环内的 slow 相遇，所以二者第一次相遇就是入口。

```python
class Solution:
    def detectCycle(self, head):
        slow = fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                start = head
                while start is not slow:
                    start = start.next
                    slow = slow.next
                return start
        return None
```

**推演**：`3→2→0→-4→2…` 的入口是值为 2 的那个节点，首次相遇点不一定是它。

| 第一阶段轮数 | slow | fast |
|---:|---|---|
| 0 | 3 | 3 |
| 1 | 2 | 0 |
| 2 | 0 | 2 |
| 3 | -4 | -4，相遇 |

第二阶段：start 回到 3，slow 留在 -4；同时走一步，都到达入口 2。这个例子 `a=1、b=2、c=3`，确实满足 `a+b=3`。

如果入口就是 head，a=0，第二阶段开始时就已经相同，直接返回。这也是 `while start is not slow` 可以自然处理的边界。

**易错点**：首次相遇后，一个指针回到头，两者都改成每次一步；不要直接返回第一次相遇点。

时间 `O(n)`，辅助空间 `O(1)`。

## 21. 合并两个有序链表

题目：[合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)。将两个升序链表合并成升序链表。

### 题目描述

给定两个按非递减顺序排列的单链表 list1、list2，将两条链中的节点拼接成一条新的非递减链表，返回结果头节点。结果必须包含两条输入链的全部节点及重复值。

**输入与约束**：任意输入链都可以为空，原链无环；相等值可以来自同一条或不同链。题目要求通过拼接输入节点组成结果，不只是返回排好序的数值数组。

```text
输入：list1=1→2→4，list2=1→3→4
输出：1→1→2→3→4→4

输入：list1=None，list2=0
输出：原 list2 的头节点
```

**状态**：`tail` 是已合并部分的尾节点；每次取两个未处理头节点中较小的一个。

### 循环前应该怎样理解三个引用？

- `dummy.next` 找到结果链的开头。
- `tail` 指向已经选定的有序前缀的最后一个节点。
- `list1、list2` 分别指向两条输入链中尚未选取的部分。

```text
两条剩余链都不空时:
    比较两个头节点
    把较小的那个接在 tail 后
    仅移动被选中链表的头
    tail 移到刚接上的节点
某条链为空后，接上另一条剩余链
```

因为输入分别有序，每条剩余链的最小值都在头部，选择两个头中较小的，必然是当前所有剩余节点中的最小值。这证明已合并前缀始终有序。

```python
class Solution:
    def mergeTwoLists(self, list1, list2):
        dummy = ListNode()
        tail = dummy
        while list1 is not None and list2 is not None:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        tail.next = list1 if list1 is not None else list2
        return dummy.next
```

**推演**：`1→3` 与 `2→4`，依次取 1、2、3，最后接上 4。两个当前头就是各自剩余部分最小值，选更小者不会破坏顺序。

| 操作 | 已确定的结果前缀 | list1 剩余 | list2 剩余 |
|---|---|---|---|
| 初始 | 空 | `1→3` | `2→4` |
| 选择 1 | 1 | 3 | `2→4` |
| 选择 2 | `1→2` | 3 | 4 |
| 选择 3 | `1→2→3` | 空 | 4 |
| 接上剩余链 | `1→2→3→4` | 空 | 空 |

表中“已确定前缀”只到 tail；复用节点时，tail.next 可能暂时仍保留原链的后继，后续连接会覆盖它，所以不要把此时从 dummy 能走到的所有节点都认为已正式合并完。

本题按两条独立输入链处理。允许值重复，遇到相等时取任意一边都能保持有序；代码使用 `<=`，优先取第一条链中的节点。

**易错点**：每接一个节点都移动 `tail`；最后可以整段接上剩余链表。实现复用并重新连接输入节点。

时间 `O(m+n)`，辅助空间 `O(1)`。

## 2. 两数相加

题目：[两数相加](https://leetcode.cn/problems/add-two-numbers/)。数字按低位到高位保存在链表中，返回相加后的链表。

### 题目描述

给定两个非空单链表 l1、l2，分别表示两个非负整数。每个节点只保存一位数字，个位在头，后面依次是十位、百位等。将两数相加，按相同的低位在前规则返回结果链表。

**输入与约束**：每个节点值为 0 到 9；除数字 0 本身外，输入整数没有前导零，即链表末尾不会是无意义的高位 0。两条链长度可以不同，结果可能多一位。

```text
输入：l1=[2,4,3]，l2=[5,6,4]
输出：[7,0,8]
解释：342+465=807，结果也按逆序保存。

输入：l1=[9,9]，l2=[1]
输出：[0,0,1]，表示 99+1=100。
```

**状态**：`carry` 是上一位产生的进位。当前位等于总和 `% 10`，下一位进位等于总和 `// 10`。

### 这是一遍从低位到高位的竖式加法

题目把个位放在头部，正好可以沿 next 按照加法的计算顺序处理，不必先反转链表或拼成一个大整数。

设当前两位为 x、y，缺失的一位视为 0：

$$
total=x+y+carry,\qquad digit=total\bmod10,\qquad new\_carry=\lfloor total/10\rfloor.
$$

`digit` 是当前结果节点值，`new_carry` 留给下一轮。`divmod(total,10)` 一次返回商与余数，因此代码写成 `carry, digit = divmod(total,10)`。

```text
只要任意一条输入链还有节点，或者仍有进位:
    读取本位的两个数字与旧进位
    算出本位结果与新进位
    创建一个结果节点，接到尾部
返回结果链头
```

```python
class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode()
        tail = dummy
        carry = 0
        while l1 is not None or l2 is not None or carry:
            total = carry
            if l1 is not None:
                total += l1.val
                l1 = l1.next
            if l2 is not None:
                total += l2.val
                l2 = l2.next
            carry, digit = divmod(total, 10)
            tail.next = ListNode(digit)
            tail = tail.next
        return dummy.next
```

**推演**：`[2,4,3] + [5,6,4]` 表示 `342+465`：个位 7，十位 10 写 0 进 1，百位 8，得到 `[7,0,8]`。

| 位数 | x | y | 输入进位 | total | 写入 digit | 输出进位 |
|---|---:|---:|---:|---:|---:|---:|
| 个位 | 2 | 5 | 0 | 7 | 7 | 0 |
| 十位 | 4 | 6 | 0 | 10 | 0 | 1 |
| 百位 | 3 | 4 | 1 | 8 | 8 | 0 |

再看 `[9,9]+[1]`：先写 0 进 1，再写 0 进 1；两条输入都结束后，还必须额外创建值为 1 的节点，得到 `[0,0,1]`。

这里创建新的结果节点，只移动输入的局部引用，原链表值和连接都不会被修改。

**易错点**：循环条件包含 `carry`，否则 `9+1` 的最高位会丢失；较短链表结束后当作该位为 0。

时间 `O(max(m,n))`；除新结果节点外辅助空间 `O(1)`，结果空间 `O(max(m,n))`。

## 19. 删除链表的倒数第 N 个结点

题目：[删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end/)。题目保证 `n` 合法。

### 题目描述

给定单链表头节点 head 和正整数 n，删除倒数第 n 个节点，返回删除后的头节点。倒数第 1 个是尾节点。

**输入与约束**：链表非空且无环，`1<=n<=链表长度`；可能删除头、尾或唯一节点。题目进阶希望用一趟扫描完成。

```text
输入：head=1→2→3→4→5，n=2
输出：1→2→3→5
解释：倒数第 2 个节点是 4。

输入：head=1，n=1
输出：None
```

**状态与思路**：两个指针从 dummy 出发，先让 `fast` 走 `n` 步，再一起走，直到 `fast` 是最后一个节点，此时 `slow` 就是待删节点的前驱。

### 为什么要找前驱，而不是只找待删节点？

删除需要修改指向待删节点的那条边：

```text
slow → 要删除的节点 → after
改成 slow → after
```

所以我们要让 slow 停在目标前面。设真实链表长度为 L，并把 dummy 视为位置 0，真实节点位置为 1..L。倒数第 n 个节点的位置是 `L-n+1`，它的前驱位置是 `L-n`。

让 fast 始终领先 slow n 条边，最终 fast 停在位置 L，slow 就停在 `L-n`，恰好符合要求。

### 操作流程

```text
两个指针都从 dummy 出发
fast 先走 n 步，建立间隔
只要 fast 后面还有节点:
    fast 与 slow 都走一步
用 slow.next = slow.next.next 跳过目标
返回 dummy.next
```

这份代码选择“领先 n 步、fast 停在尾节点”。也有“领先 n+1 步、fast 停在 None”的写法，但两个条件必须配套，不能混用。

```python
class Solution:
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0, head)
        slow = fast = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next is not None:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy.next
```

**推演**：`1→2→3→4→5`，`n=2`，最终 fast 在 5、slow 在 3，跳过 4。

| 阶段 | slow | fast |
|---|---|---|
| 初始 | dummy | dummy |
| fast 先走 2 步 | dummy | 2 |
| 同时走第 1 步 | 1 | 3 |
| 同时走第 2 步 | 2 | 4 |
| 同时走第 3 步 | 3 | 5，后继为空，停止 |

最后把 `3.next` 从 4 改成 5。若 n=L，fast 预先走到尾部，slow 仍为 dummy，于是删除 head；若只有一个节点且 n=1，结果自然为 None。

**易错点**：目标是找到前驱；使用 dummy 后，删除头节点无需特殊分支。

时间 `O(length)`，辅助空间 `O(1)`；修改输入连接。

## 24. 两两交换链表中的节点

题目：[两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)。交换节点本身，不只是交换值。

### 题目描述

给定单链表 head，每两个相邻节点为一组，交换每组中两个节点的位置，返回新头。分组从头开始，若最后只剩一个节点，则保持原位。

**输入与约束**：链表无环，可以为空。必须改变节点连接，不能仅交换节点内的值来代替节点交换。

```text
输入：head=1→2→3→4
输出：2→1→4→3

输入：head=1→2→3
输出：2→1→3，末尾单节点不变。
```

**状态**：`prev` 是下一对节点前面的节点，将 `prev→a→b→rest` 改为 `prev→b→a→rest`。

### 一对节点要改哪些连接？

```text
修改前：prev → a → b → rest
修改后：prev → b → a → rest
```

需要三条边：

1. `a.next=b.next`：a 接到这一对之后的剩余链。
2. `b.next=a`：b 接到 a 前面。
3. `prev.next=b`：之前的链表接到新组头 b。

必须先保存 a、b 引用，并在覆盖 `b.next` 前通过它找到 rest。否则可能丢失后续链表，或者错误形成 `a↔b` 的环。

### 循环继续时 prev 应该在哪？

交换后 a 是这一对的尾部，它的后面才是下一对。所以执行 `prev=a`，不应该移动到 b。

```python
class Solution:
    def swapPairs(self, head):
        dummy = ListNode(0, head)
        prev = dummy
        while prev.next is not None and prev.next.next is not None:
            a = prev.next
            b = a.next
            a.next = b.next
            b.next = a
            prev.next = b
            prev = a
        return dummy.next
```

**推演**：`1→2→3→4` 先得到 `2→1→3→4`，再得到 `2→1→4→3`。

```text
第 1 轮：prev=dummy，a=1，b=2
完成后：dummy → 2 → 1 → 3 → 4，prev=1

第 2 轮：prev=1，a=3，b=4
完成后：dummy → 2 → 1 → 4 → 3，prev=3
```

循环检查 `prev.next` 与 `prev.next.next` 都存在，保证当前至少有两个节点。余下零个或一个节点时直接结束，奇数长度的最后一个自然保留。

正确性可以按组理解：每轮只交换当前两个节点，已处理前缀保持不变，后续节点顺序也未改变，直到所有完整的二节点组都被处理。

**易错点**：交换后这一对的尾巴是 a，所以下一轮 `prev=a`；奇数长度最后一个节点保持不动。

时间 `O(n)`，辅助空间 `O(1)`。

## 25. K 个一组翻转链表

题目：[K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)。每 k 个节点反转，不足 k 个的末尾保持原样。

### 题目描述

给定单链表 head 和正整数 k，从头开始按每 k 个节点分组，分别将完整组内部的节点顺序反转，返回结果头节点。如果末尾剩余节点数不足 k，不反转这一段。

**输入与约束**：链表非空且无环，`1<=k<=链表长度`；只能改变节点连接，不能只修改节点值。进阶要求 `O(1)` 辅助空间。

```text
输入：head=1→2→3→4→5，k=2
输出：2→1→4→3→5

相同链表，k=3
输出：3→2→1→4→5，最后两个节点不足一组，保持原样。
```

**状态与思路**：`group_prev` 指向当前组前驱。先确认第 k 个节点存在，再反转半开区间 `[group_start, group_next)`，最后接回前后两段。

### 先给每个指针分工

| 指针 | 含义 |
|---|---|
| `group_prev` | 当前组之前的节点，已完成部分的尾部 |
| `kth` | 当前组第 k 个节点，反转后成为组头 |
| `group_next` | 下一组的第一个节点，也是当前反转的结束边界 |
| `old_start` | 当前组原来的头，反转后成为组尾 |
| `cur、prev` | 局部反转过程中的待处理节点与已反转部分头 |

组结构如下：

```text
group_prev → old_start → ... → kth → group_next → ...
```

反转后应该是：

```text
group_prev → kth → ... → old_start → group_next → ...
```

### 为什么分成“检查、反转、接回”三步？

1. **检查数量**：从 group_prev 向后走 k 步。若不足 k 个，必须原样保留，直接返回。
2. **局部反转**：只处理从 old_start 开始、到 group_next 之前的节点。
3. **重新连接**：group_prev 接到 kth，下一轮 group_prev 改为 old_start。

先确认完整组再修改，可以避免反转了一半才发现数量不够、还要再恢复的问题。

### 为什么 `prev` 初始为 `group_next`？

普通整链反转用 `prev=None`，因为反转后的尾部应指向空。本题的当前组后面还有链表，所以原组头反转成组尾时，应当指向 group_next。

把 prev 初值设成 group_next，第一次执行 `cur.next=prev` 就已经接好了这条尾部连接。

```python
class Solution:
    def reverseKGroup(self, head, k):
        dummy = ListNode(0, head)
        group_prev = dummy
        while True:
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if kth is None:
                    return dummy.next
            group_next = kth.next
            old_start = group_prev.next
            prev, cur = group_next, old_start
            while cur is not group_next:
                nxt = cur.next
                cur.next = prev
                prev, cur = cur, nxt
            group_prev.next = kth
            group_prev = old_start
```

**推演**：`1→2→3→4→5`，`k=2`：反转 `[1,2]`，再反转 `[3,4]`，5 不足一组，结果 `2→1→4→3→5`。

再用 k=3 看一组内部的变化，原链为 `1→2→3→4→5`：

```text
group_prev=dummy，old_start=1，kth=3，group_next=4
```

| 局部阶段 | prev 引用 | cur 引用 | 本步改动 |
|---|---|---|---|
| 初始 | 4 | 1 | 尚未反转 |
| 处理 1 后 | 1 | 2 | `1.next=4` |
| 处理 2 后 | 2 | 3 | `2.next=1` |
| 处理 3 后 | 3 | 4 | `3.next=2` |

cur 到 group_next 时停止，再让 dummy.next 指向 3，得到 `3→2→1→4→5`。下一组只剩 4、5，数量不足 3，保留原样。

group_next 必须在反转前保存，并用节点身份作为停止条件。遍历前缀、检查组长、反转虽然看起来有多层循环，但每个节点只被检查和反转常数次，总时间仍为线性。

**易错点**：先检查组长再动箭头；原组头反转后成为组尾；把初始 `prev` 设为 `group_next` 可以在反转时自然接好后半段。原题 `k>=1`。

时间 `O(n)`，辅助空间 `O(1)`。

## 138. 随机链表的复制

题目：[随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)。每个节点额外有一个 `random`，指向任意节点或空，要求深拷贝。

### 题目描述

给定随机链表头节点 head，每个节点包含 val、next、random。next 构成普通单链表，random 可以指向链表内任意节点，包括自身，也可以为 None。创建并返回整条链表的深拷贝。

**输入与约束**：链表可为空；新旧对应节点值相同，next/random 的对应关系相同，但所有新节点必须独立创建，任何新指针都不能指向旧链表节点。函数输入是头引用。

```text
原结构：A(7)→B(13)→None；A.random=None，B.random=A
输出：A'(7)→B'(13)→None；A'.random=None，B'.random=A'
解释：A'、B' 都是新对象，不能让 B'.random 指回 A。

输入：head=None
输出：None
```

平台常用 `[val,random_index]` 列表表示节点，random_index 为 None 表示空引用；这些下标只是序列化形式，不是节点内原本拥有的字段。

**状态与思路**：字典保存“旧节点 → 新节点”。先建立所有新节点，再连接 `next` 和 `random`，解决指针可能指向尚未遍历节点的问题。

### 深拷贝要求哪些信息一致？

新链表要满足：值相同、next 关系相同、random 关系相同，但所有真实节点都必须是新对象，不能有任何指针指回旧链表。

```text
旧链：A → B → C
      A.random = C
      B.random = A

新链：A' → B' → C'
      A'.random = C'
      B'.random = A'
```

如果只复制 val 和 next，把 random 直接指向旧节点，就不是深拷贝。

### 为什么两次遍历更容易理解？

第一遍只负责“创建对应物”，不急着连接：

```text
copies[A] = A'
copies[B] = B'
copies[C] = C'
```

第二遍，任何旧目标节点都已经有新对应物，可以统一写：

```text
新节点.next   = copies[旧节点.next]
新节点.random = copies[旧节点.random]
```

`copies[None]=None` 让空指针也服从同一映射规则，避免每条连接都写特殊分支。

```python
class Solution:
    def copyRandomList(self, head):
        copies = {None: None}
        cur = head
        while cur is not None:
            copies[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur is not None:
            copies[cur].next = copies[cur.next]
            copies[cur].random = copies[cur.random]
            cur = cur.next
        return copies[head]
```

**推演**：旧 A 的 random 指向旧 B，新 A 的 random 必须指向 `copies[B]`，不能指回旧 B。

| 当前旧节点 | 创建的新节点 | 新 next 应指向 | 新 random 应指向 |
|---|---|---|---|
| A | A' | `copies[B]=B'` | 若 A.random=C，则指向 C' |
| B | B' | `copies[C]=C'` | 若 B.random=A，则指向 A' |
| C | C' | None | 按 C.random 的映射决定 |

值重复不影响映射：即使 A.val 与 B.val 相同，它们仍是两个不同的键。本文提供的 Node 使用默认对象身份语义，可作为字典键。

random 可以指向自身或与其他节点构成环，算法也不会无限循环，因为遍历只沿原题中无环的 next 链推进，不沿 random 递归搜索。

**易错点**：字典键是节点身份，不是节点值，多个节点可能值相同；`random` 可以指向自身。代码不会修改原链表。

时间平均 `O(n)`，辅助字典 `O(n)`，另有 `O(n)` 新节点。穿插复制节点可以减少辅助空间，但入门先掌握映射版本。

## 148. 排序链表

题目：[排序链表](https://leetcode.cn/problems/sort-list/)。升序排列链表，使用归并排序。

### 题目描述

给定单链表 head，将所有节点按值非递减排列，返回排序后的头节点。需要保留全部元素，包括重复值。

**输入与约束**：链表可为空，节点值可为负数。进阶要求 `O(n log n)` 时间和 `O(1)` 辅助空间；本节自顶向下归并实现达到时间目标，但递归栈为 `O(log n)`，后文明确说明空间进阶的区别。

```text
输入：head=4→2→1→3
输出：1→2→3→4

输入：head=-1→5→3→4→0
输出：-1→0→3→4→5
```

**函数定义**：`sortList(head)` 返回这条链表排序后的新头。找到中点并断开，递归排序两半，再合并。

### 为什么归并排序适合链表？

链表不方便按下标随机访问，但适合沿 next 顺序扫描、拆段和连接。归并排序只需要：

1. 用快慢指针找到中点。
2. 将链表断成两个规模更小的子问题。
3. 分别排序后，用第 21 题的方法合并两条有序链。

### 递归函数应该怎样理解？

相信 `sortList(left_head)` 能把左半段排好并返回它的新头，右半同理。当前层不需要手动管理子调用内部如何排序，只需要正确拆分并合并两个返回结果。

```text
SORT(head):
    若节点数为 0 或 1，直接返回 head
    找到前半段末尾 slow
    保存后半段头 second=slow.next
    slow.next=None，将左右真正断开
    a=SORT(head)
    b=SORT(second)
    合并 a、b，返回合并后的头
```

若不执行 `slow.next=None`，左递归仍可能收到原来的完整链，问题规模不减，递归无法正常结束。

```python
class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
        second = slow.next
        slow.next = None
        a = self.sortList(head)
        b = self.sortList(second)
        dummy = ListNode()
        tail = dummy
        while a is not None and b is not None:
            if a.val <= b.val:
                tail.next = a
                a = a.next
            else:
                tail.next = b
                b = b.next
            tail = tail.next
        tail.next = a if a is not None else b
        return dummy.next
```

**推演**：`4→2→1→3` 分为 `[4,2]` 和 `[1,3]`，分别排成 `[2,4]` 和 `[1,3]`，合并成 `[1,2,3,4]`。

```text
                    [4,2,1,3]
                   /         \
                [4,2]       [1,3]
                /   \       /   \
              [4]   [2]   [1]   [3]
                \   /       \   /
                [2,4]       [1,3]
                   \         /
                    [1,2,3,4]
```

注意这些方括号是链表片段的示意，不表示代码真的转换成 Python 列表。

每层合并的节点总数为 O(n)，近似二分后共 O(log n) 层，因此时间为 O(n log n)。递归的同一时刻只保留一条调用路径，辅助栈为 O(log n)，不是所有递归调用总数。

合并时使用 `<=` 优先选择左边，能保留相等值节点原来的相对次序，即稳定排序。实现重新连接原节点，没有创建一份长度为 n 的结果链。

**易错点**：必须断开两半，否则递归问题规模不减；`fast=head.next` 让长度 2 时也正确拆成 1+1。

时间 `O(n log n)`，递归辅助空间 `O(log n)`。这是入门的自顶向下版本；题目进阶要求的 `O(1)` 辅助空间需要自底向上、按段长倍增的迭代归并。

## 23. 合并 K 个升序链表

题目：[合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)。将多个有序链表合为一个。

### 题目描述

给定数组 lists，其中每个元素是一个按非递减顺序排列的单链表头引用。将所有链表合并成一条非递减链，返回新头，保留所有输入节点对应的元素及重复次数。

**输入与约束**：链表数组可以为空，数组中的某条链也可以为空。输入各链无环，题目按独立链表处理。k 是链表条数，总节点数与 k 不一定相同。

```text
输入：lists=[1→4→5, 1→3→4, 2→6]
输出：1→1→2→3→4→4→5→6

输入：lists=[] 或 lists=[None]
输出：None
```

**状态与思路**：最小堆保存每条链表尚未处理的头节点。全局最小值必然在这些头中。取出一个后，把它的后继放入堆。

### 为什么只把每条链的头放进堆？

每条输入链已经有序。假如某条链当前剩下 `2→5→9`，它的 5、9 都不可能先于 2 被取出，所以只让 2 参与全局比较就够了。

堆中最多保存每条非空剩余链的一个代表，总大小不超过 k，而不是把全部 N 个节点一次性放进堆。

### 堆中一个元素的三个字段

```text
(node.val, 唯一序号, node)
```

- 第一项用于按节点值排序。
- 第二项在值相等时提供可比较的唯一标识。
- 第三项保留真实节点引用，取出后可以接到结果链，并读取它的 next。

Python 比较元组会从左到右逐项比较。如果只存 `(val,node)`，两个 val 相同后，就会尝试比较两个 Node 对象；节点类没有定义大小关系，因此可能报错。

### 操作伪代码

```text
将每条非空链的头加入最小堆
while 堆非空:
    取出全局最小的代表节点 node
    如果 node 有后继，把后继放入堆，成为该链的新代表
    把 node 接到结果链尾部
返回结果链头
```

```python
import heapq
from itertools import count


class Solution:
    def mergeKLists(self, lists):
        serial = count()
        heap = [(node.val, next(serial), node) for node in lists if node is not None]
        heapq.heapify(heap)
        dummy = ListNode()
        tail = dummy
        while heap:
            _, _, node = heapq.heappop(heap)
            if node.next is not None:
                heapq.heappush(heap, (node.next.val, next(serial), node.next))
            tail.next = node
            tail = node
        return dummy.next
```

**推演**：`[1→4, 1→3, 2]`，堆依次给出 `1、1、2、3、4`。

用 A、B、C 标识三条链：

| 取出节点 | 新加入代表 | 剩余代表的值（展示集合，不代表堆数组顺序） | 结果前缀 |
|---|---|---|---|
| A 的 1 | A 的 4 | 1、2、4 | 1 |
| B 的 1 | B 的 3 | 2、3、4 | 1、1 |
| C 的 2 | 无 | 3、4 | 1、1、2 |
| B 的 3 | 无 | 4 | 1、1、2、3 |
| A 的 4 | 无 | 空 | 1、1、2、3、4 |

每次取出的值不大于任何剩余节点，结果因此一直有序；后继及时补入堆，保证每个节点最终都恰好取出一次。原题按独立的升序链表处理，复用这些输入节点构造结果。

**易错点**：元组里加入唯一序号，否则值相同时 Python 会尝试比较节点对象，产生异常。

设总节点数为 `N`，链表数为 `k`，时间 `O(k + N log(k+1))`，辅助空间 `O(k)`。复用输入节点。

## 146. LRU 缓存

题目：[LRU 缓存](https://leetcode.cn/problems/lru-cache/)。支持平均 `O(1)` 的查询和写入，容量满时淘汰最久未使用的键。

### 题目描述

设计类 `LRUCache`，构造时指定正整数容量 capacity。实现 `get(key)` 与 `put(key,value)`：查询存在键时返回值并刷新使用顺序，不存在返回 -1；写入时插入或更新该键，并标记为最近使用。若新增后超过容量，淘汰上次使用距离现在最久的键。

**输入与约束**：键和值为非负整数，0 是合法值；查询和写入都要求平均 `O(1)` 时间。同一个对象连续执行多次操作，更新已有键不增加容量，未命中的查询不改变顺序。

```text
操作：LRUCache(2), put(1,1), put(2,2), get(1), put(3,3), get(2)
对应结果：[None,None,None,1,None,-1]
解释：get(1) 使键 1 最近使用，插入 3 时淘汰键 2。
```

这里 None 表示构造/写入在平台操作结果中的空返回项。完整接口说明、状态图和执行表见本节后续内容。

这是一道**数据结构设计题**：需要让同一个对象在多次操作之间保存状态。不是给定一个数组求一次答案，而是实现一个支持 `get`、`put` 的缓存类。

> **哈希表负责快速找到节点，双向链表负责维护使用顺序；每次使用节点就把它移到最近端，超出容量就淘汰最久端。**

### 1. 题意与 LRU 规则

LRU 是 **Least Recently Used**，即“最近最少使用”。在本题中，它具体指：缓存装不下新数据时，删除距离上一次使用时间最久的键。

比较的是**上一次使用的先后顺序**，不是总访问次数。某个键以前访问过很多次，但之后很久没再使用，仍可能被淘汰。

```text
容量为 2：
put(1,10) → 放入键 1
put(2,20) → 放入键 2
get(1)    → 键 1 刚被使用，键 2 变成最久未使用
put(3,30) → 超出容量，淘汰键 2
```

#### 需要实现的接口

| 接口 | 含义 | 返回值 |
|---|---|---|
| `LRUCache(capacity)` | 创建容量为 capacity 的缓存 | 构造对象 |
| `get(key)` | 查询 key；存在时更新它的使用顺序 | 存在返回 value，否则返回 -1 |
| `put(key,value)` | 插入新键或更新已有键，并更新使用顺序 | 无返回值，即 Python 的 None |

原题容量为正整数。`get` 和 `put` 都要求平均 `O(1)` 时间。

#### 哪些操作算“使用”？

- `get` 命中：算使用，移到最近端。
- `get` 未命中：没有缓存节点被使用，顺序不变。
- `put` 插入新键：新节点位于最近端。
- `put` 更新已有键：更新值，并把已有节点移到最近端。

即使 `put` 写入的值与原值相同，也需要更新使用顺序。

### 2. 为什么使用哈希表加双向链表？

#### 把需求拆成四个操作

1. 根据 key 快速找到记录。
2. 把任意一条已有记录从原位置移走。
3. 把记录放到最近使用的位置。
4. 找到并删除最久未使用的记录。

如果只用列表保存顺序，查找或删除中间元素通常需要 `O(n)`。如果只保存 `key→value`，查询很快，但不会自动维护访问顺序；Python 字典的插入顺序也不会因为调用 get 自动变化。

| 结构 | 保存的信息 | 解决的问题 |
|---|---|---|
| 哈希表 `nodes` | `key → 节点对象` | 平均 `O(1)` 找到任意节点 |
| 双向链表 | 节点的使用先后顺序 | `O(1)` 摘除已知节点、追加节点和删除两端节点 |

```text
字典：nodes[2] 指向下面的节点 [2:20]
      nodes[1] 指向下面的节点 [1:10]

链表：left ↔ [2:20] ↔ [1:10] ↔ right
             最久       最近
```

字典保存的是链表中**同一个节点对象的引用**。移动节点只改变连接，字典不需要更新数组下标，也不需要重新查找节点位置。

#### 为什么单链表不够方便？

即使字典能直接找到 node，单链表仍缺少它的前驱，通常需要从头查找才能摘除。双向链表同时有 node.prev 与 node.next，立即就能让两边相连。

因此，“哈希表定位 + 双向链表摘除”能使任意节点的刷新保持平均常数时间。

### 3. 节点、哨兵与不变量

#### 每个真实节点保存四项信息

```text
node.key    缓存键
node.value  缓存值
node.prev   前驱
node.next   后继
```

节点需要保存 key，因为淘汰时从链表拿到节点后，还要用 `node.key` 删除字典记录。

#### 固定顺序约定

```text
left ↔ 最久未使用 ↔ …… ↔ 最近使用 ↔ right
```

- `left.next` 是最久未使用的真实节点。
- `right.prev` 是最近使用的真实节点。
- 刚插入或刚使用的节点放在 right 前面。

左右也可以反过来设计，但实现中必须保持一致。

#### 两个哨兵解决头尾边界

left、right 没有业务含义，不放入字典，不占容量。

```text
空缓存：      left ↔ right
一个节点：    left ↔ node ↔ right
多个节点：    left ↔ A ↔ B ↔ right
```

每个真实节点总有前驱和后继，所以删除第一个、最后一个或唯一一个节点都能用相同操作。

哨兵默认 key 即使为 0，也不会与真实 key=0 冲突：只有真实节点才加入 nodes，定位依据是字典和节点引用。

#### 每次公开操作结束后保持的条件

1. 每个缓存 key 在字典中恰好对应一个真实节点。
2. 字典中的节点与链表中的真实节点一一对应。
3. 链表从左到右按“最久到最近”排序。
4. 真实节点数等于 `len(nodes)`，且不超过 capacity。
5. 相邻节点双向连接一致：`a.next is b` 时，`b.prev is a`。

这些就是不变量。检查它们可以发现只看 get 返回值时不容易发现的断链、重复节点和错误顺序。

### 4. 操作伪代码

先实现三个链表小操作：

```text
REMOVE(node):
    让 node 的前驱和后继直接相连

APPEND_RECENT(node):
    把 node 插到 right 前面

MOVE_TO_RECENT(node):
    REMOVE(node)
    APPEND_RECENT(node)
```

REMOVE 只负责摘除连接，不负责删除字典。它既用于“移动”也用于“淘汰”，移动时字典必须保留。

```text
GET(key):
    如果 key 不存在，返回 -1
    node = nodes[key]
    MOVE_TO_RECENT(node)
    返回 node.value

PUT(key,value):
    如果 key 已存在:
        更新原节点的 value
        MOVE_TO_RECENT(原节点)
        返回

    创建新节点，加入字典和最近端
    如果超出容量:
        old = left.next
        从链表摘除 old
        从字典删除 old.key
```

“先插入，再判断超容”让逻辑更统一。一次 put 最多新增一个节点，所以超容时只需淘汰一个，不必循环删除。

### 5. 完整 Python 实现

提交时包含 `_Entry` 和 `LRUCache` 两个类即可，不必再包装为 Solution；它们也不依赖前文的 ListNode。

```python
class _Entry:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.nodes = {}

        # 两端哨兵；真实节点放在它们中间。
        self.left = _Entry()
        self.right = _Entry()
        self.left.next = self.right
        self.right.prev = self.left

    def _remove(self, node):
        """将真实节点从链表摘除，不修改字典。"""
        before = node.prev
        after = node.next
        before.next = after
        after.prev = before

    def _append_recent(self, node):
        """把新节点或已摘除节点插到最近使用端。"""
        last = self.right.prev
        last.next = node
        node.prev = last
        node.next = self.right
        self.right.prev = node

    def _move_to_recent(self, node):
        """刷新已有节点的使用顺序。"""
        self._remove(node)
        self._append_recent(node)

    def get(self, key: int) -> int:
        if key not in self.nodes:
            return -1

        node = self.nodes[key]
        self._move_to_recent(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.nodes:
            node = self.nodes[key]
            node.value = value
            self._move_to_recent(node)
            return

        node = _Entry(key, value)
        self.nodes[key] = node
        self._append_recent(node)

        if len(self.nodes) > self.capacity:
            oldest = self.left.next
            self._remove(oldest)
            del self.nodes[oldest.key]
```

以下划线开头的类、方法表示内部实现细节。调用方只需使用 LRUCache、get、put。

### 6. 指针操作逐步解释

#### `_remove(node)`：让左右邻居绕过当前节点

```text
原来：before ↔ node ↔ after
目标：before ↔ after

before.next = after
after.prev = before
```

只改 next、不改 prev，会导致正向看似正常、反向却走回旧节点。摘除不是销毁节点，也不改变 key、value：移动时随后重设它的两个连接，淘汰时再删除字典引用。

#### `_append_recent(node)`：补齐四个方向

```text
原来：last ↔ right
目标：last ↔ node ↔ right
```

| 赋值 | 作用 |
|---|---|
| `last.next=node` | 原末节点向后指向 node |
| `node.prev=last` | node 向前指向原末节点 |
| `node.next=right` | node 向后指向右哨兵 |
| `right.prev=node` | 右哨兵向前指向 node |

空链表时 last 就是 left，同样适用。

#### 为什么移动前要先摘除？

```text
原来：left ↔ A ↔ B ↔ C ↔ right
访问 B 后：left ↔ A ↔ C ↔ B ↔ right
```

需要先让 A 与 C 直接相连，再把 B 接到最近端。不先摘除就追加，会留下 A 指向 B 的旧连接，而 B 已经被移到末尾，从而破坏链表结构。

即使 node 本来就是最近节点，摘除再追加仍然正确。容量为 1 时，会短暂变成 `left↔right`，再把唯一节点接回来。

#### 更新已有 key 为什么不新建节点？

字典已经能直接找到原节点，修改 value 并移动即可，缓存大小不变。如果新建同 key 节点却忘了摘除旧节点，字典只有一条记录、链表却有两个同 key 节点，后面的淘汰就会出错。

### 7. 完整示例与可运行调用

表格顺序始终为“最久 → 最近”，省略哨兵。

| 操作 | 返回值 | 操作后的顺序 | 说明 |
|---|---|---|---|
| `LRUCache(2)` | 创建对象 | `[]` | 容量为 2 |
| `put(1,1)` | None | `[1:1]` | 插入 1 |
| `put(2,2)` | None | `[1:1,2:2]` | 2 最新使用 |
| `get(1)` | 1 | `[2:2,1:1]` | 1 移到最近端 |
| `put(3,3)` | None | `[1:1,3:3]` | 淘汰 2 |
| `get(2)` | -1 | `[1:1,3:3]` | 未命中，顺序不变 |
| `put(4,4)` | None | `[3:3,4:4]` | 淘汰 1 |
| `get(1)` | -1 | `[3:3,4:4]` | 1 已淘汰 |
| `get(3)` | 3 | `[4:4,3:3]` | 3 刷新使用顺序 |
| `get(4)` | 4 | `[3:3,4:4]` | 4 刷新使用顺序 |

先执行 LRU 类定义，再执行这个调用示例：

```python
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
```

再看更新已有键：

```text
容量为 2
put(1,10) → [1:10]
put(2,20) → [1:10,2:20]
put(1,99) → [2:20,1:99]    只是更新，大小仍为 2
put(3,30) → [1:99,3:30]    淘汰 2
get(1)    → 返回 99，并变为 [3:30,1:99]
```

若更新时只改值、不移动顺序，后面插入 3 就会错误淘汰刚更新的 1。

### 8. 正确性与复杂度

#### 使用顺序为什么始终正确？

假设操作前的链表顺序正确：get 未命中不改顺序；get 命中、put 更新或新增，都把刚使用的节点放在最近端，其他节点相对顺序不变；超容时删掉最久端，恰好淘汰上次使用最早的节点。

初始链表为空，自然满足顺序要求。因此上述规则保证每一步之后顺序都正确。

#### 字典与链表为什么不会脱节？

插入时两边都添加，移动时字典仍指向原节点，更新值不新增节点，淘汰时两边都删除。这些规则维持了字典与真实节点的一一对应。

| 操作 | 工作 | 时间 |
|---|---|---|
| get 未命中 | 哈希查询 | 平均 `O(1)` |
| get 命中 | 查询 + 固定次数指针修改 | 平均 `O(1)` |
| put 更新 | 查询、改值、移动 | 平均 `O(1)` |
| put 新增 | 添加，必要时淘汰一个节点 | 平均 `O(1)` |

空间为 `O(capacity)`。先插入再淘汰时最多暂存 `capacity+1` 个真实节点，不改变空间量级，哨兵只有固定两个。

哈希表的常数时间按平均情况计算；这些已知节点的双链表操作只需固定次数赋值。

### 9. 常见错误与自测

1. get 命中却不移动节点，使用顺序错误。
2. put 更新时只改值、不刷新顺序。
3. 淘汰时只改链表或只改字典，两种结构不一致。
4. 字典只存 value，无法快速定位链表节点。
5. 节点不存 key，淘汰后难以直接删除字典记录。
6. 用 value 的真假判断存在性，错误处理值为 0 的缓存。
7. 满容量时更新已有键也淘汰其他节点，而更新本来不增加大小。

建议至少手算：容量为 1、同 key 连续更新、反复 get 同一键、get 不存在键、key/value 为 0，以及淘汰后重新插入相同 key。

**先独立写出摘除和追加的箭头，再组合 get、put。一个表负责“它在哪里”，一条链负责“谁最久没用”。**

## 入门练习路线与检查清单

推荐先写 206 → 21 → 19 → 24，再学快慢指针 141 → 142 → 234，最后做分组反转、排序、复制和 LRU。

### 十四道题的核心状态对比

| 题目 | 最重要的状态或操作 | 是否改变输入结构 |
|---|---|---|
| 160 相交链表 | 两个指针交换起点，对齐公共尾部 | 否 |
| 206 反转链表 | 已反转部分 prev、未处理部分 cur | 是 |
| 234 回文链表 | 中点、反转后的后半段、比较结果 | 临时改变，返回前恢复 |
| 141 环形链表 | 快慢指针是否相遇 | 否 |
| 142 环形链表 II | 相遇点、头节点同速寻找入口 | 否 |
| 21 合并两条有序链表 | 两条剩余链头、结果尾部 | 是，复用输入节点 |
| 2 两数相加 | 当前两位、进位、结果尾部 | 否，创建新结果链 |
| 19 删除倒数节点 | 两指针固定间距、待删前驱 | 是 |
| 24 两两交换 | 当前组前驱、组内两个节点 | 是 |
| 25 K 组反转 | 组前驱、原组头、组尾、组后继 | 是 |
| 138 随机链表复制 | 旧节点到新节点的映射 | 否，创建新链 |
| 148 排序链表 | 二分断链、递归返回的新头、归并 | 是 |
| 23 合并 K 条有序链表 | 每条链一个堆代表、结果尾部 | 是，复用输入节点 |
| 146 LRU | key 到节点、双向使用顺序 | 修改缓存内部状态 |

### 每道题写完后检查什么？

每道改链题至少检查：空链表、单节点、两节点、是否删除头、是否保留尾、有没有意外形成环。检查结果时不仅看节点值，还要看节点身份和连接关系。

- 需要换头时，返回的是新的 head 还是原来的旧引用？
- 修改 next 前，是否保存了还要访问的后继？
- dummy 是否被误当成结果节点返回？
- 快慢指针的初始化和终止条件是否与目标位置配套？
- 题目是复用原节点、交换节点，还是要求深拷贝？
- 回文判断等临时修改是否在所有返回分支上都正确恢复？
- 递归是否真的缩小问题规模，空间复杂度是否计入调用栈？
- 双链表操作是否同步维护两个方向？

对反转、交换、排序等复用节点的题，除了比较值，还可以检查结果节点集合与原节点集合是否相同；对深拷贝题，则应检查两组真实节点没有交集。
