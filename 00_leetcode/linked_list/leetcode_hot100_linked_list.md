# Hot100 链表：先理解节点与引用，再练指针操作

[返回总索引](../README.md)

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

## 基础三：四个高频技巧

1. **虚拟头节点 dummy**：让删除头节点和删除其他节点使用同一套逻辑，最后返回 `dummy.next`。
2. **保存后继**：修改 `cur.next` 前，先保存原来的 `nxt`，否则可能找不到后半段。
3. **快慢指针**：快指针走两步，慢指针走一步，可以找中点或判断环。
4. **画箭头**：每次修改的不是值，而是边。先画 `prev → cur → nxt`，再写赋值。

本专题按“遍历判断 → 基本改链 → 分段与排序 → 复杂结构”组织，涵盖 14 题。

## 160. 相交链表

题目：[相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)。返回两条无环链表的第一个公共节点，无交点返回 `None`。

**状态与思路**：指针 `a` 走完 A 后走 B，`b` 走完 B 后走 A。两者走过的总路径长度相同，抵消了不同长度的前缀。

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

**易错点**：两条链都含值 8 不代表相交，必须引用同一个节点。

时间 `O(m+n)`，辅助空间 `O(1)`。

## 206. 反转链表

题目：[反转链表](https://leetcode.cn/problems/reverse-linked-list/)。将链表箭头全部反向。

**状态定义**：`prev` 是已经反转部分的头；`cur` 是尚未处理部分的头。

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

**易错点**：先保存 `nxt`，再修改箭头；循环结束时 `cur` 为 `None`，新头是 `prev`。该实现修改原链表。

时间 `O(n)`，辅助空间 `O(1)`。

## 234. 回文链表

题目：[回文链表](https://leetcode.cn/problems/palindrome-linked-list/)。判断节点值序列是否正反相同。

**思路**：快慢指针找到前半段末尾，反转后半段，再逐个比较；最后恢复后半段，避免仅做判断却改变输入结构。

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

**易错点**：发现不相等也要先恢复链表再返回；比较到短的后半段结束即可。

时间 `O(n)`，辅助空间 `O(1)`。

## 141. 环形链表

题目：[环形链表](https://leetcode.cn/problems/linked-list-cycle/)。判断沿 `next` 是否会重复到达同一个节点。

**思路**：快慢指针。无环时快指针先到末尾；有环时，快指针在环上每轮相对慢指针多走一步，最终追上。

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

**易错点**：在移动之后判断相遇，否则两者初始都在头部会误判；读取 `fast.next.next` 前检查前两级引用。

时间 `O(n)`，辅助空间 `O(1)`。

## 142. 环形链表 II

题目：[环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)。返回入环的第一个节点。

**思路与推导**：设头到入口距离 `a`，入口沿环到相遇点距离 `b`，环长 `c`。快慢指针相遇时，慢指针路程可写为 `a+b+q·c`，而它也等于若干圈长，因此 `a+b` 是 `c` 的倍数。从头和相遇点各走一步，再走 `a` 步就会在入口相遇。

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

**易错点**：首次相遇后，一个指针回到头，两者都改成每次一步；不要直接返回第一次相遇点。

时间 `O(n)`，辅助空间 `O(1)`。

## 21. 合并两个有序链表

题目：[合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)。将两个升序链表合并成升序链表。

**状态**：`tail` 是已合并部分的尾节点；每次取两个未处理头节点中较小的一个。

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

**易错点**：每接一个节点都移动 `tail`；最后可以整段接上剩余链表。实现复用并重新连接输入节点。

时间 `O(m+n)`，辅助空间 `O(1)`。

## 2. 两数相加

题目：[两数相加](https://leetcode.cn/problems/add-two-numbers/)。数字按低位到高位保存在链表中，返回相加后的链表。

**状态**：`carry` 是上一位产生的进位。当前位等于总和 `% 10`，下一位进位等于总和 `// 10`。

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

**易错点**：循环条件包含 `carry`，否则 `9+1` 的最高位会丢失；较短链表结束后当作该位为 0。

时间 `O(max(m,n))`；除新结果节点外辅助空间 `O(1)`，结果空间 `O(max(m,n))`。

## 19. 删除链表的倒数第 N 个结点

题目：[删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end/)。题目保证 `n` 合法。

**状态与思路**：两个指针从 dummy 出发，先让 `fast` 走 `n` 步，再一起走，直到 `fast` 是最后一个节点，此时 `slow` 就是待删节点的前驱。

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

**易错点**：目标是找到前驱；使用 dummy 后，删除头节点无需特殊分支。

时间 `O(length)`，辅助空间 `O(1)`；修改输入连接。

## 24. 两两交换链表中的节点

题目：[两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)。交换节点本身，不只是交换值。

**状态**：`prev` 是下一对节点前面的节点，将 `prev→a→b→rest` 改为 `prev→b→a→rest`。

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

**易错点**：交换后这一对的尾巴是 a，所以下一轮 `prev=a`；奇数长度最后一个节点保持不动。

时间 `O(n)`，辅助空间 `O(1)`。

## 25. K 个一组翻转链表

题目：[K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)。每 k 个节点反转，不足 k 个的末尾保持原样。

**状态与思路**：`group_prev` 指向当前组前驱。先确认第 k 个节点存在，再反转半开区间 `[group_start, group_next)`，最后接回前后两段。

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

**易错点**：先检查组长再动箭头；原组头反转后成为组尾；把初始 `prev` 设为 `group_next` 可以在反转时自然接好后半段。原题 `k>=1`。

时间 `O(n)`，辅助空间 `O(1)`。

## 138. 随机链表的复制

题目：[随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)。每个节点额外有一个 `random`，指向任意节点或空，要求深拷贝。

**状态与思路**：字典保存“旧节点 → 新节点”。先建立所有新节点，再连接 `next` 和 `random`，解决指针可能指向尚未遍历节点的问题。

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

**易错点**：字典键是节点身份，不是节点值，多个节点可能值相同；`random` 可以指向自身。代码不会修改原链表。

时间平均 `O(n)`，辅助字典 `O(n)`，另有 `O(n)` 新节点。穿插复制节点可以减少辅助空间，但入门先掌握映射版本。

## 148. 排序链表

题目：[排序链表](https://leetcode.cn/problems/sort-list/)。升序排列链表，使用归并排序。

**函数定义**：`sortList(head)` 返回这条链表排序后的新头。找到中点并断开，递归排序两半，再合并。

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
                tail.next, a = a, a.next
            else:
                tail.next, b = b, b.next
            tail = tail.next
        tail.next = a if a is not None else b
        return dummy.next
```

**推演**：`4→2→1→3` 分为 `[4,2]` 和 `[1,3]`，分别排成 `[2,4]` 和 `[1,3]`，合并成 `[1,2,3,4]`。

**易错点**：必须断开两半，否则递归问题规模不减；`fast=head.next` 让长度 2 时也正确拆成 1+1。

时间 `O(n log n)`，递归辅助空间 `O(log n)`。这是入门的自顶向下版本；题目进阶要求的 `O(1)` 辅助空间需要自底向上、按段长倍增的迭代归并。

## 23. 合并 K 个升序链表

题目：[合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)。将多个有序链表合为一个。

**状态与思路**：最小堆保存每条链表尚未处理的头节点。全局最小值必然在这些头中。取出一个后，把它的后继放入堆。

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

**易错点**：元组里加入唯一序号，否则值相同时 Python 会尝试比较节点对象，产生异常。

设总节点数为 `N`，链表数为 `k`，时间 `O(k + N log(k+1))`，辅助空间 `O(k)`。复用输入节点。

## 146. LRU 缓存

题目：[LRU 缓存](https://leetcode.cn/problems/lru-cache/)。支持平均 `O(1)` 的查询和写入，容量满时淘汰最久未使用的键。

### 为什么需要哈希表 + 双向链表？

- 字典负责根据 key 直接找到节点。
- 双链表负责在 `O(1)` 时间移除已知节点、把它放到最近使用的位置。
- 左端是最久未用，右端是最近使用。两个哨兵节点让头尾删除都不需要特殊分支。

```python
class _Entry:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.nodes = {}
        self.left = _Entry()
        self.right = _Entry()
        self.left.next = self.right
        self.right.prev = self.left

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append_recent(self, node):
        last = self.right.prev
        last.next = node
        node.prev = last
        node.next = self.right
        self.right.prev = node

    def get(self, key):
        if key not in self.nodes:
            return -1
        node = self.nodes[key]
        self._remove(node)
        self._append_recent(node)
        return node.value

    def put(self, key, value):
        if key in self.nodes:
            node = self.nodes[key]
            node.value = value
            self._remove(node)
        else:
            node = _Entry(key, value)
            self.nodes[key] = node
        self._append_recent(node)
        if len(self.nodes) > self.capacity:
            old = self.left.next
            self._remove(old)
            del self.nodes[old.key]
```

**推演**：容量 2，`put(1,1)`、`put(2,2)` 后顺序为 `[1,2]`；`get(1)` 后顺序 `[2,1]`；`put(3,3)` 淘汰 2，顺序 `[1,3]`。

**易错点**：查询成功也算使用；更新已有键既要改值也要移到最近位置；淘汰时同时删字典和链表。

每次操作平均 `O(1)`，空间 `O(capacity)`。

## 入门练习路线与检查清单

推荐先写 206 → 21 → 19 → 24，再学快慢指针 141 → 142 → 234，最后做分组反转、排序、复制和 LRU。

每道改链题至少检查：空链表、单节点、两节点、是否删除头、是否保留尾、有没有意外形成环。检查结果时不仅看节点值，还要看节点身份和连接关系。
