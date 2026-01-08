"""
参数化测试用例

使用 unittest 内置的 subTest 实现参数化测试。
参数化测试允许用不同的输入数据运行同一段测试逻辑，
而不需要为每组数据编写独立的测试方法。

参考: https://docs.python.org/3/library/unittest.html#distinguishing-test-iterations-using-subtests
"""

import unittest
import math


# ========================
# 被测函数
# ========================
def add(a, b):
    """加法"""
    return a + b


def multiply(a, b):
    """乘法"""
    return a * b


def is_palindrome(s):
    """判断字符串是否为回文"""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def fizzbuzz(n):
    """FizzBuzz: 能被3整除返回Fizz, 能被5整除返回Buzz, 都能整除返回FizzBuzz"""
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)


def celsius_to_fahrenheit(celsius):
    """摄氏度转华氏度"""
    return celsius * 9 / 5 + 32


class Stack:
    """简单的栈实现"""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# ========================
# 参数化测试
# ========================
class TestAddParameterized(unittest.TestCase):
    """参数化测试: 加法"""

    def test_add(self):
        """使用多组数据测试加法"""
        test_cases = [
            # (a, b, expected)
            (1, 2, 3),
            (0, 0, 0),
            (-1, 1, 0),
            (-1, -1, -2),
            (100, 200, 300),
            (0.1, 0.2, 0.3),
            (1e10, 1e10, 2e10),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                result = add(a, b)
                self.assertAlmostEqual(result, expected, places=10,
                                       msg=f"add({a}, {b}) = {result}, 期望 {expected}")


class TestMultiplyParameterized(unittest.TestCase):
    """参数化测试: 乘法"""

    def test_multiply(self):
        """使用多组数据测试乘法"""
        test_cases = [
            (2, 3, 6),
            (0, 100, 0),
            (-1, 5, -5),
            (-2, -3, 6),
            (0.5, 4, 2.0),
            (1, 1, 1),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(multiply(a, b), expected)

    def test_multiply_commutative(self):
        """验证乘法交换律: a * b == b * a"""
        pairs = [(3, 7), (0, 5), (-2, 4), (1.5, 3)]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(multiply(a, b), multiply(b, a))


class TestPalindromeParameterized(unittest.TestCase):
    """参数化测试: 回文判断"""

    def test_is_palindrome_true(self):
        """测试回文字符串"""
        palindromes = [
            "racecar",
            "level",
            "madam",
            "A man a plan a canal Panama",  # 忽略空格和大小写
            "Was it a car or a cat I saw",
            "",           # 空字符串也是回文
            "a",          # 单个字符也是回文
            "aa",
            "aba",
        ]

        for s in palindromes:
            with self.subTest(string=s):
                self.assertTrue(is_palindrome(s), f"'{s}' 应该是回文")

    def test_is_palindrome_false(self):
        """测试非回文字符串"""
        non_palindromes = [
            "hello",
            "python",
            "unittest",
            "ab",
            "abc",
        ]

        for s in non_palindromes:
            with self.subTest(string=s):
                self.assertFalse(is_palindrome(s), f"'{s}' 不应该是回文")


class TestFizzBuzzParameterized(unittest.TestCase):
    """参数化测试: FizzBuzz"""

    def test_fizzbuzz(self):
        """完整的 FizzBuzz 测试"""
        test_cases = {
            1: "1",
            2: "2",
            3: "Fizz",
            4: "4",
            5: "Buzz",
            6: "Fizz",
            7: "7",
            10: "Buzz",
            12: "Fizz",
            15: "FizzBuzz",
            30: "FizzBuzz",
            45: "FizzBuzz",
            98: "98",
            99: "Fizz",
            100: "Buzz",
        }

        for n, expected in test_cases.items():
            with self.subTest(n=n):
                self.assertEqual(fizzbuzz(n), expected)


class TestTemperatureConversion(unittest.TestCase):
    """参数化测试: 温度转换"""

    def test_celsius_to_fahrenheit(self):
        """摄氏度转华氏度"""
        conversions = [
            # (摄氏度, 华氏度)
            (0, 32),
            (100, 212),
            (-40, -40),     # -40 摄氏度 == -40 华氏度
            (37, 98.6),     # 体温
            (-273.15, -459.67),  # 绝对零度
        ]

        for celsius, expected_fahrenheit in conversions:
            with self.subTest(celsius=celsius):
                result = celsius_to_fahrenheit(celsius)
                self.assertAlmostEqual(result, expected_fahrenheit, places=2,
                                       msg=f"{celsius}°C 应该等于 {expected_fahrenheit}°F")


class TestMathFunctions(unittest.TestCase):
    """参数化测试: 数学函数"""

    def test_sqrt(self):
        """平方根"""
        test_cases = [
            (0, 0),
            (1, 1),
            (4, 2),
            (9, 3),
            (16, 4),
            (2, math.sqrt(2)),
        ]

        for value, expected in test_cases:
            with self.subTest(value=value):
                self.assertAlmostEqual(math.sqrt(value), expected, places=10)

    def test_pow(self):
        """幂运算"""
        test_cases = [
            (2, 0, 1),
            (2, 1, 2),
            (2, 10, 1024),
            (3, 3, 27),
            (10, 3, 1000),
            (0, 0, 1),     # 0^0 = 1 (Python 约定)
        ]

        for base, exp, expected in test_cases:
            with self.subTest(base=base, exp=exp):
                self.assertEqual(pow(base, exp), expected)


class TestStackParameterized(unittest.TestCase):
    """参数化测试: 栈操作"""

    def test_push_and_pop(self):
        """测试 push 后 pop 得到正确的值"""
        items = [1, "hello", 3.14, None, True, [1, 2], {"key": "value"}]

        for item in items:
            with self.subTest(item=item):
                stack = Stack()
                stack.push(item)
                self.assertEqual(stack.pop(), item)
                self.assertTrue(stack.is_empty())

    def test_push_multiple_and_pop_order(self):
        """测试多次 push 后 pop 的顺序 (LIFO)"""
        sequences = [
            [1, 2, 3],
            ["a", "b", "c", "d"],
            [True, False, None],
        ]

        for seq in sequences:
            with self.subTest(sequence=seq):
                stack = Stack()
                for item in seq:
                    stack.push(item)

                # pop 顺序应该是反序
                for expected in reversed(seq):
                    self.assertEqual(stack.pop(), expected)

                self.assertTrue(stack.is_empty())

    def test_size_after_operations(self):
        """测试各种操作后的栈大小"""
        operations_and_expected_sizes = [
            # (操作序列, 预期最终大小)
            (["push"] * 5, 5),
            (["push"] * 3 + ["pop"] * 2, 1),
            (["push", "pop"] * 4, 0),
            (["push"] * 10 + ["pop"] * 10, 0),
        ]

        for ops, expected_size in operations_and_expected_sizes:
            with self.subTest(ops=ops, expected_size=expected_size):
                stack = Stack()
                push_counter = 0
                for op in ops:
                    if op == "push":
                        stack.push(push_counter)
                        push_counter += 1
                    elif op == "pop":
                        stack.pop()
                self.assertEqual(stack.size(), expected_size)


if __name__ == '__main__':
    unittest.main(verbosity=2)
