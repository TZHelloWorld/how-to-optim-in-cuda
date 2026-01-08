"""
常见断言方法测试用例

演示 unittest 中各类 assert* 方法的用法，包括：
- 基本相等/不等断言
- 布尔断言
- 身份断言 (is/is not)
- None 断言
- 成员断言 (in/not in)
- 类型断言
- 比较断言 (大于/小于/近似相等)
- 集合/序列断言
- 字符串/正则断言

参考: https://docs.python.org/3/library/unittest.html#assert-methods
"""

import unittest


class TestBasicAssertions(unittest.TestCase):
    """基本断言方法"""

    # ========================
    # 相等 / 不等
    # ========================
    def test_assertEqual(self):
        """assertEqual: 检查 a == b"""
        self.assertEqual(1 + 1, 2)
        self.assertEqual("hello", "hello")
        self.assertEqual([1, 2, 3], [1, 2, 3])

    def test_assertNotEqual(self):
        """assertNotEqual: 检查 a != b"""
        self.assertNotEqual(1, 2)
        self.assertNotEqual("hello", "world")

    # ========================
    # 布尔值
    # ========================
    def test_assertTrue(self):
        """assertTrue: 检查 bool(x) is True"""
        self.assertTrue(True)
        self.assertTrue(1)          # 非零整数为真
        self.assertTrue("non-empty")  # 非空字符串为真
        self.assertTrue([1])        # 非空列表为真

    def test_assertFalse(self):
        """assertFalse: 检查 bool(x) is False"""
        self.assertFalse(False)
        self.assertFalse(0)         # 零为假
        self.assertFalse("")        # 空字符串为假
        self.assertFalse([])        # 空列表为假
        self.assertFalse(None)      # None 为假

    # ========================
    # 身份 (is / is not)
    # ========================
    def test_assertIs(self):
        """assertIs: 检查 a is b (同一对象)"""
        a = [1, 2, 3]
        b = a  # b 指向同一个对象
        self.assertIs(a, b)
        self.assertIs(True, True)
        self.assertIs(None, None)

    def test_assertIsNot(self):
        """assertIsNot: 检查 a is not b (不是同一对象)"""
        a = [1, 2, 3]
        b = [1, 2, 3]  # b 是不同对象，虽然值相等
        self.assertIsNot(a, b)
        # 注意: 1.0 和 1 虽然 == 但不是同一对象
        self.assertIsNot(1.0, 1)

    # ========================
    # None 检查
    # ========================
    def test_assertIsNone(self):
        """assertIsNone: 检查 x is None"""
        self.assertIsNone(None)
        result = {}.get("missing_key")
        self.assertIsNone(result)

    def test_assertIsNotNone(self):
        """assertIsNotNone: 检查 x is not None"""
        self.assertIsNotNone(0)     # 0 不是 None
        self.assertIsNotNone("")    # 空字符串不是 None
        self.assertIsNotNone(False) # False 不是 None

    # ========================
    # 成员关系 (in / not in)
    # ========================
    def test_assertIn(self):
        """assertIn: 检查 a in b"""
        self.assertIn(3, [1, 2, 3, 4])
        self.assertIn("hello", "hello world")
        self.assertIn("key", {"key": "value"})

    def test_assertNotIn(self):
        """assertNotIn: 检查 a not in b"""
        self.assertNotIn(5, [1, 2, 3])
        self.assertNotIn("xyz", "hello world")

    # ========================
    # 类型检查
    # ========================
    def test_assertIsInstance(self):
        """assertIsInstance: 检查 isinstance(a, b)"""
        self.assertIsInstance(42, int)
        self.assertIsInstance("hello", str)
        self.assertIsInstance([1, 2], list)
        self.assertIsInstance(3.14, (int, float))  # 可以传入类型元组

    def test_assertNotIsInstance(self):
        """assertNotIsInstance: 检查 not isinstance(a, b)"""
        self.assertNotIsInstance("hello", int)
        self.assertNotIsInstance(42, str)


class TestComparisonAssertions(unittest.TestCase):
    """比较断言方法"""

    def test_assertGreater(self):
        """assertGreater: 检查 a > b"""
        self.assertGreater(10, 5)
        self.assertGreater(3.14, 3.0)

    def test_assertGreaterEqual(self):
        """assertGreaterEqual: 检查 a >= b"""
        self.assertGreaterEqual(10, 10)
        self.assertGreaterEqual(10, 5)

    def test_assertLess(self):
        """assertLess: 检查 a < b"""
        self.assertLess(5, 10)
        self.assertLess(-1, 0)

    def test_assertLessEqual(self):
        """assertLessEqual: 检查 a <= b"""
        self.assertLessEqual(5, 5)
        self.assertLessEqual(5, 10)

    def test_assertAlmostEqual(self):
        """assertAlmostEqual: 检查 round(a-b, places) == 0, 默认 places=7"""
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)
        self.assertAlmostEqual(1.0000001, 1.0, places=5)

    def test_assertNotAlmostEqual(self):
        """assertNotAlmostEqual: 检查 round(a-b, places) != 0"""
        self.assertNotAlmostEqual(0.1, 0.2)
        self.assertNotAlmostEqual(1.0, 1.1, places=1)


class TestCollectionAssertions(unittest.TestCase):
    """集合/序列断言方法"""

    def test_assertListEqual(self):
        """assertListEqual: 检查两个列表相等"""
        self.assertListEqual([1, 2, 3], [1, 2, 3])

    def test_assertTupleEqual(self):
        """assertTupleEqual: 检查两个元组相等"""
        self.assertTupleEqual((1, 2, 3), (1, 2, 3))

    def test_assertSetEqual(self):
        """assertSetEqual: 检查两个集合相等"""
        self.assertSetEqual({1, 2, 3}, {3, 2, 1})

    def test_assertDictEqual(self):
        """assertDictEqual: 检查两个字典相等"""
        self.assertDictEqual(
            {"name": "Alice", "age": 30},
            {"age": 30, "name": "Alice"}
        )

    def test_assertSequenceEqual(self):
        """assertSequenceEqual: 检查两个序列相等"""
        self.assertSequenceEqual([1, 2, 3], [1, 2, 3])
        self.assertSequenceEqual("abc", "abc")

    def test_assertCountEqual(self):
        """assertCountEqual: 检查两个序列包含相同元素(不考虑顺序)"""
        self.assertCountEqual([1, 2, 3], [3, 1, 2])
        self.assertCountEqual("abc", "cba")


class TestStringAssertions(unittest.TestCase):
    """字符串/正则断言方法"""

    def test_assertRegex(self):
        """assertRegex: 检查 regex.search(text) 匹配"""
        self.assertRegex("hello world", r"hello")
        self.assertRegex("2024-01-01", r"\d{4}-\d{2}-\d{2}")

    def test_assertNotRegex(self):
        """assertNotRegex: 检查 regex.search(text) 不匹配"""
        self.assertNotRegex("hello", r"\d+")

    def test_assertMultiLineEqual(self):
        """assertMultiLineEqual: 多行字符串相等（提供更清晰的 diff 输出）"""
        text1 = "line1\nline2\nline3"
        text2 = "line1\nline2\nline3"
        self.assertMultiLineEqual(text1, text2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
