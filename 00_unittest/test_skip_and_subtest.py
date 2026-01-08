"""
跳过测试、预期失败、子测试 (subTest) 用例

演示 unittest 中控制测试执行的高级功能：
- @unittest.skip: 无条件跳过
- @unittest.skipIf: 条件跳过
- @unittest.skipUnless: 条件不满足则跳过
- self.skipTest(): 在运行时动态跳过
- @unittest.expectedFailure: 预期失败
- self.subTest(): 子测试/参数化断言

参考:
- https://docs.python.org/3/library/unittest.html#skipping-tests-and-expected-failures
- https://docs.python.org/3/library/unittest.html#distinguishing-test-iterations-using-subtests
"""

import unittest
import sys
import os
import time


# ========================
# 跳过测试
# ========================
class TestSkipping(unittest.TestCase):
    """演示各种跳过测试的方式"""

    @unittest.skip("演示: 无条件跳过此测试")
    def test_unconditional_skip(self):
        """这个测试永远不会执行"""
        self.fail("不应该执行到这里")

    @unittest.skipIf(sys.version_info < (3, 6), "需要 Python 3.6 或更高版本")
    def test_skip_if_old_python(self):
        """仅在 Python 3.6+ 上运行"""
        # 使用 f-string (Python 3.6+ 特性)
        name = "unittest"
        self.assertEqual(f"hello {name}", "hello unittest")

    @unittest.skipUnless(sys.platform.startswith("linux"), "仅在 Linux 上运行")
    def test_linux_only(self):
        """仅在 Linux 平台上运行"""
        self.assertTrue(os.path.exists("/proc"))

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TESTS") == "1",
        "设置 RUN_SLOW_TESTS=1 环境变量以运行慢速测试"
    )
    def test_slow_operation(self):
        """耗时测试，默认跳过，需要设置环境变量才运行"""
        time.sleep(0.01)  # 模拟耗时操作
        self.assertTrue(True)

    def test_dynamic_skip(self):
        """运行时动态决定是否跳过"""
        # 模拟检查外部资源是否可用
        external_resource_available = False
        if not external_resource_available:
            self.skipTest("外部资源不可用，跳过此测试")
        # 如果资源可用，继续测试
        self.assertTrue(True)


@unittest.skip("演示: 跳过整个测试类")
class TestSkippedClass(unittest.TestCase):
    """整个类被跳过，其中所有测试方法都不会执行"""

    def test_method_1(self):
        self.fail("不会执行")

    def test_method_2(self):
        self.fail("不会执行")


# ========================
# 预期失败
# ========================
class TestExpectedFailure(unittest.TestCase):
    """演示预期失败"""

    @unittest.expectedFailure
    def test_known_bug(self):
        """已知的 bug，标记为预期失败"""
        # 这个测试"失败"了，但因为标记了 expectedFailure，所以算通过
        self.assertEqual(1, 0, "这是一个已知的 bug")

    @unittest.expectedFailure
    def test_unimplemented_feature(self):
        """尚未实现的功能"""
        raise NotImplementedError("功能待实现")


# ========================
# 子测试 (subTest)
# ========================
class TestSubTest(unittest.TestCase):
    """演示 subTest 的用法 — 让单个测试方法包含多个独立的子测试"""

    def test_is_even(self):
        """使用 subTest 测试多个偶数"""
        even_numbers = [0, 2, 4, 6, 8, 10, 100, -2, -4]
        for num in even_numbers:
            with self.subTest(num=num):
                self.assertEqual(num % 2, 0, f"{num} 应该是偶数")

    def test_string_operations(self):
        """使用 subTest 测试多种字符串操作"""
        test_cases = [
            ("hello", "HELLO", str.upper),
            ("HELLO", "hello", str.lower),
            ("  hello  ", "hello", str.strip),
            ("hello world", "Hello World", str.title),
        ]
        for input_str, expected, operation in test_cases:
            with self.subTest(input=input_str, operation=operation.__name__):
                self.assertEqual(operation(input_str), expected)

    def test_type_conversions(self):
        """使用 subTest 测试类型转换"""
        conversions = [
            ("42", int, 42),
            ("3.14", float, 3.14),
            ("True", bool, True),
            ("", bool, False),      # 空字符串的 bool 为 False
            ("hello", str, "hello"),
        ]
        for value, target_type, expected in conversions:
            with self.subTest(value=value, target_type=target_type.__name__):
                result = target_type(value)
                self.assertEqual(result, expected)

    def test_dict_keys(self):
        """使用 subTest 检查字典中是否包含所有必需的键"""
        config = {
            "host": "localhost",
            "port": 8080,
            "debug": True,
            "database": "mydb",
        }
        required_keys = ["host", "port", "debug", "database"]

        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, config, f"配置缺少必需的键: {key}")

    def test_subtest_with_msg(self):
        """subTest 可以传入 msg 参数用于标识"""
        data = {"a": 1, "b": 2, "c": 3}
        for key, value in data.items():
            with self.subTest(msg=f"检查键 '{key}' 的值"):
                self.assertIsInstance(value, int)
                self.assertGreater(value, 0)


class TestSubTestAdvanced(unittest.TestCase):
    """subTest 的高级用法"""

    def test_nested_subtests(self):
        """嵌套子测试"""
        operations = {
            "add": lambda a, b: a + b,
            "mul": lambda a, b: a * b,
        }
        test_data = [(2, 3), (0, 5), (10, 10)]

        expected_add = [5, 5, 20]
        expected_mul = [6, 0, 100]

        for i, (a, b) in enumerate(test_data):
            with self.subTest(operation="add", a=a, b=b):
                self.assertEqual(operations["add"](a, b), expected_add[i])
            with self.subTest(operation="mul", a=a, b=b):
                self.assertEqual(operations["mul"](a, b), expected_mul[i])

    def test_fibonacci(self):
        """使用 subTest 测试斐波那契数列"""

        def fibonacci(n):
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

        expected = {
            0: 0, 1: 1, 2: 1, 3: 2, 4: 3,
            5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55,
        }

        for n, expected_value in expected.items():
            with self.subTest(n=n):
                self.assertEqual(fibonacci(n), expected_value,
                                 f"fibonacci({n}) 应该等于 {expected_value}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
