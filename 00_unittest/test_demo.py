"""
基础入门测试用例

演示 unittest 的最小可用结构：
- 测试类继承 unittest.TestCase
- setUp / tearDown 在每个测试方法前后自动调用
- 常用断言方法 assertEqual / assertTrue / assertIs
- 用 @unittest.expectedFailure 标记"预期会失败"的演示用例

参考: https://docs.python.org/3/library/unittest.html
"""

import unittest


class TestFun(unittest.TestCase):

    def setUp(self):
        """每个测试方法执行前调用"""
        print(f"[Test Method] {self._testMethodName}: 测试前的准备工作", flush=True)

    def tearDown(self):
        """每个测试方法执行后调用（无论测试是否通过）"""
        print(f"[Test Method] {self._testMethodName}: 测试后的清理工作\n", flush=True)

    def test_placeholder_one(self):
        """功能测试 1：占位用例，观察 setUp/tearDown 的执行顺序"""
        print("test_placeholder_one exec...", flush=True)

    def test_placeholder_two(self):
        """功能测试 2：占位用例"""
        print("test_placeholder_two exec...", flush=True)

    def test_basic_assertions(self):
        """基本断言方法演示（全部可通过）"""
        a = 1.0
        b = 1

        self.assertEqual(a, b)       # 检查 a == b（1.0 == 1 为 True）
        self.assertTrue(bool(b))     # 检查表达式为 True；assertFalse(x) 检查为 False
        self.assertIsNot(a, b)       # a 与 b 不是同一个对象（float 与 int）

    @unittest.expectedFailure
    def test_expected_failure_demo(self):
        """演示失败断言的规范写法：用 expectedFailure 标记，整体结果仍为通过

        注意：assertIs 检查的是"同一个对象"（is），1.0 与 1 虽然相等（==）
        但不是同一个对象，因此该断言必然失败。
        """
        a = 1.0
        b = 1
        self.assertIs(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
