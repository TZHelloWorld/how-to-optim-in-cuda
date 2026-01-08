"""
异常和警告测试用例

演示如何使用 unittest 测试异常和警告：
- assertRaises: 检查是否抛出指定异常
- assertRaisesRegex: 检查异常及其消息
- assertWarns: 检查是否触发指定警告
- assertWarnsRegex: 检查警告及其消息
- assertLogs: 检查日志输出

参考: https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertRaises
"""

import unittest
import warnings
import logging
import math


# ========================
# 被测函数
# ========================
def divide(a, b):
    """除法函数，除数为零时抛出 ValueError"""
    if b == 0:
        raise ValueError(f"除数不能为零: {a} / {b}")
    return a / b


def parse_int(value):
    """解析整数，输入不合法时抛出 TypeError 或 ValueError"""
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"不支持的类型: {type(value).__name__}")
    return int(value)


def deprecated_function():
    """模拟一个已废弃的函数"""
    warnings.warn("deprecated_function() 已废弃，请使用 new_function()", DeprecationWarning)
    return 42


def unsafe_operation(value):
    """模拟一个可能产生运行时警告的操作"""
    if value < 0:
        warnings.warn(f"输入值 {value} 为负数，可能导致不可预期的结果", RuntimeWarning)
    return abs(value)


class Calculator:
    """简单的计算器类，用于演示异常测试"""

    def sqrt(self, x):
        if x < 0:
            raise ValueError("不能对负数求平方根")
        return math.sqrt(x)

    def factorial(self, n):
        if not isinstance(n, int):
            raise TypeError("阶乘只接受整数参数")
        if n < 0:
            raise ValueError("阶乘只接受非负整数")
        return math.factorial(n)


# ========================
# 测试类
# ========================
class TestAssertRaises(unittest.TestCase):
    """测试 assertRaises 的各种用法"""

    def test_raises_as_callable(self):
        """用法一: assertRaises(异常类型, 可调用对象, *参数)"""
        self.assertRaises(ValueError, divide, 10, 0)
        self.assertRaises(TypeError, parse_int, [1, 2, 3])

    def test_raises_as_context_manager(self):
        """用法二: 作为上下文管理器使用 (推荐方式)"""
        with self.assertRaises(ValueError):
            divide(10, 0)

        with self.assertRaises(ZeroDivisionError):
            1 / 0

    def test_raises_check_exception_details(self):
        """获取异常对象并进行更详细的检查"""
        with self.assertRaises(ValueError) as context:
            divide(10, 0)

        # 访问异常对象
        exception = context.exception
        self.assertIn("除数不能为零", str(exception))
        self.assertIn("10", str(exception))

    def test_raises_multiple_exceptions(self):
        """测试函数在不同输入下抛出不同类型的异常"""
        calc = Calculator()

        with self.assertRaises(ValueError):
            calc.sqrt(-1)

        with self.assertRaises(TypeError):
            calc.factorial(3.14)

        with self.assertRaises(ValueError):
            calc.factorial(-5)

    def test_no_exception_for_valid_input(self):
        """确保合法输入不会抛出异常"""
        # 如果下面的代码抛出异常，测试会自动失败
        result = divide(10, 2)
        self.assertEqual(result, 5.0)


class TestAssertRaisesRegex(unittest.TestCase):
    """测试 assertRaisesRegex: 同时检查异常类型和消息内容"""

    def test_raises_regex_as_callable(self):
        """用法一: assertRaisesRegex(异常类型, 正则表达式, 可调用对象, *参数)"""
        self.assertRaisesRegex(ValueError, r"除数不能为零", divide, 10, 0)

    def test_raises_regex_as_context_manager(self):
        """用法二: 作为上下文管理器"""
        with self.assertRaisesRegex(ValueError, r"除数不能为零.*10 / 0"):
            divide(10, 0)

    def test_raises_regex_pattern(self):
        """使用正则表达式匹配异常消息"""
        with self.assertRaisesRegex(TypeError, r"不支持的类型: \w+"):
            parse_int([1, 2, 3])

    def test_calculator_error_messages(self):
        """测试计算器的错误消息是否符合预期"""
        calc = Calculator()

        with self.assertRaisesRegex(ValueError, "负数"):
            calc.sqrt(-4)

        with self.assertRaisesRegex(TypeError, "整数"):
            calc.factorial(2.5)


class TestAssertWarns(unittest.TestCase):
    """测试 assertWarns: 检查是否触发指定类型的警告"""

    def test_warns_as_callable(self):
        """用法一: assertWarns(警告类型, 可调用对象, *参数)"""
        self.assertWarns(DeprecationWarning, deprecated_function)

    def test_warns_as_context_manager(self):
        """用法二: 作为上下文管理器"""
        with self.assertWarns(DeprecationWarning):
            deprecated_function()

    def test_warns_check_details(self):
        """获取警告对象并检查详细信息"""
        with self.assertWarns(DeprecationWarning) as cm:
            deprecated_function()

        # 检查警告消息
        self.assertIn("已废弃", str(cm.warning))
        # 检查警告发生的文件
        self.assertIn("test_exception.py", cm.filename)

    def test_warns_runtime_warning(self):
        """测试运行时警告"""
        with self.assertWarns(RuntimeWarning):
            unsafe_operation(-5)


class TestAssertWarnsRegex(unittest.TestCase):
    """测试 assertWarnsRegex: 同时检查警告类型和消息内容"""

    def test_warns_regex(self):
        """检查警告消息是否匹配正则表达式"""
        with self.assertWarnsRegex(DeprecationWarning, r"已废弃.*new_function"):
            deprecated_function()

    def test_warns_regex_runtime(self):
        """检查运行时警告的消息"""
        with self.assertWarnsRegex(RuntimeWarning, r"负数"):
            unsafe_operation(-10)


class TestAssertLogs(unittest.TestCase):
    """测试 assertLogs: 检查日志输出"""

    def test_logs_basic(self):
        """检查是否有日志输出"""
        logger = logging.getLogger("test_logger")
        with self.assertLogs("test_logger", level="INFO") as cm:
            logger.info("这是一条信息日志")
            logger.warning("这是一条警告日志")

        # cm.output 包含格式化后的日志消息列表
        self.assertEqual(len(cm.output), 2)
        self.assertIn("INFO:test_logger:这是一条信息日志", cm.output)
        self.assertIn("WARNING:test_logger:这是一条警告日志", cm.output)

    def test_logs_level_filter(self):
        """检查日志级别过滤"""
        logger = logging.getLogger("level_test")
        with self.assertLogs("level_test", level="WARNING") as cm:
            logger.warning("警告消息")
            logger.error("错误消息")

        # 只会捕获 WARNING 及以上级别的日志
        self.assertTrue(all("WARNING" in msg or "ERROR" in msg for msg in cm.output))

    def test_logs_records(self):
        """通过 records 属性访问 LogRecord 对象"""
        logger = logging.getLogger("record_test")
        with self.assertLogs("record_test", level="DEBUG") as cm:
            logger.debug("调试消息")

        # cm.records 包含 LogRecord 对象列表
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].getMessage(), "调试消息")
        self.assertEqual(cm.records[0].levelname, "DEBUG")


if __name__ == '__main__':
    unittest.main(verbosity=2)
