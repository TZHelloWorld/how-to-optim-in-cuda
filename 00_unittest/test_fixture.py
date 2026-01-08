"""
测试夹具 (Test Fixture) 用法

演示 unittest 中不同层级的夹具机制：
- 方法级别: setUp() / tearDown()     — 每个测试方法前后调用
- 类级别:   setUpClass() / tearDownClass() — 整个测试类前后调用一次
- 模块级别: setUpModule() / tearDownModule() — 整个模块前后调用一次

执行顺序:
  setUpModule -> setUpClass -> setUp -> test -> tearDown -> ... -> tearDownClass -> tearDownModule

参考: https://docs.python.org/3/library/unittest.html#organizing-test-code
"""

import unittest

# 用于记录执行顺序的列表
execution_log = []


def setUpModule():
    """模块级别: 在模块中所有测试开始前调用一次"""
    execution_log.append("setUpModule")
    print("\n[Module] setUpModule - 模块初始化")


def tearDownModule():
    """模块级别: 在模块中所有测试结束后调用一次"""
    execution_log.append("tearDownModule")
    print("[Module] tearDownModule - 模块清理")
    # 打印完整的执行顺序
    print(f"\n执行顺序记录: {' -> '.join(execution_log)}")


class TestFixtureMethodLevel(unittest.TestCase):
    """演示方法级别的 setUp/tearDown"""

    def setUp(self):
        """每个测试方法执行前调用"""
        execution_log.append(f"setUp({self._testMethodName})")
        self.data = {"name": "Alice", "age": 30}
        print(f"  [Method] setUp - 准备测试数据: {self.data}")

    def tearDown(self):
        """每个测试方法执行后调用（无论测试是否通过）"""
        execution_log.append(f"tearDown({self._testMethodName})")
        self.data = None
        print("  [Method] tearDown - 清理测试数据")

    def test_name(self):
        """测试名字字段"""
        execution_log.append("test_name")
        self.assertEqual(self.data["name"], "Alice")
        print("    [Test] test_name 通过")

    def test_age(self):
        """测试年龄字段"""
        execution_log.append("test_age")
        self.assertEqual(self.data["age"], 30)
        print("    [Test] test_age 通过")


class TestFixtureClassLevel(unittest.TestCase):
    """演示类级别的 setUpClass/tearDownClass"""

    # 类属性，由 setUpClass 初始化
    shared_resource = None

    @classmethod
    def setUpClass(cls):
        """整个测试类开始前调用一次 — 适合初始化昂贵的共享资源"""
        execution_log.append("setUpClass")
        cls.shared_resource = list(range(100))
        print(f"\n  [Class] setUpClass - 初始化共享资源 (长度={len(cls.shared_resource)})")

    @classmethod
    def tearDownClass(cls):
        """整个测试类结束后调用一次 — 适合清理共享资源"""
        execution_log.append("tearDownClass")
        cls.shared_resource = None
        print("  [Class] tearDownClass - 释放共享资源")

    def test_shared_resource_exists(self):
        """验证共享资源已被初始化"""
        execution_log.append("test_shared_resource_exists")
        self.assertIsNotNone(self.shared_resource)
        self.assertEqual(len(self.shared_resource), 100)

    def test_shared_resource_content(self):
        """验证共享资源的内容"""
        execution_log.append("test_shared_resource_content")
        self.assertIn(50, self.shared_resource)
        self.assertEqual(self.shared_resource[0], 0)
        self.assertEqual(self.shared_resource[-1], 99)


class TestFixtureCombined(unittest.TestCase):
    """演示同时使用类级别和方法级别夹具"""

    items = None

    @classmethod
    def setUpClass(cls):
        """初始化一个共享列表"""
        cls.items = []
        print("\n  [Class] setUpClass - 初始化共享列表")

    @classmethod
    def tearDownClass(cls):
        """打印最终结果并清理"""
        print(f"  [Class] tearDownClass - 最终列表: {cls.items}")
        cls.items = None

    def setUp(self):
        """每个测试前记录当前列表长度"""
        self.initial_length = len(self.items)
        print(f"  [Method] setUp - 当前列表长度: {self.initial_length}")

    def tearDown(self):
        """每个测试后验证列表确实增长了"""
        print(f"  [Method] tearDown - 当前列表长度: {len(self.items)}")

    def test_add_apple(self):
        """往共享列表添加苹果"""
        self.items.append("apple")
        self.assertIn("apple", self.items)

    def test_add_banana(self):
        """往共享列表添加香蕉"""
        self.items.append("banana")
        # 注意: 测试方法的执行顺序按方法名排序
        # 所以 test_add_apple 会先于 test_add_banana 执行
        self.assertIn("banana", self.items)


class TestAddCleanup(unittest.TestCase):
    """演示 addCleanup 的用法 — 注册清理函数，在 tearDown 之后执行"""

    def setUp(self):
        self.temp_data = "important data"
        # addCleanup 注册的函数在 tearDown 之后按 LIFO 顺序执行
        self.addCleanup(self._cleanup_step_1)
        self.addCleanup(self._cleanup_step_2)

    def _cleanup_step_1(self):
        print("    [Cleanup] 清理步骤 1 (后注册，先执行? 不，LIFO 所以这个后执行)")

    def _cleanup_step_2(self):
        print("    [Cleanup] 清理步骤 2 (最后注册，先执行)")

    def test_cleanup_order(self):
        """addCleanup 按 LIFO 顺序执行"""
        self.assertIsNotNone(self.temp_data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
