import unittest

class TestFun(unittest.TestCase):

    def setUp(self):
        print(f"[Test Method] {self._testMethodName}： 测试前做的事情", flush=True)

    def tearDown(self):
        print(f"[Test Method] {self._testMethodName}： 测试后结束后的事情\n", flush=True)

    # 功能测试1
    def test1(self):
        print(f"test1 exec...", flush=True)

    # 功能测试2
    def test2(self):
        print(f"test2 exec...", flush=True)

    # 内置方法调用测试
    def testfun(self):
        a=1.0
        b=1
        x = False

        self.assertEqual(a, b)  # 检查a和b是否相等。
        self.assertTrue(x)      # 检查x是否为True； assertFalse(x)检查x是否为False
        self.assertIs(a, b)     # 检查a和b是否是同一个对象; assertIsNot(a, b) 检查a和b是否不是同一个对象

if __name__ == '__main__':
    unittest.main()