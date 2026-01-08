"""
Mock 对象测试用例

演示 unittest.mock 模块的常见用法：
- Mock 基本用法: return_value, side_effect
- Mock 断言: assert_called, assert_called_with 等
- patch 装饰器和上下文管理器
- MagicMock 的魔术方法支持
- spec 参数限制 Mock 的属性

参考: https://docs.python.org/3/library/unittest.mock.html
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import os


# ========================
# 被测试的模块/类
# ========================
class DatabaseClient:
    """模拟的数据库客户端"""

    def connect(self, host, port):
        """连接数据库"""
        # 实际中这里会建立真实连接
        raise ConnectionError("无法连接到数据库")

    def query(self, sql):
        """执行查询"""
        raise NotImplementedError("需要真实的数据库连接")

    def close(self):
        """关闭连接"""
        pass


class UserService:
    """用户服务，依赖 DatabaseClient"""

    def __init__(self, db_client):
        self.db = db_client

    def get_user(self, user_id):
        """获取用户信息"""
        result = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        if result:
            return {"id": user_id, "name": result[0]}
        return None

    def get_all_users(self):
        """获取所有用户"""
        return self.db.query("SELECT * FROM users")


def fetch_data_from_api(url):
    """模拟从 API 获取数据的函数"""
    # 实际中这里会发起 HTTP 请求
    raise ConnectionError(f"无法连接到 {url}")


def process_data():
    """处理数据的函数，依赖 fetch_data_from_api"""
    data = fetch_data_from_api("https://api.example.com/data")
    return [item.upper() for item in data]


# ========================
# 测试类
# ========================
class TestMockBasic(unittest.TestCase):
    """Mock 基本用法"""

    def test_mock_return_value(self):
        """设置 Mock 的返回值"""
        mock_func = Mock(return_value=42)
        result = mock_func()
        self.assertEqual(result, 42)

    def test_mock_return_value_with_args(self):
        """Mock 可以接受任意参数"""
        mock_func = Mock(return_value="hello")
        result = mock_func(1, 2, key="value")
        self.assertEqual(result, "hello")

    def test_mock_side_effect_function(self):
        """side_effect: 用函数替代返回值"""
        mock_func = Mock(side_effect=lambda x: x * 2)
        self.assertEqual(mock_func(3), 6)
        self.assertEqual(mock_func(5), 10)

    def test_mock_side_effect_exception(self):
        """side_effect: 抛出异常"""
        mock_func = Mock(side_effect=ValueError("出错了"))
        with self.assertRaises(ValueError):
            mock_func()

    def test_mock_side_effect_iterable(self):
        """side_effect: 使用可迭代对象依次返回值"""
        mock_func = Mock(side_effect=[1, 2, 3])
        self.assertEqual(mock_func(), 1)
        self.assertEqual(mock_func(), 2)
        self.assertEqual(mock_func(), 3)
        # 耗尽后再调用会抛出 StopIteration
        with self.assertRaises(StopIteration):
            mock_func()

    def test_mock_attribute_access(self):
        """Mock 支持任意属性访问，自动创建子 Mock"""
        mock_obj = Mock()
        mock_obj.name = "test"
        mock_obj.value = 42

        self.assertEqual(mock_obj.name, "test")
        self.assertEqual(mock_obj.value, 42)
        # 未设置的属性返回新的 Mock 对象
        self.assertIsInstance(mock_obj.undefined_attr, Mock)

    def test_mock_chained_calls(self):
        """Mock 支持链式调用"""
        mock_obj = Mock()
        mock_obj.method1.return_value.method2.return_value = "chained result"

        result = mock_obj.method1().method2()
        self.assertEqual(result, "chained result")


class TestMockAssertions(unittest.TestCase):
    """Mock 的断言方法"""

    def test_assert_called(self):
        """assert_called: 断言 Mock 被调用过"""
        mock_func = Mock()
        mock_func()
        mock_func.assert_called()

    def test_assert_called_once(self):
        """assert_called_once: 断言只被调用一次"""
        mock_func = Mock()
        mock_func()
        mock_func.assert_called_once()

    def test_assert_called_with(self):
        """assert_called_with: 断言最后一次以指定参数调用"""
        mock_func = Mock()
        mock_func(1, 2, key="value")
        mock_func.assert_called_with(1, 2, key="value")

    def test_assert_called_once_with(self):
        """assert_called_once_with: 断言只被以指定参数调用一次"""
        mock_func = Mock()
        mock_func("hello", count=3)
        mock_func.assert_called_once_with("hello", count=3)

    def test_call_count(self):
        """call_count: 获取调用次数"""
        mock_func = Mock()
        mock_func()
        mock_func()
        mock_func()
        self.assertEqual(mock_func.call_count, 3)

    def test_call_args(self):
        """call_args: 获取最后一次调用的参数"""
        mock_func = Mock()
        mock_func("first")
        mock_func("second", key="value")

        # call_args 是一个 (args, kwargs) 的元组
        args, kwargs = mock_func.call_args
        self.assertEqual(args, ("second",))
        self.assertEqual(kwargs, {"key": "value"})

    def test_call_args_list(self):
        """call_args_list: 获取所有调用的参数列表"""
        mock_func = Mock()
        mock_func(1)
        mock_func(2, key="a")
        mock_func(3)

        expected_calls = [call(1), call(2, key="a"), call(3)]
        self.assertEqual(mock_func.call_args_list, expected_calls)

    def test_assert_not_called(self):
        """assert_not_called: 断言从未被调用"""
        mock_func = Mock()
        mock_func.assert_not_called()

    def test_assert_any_call(self):
        """assert_any_call: 断言曾以指定参数调用过(不要求是最后一次)"""
        mock_func = Mock()
        mock_func(1)
        mock_func(2)
        mock_func(3)
        mock_func.assert_any_call(2)  # 曾经以参数 2 调用过

    def test_reset_mock(self):
        """reset_mock: 重置调用记录"""
        mock_func = Mock()
        mock_func(1)
        mock_func(2)
        self.assertEqual(mock_func.call_count, 2)

        mock_func.reset_mock()
        self.assertEqual(mock_func.call_count, 0)
        mock_func.assert_not_called()


class TestMockWithDependency(unittest.TestCase):
    """使用 Mock 替代依赖进行测试"""

    def test_user_service_get_user(self):
        """用 Mock 替代数据库客户端测试 UserService"""
        # 创建 Mock 数据库客户端
        mock_db = Mock(spec=DatabaseClient)
        mock_db.query.return_value = ["Alice"]

        # 注入 Mock 依赖
        service = UserService(mock_db)
        user = service.get_user(1)

        # 验证结果
        self.assertEqual(user, {"id": 1, "name": "Alice"})
        # 验证 query 被以正确的 SQL 调用
        mock_db.query.assert_called_once_with("SELECT * FROM users WHERE id = 1")

    def test_user_service_get_user_not_found(self):
        """测试用户不存在的情况"""
        mock_db = Mock(spec=DatabaseClient)
        mock_db.query.return_value = []

        service = UserService(mock_db)
        user = service.get_user(999)

        self.assertIsNone(user)

    def test_user_service_get_all_users(self):
        """测试获取所有用户"""
        mock_db = Mock(spec=DatabaseClient)
        mock_db.query.return_value = ["Alice", "Bob", "Charlie"]

        service = UserService(mock_db)
        users = service.get_all_users()

        self.assertEqual(users, ["Alice", "Bob", "Charlie"])
        mock_db.query.assert_called_once_with("SELECT * FROM users")


class TestPatch(unittest.TestCase):
    """演示 patch 装饰器和上下文管理器"""

    @patch("os.path.exists")
    def test_patch_decorator(self, mock_exists):
        """使用 @patch 装饰器替换 os.path.exists"""
        mock_exists.return_value = True

        result = os.path.exists("/fake/path")
        self.assertTrue(result)
        mock_exists.assert_called_once_with("/fake/path")

    def test_patch_context_manager(self):
        """使用 with patch() 上下文管理器"""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            result = os.path.exists("/another/fake/path")
            self.assertFalse(result)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_multiple_patches(self, mock_isfile, mock_exists):
        """多个 @patch 装饰器（注意参数顺序：从下到上）"""
        self.assertTrue(os.path.exists("/fake"))
        self.assertTrue(os.path.isfile("/fake"))

    def test_patch_object(self):
        """使用 patch.object 替换对象的方法"""
        db = DatabaseClient()
        with patch.object(db, "query", return_value=["mocked result"]):
            result = db.query("SELECT 1")
            self.assertEqual(result, ["mocked result"])

    def test_patch_module_function(self):
        """使用 patch 替换本模块的函数，测试依赖它的 process_data

        注意 patch 的目标字符串要指向"被使用处"的名字：
        这里用 f"{__name__}.fetch_data_from_api"，无论本文件作为模块导入
        还是直接运行（__main__），都能正确定位。
        """
        with patch(f"{__name__}.fetch_data_from_api", return_value=["alice", "bob"]):
            result = process_data()
        self.assertEqual(result, ["ALICE", "BOB"])

    def test_patch_dict(self):
        """使用 patch.dict 临时修改字典"""
        marker_before = os.environ.get("MY_VAR")   # 记录进入前的值（通常为 None）
        with patch.dict(os.environ, {"MY_VAR": "test_value"}, clear=False):
            self.assertEqual(os.environ.get("MY_VAR"), "test_value")
        # 退出上下文后环境变量恢复为进入前的值
        self.assertEqual(os.environ.get("MY_VAR"), marker_before)


class TestMagicMock(unittest.TestCase):
    """演示 MagicMock — 支持魔术方法的 Mock"""

    def test_magic_mock_len(self):
        """MagicMock 支持 __len__"""
        mock_obj = MagicMock()
        mock_obj.__len__.return_value = 5
        self.assertEqual(len(mock_obj), 5)

    def test_magic_mock_iter(self):
        """MagicMock 支持 __iter__"""
        mock_obj = MagicMock()
        mock_obj.__iter__.return_value = iter([1, 2, 3])
        self.assertEqual(list(mock_obj), [1, 2, 3])

    def test_magic_mock_context_manager(self):
        """MagicMock 支持上下文管理器协议"""
        mock_obj = MagicMock()
        mock_obj.__enter__.return_value = "resource"

        with mock_obj as resource:
            self.assertEqual(resource, "resource")

        mock_obj.__enter__.assert_called_once()
        mock_obj.__exit__.assert_called_once()

    def test_magic_mock_contains(self):
        """MagicMock 支持 __contains__ (in 运算符)"""
        mock_obj = MagicMock()
        mock_obj.__contains__.return_value = True
        self.assertIn("anything", mock_obj)

    def test_magic_mock_str(self):
        """MagicMock 支持 __str__"""
        mock_obj = MagicMock()
        mock_obj.__str__.return_value = "mocked string"
        self.assertEqual(str(mock_obj), "mocked string")


class TestMockSpec(unittest.TestCase):
    """演示 spec 参数 — 限制 Mock 只能访问真实对象的属性"""

    def test_spec_basic(self):
        """spec 限制了 Mock 的可用属性"""
        mock_db = Mock(spec=DatabaseClient)

        # 可以访问 DatabaseClient 的方法
        mock_db.connect("localhost", 5432)
        mock_db.query("SELECT 1")
        mock_db.close()

        # 访问不存在的属性会抛出 AttributeError
        with self.assertRaises(AttributeError):
            mock_db.nonexistent_method()

    def test_spec_isinstance(self):
        """使用 spec 创建的 Mock 会通过 isinstance 检查"""
        mock_db = Mock(spec=DatabaseClient)
        self.assertIsInstance(mock_db, DatabaseClient)


if __name__ == '__main__':
    unittest.main(verbosity=2)
