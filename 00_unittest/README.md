# Python unittest 单元测试指南

> 本文系统介绍 Python 标准库的 unittest 单元测试框架：从核心概念与最小可用结构出发，依次讲解断言方法体系、测试夹具（Fixture）、跳过/预期失败/子测试、Mock 与 patch，最后覆盖命令行用法与测试发现机制。全文配有 7 个可运行的测试文件，各章节与文件一一对应。
>
> 参考官方文档：
> - unittest: https://docs.python.org/3/library/unittest.html
> - unittest.mock: https://docs.python.org/3/library/unittest.mock.html

---

## 目录

- [第 1 章 概述与核心概念](#第-1-章-概述与核心概念)
- [第 2 章 快速上手](#第-2-章-快速上手)
- [第 3 章 断言方法体系](#第-3-章-断言方法体系)
- [第 4 章 测试夹具（Test Fixture）](#第-4-章-测试夹具test-fixture)
- [第 5 章 跳过测试、预期失败与子测试](#第-5-章-跳过测试预期失败与子测试)
- [第 6 章 Mock 与 patch](#第-6-章-mock-与-patch)
- [第 7 章 命令行与测试发现](#第-7-章-命令行与测试发现)
- [第 8 章 配套测试文件](#第-8-章-配套测试文件)

---

## 第 1 章 概述与核心概念

### 1.1 定位

`unittest` 是 Python 标准库内置的单元测试框架（灵感源自 Java 的 JUnit），零依赖即可使用。它支持：

- **测试自动化**：自动运行测试并收集结果；
- **共享 setUp/tearDown 代码**：测试前后的初始化与清理；
- **测试聚合**：将多个测试组织成测试套件（TestSuite）；
- **独立于报告框架**：测试执行与结果报告解耦。

在深度学习工程中，PyTorch、Transformers、SGLang 等主流框架的测试体系都构建在 unittest（或与其兼容的 pytest）之上，读懂并会写 unittest 测试是参与这些项目开发的基础能力。

### 1.2 四个核心概念

| 概念 | 说明 |
|------|------|
| **Test Fixture（测试夹具）** | 执行测试所需的准备工作和清理操作，例如创建临时数据库、目录或启动服务进程 |
| **Test Case（测试用例）** | 最小的测试单元，检查特定输入的特定响应。通过继承 `unittest.TestCase` 创建 |
| **Test Suite（测试套件）** | 测试用例或测试套件的集合，用于聚合需要一起执行的测试 |
| **Test Runner（测试运行器）** | 负责编排测试执行并向用户提供结果的组件 |

---

## 第 2 章 快速上手

> 配套文件：[`test_demo.py`](./test_demo.py)

### 2.1 最简单的测试

```python
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # 检查当分隔符不是字符串时 s.split 会抛出 TypeError
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
```

**四条关键规则**：

1. 测试类必须继承 `unittest.TestCase`；
2. 测试方法名必须以 `test` 开头（框架据此发现用例）；
3. 使用 `assert*` 方法进行断言，而不是 Python 内置的 `assert` 语句（前者失败时给出结构化的差异信息，且不受 `-O` 优化开关影响）；
4. 使用 `unittest.main()` 作为入口运行测试（推荐 `unittest.main(verbosity=2)` 获得详细输出）。

### 2.2 运行结果

```
...
----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK
```

使用 `-v` 选项可以看到详细输出：

```
test_isupper (__main__.TestStringMethods.test_isupper) ... ok
test_split (__main__.TestStringMethods.test_split) ... ok
test_upper (__main__.TestStringMethods.test_upper) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
```

结果符号约定：`.` 通过、`F` 断言失败、`E` 出错（抛出非断言异常）、`s` 跳过、`x` 预期失败。

---

## 第 3 章 断言方法体系

> 配套文件：[`test_assert_methods.py`](./test_assert_methods.py)（按类别分组的完整可运行示例）、[`test_exception.py`](./test_exception.py)（异常/警告/日志断言）

### 3.1 基本断言

| 方法 | 检查内容 |
|------|----------|
| `assertEqual(a, b)` | `a == b` |
| `assertNotEqual(a, b)` | `a != b` |
| `assertTrue(x)` | `bool(x) is True` |
| `assertFalse(x)` | `bool(x) is False` |
| `assertIs(a, b)` | `a is b`（同一对象） |
| `assertIsNot(a, b)` | `a is not b` |
| `assertIsNone(x)` | `x is None` |
| `assertIsNotNone(x)` | `x is not None` |
| `assertIn(a, b)` | `a in b` |
| `assertNotIn(a, b)` | `a not in b` |
| `assertIsInstance(a, b)` | `isinstance(a, b)` |
| `assertNotIsInstance(a, b)` | `not isinstance(a, b)` |

> 注意 `assertEqual` 与 `assertIs` 的区别：`1.0 == 1` 为 True，但 `1.0 is 1` 为 False（不同对象）。配套文件 `test_demo.py` 中有一个用 `@unittest.expectedFailure` 标记的用例专门演示这一点。

### 3.2 比较断言

| 方法 | 检查内容 |
|------|----------|
| `assertAlmostEqual(a, b)` | `round(a-b, 7) == 0`（浮点近似相等） |
| `assertNotAlmostEqual(a, b)` | `round(a-b, 7) != 0` |
| `assertGreater(a, b)` | `a > b` |
| `assertGreaterEqual(a, b)` | `a >= b` |
| `assertLess(a, b)` | `a < b` |
| `assertLessEqual(a, b)` | `a <= b` |

浮点比较务必用 `assertAlmostEqual`（可用 `places=` 控制精度），如 `assertAlmostEqual(0.1 + 0.2, 0.3, places=7)` 通过，而 `assertEqual(0.1 + 0.2, 0.3)` 会失败。

### 3.3 集合/序列断言

| 方法 | 检查内容 |
|------|----------|
| `assertListEqual(a, b)` | 两个列表相等 |
| `assertTupleEqual(a, b)` | 两个元组相等 |
| `assertSetEqual(a, b)` | 两个集合相等 |
| `assertDictEqual(a, b)` | 两个字典相等 |
| `assertSequenceEqual(a, b)` | 两个序列相等 |
| `assertCountEqual(a, b)` | 两个序列包含相同的元素（不考虑顺序） |

实践中直接用 `assertEqual` 即可——它会根据类型自动分派到对应的专用断言，并给出结构化 diff。

### 3.4 异常/警告断言

| 方法 | 检查内容 |
|------|----------|
| `assertRaises(exc, fun, *args, **kwds)` | `fun(*args, **kwds)` 抛出 `exc` |
| `assertRaisesRegex(exc, r, fun, *args)` | 抛出 `exc` 且消息匹配正则 `r` |
| `assertWarns(warn, fun, *args, **kwds)` | `fun(*args, **kwds)` 触发 `warn` |
| `assertWarnsRegex(warn, r, fun, *args)` | 触发 `warn` 且消息匹配正则 `r` |

两种等价用法（上下文管理器形式更常用，还能进一步检查异常对象）：

```python
# 函数式
self.assertRaises(ZeroDivisionError, divide, 1, 0)

# 上下文管理器（推荐）：可通过 cm.exception 访问异常对象
with self.assertRaises(ValueError) as cm:
    parse_int("abc")
self.assertIn("abc", str(cm.exception))
```

**日志断言 `assertLogs`**（配套文件 `test_exception.py` 的 `TestAssertLogs`）：

```python
with self.assertLogs('my.logger', level='WARNING') as cm:
    do_something()
self.assertIn('warning message', cm.output[0])   # cm.records 可访问原始 LogRecord
```

### 3.5 字符串断言

| 方法 | 检查内容 |
|------|----------|
| `assertRegex(text, regex)` | `regex.search(text)` 匹配 |
| `assertNotRegex(text, regex)` | `regex.search(text)` 不匹配 |
| `assertMultiLineEqual(a, b)` | 多行字符串相等（失败时给出逐行 diff） |

---

## 第 4 章 测试夹具（Test Fixture）

> 配套文件：[`test_fixture.py`](./test_fixture.py)（含完整执行顺序打印）

夹具用于在测试前后执行准备和清理工作，unittest 提供**方法、类、模块**三个层级。

### 4.1 方法级别：setUp / tearDown

每个测试方法执行前后都会调用：

```python
class TestExample(unittest.TestCase):
    def setUp(self):
        """每个测试方法执行前调用"""
        self.data = [1, 2, 3]

    def tearDown(self):
        """每个测试方法执行后调用（无论测试是否通过）"""
        self.data = None

    def test_length(self):
        self.assertEqual(len(self.data), 3)

    def test_contains(self):
        self.assertIn(2, self.data)
```

### 4.2 类级别：setUpClass / tearDownClass

整个测试类执行前后只调用一次（注意必须是 `@classmethod`），适合昂贵的共享资源（数据库连接、加载模型等）：

```python
class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """整个测试类开始前调用一次"""
        cls.db = connect_to_database()

    @classmethod
    def tearDownClass(cls):
        """整个测试类结束后调用一次"""
        cls.db.close()
```

### 4.3 模块级别：setUpModule / tearDownModule

整个测试模块执行前后只调用一次（模块级普通函数）：

```python
def setUpModule():
    """模块中所有测试开始前调用"""
    pass

def tearDownModule():
    """模块中所有测试结束后调用"""
    pass
```

### 4.4 执行顺序

```
setUpModule()
  setUpClass()
    setUp()
      test_method_1()
    tearDown()
    setUp()
      test_method_2()
    tearDown()
  tearDownClass()
tearDownModule()
```

### 4.5 addCleanup：更灵活的清理注册

`tearDown` 之外，还可以在测试运行过程中**动态注册**清理函数，按 **LIFO（后注册先执行）** 顺序在测试结束后调用——即使 setUp 或测试中途失败，已注册的清理仍会执行，比 tearDown 更健壮：

```python
class TestAddCleanup(unittest.TestCase):
    def test_cleanup_order(self):
        resource_a = acquire_a()
        self.addCleanup(resource_a.release)     # 后执行

        resource_b = acquire_b()
        self.addCleanup(resource_b.release)     # 先执行（LIFO）
```

---

## 第 5 章 跳过测试、预期失败与子测试

> 配套文件：[`test_skip_and_subtest.py`](./test_skip_and_subtest.py)

### 5.1 跳过测试

四种跳过方式，覆盖"无条件 / 按条件 / 按平台 / 运行时动态"：

```python
import sys

class TestSkipping(unittest.TestCase):

    @unittest.skip("演示无条件跳过")
    def test_nothing(self):
        self.fail("不会执行到这里")

    @unittest.skipIf(sys.version_info < (3, 8), "需要 Python 3.8+")
    def test_new_feature(self):
        pass

    @unittest.skipUnless(sys.platform.startswith("linux"), "仅限 Linux")
    def test_linux_only(self):
        pass

    def test_maybe_skip(self):
        if not some_condition():
            self.skipTest("条件不满足，跳过")
        # 继续测试...
```

装饰器也可以作用于**整个测试类**（`@unittest.skip(...)` 加在 class 上）。工程中常见的用法是用环境变量门控慢速/需 GPU 的测试：

```python
@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "设置 RUN_SLOW_TESTS=1 以运行")
def test_slow_operation(self):
    ...
```

### 5.2 预期失败

已知 bug 尚未修复时，用 `@unittest.expectedFailure` 标记——用例失败时整体结果仍为通过（报告中记为 expected failure）；若它意外通过了，反而报 unexpected success 提醒你移除标记：

```python
class TestExpectedFailure(unittest.TestCase):

    @unittest.expectedFailure
    def test_known_bug(self):
        self.assertEqual(1, 0, "已知的 bug，等待修复")
```

### 5.3 子测试（subTest）

用不同参数测试同一逻辑时，`subTest` 让**单个参数的失败不会终止其余参数的执行**，且报告中清楚标识失败的参数：

```python
class TestNumbers(unittest.TestCase):

    def test_even(self):
        for i in range(0, 6):
            with self.subTest(i=i):
                self.assertEqual(i % 2, 0)
```

subTest 是 unittest 内置的**参数化测试**手段（配套文件 [`test_parameterized.py`](./test_parameterized.py) 用它实现了加法/回文/FizzBuzz/栈操作等多组参数化用例）。`subTest` 支持多个关键字参数与 `msg`，可嵌套使用以表达多维参数组合。

---

## 第 6 章 Mock 与 patch

> 配套文件：[`test_mock.py`](./test_mock.py)

`unittest.mock` 用**替身对象**隔离被测代码的外部依赖（数据库、网络、文件系统），使单元测试快速、确定、可离线运行。

### 6.1 Mock 基本用法

```python
from unittest.mock import Mock, patch

class TestWithMock(unittest.TestCase):

    def test_mock_return_value(self):
        mock_func = Mock(return_value=42)
        result = mock_func(1, 2, 3)
        self.assertEqual(result, 42)
        mock_func.assert_called_once_with(1, 2, 3)

    @patch('module.ClassName')
    def test_patch(self, MockClass):
        instance = MockClass.return_value
        instance.method.return_value = 'mocked'
        # 测试使用 module.ClassName 的代码
```

**常用 Mock 功能速查**：

| 功能 | 说明 |
|------|------|
| `Mock(return_value=x)` | 设置返回值 |
| `Mock(side_effect=func)` | 设置副作用：传函数则调用它、传异常则抛出、传可迭代对象则依次返回 |
| `mock.assert_called()` | 断言被调用过 |
| `mock.assert_called_once()` | 断言只被调用一次 |
| `mock.assert_called_with(...)` | 断言最后一次以指定参数被调用 |
| `mock.assert_any_call(...)` | 断言曾以指定参数被调用过 |
| `mock.call_count` | 被调用的次数 |
| `mock.call_args` | 最后一次调用的参数 |
| `mock.call_args_list` | 所有调用的参数列表 |
| `mock.reset_mock()` | 重置调用记录 |

### 6.2 依赖注入：用 Mock 替代真实依赖

典型模式——被测服务依赖数据库客户端，测试时注入 Mock：

```python
def test_user_service_get_user(self):
    mock_db = Mock()
    mock_db.query.return_value = ["Alice"]

    service = UserService(mock_db)          # 注入 Mock 而非真实 DatabaseClient
    user = service.get_user(1)

    self.assertEqual(user, {"id": 1, "name": "Alice"})
    mock_db.query.assert_called_once()
```

### 6.3 patch 的四种形态

| 形态 | 写法 | 适用场景 |
|------|------|---------|
| 装饰器 | `@patch("os.path.exists")` | 整个测试方法内生效，Mock 作为参数注入 |
| 上下文管理器 | `with patch(...) as m:` | 只在局部代码块内替换 |
| `patch.object` | `patch.object(obj, "method", ...)` | 替换已有对象的方法/属性 |
| `patch.dict` | `patch.dict(os.environ, {...})` | 临时修改字典（如环境变量），退出后自动恢复 |

**patch 目标字符串的定位规则**：patch 的目标要指向**被使用处**的名字，而不是定义处。对本模块内的函数，用 `f"{__name__}.func_name"` 可同时兼容"作为模块导入"和"直接运行（`__main__`）"两种场景（配套文件的 `test_patch_module_function` 演示了这一写法）。

### 6.4 MagicMock 与 spec

- **`MagicMock`**：在 `Mock` 基础上预置了魔术方法支持（`__len__`、`__iter__`、`__enter__/__exit__`、`__contains__`、`__str__` 等），可直接用于 `len(m)`、`for x in m`、`with m:` 等场景；
- **`spec` 参数**：`Mock(spec=SomeClass)` 将 Mock 的属性限制为目标类真实存在的属性——访问不存在的属性会抛 `AttributeError`，且 `isinstance` 检查通过。它能防止"测试用了拼错的方法名却照样通过"这类假阳性。

---

## 第 7 章 命令行与测试发现

### 7.1 命令行运行

```bash
# 运行指定模块
python -m unittest test_module

# 运行指定类
python -m unittest test_module.TestClass

# 运行指定方法
python -m unittest test_module.TestClass.test_method

# 详细输出 (-v)
python -m unittest -v test_module

# 遇到第一个失败就停止 (-f)
python -m unittest -f test_module

# 缓冲输出 (-b)，通过的测试不显示 stdout
python -m unittest -b test_module

# 按关键字过滤测试 (-k)
python -m unittest -k "test_add" test_module

# 显示最慢的 N 个测试 (Python 3.12+)
python -m unittest --durations 5 test_module
```

### 7.2 测试发现（Test Discovery）

```bash
# 自动发现并运行当前目录下所有测试
python -m unittest discover

# 指定起始目录和文件模式
python -m unittest discover -s tests -p "test_*.py"

# 等价的简短写法
python -m unittest discover tests "test_*.py"
```

**发现规则**：

- 默认在当前目录查找；
- 默认匹配 `test*.py` 模式；
- 测试文件必须是可导入的模块；
- 子目录需要包含 `__init__.py`（Python 3.11+ 支持 namespace package 例外）。

---

## 第 8 章 配套测试文件

### 8.1 文件索引

| 文件 | 内容 | 对应章节 |
|------|------|---------|
| [`test_demo.py`](./test_demo.py) | 最小结构：setUp/tearDown、基本断言、expectedFailure 演示 `==` vs `is` | 第 2 章 |
| [`test_assert_methods.py`](./test_assert_methods.py) | 四类断言（基本/比较/集合/字符串）的完整分组示例 | 第 3 章 |
| [`test_exception.py`](./test_exception.py) | assertRaises[Regex] / assertWarns[Regex] / assertLogs 的全部形态 | 第 3.4 节 |
| [`test_fixture.py`](./test_fixture.py) | 方法/类/模块三级夹具的执行顺序、addCleanup 的 LIFO 清理 | 第 4 章 |
| [`test_skip_and_subtest.py`](./test_skip_and_subtest.py) | skip/skipIf/skipUnless/环境变量门控、expectedFailure、subTest 各用法 | 第 5 章 |
| [`test_mock.py`](./test_mock.py) | Mock/断言族/依赖注入/patch 四形态/MagicMock/spec | 第 6 章 |
| [`test_parameterized.py`](./test_parameterized.py) | 用 subTest 实现参数化测试（多组数据驱动同一逻辑） | 第 5.3 节 |

### 8.2 运行方式

```bash
cd 00_unittest

# 运行全部测试（推荐）
python -m unittest discover -v

# 运行单个测试文件
python -m unittest test_assert_methods -v

# 直接执行单个文件（各文件均有 unittest.main 入口）
python test_fixture.py
```
