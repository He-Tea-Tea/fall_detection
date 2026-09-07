# -*- coding: utf-8 -*-
"""环境工具回归测试：使用合成依赖和临时wheel，不联网、不安装软件、不访问相机。"""

import tempfile
import unittest
import zipfile
from pathlib import Path

from environment_tool import dependency_closure, scan_imports, validate_platform, wheel_requirements

class FakeDistribution:
    """模拟发行包元数据，用于测试依赖解析，不依赖本机是否安装相应软件。"""

    def __init__(self, version: str, requirements=()):
        self.version = version
        self.requires = list(requirements)

class EnvironmentToolTests(unittest.TestCase):
    """验证依赖完整性、平台隔离和wheel完整性这三类部署风险。"""

    def test_recursive_dependencies_and_extras(self):
        """同一依赖后来新增extra时，需要重新展开其额外依赖。"""
        packages = {
            "app": FakeDistribution("1.0", ["base>=1", "addon"]),
            "addon": FakeDistribution("1.0", ["base[vision]"]),
            "base": FakeDistribution("1.2", ["image; extra == 'vision'"]),
            "image": FakeDistribution("2.0"),
        }
        selected, extras, errors = dependency_closure(["app"], packages)
        self.assertEqual(set(selected), set(packages))
        self.assertEqual(extras["base"], ["vision"])
        self.assertFalse(errors)

    def test_conflicts_missing_and_markers(self):
        """缺失包和版本冲突必须报错；不适用当前Python的依赖无需加入。"""
        packages = {
            "app": FakeDistribution("1", ["base<2", "absent", "ancient; python_version < '2'"]),
            "base": FakeDistribution("3"),
        }
        selected, _, errors = dependency_closure(["app"], packages)
        self.assertEqual(len(errors), 2)
        self.assertNotIn("ancient", selected)

    def test_cycles_terminate(self):
        """依赖环不应使采集程序无限循环。"""
        packages = {"a": FakeDistribution("1", ["b"]), "b": FakeDistribution("1", ["a"])}
        selected, _, errors = dependency_closure(["a"], packages)
        self.assertEqual(set(selected), {"a", "b"})
        self.assertFalse(errors)

    def test_platform_mismatch(self):
        """架构和Python小版本不同就停止，不能误装Windows或其他架构的基线。"""
        reference = {
            "system": "Linux", "architecture": "x86_64", "os_id": "ubuntu", "os_version": "22.04",
            "python_minor": [3, 10], "implementation": "CPython", "libc": ["glibc", "2.35"],
        }
        validate_platform(reference, dict(reference))
        with self.assertRaises(RuntimeError):
            validate_platform(reference, dict(reference, architecture="aarch64"))
        with self.assertRaises(RuntimeError):
            validate_platform(reference, dict(reference, python_minor=[3, 12]))
        with self.assertRaises(RuntimeError):
            validate_platform(reference, dict(reference, libc=["glibc", "2.31"]))

    @staticmethod
    def make_wheel(directory: Path, version="1.0", requirements=(), payload="original") -> Path:
        """创建只含测试元数据的wheel；不会交给pip安装。"""
        path = directory / "demo-1.0-py3-none-any.whl"
        text = f"Metadata-Version: 2.1\nName: demo\nVersion: {version}\n"
        text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA", text)
            archive.writestr("demo.py", payload)
        return path

    def test_hash_tracks_wheel_content(self):
        """即使名称和版本不变，只要安装包内容改变，锁定哈希也必须改变。"""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = {"packages": {"demo": "1.0"}, "extras": {}}
            self.make_wheel(directory)
            first = wheel_requirements(directory, report)
            self.make_wheel(directory, payload="modified")
            second = wheel_requirements(directory, report)
            self.assertIn("demo==1.0 --hash=sha256:", first)
            self.assertNotEqual(first, second)

    def test_wheel_version_and_dependency_mismatch(self):
        """索引上相同名字但版本或依赖不一致的文件不能作为原环境安装包。"""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = {"packages": {"demo": "1.0"}, "extras": {}}
            self.make_wheel(directory, version="2.0")
            with self.assertRaises(RuntimeError):
                wheel_requirements(directory, report)
            self.make_wheel(directory, requirements=["unrecorded>=1"])
            with self.assertRaises(RuntimeError):
                wheel_requirements(directory, report)

    def test_missing_wheel(self):
        """安装包不齐时不能生成看似完整的离线清单。"""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                wheel_requirements(Path(temporary), {"packages": {"demo": "1.0"}, "extras": {}})

    def test_scan_delayed_imports(self):
        """函数内相机/模型导入也属于运行依赖；相对导入和标准库无需pip安装。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "fall_detection"
            source_dir.mkdir()
            (source_dir / "example.py").write_text(
                "import os\nimport numpy as np\nfrom . import local\n"
                "def camera():\n    from pyorbbecsdk import Pipeline\n", encoding="utf-8",
            )
            self.assertEqual(scan_imports(root), ["numpy", "pyorbbecsdk"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
