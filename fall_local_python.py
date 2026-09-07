# -*- coding: utf-8 -*-
"""fall项目的本地依赖启动器：仍用系统Python，第三方库从本工程python_packages读取。

用法：/usr/bin/python3 -I -S fall_local_python.py main.py --self-test
-I忽略外部PYTHONPATH及用户site；-S禁止自动加载系统site和.pth。
本程序随后明确加入工程目录和选定的本地依赖目录，不创建venv或Conda环境。
此机制限定Python搜索路径，不隔离操作系统动态库、设备或网络。
"""

import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import runpy
import site
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGES_ROOT = PROJECT_ROOT / "python_packages"
DEFAULT_DIRECTORY = "v2"
CHECK_MODULES = {
    "numpy": "numpy", "scipy": "scipy", "opencv-python": "cv2", "open3d": "open3d",
    "torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics",
    "pyorbbecsdk2": "pyorbbecsdk", "openai": "openai", "PyYAML": "yaml", "lap": "lap",
    "nvidia-ml-py": "pynvml",
     "onnxruntime": "onnxruntime",
}

def configure_paths(directory_name: str) -> Path:
    """选择python_packages里的一个依赖版本，禁止从其他项目或共享目录补包。"""
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("请用 /usr/bin/python3 -I -S fall_local_python.py 启动")
    dependencies = (PACKAGES_ROOT / directory_name).resolve()
    if not dependencies.is_relative_to(PACKAGES_ROOT.resolve()) or dependencies == PACKAGES_ROOT.resolve():
        raise ValueError("--deps必须指定本工程python_packages中的一个子目录")
    if not dependencies.is_dir():
        raise FileNotFoundError(f"本地依赖目录不存在，请先安装：{dependencies}")

    # 在-I -S下，此时的路径应只包含解释器自带的标准库和动态扩展目录。
    standard_paths = list(sys.path)
    for value in standard_paths:
        if {"site-packages", "dist-packages"}.intersection(Path(value).parts):
            raise RuntimeError(f"启动时意外读到了共享第三方目录：{value}")
    sys.path[:] = [str(dependencies), str(PROJECT_ROOT)] + standard_paths
    site.addsitedir(str(dependencies))

    # 允许wheel自己的.pth，但拒绝.pth把其他工程、用户或系统第三方目录加入搜索路径。
    allowed_exact = {str(Path(value).resolve()) for value in standard_paths} | {str(PROJECT_ROOT)}
    for value in sys.path:
        path = Path(value).resolve()
        if str(path) not in allowed_exact and not path.is_relative_to(dependencies):
            raise RuntimeError(f"本地.pth加入了外部路径，请检查安装内容：{path}")

    # 这些设置只作用于本进程及其子进程，不写入.bashrc，不改变其他终端。
    # 普通子进程仍可能读取系统site；新业务Python子进程应使用同一个启动器。
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(dependencies), str(PROJECT_ROOT)])
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / "runtime_local" / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "runtime_local" / "matplotlib")
    os.chdir(PROJECT_ROOT)
    return dependencies

def print_environment(dependencies: Path) -> None:
    """显示解释器、选中目录和搜索路径，帮助确认没有加载共享Python包。"""
    print(f"Python：{sys.executable}")
    print(f"Python版本：{sys.version.split()[0]}")
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"依赖目录：{dependencies}")
    print("搜索路径：")
    for value in sys.path:
        print(f"  {value}")

def check_modules(dependencies: Path) -> None:
    """检查包版本、真实导入路径和基本二进制调用；不打开相机或向AI发送请求。"""
    print_environment(dependencies)
    failures = []
    for distribution, module_name in CHECK_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(dependencies):
                raise RuntimeError(f"加载到了本地目录之外：{path}")
            print(f"[OK] {distribution}={metadata.version(distribution)}，文件={path}")
        except Exception as error:
            failures.append(f"{distribution}：{type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("依赖导入检查失败：\n" + "\n".join(failures))

    # 覆盖报告中NumPy/SciPy二进制不匹配，以及torchvision扩展与torch不匹配的风险。
    import numpy as np
    import scipy.sparse
    import torch
    from torchvision.ops import nms

    matrix = scipy.sparse.csr_matrix(np.eye(2, dtype=np.float32))
    assert np.allclose(matrix.toarray(), np.eye(2))
    assert np.array_equal(torch.from_numpy(np.zeros(2, dtype=np.float32)).numpy(), [0.0, 0.0])
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
    print("导入路径与基础二进制检查通过；还需运行项目自测试和真实相机验收。")

def run_target(arguments: list) -> None:
    """像python一样执行项目脚本或-m模块，并保留目标程序自己的命令行参数。"""
    if arguments and arguments[0] == "-m":
        if len(arguments) < 2:
            raise ValueError("-m后面需要模块名，例如pip或fall_detection.tools.ground_detector")
        sys.argv = [arguments[1]] + arguments[2:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
        return
    arguments = arguments or ["main.py"]
    script = (PROJECT_ROOT / arguments[0]).resolve()
    if not script.is_relative_to(PROJECT_ROOT) or not script.is_file():
        raise ValueError("只能执行本工程内部已存在的Python脚本；请检查相对路径")
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + arguments[1:]
    runpy.run_path(str(script), run_name="__main__")

def main() -> None:
    """解析启动器参数：--deps选择目录，--info查看路径，--check验证实际导入。"""
    arguments = sys.argv[1:]
    directory_name = DEFAULT_DIRECTORY
    if arguments[:1] == ["--deps"]:
        if len(arguments) < 2:
            raise ValueError("--deps后需要目录名，例如v1或v2")
        directory_name, arguments = arguments[1], arguments[2:]
    dependencies = configure_paths(directory_name)
    if arguments == ["--info"]:
        print_environment(dependencies)
    elif arguments == ["--check"]:
        check_modules(dependencies)
    else:
        run_target(arguments)

if __name__ == "__main__":
    main()
