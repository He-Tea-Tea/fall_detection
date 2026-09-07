# -*- coding: utf-8 -*-
"""v0.96环境工具：采集已验证版本、准备安装包、部署前检查和安装后验收。

本文件只管理环境，不启动相机、不调用AI、不修改跌倒算法。
capture必须在实际跑通项目的Ubuntu上执行；不能用另一台机器的版本代替。
Python依赖从当前安装元数据递归收集，系统软件清单仅作审计，不直接整机重装。
"""

import argparse
import ast
import email
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# 这里使用发行包名称；Python导入名称并不总是与pip安装名称相同。
CORE_PACKAGES = ("numpy", "PyYAML", "opencv-python", "torch", "torchvision", "ultralytics", "openai", "lap")
OPTIONAL_PACKAGES = {
    "pointcloud": ("open3d",),
    "onnx": ("onnxruntime",),
    "export": ("onnx", "onnxslim", "onnxruntime"),
}
MODULES = {
    "numpy": "numpy", "pyyaml": "yaml", "opencv-python": "cv2", "torch": "torch",
    "torchvision": "torchvision", "ultralytics": "ultralytics", "openai": "openai", "lap": "lap",
    "pyorbbecsdk2": "pyorbbecsdk", "pyorbbecsdk": "pyorbbecsdk", "open3d": "open3d",
    "onnxruntime": "onnxruntime", "onnx": "onnx", "onnxslim": "onnxslim",
}
CV_PACKAGES = ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")

def canonical_name(name: str) -> str:
    """统一发行包名称中的大小写、连字符、下划线和点，方便比较。"""
    return re.sub(r"[-_.]+", "-", name).lower()

def packaging_api():
    """按需加载依赖解析工具；部署前平台检查只需要Python标准库。"""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.version import Version
    return Requirement, Version

def run_command(arguments: list, timeout_s: int = 120) -> subprocess.CompletedProcess:
    """使用参数列表调用命令，避免路径空格和shell转义引起意外执行。"""
    return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout_s, check=False)

def system_identity() -> dict:
    """记录平台、Python、libc和ROS名称；不读取API密钥或完整环境变量。"""
    release = {}
    release_file = Path("/etc/os-release")
    if release_file.is_file():
        for line in release_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                name, value = line.split("=", 1)
                tokens = shlex.split(value)
                release[name] = tokens[0] if tokens else ""
    return {
        "system": platform.system(), "architecture": platform.machine(),
        "os_id": release.get("ID", ""), "os_version": release.get("VERSION_ID", ""),
        "python": platform.python_version(), "python_minor": list(sys.version_info[:2]),
        "implementation": platform.python_implementation(), "executable": sys.executable,
        "libc": list(platform.libc_ver()), "kernel": platform.release(),
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
    }

def installed_packages() -> dict:
    """记录当前解释器可见的发行包，保留与metadata.distribution一致的查找优先级。"""
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and canonical_name(name) not in packages:
            packages[canonical_name(name)] = distribution
    return packages

def dependency_closure(roots: list, packages: dict) -> tuple:
    """递归收集项目的直接和间接依赖，并检查已安装版本是否满足依赖要求。

    根据当前平台解释环境标记；extras会沿依赖传递，但不凭空启用训练、开发等可选组。
    返回选中包、各包启用的extras和错误列表，不安装或升级任何软件。
    """
    Requirement, _ = packaging_api()
    queue = deque(Requirement(name) for name in roots)
    selected, extras_by_name, visited, errors = {}, {}, set(), []
    while queue:
        requirement = queue.popleft()
        name = canonical_name(requirement.name)
        distribution = packages.get(name)
        if distribution is None:
            errors.append(f"缺少依赖：{requirement}")
            continue
        if requirement.url:
            errors.append(f"{name}使用URL依赖，需要人工转换成经过验证的本地wheel")
        if requirement.specifier and not requirement.specifier.contains(distribution.version, prereleases=True):
            errors.append(f"版本冲突：{requirement}，当前为{distribution.version}")
        extras = extras_by_name.setdefault(name, set())
        extras.update(requirement.extras)
        visit_key = (name, frozenset(extras))
        if visit_key in visited:
            continue
        visited.add(visit_key)
        selected[name] = distribution
        for raw_requirement in distribution.requires or []:
            child = Requirement(raw_requirement)
            if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in {""} | extras):
                queue.append(child)
    return selected, {name: sorted(values) for name, values in extras_by_name.items()}, sorted(set(errors))

def scan_imports(project_root: Path) -> list:
    """静态扫描包括函数内延迟导入在内的第三方模块；动态加载依赖仍需额外声明。"""
    names = set()
    for source in (project_root / "fall_detection").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    standard_names = set(sys.stdlib_module_names) | {"winsound", "fall_detection"}
    return sorted(names - standard_names)

def probe_modules(module_names: list) -> tuple:
    """在独立子进程中导入关键模块，发现动态库缺失、ABI错误和实际导入位置。

    不调用Pipeline.start、不执行YOLO推理；检查成功不代表相机或真实模型已经验收。
    """
    probes, errors = {}, []
    for name in sorted(set(module_names)):
        lines_of_code = [
            "import importlib", "import json", f"module = importlib.import_module({name!r})",
            "data = {'file': str(getattr(module, '__file__', '')), "
            "'version': str(getattr(module, '__version__', ''))}",
        ]
        if name == "torch":
            lines_of_code.append(
                "data.update(cuda_build=module.version.cuda, cuda_available=module.cuda.is_available())"
            )
        if name == "cv2":
            lines_of_code.append(
                "data['gui'] = [line.strip() for line in module.getBuildInformation().splitlines() if 'GUI:' in line]"
            )
        lines_of_code.append("print('ENV_PROBE=' + json.dumps(data))")
        code = "\n".join(lines_of_code)
        try:
            result = run_command([sys.executable, "-c", code], timeout_s=90)
            lines = [line for line in result.stdout.splitlines() if line.startswith("ENV_PROBE=")]
            if result.returncode != 0 or not lines:
                detail = (result.stderr or result.stdout).strip()[-1500:]
                errors.append(f"无法导入{name}：{detail}")
            else:
                probes[name] = json.loads(lines[-1].split("=", 1)[1])
        except subprocess.TimeoutExpired:
            errors.append(f"导入{name}超过90秒")
    return probes, errors

def write_json(path: Path, data: dict) -> None:
    """以UTF-8保存可阅读的JSON报告；每次采集写入新的目录。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def ensure_new_directory(path: Path) -> None:
    """拒绝覆盖已有环境记录，避免新采集污染已经验证的部署基线。"""
    if path.exists():
        raise RuntimeError(f"目录已存在，请换一个新目录：{path}")
    path.mkdir(parents=True)

def capture_environment(args) -> None:
    """从用户当前解释器采集版本和项目依赖，检查通过后才生成可安装清单。"""
    project = Path(args.project_root).resolve()
    if not (project / "fall_detection" / "app" / "main.py").is_file():
        raise RuntimeError("--project-root必须指向包含fall_detection/app/main.py的v0.96项目根目录")
    packages = installed_packages()
    roots = list(CORE_PACKAGES)
    sdk_candidates = [name for name in ("pyorbbecsdk2", "pyorbbecsdk") if name in packages]
    if len(sdk_candidates) != 1:
        raise RuntimeError("需要唯一可识别的Orbbec发行包；手动复制的.so必须先整理为匹配平台的wheel")
    roots.extend(sdk_candidates)
    for group in args.include:
        roots.extend(OPTIONAL_PACKAGES[group])
    roots.extend(args.extra_package)
    imports = scan_imports(project)
    # 将源码直接导入的已知模块也纳入锁定，避免只检查导入却漏打包。
    for package_name, module_name in MODULES.items():
        if module_name in imports and package_name in packages:
            roots.append(package_name)
    selected, extras, errors = dependency_closure(roots, packages)
    cv_variants = [name for name in CV_PACKAGES if name in packages]
    if cv_variants != ["opencv-python"]:
        errors.append(f"当前GUI部署基线要求唯一opencv-python，检测到：{cv_variants}")
    known_modules = set(MODULES.values())
    for name in imports:
        if name not in known_modules:
            errors.append(f"发现未登记模块{name}，请在MODULES/依赖声明中补充其发行包映射")
    module_names = imports + [MODULES[name] for name in selected if name in MODULES]
    probes, probe_errors = probe_modules(module_names)
    errors.extend(probe_errors)
    pip_check = run_command([sys.executable, "-m", "pip", "check"])
    if pip_check.returncode:
        errors.append("pip check未通过；请先查看报告并解决共享Python环境中的依赖冲突")
    source_notes = []
    for name, distribution in selected.items():
        direct = distribution.read_text("direct_url.json")
        if direct:
            data = json.loads(direct)
            if data.get("vcs_info") or data.get("dir_info"):
                errors.append(f"{name}来自源码或可编辑目录，必须先制作并验证wheel，再建立部署基线")
            else:
                source_notes.append(f"{name}可能来自本地安装包，请保留原始wheel用于prepare --find-links")
    output = Path(args.output).resolve()
    ensure_new_directory(output)
    report = {
        "schema": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
        "system": system_identity(), "roots": sorted(set(roots)), "extras": extras,
        "packages": {name: item.version for name, item in sorted(selected.items())},
        "imports": imports, "probes": probes, "source_notes": source_notes,
        "pip_check": (pip_check.stdout + pip_check.stderr).strip(),
        "errors": sorted(set(errors)), "ready": not errors,
    }
    write_json(output / "environment.json", report)
    write_json(output / "inventory.json", {name: item.version for name, item in sorted(packages.items())})
    if platform.system() == "Linux" and Path("/usr/bin/dpkg-query").is_file():
        apt = run_command(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"])
        (output / "apt-inventory.tsv").write_text(apt.stdout, encoding="utf-8")
    if errors:
        raise RuntimeError(f"采集存在{len(set(errors))}项问题，未生成安装清单；请查看{output / 'environment.json'}")
    lock = [f"{name}=={item.version}" for name, item in sorted(selected.items())]
    direct_names = {canonical_name(packaging_api()[0](name).name) for name in roots}
    direct_lock = [line for line in lock if line.split("==", 1)[0] in direct_names]
    header = "# 从实际验证环境采集；只适用于environment.json记录的平台和Python版本。\n"
    (output / "requirements.lock.txt").write_text(header + "\n".join(lock) + "\n", encoding="utf-8")
    (output / "requirements.direct.txt").write_text(header + "\n".join(direct_lock) + "\n", encoding="utf-8")
    print(f"采集完成：{output}，锁定{len(selected)}个项目相关发行包")
    print("仍需完成真实相机、模型、音频、HTTP和AI验收，再将该目录作为正式发布基线。")

def load_snapshot(path: str) -> tuple:
    """加载成功采集的基线；错误报告不能被误用为安装清单。"""
    directory = Path(path).resolve()
    report = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    if not report.get("ready"):
        raise RuntimeError("该基线采集未通过，请先处理environment.json中的errors")
    return directory, report

def validate_platform(reference: dict, current: dict) -> None:
    """拒绝把Windows、ARM或不同Python小版本的安装包部署到当前平台。"""
    keys = ("system", "architecture", "os_id", "os_version", "python_minor", "implementation")
    differences = [f"{key}: 基线={reference[key]}，当前={current[key]}" for key in keys if reference[key] != current[key]]
    if differences:
        raise RuntimeError("平台不一致，需要单独采集和验收：" + "；".join(differences))
    if reference["libc"][0] == current["libc"][0] == "glibc":
        old = tuple(int(item) for item in reference["libc"][1].split("."))
        new = tuple(int(item) for item in current["libc"][1].split("."))
        if new < old:
            raise RuntimeError("当前glibc低于基线，需要匹配的系统版本或重新构建安装包")

def preflight(args) -> None:
    """部署前只检查平台，不安装软件，也不要求当前依赖已经齐全。"""
    _, report = load_snapshot(args.snapshot)
    validate_platform(report["system"], system_identity())
    print("平台检查通过：系统、CPU架构和Python小版本与基线一致")

def wheel_requirements(directory: Path, report: dict) -> str:
    """校验wheel内元数据与基线依赖闭包，再生成每个安装包的SHA256锁定清单。"""
    Requirement, Version = packaging_api()
    found = {}
    for path in sorted(directory.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(candidates) != 1:
                raise RuntimeError(f"wheel元数据异常：{path.name}")
            info = email.message_from_bytes(archive.read(candidates[0]))
        name = canonical_name(info["Name"])
        if name not in report["packages"] or Version(info["Version"]) != Version(report["packages"][name]):
            raise RuntimeError(f"wheel不属于当前基线：{path.name}")
        if name in found:
            raise RuntimeError(f"同一个发行包有多个wheel，需明确唯一目标平台文件：{name}")
        extras = set(report["extras"].get(name, [])) | {""}
        for text in info.get_all("Requires-Dist", []):
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": value}) for value in extras):
                continue
            child = canonical_name(requirement.name)
            installed = report["packages"].get(child)
            if requirement.url or installed is None:
                raise RuntimeError(f"wheel的依赖与采集基线不一致：{name}需要{requirement}")
            if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f"wheel依赖版本冲突：{name}需要{requirement}，基线为{installed}")
            if not set(requirement.extras).issubset(set(report["extras"].get(child, []))):
                raise RuntimeError(f"wheel需要基线未采集的额外依赖组：{requirement}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        found[name] = digest.hexdigest()
    missing = set(report["packages"]) - set(found)
    if missing:
        raise RuntimeError("缺少安装包：" + ", ".join(sorted(missing)))
    lines = ["# 固定版本和wheel哈希；必须配合当前wheelhouse使用。"]
    for name, version in sorted(report["packages"].items()):
        lines.append(f"{name}=={version} --hash=sha256:{found[name]}")
    return "\n".join(lines) + "\n"

def prepare_wheels(args) -> None:
    """在原机器上下载固定版本wheel，成功后生成带哈希的离线安装清单。"""
    directory, report = load_snapshot(args.snapshot)
    validate_platform(report["system"], system_identity())
    wheelhouse = directory / "wheelhouse"
    if (directory / "requirements.hashed.txt").exists():
        raise RuntimeError("当前wheel基线已经完成；重新准备时请重新采集到新目录")
    wheelhouse.mkdir(exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
        "--index-url", args.index_url, "--dest", str(wheelhouse), "-r", str(directory / "requirements.lock.txt"),
    ]
    # CPU/CUDA构建后缀属于PyTorch版本的一部分，必须保留并使用相应官方源。
    torch_version = report["packages"].get("torch", "")
    torch_suffix = torch_version.split("+", 1)[-1] if "+" in torch_version else ""
    indexes = list(args.extra_index_url)
    if torch_suffix == "cpu" or re.fullmatch(r"cu\d+", torch_suffix):
        indexes.append(f"https://download.pytorch.org/whl/{torch_suffix}")
    for index in sorted(set(indexes)):
        command.extend(["--extra-index-url", index])
    for source in args.find_links:
        command.extend(["--find-links", str(Path(source).resolve())])
    subprocess.run(command, check=True)
    content = wheel_requirements(wheelhouse, report)
    (directory / "requirements.hashed.txt").write_text(content, encoding="utf-8")
    print("安装包和哈希清单已准备完成；wheelhouse只包含Python依赖，不包含apt包或模型。")

def verify_environment(args) -> None:
    """比较安装后的发行包版本和导入结果，检查ROS共存；不假称已通过真实相机验收。"""
    _, report = load_snapshot(args.snapshot)
    validate_platform(report["system"], system_identity())
    packages = installed_packages()
    errors = []
    for name, version in report["packages"].items():
        if name not in packages or packages[name].version != version:
            actual = packages[name].version if name in packages else "未安装"
            errors.append(f"{name}：需要{version}，当前{actual}")
    _, _, dependency_errors = dependency_closure(report["roots"], packages)
    errors.extend(dependency_errors)
    modules = list(report["probes"])
    if args.ros:
        modules.extend(["rclpy", "cv_bridge"])
    probes, probe_errors = probe_modules(modules)
    errors.extend(probe_errors)
    for name, details in probes.items():
        print(f"导入检查：{name}，版本={details.get('version', '')}，路径={details['file']}")
        expected_version = report["probes"].get(name, {}).get("version", "")
        if expected_version and details.get("version") != expected_version:
            errors.append(f"{name}实际导入版本与基线不同，可能存在路径遮蔽或二进制混装")
    if "torch" in probes:
        if probes["torch"].get("cuda_build") != report["probes"].get("torch", {}).get("cuda_build"):
            errors.append("PyTorch实际CUDA构建与基线不同")
        if report["probes"].get("torch", {}).get("cuda_available") and not probes["torch"].get("cuda_available"):
            errors.append("基线使用可用CUDA，但目标机器CUDA不可用；需要检查GPU和驱动")
    variants = [name for name in CV_PACKAGES if name in packages]
    if variants != ["opencv-python"]:
        errors.append(f"OpenCV发行包冲突：{variants}")
    pip_check = run_command([sys.executable, "-m", "pip", "check"])
    if pip_check.returncode:
        errors.append("pip check失败：" + (pip_check.stdout + pip_check.stderr).strip())
    if errors:
        raise RuntimeError("安装后检查失败：\n" + "\n".join(errors))
    print("版本、导入和依赖检查通过。真实相机取帧、D2C、模型推理和外部通讯仍需现场验收。")

def install_environment(args) -> None:
    """从校验过的本地wheel向当前用户安装；不使用sudo pip或修改系统Python版本。"""
    directory, report = load_snapshot(args.snapshot)
    validate_platform(report["system"], system_identity())
    if platform.system() != "Linux":
        raise RuntimeError("本安装命令用于Linux系统；Windows需要单独的安装流程")
    if os.geteuid() == 0 or sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("请使用普通用户的系统Python运行本部署流程，不要sudo运行Python或激活conda/venv")
    installed = installed_packages()
    conflicts = [name for name in CV_PACKAGES if name in installed and name != "opencv-python"]
    if conflicts:
        raise RuntimeError("当前用户环境已有其他cv2发行包，请先人工处理冲突：" + ", ".join(conflicts))
    hashed_file = directory / "requirements.hashed.txt"
    if not hashed_file.is_file():
        raise RuntimeError("尚未准备离线wheel；请先在原机器执行prepare")
    expected = wheel_requirements(directory / "wheelhouse", report)
    if expected != hashed_file.read_text(encoding="utf-8"):
        raise RuntimeError("安装包哈希或清单发生变化，请恢复原始部署基线")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--user", "--no-index", "--no-deps", "--require-hashes",
        "--find-links", str(directory / "wheelhouse"), "-r", str(hashed_file),
    ], check=True)
    verify_environment(args)

def build_parser() -> argparse.ArgumentParser:
    """定义采集、安装包准备、平台检查、安装和验收五个命令。"""
    parser = argparse.ArgumentParser(description="v0.96环境采集与固定版本部署")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="在已经跑通项目的机器上采集")
    capture.add_argument("--project-root", default=".", help="项目根目录")
    capture.add_argument("--output", required=True, help="新的环境基线目录")
    capture.add_argument("--include", action="append", choices=tuple(OPTIONAL_PACKAGES), default=[])
    capture.add_argument("--extra-package", action="append", default=[], help="额外发行包名称，可重复")
    capture.set_defaults(function=capture_environment)
    prepare = commands.add_parser("prepare", help="下载与采集版本一致的安装包")
    prepare.add_argument("--snapshot", required=True, help="环境基线目录")
    prepare.add_argument("--index-url", default="https://pypi.org/simple", help="主Python包索引")
    prepare.add_argument("--extra-index-url", action="append", default=[], help="附加可信索引")
    prepare.add_argument("--find-links", action="append", default=[], help="保存原始SDK等wheel的本地目录")
    prepare.set_defaults(function=prepare_wheels)
    for name, function in (("preflight", preflight), ("install", install_environment), ("verify", verify_environment)):
        command = commands.add_parser(name, help=f"执行{name}步骤")
        command.add_argument("--snapshot", required=True, help="环境基线目录")
        command.add_argument("--ros", action="store_true", help="额外检查rclpy和cv_bridge能否导入")
        command.set_defaults(function=function)
    return parser

def main() -> None:
    """统一处理命令和错误；失败时返回非零退出码，让部署脚本立即停止。"""
    args = build_parser().parse_args()
    try:
        args.function(args)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"环境工具失败：{error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
