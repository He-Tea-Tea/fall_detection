#!/usr/bin/env bash
# 给fall创建项目内依赖；不卸载或升级共享用户目录中的Python包。
# 用法：bash install_fall_local.sh v1；升级时使用一个新名称，例如v2。
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_name="${1:-v1}"
if [[ $# -gt 1 || ! "$release_name" =~ ^v[0-9]+$ ]]; then
    echo "目录名必须是v加数字，例如：bash install_fall_local.sh v1" >&2
    exit 1
fi
if [[ "$EUID" -eq 0 || -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
    echo "请使用普通用户的系统Python，不要sudo运行本脚本或激活其他Python环境。" >&2
    exit 1
fi

# 该候选清单按用户报告的平台制作；其他系统需要另行选择依赖。
/usr/bin/python3 -I -S - <<'PY'
import platform
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 10), "需要系统Python 3.10"
assert platform.system() == "Linux" and platform.machine() == "x86_64", "需要Linux x86_64"
release = Path('/etc/os-release').read_text(encoding='utf-8')
assert 'ID=ubuntu' in release and 'VERSION_ID="22.04"' in release, "需要Ubuntu 22.04"
PY

cd -- "$project_dir"
test -f fall_local_python.py
test -f requirements.fall-local.txt
target_dir="$project_dir/python_packages/$release_name"
if [[ -e "$target_dir" ]]; then
    echo "目录已存在，拒绝混装：$target_dir；请换一个新版本名。" >&2
    exit 1
fi
if ! /usr/bin/python3 -I -m pip --version; then
    echo "系统Python没有pip，请先通过apt安装python3-pip。" >&2
    exit 1
fi
mkdir -p -- "$project_dir/python_packages" "$project_dir/deploy"

# 先把构建工具安装到工程内独立目录，避免使用共享环境中的setuptools 84。
# 安装的都是wheel；业务依赖安装再显式关闭构建隔离，不创建venv/Conda环境。
tools_dir="$(mktemp -d "$project_dir/python_packages/build_tools.XXXXXX")"
tools_name="$(basename -- "$tools_dir")"
echo "[1/3] 准备项目内构建工具：$tools_dir"
/usr/bin/python3 -I -m pip --isolated install \
    --target "$tools_dir" \
    --ignore-installed \
    --only-binary=:all: \
    --index-url https://pypi.org/simple \
    "pip==25.1.1" "setuptools==79.0.1" "wheel==0.45.1" "packaging==25.0"

# --target明确安装位置；--ignore-installed要求依赖完整落入新目录，不借用系统包。
# Linux的pynput可能依赖evdev源码包，因此仅允许evdev编译；其他包要求预编译wheel。
# --no-build-isolation让evdev使用上面准备的项目内构建工具。
echo "[2/3] 安装业务依赖：$target_dir"
/usr/bin/python3 -I -S fall_local_python.py --deps "$tools_name" -m pip --isolated install \
    --target "$target_dir" \
    --ignore-installed \
    --no-build-isolation \
    --only-binary=:all: \
    --no-binary=evdev \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --report "$project_dir/deploy/install-$release_name.json" \
    -r "$project_dir/requirements.fall-local.txt"

# 在同一个严格限制搜索路径的启动方式下检查，只检查fall本地依赖，不读取colcon环境。
echo "[3/3] 检查本地依赖关系"
/usr/bin/python3 -I -S fall_local_python.py --deps "$release_name" -m pip check
echo "安装完成。下一步请用fall_local_python.py的--check检查真实导入，再运行main.py --self-test。"
