#!/usr/bin/env bash
# Ubuntu 22.04部署入口：普通用户运行；系统库使用apt，Python包安装到当前用户。
# 用法：bash deploy/deploy_ubuntu.sh deploy/ubuntu22_py310_cpu
# 已预装系统库的离线机器：bash deploy/deploy_ubuntu.sh deploy/ubuntu22_py310_cpu --skip-apt
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
snapshot_arg="${1:-}"
apt_option="${2:-}"
if [[ -z "$snapshot_arg" || $# -gt 2 || ( -n "$apt_option" && "$apt_option" != "--skip-apt" ) ]]; then
    echo "用法：bash deploy/deploy_ubuntu.sh <环境基线目录> [--skip-apt]" >&2
    exit 1
fi
if [[ "$EUID" -eq 0 || -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
    echo "请用普通用户运行，不要sudo运行本脚本，也不要激活conda/venv。" >&2
    exit 1
fi
if [[ ! -x /usr/bin/python3 || ! -f /etc/os-release ]]; then
    echo "未找到Ubuntu系统Python或系统信息。" >&2
    exit 1
fi
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "本脚本只面向Ubuntu 22.04；其他系统需要单独建立并验证环境基线。" >&2
    exit 1
fi
python_minor="$(/usr/bin/python3 -c '
import sys
print(".".join(map(str, sys.version_info[:2])))
')"
if [[ "$python_minor" != "3.10" ]]; then
    echo "需要Ubuntu 22.04自带的Python 3.10，请不要替换/usr/bin/python3。" >&2
    exit 1
fi

# 先按调用者当前目录解析基线路径，再进入项目目录，避免相对路径指向错误。
snapshot_dir="$(/usr/bin/python3 -c '
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).resolve())
' "$snapshot_arg")"
/usr/bin/python3 "$script_dir/environment_tool.py" preflight --snapshot "$snapshot_dir"

# apt只安装必要运行库和Python安装工具；不卸载ROS，不升级整机，不安装模型或相机SDK。
if [[ "$apt_option" != "--skip-apt" ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-packaging python3-yaml \
        libgl1 libglib2.0-0 libgomp1 libusb-1.0-0 libudev1 \
        libsm6 libxext6 libxrender1 libxkbcommon-x11-0 libxcb-xinerama0 alsa-utils
fi

# 仅在已安装Humble时加载ROS，并追加共存检查。此脚本不会自动安装ROS。
ros_options=()
if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
    ros_options=(--ros)
fi
/usr/bin/python3 "$script_dir/environment_tool.py" install --snapshot "$snapshot_dir" "${ros_options[@]}"

# 无相机自测试检查源码接口；硬件、标定、模型、AI和HTTP仍由用户现场验收。
cd -- "$project_dir"
/usr/bin/python3 main.py --self-test
echo "依赖与无硬件测试完成。请按说明完成相机权限、模型文件及真实设备验收，再运行python3 main.py。"
