# 跌倒检测系统 v0.91：Ubuntu 22.04 + ROS 2 Humble 部署与使用指南

## 1. 文档目的

本文用于把已经在 Windows 上验证的跌倒检测系统 v0.91，以压缩包方式复制到 Ubuntu 22.04 工控机，并在不使用 Conda、venv 或其他 Python 虚拟环境的前提下完成安装、测试、运行和开机启动。

本文默认条件如下：

- 操作系统：Ubuntu 22.04 x86_64。
- ROS版本：ROS 2 Humble。
- Python版本：Ubuntu 22.04自带的Python 3.10。
- 深度相机：Orbbec Gemini 335Le，以太网/PoE连接。
- 人体模型：YOLO26 Pose。
- 场景模型：YOLO26 Seg。
- 当前模型格式：优先使用Windows已经验证的PT模型。
- AI复核：火山方舟视觉模型。
- 对外通讯：HTTP POST JSON，发送最终true或false。

## 2. 先理解当前程序与ROS 2的关系

当前v0.91不是ROS 2节点。`main.py`直接使用`pyorbbecsdk`打开Gemini 335Le，直接运行YOLO模型，再通过HTTP输出最终结果。

因此，当前部署方式是：

```text
Ubuntu 22.04 + ROS 2 Humble系统
└── 独立Python程序main.py
    ├── 直接打开Gemini 335Le
    ├── YOLO Pose/Seg推理
    ├── P/H/V/S/C五维本地判断
    ├── 豆包视觉AI复核
    ├── 最终结果融合
    ├── 日志记录
    └── HTTP发送true/false
```

这意味着：

1. 可以在已经安装ROS 2 Humble的工控机上直接运行`python3 main.py`。
2. 当前版本不需要`colcon build`，也不需要放进ROS 2工作空间。
3. 当前版本不会发布或订阅ROS 2话题。
4. 不要同时启动Orbbec ROS 2相机节点和`main.py`去占用同一台相机。
5. 后续如果抓取模块也要使用这台相机，应改成“一个相机采集节点，多个消费者共享RGB/Depth”，不能让多个程序各自创建相机Pipeline。

ROS 2 Humble官方维护期到2027年5月。当前部署可以继续使用Humble，但正式产品应提前安排后续ROS版本迁移评估。

## 3. 不使用虚拟环境时采用什么安装方式

本文采用：

```bash
python3 -m pip install --user ...
```

`--user`表示把Python第三方包安装到当前Linux用户的`~/.local/lib/python3.10/site-packages`，而不是安装到虚拟环境，也不直接覆盖`/usr/lib/python3`中的Ubuntu/ROS系统包。

必须遵守以下规则：

1. 不使用`sudo pip install`。
2. 所有依赖由将来实际运行跌倒检测程序的同一个Linux用户安装。
3. systemd服务的`User=`必须与安装这些`--user`依赖的用户相同。
4. 不要在同一个用户环境中随意升级NumPy、OpenCV、PyTorch和Ultralytics。
5. 安装完成后保存版本锁定清单，后续升级前先备份。

这种方式符合“不使用虚拟环境”的要求，但隔离性弱于venv。为了稳定，建议工控机使用一个固定的机器人运行账户，不要让不同项目反复修改该账户的Python包。

## 4. Windows端压缩前检查

### 4.1 压缩包必须包含

至少确认以下文件存在：

```text
main.py
config.yaml
logging_utils.py
ground_detector.py
get_intrinsics.py
pose_2D.py
pose_3D.py
height.py
velocity.py
static.py
scene.py
fall_detector.py
ai_verifier.py
ai_test.py
decision_fusion.py
alert_manager.py
http_bridge.py
receiver.py
yolo26s-pose.pt
yolo26s-seg.pt
```

如果模型放在单独的`models`目录，也可以，但必须同步修改`config.yaml`中的模型路径。

### 4.2 不要打包的内容

以下内容不需要复制到Ubuntu：

```text
fall_env
.venv
__pycache__
*.pyc
Log目录中的旧日志
debug_ai目录中的调试图片
Windows的site-packages
Windows版DLL或WHL文件
真实API Key
```

Windows的Python环境不能直接复制到Linux，因为二进制依赖、OpenCV、PyTorch和Orbbec SDK的平台格式不同。

### 4.3 校验模型文件

在Windows PowerShell执行：

```powershell
Get-FileHash .\yolo26s-pose.pt -Algorithm SHA256
Get-FileHash .\yolo26s-seg.pt -Algorithm SHA256
```

保存两个SHA-256值。复制到Ubuntu后再次计算，两个系统的结果应完全一致。

### 4.4 Linux文件名区分大小写

代码中的文件名必须与导入名称完全一致：

```text
pose_2D.py
pose_3D.py
```

不要随意改成`pose_2d.py`或`pose_3d.py`，否则Linux可能出现`ModuleNotFoundError`。

## 5. 工控机硬件和系统检查

先打开终端执行：

```bash
uname -m
lsb_release -a
python3 --version
whoami
echo "$ROS_DISTRO"
source /opt/ros/humble/setup.bash
echo "$ROS_DISTRO"
lscpu | sed -n '1,25p'
free -h
lspci | grep -Ei 'vga|3d|display|ethernet'
nvidia-smi
```

预期结果：

| 检查项 | 正常结果 | 作用 |
|---|---|---|
| CPU架构 | `x86_64` | 决定Orbbec和PyTorch安装包架构 |
| Ubuntu | `22.04` | 与ROS 2 Humble官方目标系统一致 |
| Python | `3.10.x` | 与Ubuntu 22.04、ROS 2 Humble匹配 |
| ROS_DISTRO | `humble` | 说明ROS环境加载成功 |
| NVIDIA GPU | `nvidia-smi`能显示型号 | 可使用CUDA加速 |
| 无NVIDIA GPU | `nvidia-smi`命令失败 | 使用CPU，后续再评估ONNX/OpenVINO |

如果`ROS_DISTRO`为空，在`~/.bashrc`末尾加入：

```bash
source /opt/ros/humble/setup.bash
```

然后执行：

```bash
source ~/.bashrc
echo "$ROS_DISTRO"
```

## 6. 安装Ubuntu系统依赖

执行：

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-dev \
  build-essential cmake git unzip curl ca-certificates \
  libgl1 libglib2.0-0 libgtk-3-0 libusb-1.0-0 \
  usbutils ethtool net-tools
```

说明：

- `python3-pip`用于安装项目Python依赖。
- `python3-dev`和`build-essential`用于少量需要本机编译的包。
- `libgl1`、`libglib2.0-0`和`libgtk-3-0`用于OpenCV显示窗口。
- `ethtool`和`net-tools`用于检查Gemini 335Le以太网连接。
- 当前YOLO Pose版本不需要MediaPipe，不要额外安装`mediapipe`。
- 当前HTTP代码使用Python标准库，不需要安装Flask或requests。

## 7. 配置Gemini 335Le以太网

Gemini 335Le使用千兆以太网/PoE。官方快速指南给出的默认相机IP通常是：

```text
192.168.1.10
```

### 7.1 推荐网络结构

工控机最好有两条独立网络：

```text
网卡1：连接Gemini 335Le，只负责相机数据
网卡2或Wi-Fi：连接互联网，只负责豆包API和其他网络服务
```

相机网卡不要设置默认网关，否则互联网流量可能错误地走相机网卡，造成AI请求超时。

### 7.2 图形界面配置方法

在Ubuntu“设置→网络→对应有线网卡→IPv4”中设置：

```text
IPv4方式：手动
地址：192.168.1.100
子网掩码：255.255.255.0
网关：留空
DNS：留空
```

保存后重新连接该网卡。

### 7.3 命令行配置方法

先查看网卡名称：

```bash
ip -br link
nmcli device status
```

假设连接相机的网卡名为`enp3s0`。下面的`enp3s0`必须换成工控机的真实网卡名：

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enp3s0 \
  con-name gemini-camera \
  ipv4.method manual \
  ipv4.addresses 192.168.1.100/24 \
  ipv4.never-default yes \
  ipv6.method disabled

sudo nmcli connection up gemini-camera
```

如果该网卡已经存在连接配置，不要重复创建同名连接，可在图形界面修改原连接。

### 7.4 网络验证

执行：

```bash
ip -br address
ip route
ping -c 4 192.168.1.10
ip route get 192.168.1.10
curl -I --max-time 5 https://ark.cn-beijing.volces.com
```

合格条件：

1. 相机IP可以连通。
2. 访问相机IP时走相机专用网卡。
3. 互联网请求走另一张网卡或Wi-Fi。
4. 相机网卡是千兆连接，可用以下命令检查：

```bash
sudo ethtool enp3s0 | grep -E 'Speed|Duplex|Link detected'
```

正常应优先看到`Speed: 1000Mb/s`、`Duplex: Full`和`Link detected: yes`。

## 8. 解压v0.91项目

假设压缩包已经复制到Ubuntu的“下载”目录：

```bash
unzip -l ~/下载/fall_detection_v0.91.zip | head -50
mkdir -p ~/fall_detection_v0.91
unzip ~/下载/fall_detection_v0.91.zip -d ~/fall_detection_v0.91
find ~/fall_detection_v0.91 -maxdepth 3 -name main.py -print
```

根据`find`输出进入真正包含`main.py`的目录。例如：

```bash
cd ~/fall_detection_v0.91
ls
```

如果压缩包内部还有一层目录，就进入那一层，不要在找不到`main.py`的目录运行程序。

计算Ubuntu模型哈希：

```bash
sha256sum yolo26s-pose.pt
sha256sum yolo26s-seg.pt
```

与Windows记录的SHA-256比较，必须一致。

## 9. 安装Python依赖：不使用虚拟环境

### 9.1 查看当前用户环境

```bash
python3 -m pip --version
python3 -m pip list --user
python3 -m site --user-site
```

### 9.2 升级当前用户的安装工具

```bash
python3 -m pip install --user --upgrade pip setuptools wheel
```

不要在命令前加`sudo`。

### 9.3 先安装PyTorch

Ultralytics官方建议先根据工控机的计算平台安装PyTorch。

#### 情况A：NVIDIA GPU

先确保：

```bash
nvidia-smi
```

如果没有驱动，可先查看Ubuntu推荐驱动：

```bash
ubuntu-drivers devices
```

安装驱动和重启属于系统级操作，应选择Ubuntu为该GPU推荐的驱动版本。驱动安装完成并重启后，`nvidia-smi`必须正常。

然后访问PyTorch官方安装选择器，选择：

```text
OS：Linux
Package：Pip
Language：Python
Compute Platform：与NVIDIA驱动兼容的CUDA版本
```

把官方生成命令中的`pip3`改成`python3 -m pip install --user`后执行。不要根据网上旧教程随便固定CUDA版本。

安装后验证：

```bash
python3 - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

`cuda available`必须为`True`，否则`runtime.device: auto`仍会回退CPU。

#### 情况B：只有CPU

按PyTorch官方页面选择CPU安装命令。常见形式如下，但正式安装时仍以官方页面当前生成的命令为准：

```bash
python3 -m pip install --user torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
```

CPU环境中`torch.cuda.is_available()`返回`False`是正常现象。

### 9.4 安装v0.91基础依赖

为了先复现Windows验证环境，建议先使用已验证版本：

```bash
python3 -m pip install --user --upgrade-strategy only-if-needed \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "PyYAML>=6.0,<7.0" \
  "ultralytics==8.4.135" \
  "openai>=1.0,<3.0"
```

注意：

1. 不要同时安装`opencv-python`、`opencv-contrib-python`和`opencv-python-headless`。
2. 当前需要OpenCV窗口，所以使用`opencv-python`。
3. 正式无界面运行也可继续保留该版本，先不要在第一次迁移时改成headless。
4. NumPy先保持1.26.4，避免Orbbec SDK和既有代码受到NumPy 2.x变化影响。
5. 不要在第一次迁移时直接安装Ultralytics最新版，先复现v0.91已验证版本。

### 9.5 安装Orbbec Python SDK

当前代码导入：

```python
from pyorbbecsdk import Pipeline, AlignFilter, OBStreamType
```

PyPI包名是`pyorbbecsdk2`，导入模块名仍然是`pyorbbecsdk`。

先查看Windows已验证版本。如果Windows使用的是2.1.2，Ubuntu首先尝试安装相同版本：

```bash
python3 -m pip install --user --upgrade-strategy only-if-needed \
  "numpy==1.26.4" \
  "pyorbbecsdk2==2.1.2"
```

如果提示没有匹配的安装包：

1. 确认`uname -m`是不是`x86_64`。
2. 确认`python3 --version`是不是3.10。
3. 从Orbbec官方`pyorbbecsdk` Release下载包含`cp310`和`linux_x86_64`的WHL。
4. 使用下面格式安装真实文件：

```bash
python3 -m pip install --user /实际路径/pyorbbecsdk-版本-cp310-cp310-linux_x86_64.whl
```

不要把Windows的`win_amd64.whl`复制到Ubuntu使用。

### 9.6 执行Orbbec官方环境配置

安装完成后执行：

```bash
ORB_PACKAGE_DIR=$(python3 -c "import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))")
python3 "$ORB_PACKAGE_DIR/shared/setup_env.py"
```

该步骤用于完成Orbbec在Linux上的一次性系统环境配置。执行完成后重新连接或重启相机。

### 9.7 检查依赖冲突

```bash
python3 -m pip check
python3 -m pip list --user | grep -E 'numpy|opencv|torch|ultralytics|openai|orbbec|PyYAML'
```

如果`pip check`出现冲突，不要继续安装更多包，应先根据报错统一NumPy、OpenCV和PyTorch版本。

### 9.8 保存Ubuntu版本锁定文件

依赖验证通过后执行：

```bash
cd ~/fall_detection_v0.91
python3 -m pip freeze --user > requirements-v0.91-ubuntu.lock.txt
```

以后迁移另一台相同工控机时，以该文件作为版本参考，但GPU版PyTorch仍需根据目标机器驱动安装。

## 10. 一次性导入验证

在项目目录执行：

```bash
source /opt/ros/humble/setup.bash

python3 - <<'PY'
import importlib

modules = [
    "cv2",
    "numpy",
    "openai",
    "torch",
    "ultralytics",
    "yaml",
    "pyorbbecsdk",
]

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "未提供版本号")
        print(f"[PASS] {name}: {version}")
    except Exception as error:
        print(f"[FAIL] {name}: {error}")

try:
    from pyorbbecsdk import Pipeline, AlignFilter, OBStreamType
    print("[PASS] pyorbbecsdk核心类导入成功")
except Exception as error:
    print(f"[FAIL] pyorbbecsdk核心类导入失败: {error}")

try:
    import torch
    print("[INFO] CUDA available:", torch.cuda.is_available())
except Exception:
    pass
PY
```

如果这里失败，先解决依赖，不要直接运行`main.py`。

## 11. 检查和修改config.yaml

### 11.1 模型路径

如果模型和`main.py`位于同一目录，可保持：

```yaml
models:
  pose_model: yolo26s-pose.pt
  scene_model: yolo26s-seg.pt
```

如果放在`models`子目录，修改为：

```yaml
models:
  pose_model: models/yolo26s-pose.pt
  scene_model: models/yolo26s-seg.pt
```

相对路径以`config.yaml`所在目录为基准。

### 11.2 推理设备

初次部署建议：

```yaml
runtime:
  device: auto
  use_half: true
  pose_imgsz: 640
  scene_imgsz: 640
```

含义：

- 有可用CUDA时自动使用第一张NVIDIA GPU。
- 没有CUDA时自动使用CPU。
- 只有PT模型加CUDA时才真正启用FP16。
- CPU和ONNX不会由主程序强制使用FP16。

如果要明确强制CPU：

```yaml
runtime:
  device: cpu
```

如果要指定第一张NVIDIA GPU：

```yaml
runtime:
  device: 0
```

### 11.3 显示模式

第一次现场调试保持：

```yaml
display:
  enabled: true
```

部署成无界面systemd服务前改为：

```yaml
display:
  enabled: false
```

### 11.4 日志配置

建议保持v0.91默认设置：

```yaml
logging:
  enabled: true
  console_enabled: true
  level: INFO
  directory: Log
  filename: fall_detection.log
  max_file_size_mb: 10
  backup_count: 5
  repeated_warning_interval_s: 5.0
```

日志保存到项目目录下的`Log/fall_detection.log`，单个文件达到10MB后自动轮转，最多保留5份历史日志。

### 11.5 AI模型配置

确认以下字段使用的是火山方舟控制台中已经开通并且支持视觉输入的真实模型或推理接入点ID：

```yaml
ai:
  enabled: true
  base_url: https://ark.cn-beijing.volces.com/api/v3
  api_key_env: ARK_API_KEY
  api_key: ""
  model: 这里填写已开通的视觉模型或Endpoint ID
```

如果出现`HTTP 404 ModelNotOpen`，说明账号没有开通当前模型，或者`ai.model`填写错误，不是程序的图片编码问题。

如果工控机暂时没有互联网，可先设置：

```yaml
ai:
  enabled: false
```

系统会继续执行本地判断。

### 11.6 HTTP配置

同一台工控机上的另一个程序接收结果时，保持环回地址：

```yaml
http_bridge:
  enabled: true
  target_url: http://127.0.0.1:8080/fall
  listen_host: 127.0.0.1
  listen_port: 8123
  timeout_s: 2.0
  retry_count: 2
  retry_interval_s: 0.3
```

如果暂时不接收HTTP结果，可设置：

```yaml
http_bridge:
  enabled: false
```

## 12. 配置API Key

### 12.1 当前终端临时配置

```bash
export ARK_API_KEY='替换成你重新生成的真实密钥'
```

验证变量是否存在，但不要输出密钥内容：

```bash
python3 - <<'PY'
import os
value = os.getenv("ARK_API_KEY", "")
print("ARK_API_KEY configured:", bool(value.strip()))
PY
```

不要把真实密钥写进Git、截图、日志或部署文档。

### 12.2 systemd使用的永久配置

创建仅管理员和服务可读取的环境文件：

```bash
sudo install -m 600 /dev/null /etc/fall_detection.env
sudo nano /etc/fall_detection.env
```

文件内容：

```text
ARK_API_KEY=替换成真实密钥
```

不要在等号两边加空格。

## 13. 先验证相机，再运行算法

### 13.1 不要同时运行ROS 2相机驱动

运行下面测试前，先停止可能正在使用相机的ROS 2节点：

```bash
ros2 node list
```

如果已经启动Orbbec ROS 2相机节点，应先关闭对应launch终端。当前`main.py`会自己打开相机。

### 13.2 检查相机内参

在项目目录运行：

```bash
python3 get_intrinsics.py
```

如果无法打开相机，先检查：

```bash
ping -c 4 192.168.1.10
ip route get 192.168.1.10
```

不要先修改跌倒算法阈值。

### 13.3 Ubuntu现场重新标定地面

压缩包中的Windows版`ground.yaml`只能作为备份。相机搬到工控机和机器人后，安装位置、云台角度或相机姿态可能变化，因此必须在最终安装姿态下重新运行：

```bash
python3 ground_detector.py
```

标定时要求：

1. 机器人停稳。
2. 云台Yaw和Pitch固定到计划运行的零位。
3. 地面区域尽量完整、平整、无遮挡。
4. RGB和Depth分辨率保持正式运行设置。
5. 标定完成后检查新的`ground.yaml`是否生成。

当前`strict_ground_calibration`只能检查分辨率和内参是否一致，不能自动发现相机外部姿态发生变化。因此，即使程序没有报标定错误，相机Pitch/Roll变化后地面平面也可能已经失效。

## 14. 无相机自测试

在项目目录执行：

```bash
python3 main.py --self-test
```

它会依次检查：

1. P3D三维姿态。
2. P2D二维姿态。
3. H高度评分。
4. V速度评分。
5. S静止评分。
6. C场景评分。
7. 本地二值跌倒状态机。
8. AI请求封装和断网降级。
9. 决策融合和告警。
10. Depth缺失时的P2D降级路径。
11. 主流程公共接口。

所有模块都应出现`PASS`。如果自测试失败，不要继续真实相机测试。

也可以分别运行：

```bash
python3 pose_3D.py --self-test
python3 pose_2D.py --self-test
python3 height.py --self-test
python3 velocity.py --self-test
python3 static.py --self-test
python3 scene.py --self-test
python3 fall_detector.py --self-test
python3 ai_verifier.py --self-test
```

## 15. AI单图测试

准备一张测试图片后执行：

```bash
python3 ai_verifier.py --api-test /实际路径/test.jpg
```

合格条件：

1. 图片能够完成JPEG编码。
2. 请求成功返回。
3. 返回值可以解析为`true`、`false`或`uncertain`。
4. 日志中有模型名称、耗时、图片大小和结论。
5. 日志中没有API Key和Base64图片正文。

如果超时：

```bash
ip route
curl -I --max-time 5 https://ark.cn-beijing.volces.com
```

重点检查相机专用网卡是否错误地成为默认网关。

## 16. HTTP通讯测试

### 16.1 启动示例接收端

打开终端A：

```bash
cd ~/fall_detection_v0.91
python3 receiver.py
```

如果项目多套了一层目录，应进入实际含`receiver.py`的目录。

### 16.2 手动发送结果

打开终端B：

```bash
curl -X POST http://127.0.0.1:8080/fall \
  -H 'Content-Type: application/json' \
  -d '{"fall":true}'

curl -X POST http://127.0.0.1:8080/fall \
  -H 'Content-Type: application/json' \
  -d '{"fall":false}'
```

### 16.3 测试重置接口

`main.py`运行后，可以重置全部人员：

```bash
curl -X POST http://127.0.0.1:8123/reset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

也可以重置指定人员：

```bash
curl -X POST http://127.0.0.1:8123/reset \
  -H 'Content-Type: application/json' \
  -d '{"person_id":2}'
```

HTTP发送规则是“最终状态发生变化时发送一次”，不是每帧重复发送。

## 17. 按模块打开相机测试

建议按照以下顺序测试：

```bash
python3 main.py --stage pose_2d
python3 main.py --stage pose_3d
python3 main.py --stage height
python3 main.py --stage velocity
python3 main.py --stage static
python3 main.py --stage scene
python3 main.py --stage fall_detector
python3 main.py --stage ai
python3 main.py --stage full
```

窗口中按`Esc`退出。

也可以直接运行各文件打开对应功能窗口：

```bash
python3 pose_2D.py
python3 pose_3D.py
python3 height.py
python3 velocity.py
python3 static.py
python3 scene.py
python3 fall_detector.py
python3 ai_verifier.py
```

## 18. 正式完整运行

开发调试模式：

```bash
cd ~/fall_detection_v0.91
export ARK_API_KEY='替换成真实密钥'
python3 main.py --stage full
```

默认`--stage`就是`full`，所以也可以：

```bash
python3 main.py
```

程序正常启动后，日志应显示：

1. 日志系统初始化完成。
2. 当前stage为full。
3. Pose和Seg模型路径。
4. 推理设备是CPU或CUDA。
5. FP16是否启用。
6. HTTP reset服务地址。
7. 相机流程启动成功。

## 19. 查看v0.91日志

实时查看日志文件：

```bash
tail -f Log/fall_detection.log
```

查看最近100行：

```bash
tail -n 100 Log/fall_detection.log
```

只查看警告和错误：

```bash
grep -E 'WARNING|ERROR|CRITICAL' Log/fall_detection.log
```

日志轮转后可能出现：

```text
fall_detection.log
fall_detection.log.1
fall_detection.log.2
```

当前日志不会记录API Key、完整Base64图片或完整AI请求内容。

## 20. 工控机性能检查

运行完整流程时另开终端：

```bash
htop
watch -n 1 nvidia-smi
```

如果没有`htop`：

```bash
sudo apt install -y htop
```

重点记录：

| 项目 | 观察内容 |
|---|---|
| CPU | 是否长时间接近100% |
| GPU利用率 | CUDA是否真正工作 |
| GPU显存 | Pose和Seg是否导致显存不足 |
| 内存 | 是否持续增长 |
| 温度 | 长时间运行是否降频 |
| 实际FPS | 是否满足跌倒检测时序要求 |
| AI耗时 | 是否经常超时 |
| 网络 | RGB-D是否丢帧、AI是否断网 |

第一次Ubuntu迁移建议保留PT模型和640输入尺寸，先验证结果一致，再单独测试ONNX、TensorRT或更小模型。不要同时更换操作系统、模型格式、输入尺寸和判断阈值，否则出现差异时难以定位原因。

## 21. systemd开机自动运行

只有在终端连续稳定运行数小时后，再配置systemd。

### 21.1 准备绝对路径

执行：

```bash
whoami
readlink -f ~/fall_detection_v0.91
python3 -m site --user-site
```

记录真实用户名、项目绝对路径和用户Python包路径。

### 21.2 关闭OpenCV窗口

编辑`config.yaml`：

```yaml
display:
  enabled: false
```

### 21.3 创建服务文件

```bash
sudo nano /etc/systemd/system/fall-detection.service
```

示例内容如下。`robot`和`/home/robot/fall_detection_v0.91`必须替换成真实值：

```ini
[Unit]
Description=Fall Detection v0.91
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=robot
Group=robot
WorkingDirectory=/home/robot/fall_detection_v0.91
EnvironmentFile=/etc/fall_detection.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /home/robot/fall_detection_v0.91/main.py --stage full
Restart=on-failure
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
```

因为依赖使用`pip --user`安装，`User=robot`必须就是执行安装命令的账户。

### 21.4 先用服务账户验证导入

将`robot`替换成真实用户：

```bash
sudo -u robot -H /usr/bin/python3 - <<'PY'
import cv2
import torch
import ultralytics
from pyorbbecsdk import Pipeline
print("service user imports: PASS")
PY
```

### 21.5 启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable fall-detection.service
sudo systemctl start fall-detection.service
systemctl status fall-detection.service
```

实时查看systemd日志：

```bash
journalctl -u fall-detection.service -f
```

停止和重启：

```bash
sudo systemctl stop fall-detection.service
sudo systemctl restart fall-detection.service
```

如果服务无法导入用户安装的Python包，先确认`User=`是否正确，再运行`python3 -m site --user-site`检查路径。不要直接改成root运行。

## 22. ROS 2 Humble后续接入方式

当前v0.91先作为独立程序运行，最终结果通过HTTP发送给另一个模块。后续接入ROS 2时建议选择以下结构：

```text
Orbbec Camera Manager ROS 2节点
├── 发布RGB、Depth、CameraInfo和相机状态
├── 跌倒检测节点订阅
└── 抓取识别节点订阅
```

建议的ROS 2结果接口：

```text
/fall_detection/result
```

消息至少应包含：

```text
timestamp
person_id
is_fall
confidence
source
event_id
```

但这属于后续ROS化工作。当前v0.91仍按HTTP发送：

```json
{"fall": true}
```

或：

```json
{"fall": false}
```

## 23. 移动机器人和云台的限制

这是当前版本部署时最重要的边界。

### 23.1 当前版本适合什么状态

v0.91适合先在以下条件下验收：

1. 机器人底盘静止。
2. 相机安装结构固定。
3. 云台Yaw/Pitch固定。
4. 相机姿态与生成`ground.yaml`时一致。

### 23.2 为什么移动时可能不准

- 云台Pitch或机器人Roll/Pitch变化会让旧地面法向量失效，影响P3D和H。
- 机器人自身前后运动会改变人体在相机坐标系中的位置，影响V和S。
- 快速转头可能造成关键点模糊、Track ID变化和Depth对齐质量下降。
- 只检查相机内参无法发现相机外部姿态变化。

### 23.3 正式移动版本应增加

1. 底盘IMU姿态。
2. 云台Yaw/Pitch编码器角度。
3. 相机到云台、云台到机器人基座的固定外参。
4. 相机帧、IMU和关节角的时间同步。
5. 每帧动态坐标变换，把3D数据转换到重力对齐坐标系。
6. 机器人快速运动时降低V和S的权重或暂时标记无效。
7. 相机只由一个Camera Manager管理，并向抓取和跌倒检测共享数据。

在这些补偿完成前，不建议把移动状态下的V和S作为高可信告警证据。

## 24. 常见问题处理

### 24.1 ModuleNotFoundError

先确认当前目录：

```bash
pwd
ls
```

必须在包含`main.py`、`scene.py`、`pose_2D.py`和`pose_3D.py`的目录运行。

再确认文件名大小写完全一致。

### 24.2 找不到pyorbbecsdk

```bash
python3 -m pip show pyorbbecsdk2
python3 -c "import pyorbbecsdk; print(pyorbbecsdk.__file__)"
```

如果第一条有结果、第二条失败，检查安装用户和运行用户是否一致。

### 24.3 相机IP能ping通但程序无图像

依次检查：

1. Orbbec Viewer能否打开彩色和Depth流。
2. 是否有ROS 2相机节点或其他程序占用相机。
3. 网卡是否千兆全双工。
4. 相机和工控机是否在同一网段。
5. Orbbec SDK、相机固件和设备型号是否匹配。
6. 防火墙是否阻止相机数据。

### 24.4 OpenCV窗口打不开

确认：

```bash
echo "$DISPLAY"
python3 -c "import cv2; print(cv2.__version__)"
```

如果通过SSH运行，需要正确的图形转发；生产systemd模式应设置`display.enabled: false`。

### 24.5 AI返回ModelNotOpen

原因通常是：

1. `ai.model`填写的模型未开通。
2. 填写了错误的模型名而不是实际Endpoint ID。
3. API Key所属账号无权访问该模型。

进入火山方舟控制台确认已开通的视觉模型和接入点ID。

### 24.6 AI经常Request timed out

优先检查：

1. 相机网卡有没有错误的默认网关。
2. 工控机是否能访问火山方舟域名。
3. DNS是否正常。
4. `ai.request.timeout_s`是否过短。
5. 图片JPEG大小是否异常。

不要因为AI超时停止本地检测。v0.91会记录错误并继续使用本地判断。

### 24.7 日志文件没有生成

检查：

```bash
grep -A8 '^logging:' config.yaml
ls -ld . Log
```

运行用户必须对项目目录和`Log`目录有写权限。

### 24.8 ROS 2程序和main.py抢相机

只能保留一个相机拥有者。当前阶段关闭Orbbec ROS 2节点，只运行`main.py`。后续改为统一Camera Manager后，再让多个模块订阅同一份数据。

## 25. 首次部署验收顺序

严格按以下顺序执行：

```text
1. 检查Ubuntu、Python、ROS、CPU和GPU
2. 安装系统依赖
3. 配置335Le相机专用网卡
4. 解压v0.91并校验模型哈希
5. 使用pip --user安装PyTorch和项目依赖
6. 安装并配置pyorbbecsdk2
7. 完成全部Python导入测试
8. 检查config.yaml模型、设备、AI、HTTP和日志参数
9. 运行get_intrinsics.py验证相机
10. 在最终安装姿态重新运行ground_detector.py
11. 运行main.py --self-test
12. 运行AI单图测试
13. 运行HTTP收发和reset测试
14. 逐个运行Pose、Height、Velocity、Static和Scene窗口
15. 运行main.py --stage full
16. 检查终端日志和Log/fall_detection.log
17. 连续运行至少数小时
18. 关闭显示后配置systemd
```

## 26. 最终验收表

| 类别 | 合格条件 |
|---|---|
| Python环境 | 使用Python 3.10，`pip check`无冲突 |
| ROS环境 | `ROS_DISTRO=humble`，但当前程序独立运行 |
| 相机网络 | 335Le可访问，相机网卡千兆且没有默认网关 |
| RGB-D | RGB和Depth持续稳定，D2C后尺寸一致 |
| 模型 | Pose/Seg文件哈希与Windows一致 |
| 推理设备 | 日志中的CPU/CUDA选择符合预期 |
| P2D/P3D | 站立和躺卧角度方向正确 |
| H/V/S/C | 变化趋势符合真实动作和场景 |
| 降级 | Depth缺失时P2D路径仍能更新，程序不崩溃 |
| AI | 单图请求成功，断网时自动使用本地判断 |
| HTTP | 最终状态变化时发送一次true/false |
| Reset | 能重置全部人员或指定person_id |
| 日志 | 终端和文件均有记录，轮转配置生效 |
| 稳定性 | 长时间运行无持续内存增长、无频繁相机断流 |
| 安全 | API Key未写入代码、日志和Git |

## 27. 部署完成后的备份

保存Ubuntu现场配置和地面标定：

```bash
cp config.yaml config.v0.91.ubuntu.yaml
cp ground.yaml ground.v0.91.ubuntu.yaml
sha256sum yolo26s-pose.pt yolo26s-seg.pt > model-sha256.txt
python3 -m pip freeze --user > requirements-v0.91-ubuntu.lock.txt
```

备份内容至少包括：

```text
config.v0.91.ubuntu.yaml
ground.v0.91.ubuntu.yaml
model-sha256.txt
requirements-v0.91-ubuntu.lock.txt
systemd服务文件
```

API Key不要放入普通备份包。

## 28. 官方参考

1. ROS 2 Humble版本与维护周期：https://docs.ros.org/en/humble/Releases.html
2. Orbbec pyorbbecsdk官方仓库：https://github.com/orbbec/pyorbbecsdk
3. Gemini 335Le以太网快速指南：https://www.orbbec.com/docs/gemini-335le-quick-start/
4. Gemini 335Le产品规格：https://www.orbbec.com/gemini-335le/
5. PyTorch Linux安装选择器：https://pytorch.org/get-started/locally/
6. Ultralytics安装说明：https://docs.ultralytics.com/quickstart

## 29. 当前v0.91部署结论

当前v0.91可以在Ubuntu 22.04和ROS 2 Humble工控机上运行，并且不要求使用虚拟环境。推荐使用系统Python 3.10和当前运行用户的`pip --user`目录安装依赖，运行时使用`python3 main.py`。

第一次部署应保持PT模型、原输入尺寸和原判断阈值不变，先完成Windows与Ubuntu结果一致性验证。当前版本虽然能在ROS 2系统中运行，但仍是独立Python程序；相机必须由它独占。机器人移动、底盘倾斜或云台Pitch变化时，P3D、H、V和S可能受到影响，正式移动运行前仍需接入IMU、云台角度和动态坐标变换。
