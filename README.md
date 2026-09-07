# Fall Detection v0.96 部署说明

## 项目概述

基于 Orbbec Gemini 335Le 深度相机的跌倒检测系统，集成了 YOLO Pose/Seg、本地评分机制（P/H/V/S/C）、可选的视觉AI验证以及HTTP告警功能。

---

## 一、快速开始

在项目根目录下执行：

```bash
python main.py --config config.yaml
```

运行硬件无关的自检：

```bash
python main.py --self-test --config config.yaml
python -m unittest discover -s tests -v
python -m compileall -q .
```

---

## 二、目录结构总览

```
C:\Users\HTT\Desktop\v0.96\
├─ assets\                      资源文件目录
│  ├─ models\                   YOLO模型文件
│  ├─ calibration\              地面标定文件
│  └─ images\                   测试图片/回放图片
├─ audio\                       音频文件（告警语音）
├─ Log\                         运行日志（程序自动生成）
├─ debug_ai\                    AI调试图片（程序自动生成）
├─ fall_detection\              全部Python代码
├─ tests\                       单元测试代码
├─ docs\                        项目说明文档
├─ config.yaml                  运行配置文件
└─ main.py                      程序启动入口
```

---

## 三、各目录详细说明

### 1. 模型文件（YOLO）

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\assets\models\
```

**文件示例：**

```
yolo26s-pose.pt          # 姿态估计模型（默认）
yolo26s-seg.pt           # 场景分割模型（默认）
yolo26s-pose.onnx        # ONNX格式姿态模型
yolo26s-seg.onnx         # ONNX格式分割模型
```

**配置说明：**

当前配置默认读取：
- `assets/models/yolo26s-pose.pt`
- `assets/models/yolo26s-seg.pt`

如需改用 ONNX 格式，需同步修改 `config.yaml` 中的：
- `models.pose_model`
- `models.scene_model`

---

### 2. 地面标定文件

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\assets\calibration\ground.yaml
```

**说明：**

这是当前相机对应的地面平面、相机高度、深度坐标等标定数据。

⚠️ **重要**：不能直接使用其他相机的 `ground.yaml`，每台相机需单独标定。

---

### 3. 测试图片 / 回放图片

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\assets\images\
```

**支持格式：**

- `.jpg`
- `.jpeg`
- `.png`

**用途：**

- 离线测试
- 人工验收
- 算法回放

**注意：** 正式运行时此目录可以为空。

---

### 4. 音频文件（告警语音）

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\audio\
```

**当前配置对应文件：**

| 文件名 | 用途 | 配置项 |
|--------|------|--------|
| `Are you ok.wav` | 检测到最终跌倒时播放 | `audio.fall_file` |
| `care.wav` | 从 FALL 恢复到 NO_FALL 时播放 | `audio.recovery_file` |

**格式要求：** 推荐使用 WAV 格式。

**注意：** 文件名可自定义，修改后需同步更新 `config.yaml` 中对应的配置项。

---

### 5. 日志文件（程序自动生成）

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\Log\
```

**主要文件：**

- `fall_detection.log` - 主程序日志
- `OrbbecSDK.log.txt` - 相机SDK日志

⚠️ **重要**：`Log` 目录属于运行生成目录，不属于源代码，无需手动创建或放置文件。

---

### 6. AI 调试图片（程序自动生成）

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\debug_ai\
```

**启用方式：**

在 `config.yaml` 中设置：

```yaml
ai.debug.save_trigger_image: true
```

**说明：**

- 开启后，触发检测时的图像会被保存到该目录
- 用于调试和验证AI判断逻辑

⚠️ **正式部署建议关闭**：避免保存包含人员隐私的图像。

---

### 7. API 密钥

**安全要求：**

❌ 不要放在：
- Python 源码中
- `config.yaml` 中
- 普通文本文件中
- Git 仓库中

✅ 正确做法：

设置环境变量：

```bash
export ARK_API_KEY="你的API密钥"
```

或在 Windows 中设置系统环境变量 `ARK_API_KEY`。

---

### 8. 项目文档

**存放路径：**

```
C:\Users\HTT\Desktop\v0.96\docs\
```

**文档清单：**

- `ARCHITECTURE.md` - 系统架构说明
- `DEVELOPMENT_GUIDE.md` - 开发指南
- `AI_SETUP.md` - AI功能配置说明
- `V0.95_CHANGELOG.md` - v0.95版本更新日志
- `V0.96_STRUCTURE.txt` - v0.96目录结构说明

---

## 四、代码目录（仅放 Python 代码）

```
C:\Users\HTT\Desktop\v0.96\fall_detection\
```

**子目录：**

```
fall_detection/
├─ app/               应用层代码
├─ domain/            领域模型
├─ features/          功能模块
├─ perception/        感知模块
├─ integrations/      外部集成
└─ infrastructure/    基础设施
```

⚠️ **重要**：以下目录**只放 Python 代码**，不要放入模型、图片、音频或标定文件：

```
fall_detection/app
fall_detection/domain
fall_detection/features
fall_detection/perception
fall_detection/integrations
fall_detection/infrastructure
```

---

## 五、外部资源路径配置（config.yaml）

所有外部资源路径均在 `config.yaml` 中配置：

| 资源类型 | 配置项 | 默认路径 |
|----------|--------|----------|
| 姿态模型 | `models.pose_model` | `assets/models/yolo26s-pose.pt` |
| 场景模型 | `models.scene_model` | `assets/models/yolo26s-seg.pt` |
| 地面标定 | `calibration.ground_file` | `assets/calibration/ground.yaml` |
| 跌倒音频 | `audio.fall_file` | `audio/Are you ok.wav` |
| 恢复音频 | `audio.recovery_file` | `audio/care.wav` |

---

## 六、最终目录结构（完整版）

```
C:\Users\HTT\Desktop\v0.96\
│
├─ assets\
│  ├─ models\                      # YOLO模型文件
│  │  ├─ yolo26s-pose.pt
│  │  ├─ yolo26s-seg.pt
│  │  ├─ yolo26s-pose.onnx        # (可选)
│  │  └─ yolo26s-seg.onnx         # (可选)
│  ├─ calibration\
│  │  └─ ground.yaml              # 地面标定（每台相机独立）
│  └─ images\                     # 测试图片（运行时可为空）
│
├─ audio\                          # 告警音频
│  ├─ Are you ok.wav
│  └─ care.wav
│
├─ fall_detection\                # Python源码（仅代码）
│  ├─ app\
│  ├─ domain\
│  ├─ features\
│  ├─ perception\
│  ├─ integrations\
│  └─ infrastructure\
│
├─ tests\                          # 单元测试
│
├─ docs\                          # 项目文档
│  ├─ ARCHITECTURE.md
│  ├─ DEVELOPMENT_GUIDE.md
│  ├─ AI_SETUP.md
│  ├─ V0.95_CHANGELOG.md
│  └─ V0.96_STRUCTURE.txt
│
├─ Log\                           # 运行时日志（自动生成）
├─ debug_ai\                      # AI调试图片（自动生成）
├─ config.yaml                    # 主配置文件
└─ main.py                        # 程序入口
```

---

## 七、重要提醒

1. **模型文件**：确保 `assets/models/` 下有正确格式的模型文件（`.pt` 或 `.onnx`）。

2. **地面标定**：更换相机或改变安装位置后，必须重新运行地面标定工具生成新的 `ground.yaml`。

3. **API密钥**：通过环境变量 `ARK_API_KEY` 注入，不要硬编码。

4. **日志管理**：`Log/` 目录下日志会持续增长，建议定期清理或配置日志轮转。

5. **隐私保护**：生产环境务必关闭 `ai.debug.save_trigger_image`。

6. **音频文件**：如需自定义语音，确保文件名与 `config.yaml` 中的配置一致。

7. **路径兼容性**：代码中的路径均使用相对路径，只要从项目根目录启动即可正常工作。

---

> 文档版本：v0.96  
> 更新日期：2026-09-07  
> 适用范围：Orbbec Gemini 335Le + YOLO Pose/Seg 跌倒检测系统