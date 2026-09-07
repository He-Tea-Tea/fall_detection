# v0.96 架构说明

## 适用范围

本版本为单进程、帧驱动的应用程序。它直接控制 Orbbec 相机数据流，运行 YOLO 推理，计算本地跌倒证据，可选地发送异步视觉AI请求，并通过 HTTP 和音频适配器发布状态变化事件。

当前版本**尚未**实现 ROS 2 节点，也**尚未**提供里程计/IMU/云台运动补偿功能。当前的运行前提是相机固定或已进行补偿。

---

## 目录职责

```
v0.96/
├── main.py                         兼容性启动入口
├── config.yaml                     运行时配置
├── VERSION                         版本标记文件
├── pyproject.toml                  Python包元数据
├── assets/
│   ├── models/                     本地模型资源
│   ├── calibration/                本地相机/地面标定资源
│   └── images/                     本地回放和测试资源
├── fall_detection/
│   ├── app/                        应用生命周期与编排
│   ├── domain/                     跌倒状态机与决策融合
│   ├── features/                   高度、速度和静止评分
│   ├── perception/                 姿态、场景、地面及测量质量门控
│   ├── integrations/               AI、HTTP和告警适配器
│   ├── infrastructure/             日志和音频运行时服务
│   └── tools/                      标定、转换和离线工具
├── tests/                          标准测试入口
└── docs/                           架构、部署和开发文档
```

---

## 运行时数据流

```
Orbbec RGB-D
    -> 帧对齐与转换
    -> YOLO Pose 追踪 + YOLO Seg 家具掩码
    -> PersonMeasurement（人体测量数据）
    -> P3D/P2D, H, V, S, C 五项评分
    -> FallDetector（本地 FALL/NO_FALL 状态）
    -> AIFallCoordinator（可选异步单图验证）
    -> DecisionFusion（决策融合）
    -> AlertManager（告警管理）
    -> HTTP/音频/控制台处理器
```

**重置路径**严格遵循跨线程边界的单向设计：

```
HTTP /reset 处理器 -> 重置队列 -> 主循环 -> 各模块 reset_person
```

HTTP 线程绝不直接修改检测器的历史记录。

---

## 依赖方向

推荐的依赖方向为：

```
perception（感知层） -> features（特征层） -> domain（领域层） -> app（应用层）
```

`integrations` 和 `infrastructure` 作为适配器，由 `app`/`domain` 层事件触发调用。

**设计约束：**

- **领域层（domain）**不得导入相机SDK、OpenCV窗口、HTTP服务器或AI SDK。
- **集成适配器（integrations）**不得访问检测器内部状态。

---

## 当前模块边界

| 模块 | 职责 |
|------|------|
| `pose_2d.py` / `pose_3d.py` | 计算独立的姿态证据 |
| `height.py` / `velocity.py` / `static.py` | 管理每个追踪目标的历史数据（高度、速度、静止） |
| `scene.py` | 分类地面/家具空间关系，输出场景评分（C） |
| `fall_detector.py` | 仅消费归一化的证据数据，拥有二值状态（FALL/NO_FALL） |
| `ai_verifier.py` | 负责请求调度、解析、超时和退避策略 |
| `decision_fusion.py` | 负责信源优先级和结果保持时间 |
| `alert_manager.py` | 仅发射状态转换事件，避免重复告警 |

---

## 未来演进规划

当引入移动机器人支持时：

1. 在感知侧添加**运动补偿服务**。
2. 传入包含采集时间、`T_odom_camera`、重力方向和补偿有效性标志的**帧上下文**。
3. 高度（H）/速度（V）/场景（S）及地面关系应基于**连续的重力对齐坐标系**进行计算。

如果引入 ROS 2：

- 保持**单一相机所有者**。
- 对外暴露 RGB、深度图和相机信息（CameraInfo）供消费者使用。
- **避免**从多个进程同时打开同一台相机。

