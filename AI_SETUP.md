# 豆包AI跌倒复核配置与运行说明

## 1. 当前版本做了什么

本地YOLO Pose、YOLO Seg、Depth和P/H/V/S/C仍然持续运行。只有本地出现疑似风险时，系统才把最近几秒的结构化时间序列提交给豆包；请求在后台线程执行，不会等待网络或阻塞相机。

当前模型`doubao-1-5-pro-32k-250115`是文本模型，不能直接分析图片。它读取的内容包括：

- 本地`FALL/NO_FALL`和`FallScore`；
- P姿态、H高度、V下降速度、S静止、C场景五项分数；
- 本帧哪些维度有效，以及是否处于2D降级模式；
- 2D角度、3D角度、髋部高度、垂直速度和静止时间；
- 人物与地面、床、沙发或椅子的关系。

该历史模型已经进入官方下线计划。当前API仍可用时本项目可以继续调用，但正式部署前应在火山方舟控制台迁移到仍受支持的模型或推理接入点。模型名称全部来自`config.yaml`，迁移不需要修改Python业务代码。

## 2. 设置API密钥

密钥不能写入Python代码或`config.yaml`，默认从环境变量`ARK_API_KEY`读取。

Windows PowerShell当前窗口：

```powershell
$env:ARK_API_KEY="你的API密钥"
```

Ubuntu当前终端：

```bash
export ARK_API_KEY="你的API密钥"
```

如果火山方舟要求使用推理接入点ID，把`config.yaml`中的`ai.model`改成控制台提供的`ep-...`，其他代码不用修改。

本版使用Python标准库发送HTTPS请求，不需要额外安装`openai`或火山方舟SDK。

## 3. 推荐测试顺序

先运行全部离线测试，测试过程不会访问真实API：

```bash
python main.py --self-test
```

再单独发送一次真实API测试：

```bash
python ai_verifier.py --api-test
```

打开相机测试本地检测、豆包复核和融合画面：

```bash
python main.py --stage ai
```

运行完整流程：

```bash
python main.py
```

## 4. AI何时调用

系统不固定每秒上传。默认满足以下条件后才调用：

1. 本地总分达到`ai.trigger.fall_score`；
2. 姿态分达到`ai.trigger.min_pose_score`；
3. 数据质量达到`ai.trigger.min_data_quality`；
4. 上述风险连续达到规定帧数，或者本地已经确认FALL；
5. 同一人员当前没有未完成请求，并且已经超过冷却时间。

本地已经确认FALL后，默认每20秒最多复核一次，不会每帧重复请求。

这种方式比固定每秒调用更省费用，也能减少正常画面上传。

## 5. 画面字段

| 字段 | 含义 |
|---|---|
| `AI=IDLE` | 等待本地疑似事件 |
| `AI=PENDING` | 后台正在调用豆包，主循环仍继续运行 |
| `AI=FALL` | AI认为像跌倒 |
| `AI=NO_FALL` | AI认为不像跌倒 |
| `AI=UNCERTAIN` | AI认为证据不足或互相矛盾 |
| `AI=NO_KEY` | 没有设置API密钥，当前只使用本地判断 |
| `AI=ERROR/BACKOFF` | 请求失败或处于网络退避时间 |
| `input=DATA` | 当前文本模型只读取结构化数据 |
| `input=VISION` | 后续视觉模型实际接收了JPEG图片 |
| `FINAL` | 本地和AI融合后的最终二值结果 |
| `source=LOCAL` | 由本地状态机确认 |
| `source=AI_ASSISTED` | 中高本地风险获得AI支持后确认 |

## 6. 后续切换视觉模型

在火山方舟开通一个当前可用、支持图片理解的模型或推理接入点，然后修改：

```yaml
ai:
  model: 你的视觉模型或ep接入点ID
  supports_vision: true
```

系统会自动按`ai.buffer.sample_interval_s`裁剪人体及周边场景，保留最近事件，并为一次请求选择`keyframe_count`张有时间顺序的JPEG。不要给不支持视觉的文本模型开启`supports_vision`，否则API会拒绝图片输入。

## 7. 安全边界

- AI不能把本地已经确认的FALL改成NO_FALL。
- AI返回不确定、超时或断网时，最终判断退回本地结果。
- 文本AI只有在本地已经达到中高风险时才能辅助确认。
- 当前控制台告警不会自动停车、转头、说话或发送通知。
- 后续通过`AlertManager.register_handler()`注册机器人动作和通知处理器。
- 移动机器人仍需完成IMU、底盘里程计和云台编码器运动补偿，否则H/V/S可能受到机器人自身运动影响。
