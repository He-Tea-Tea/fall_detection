# 豆包视觉单图跌倒复核配置与运行说明

## 1. 当前执行逻辑

本地YOLO Pose、YOLO Seg、Depth和P/H/V/S/C始终连续运行。某个人的本地`FallScore`达到`0.50`时，系统只取触发时刻的一张RGB图片，在后台调用火山方舟视觉模型：

1. `FallScore < 0.50`：不编码图片、不访问网络。
2. `FallScore >= 0.50`：获取当前一张干净RGB画面。
3. 图片最长边超过640像素时等比例缩小，然后以质量75编码为JPEG。
4. 后台线程调用`OpenAI().responses.create()`，相机和本地检测不等待。
5. AI明确回答`true`或`false`时，在配置的保持时间内以AI结论为主。
6. AI回答`uncertain`、调用失败、超时、断网或没有密钥时，直接使用本地结果。
7. 同一人持续高风险时最多每5秒复核一次，本地已确认FALL时最多每10秒复核一次。

默认上传完整画面，因为AI需要看清人究竟倒在地板上，还是正常躺在床、沙发等家具上。若以后改用高分辨率相机，可把`ai.image.mode`改成`person_context`，仅上传人体及周边区域。

## 2. 需要安装的依赖

在原来的`fall_env`环境中执行：

```powershell
python -m pip install -U openai
```

也可以执行：

```powershell
python -m pip install -r requirements-ai.txt
```

## 3. API Key

推荐使用环境变量：

```powershell
$env:ARK_API_KEY="你的新API Key"
```

如果确实要写在本机配置中：

```yaml
ai:
  api_key_env: ARK_API_KEY
  api_key: "你的新API Key"
```

程序优先读取`ai.api_key`，该项留空时再读取环境变量。不要把包含真实密钥的`config.yaml`发送给别人或提交到Git。已经在聊天、截图或仓库中暴露的密钥应立即在方舟控制台作废并重新生成。

## 4. 模型配置

`ai.model`必须填写账号已经开通、支持图片理解和Responses API的模型ID或`ep-...`接入点ID：

```yaml
ai:
  model: 你的视觉模型或ep接入点ID
  supports_vision: true
```

若出现`ModelNotOpen`，表示这个模型名称虽然能被方舟识别，但当前账号没有开通。需要在方舟控制台开通它，或者替换成账号已经开通的视觉模型/接入点。

## 5. 测试顺序

先运行完全离线的代码测试：

```powershell
python ai_verifier.py --self-test
python decision_fusion.py --self-test
python main.py --self-test
```

再上传一张真实图片测试视觉API：

```powershell
python ai_verifier.py --api-test test.jpg
```

图片也可以是URL：

```powershell
python ai_verifier.py --api-test "https://example.com/test.jpg"
```

最后打开相机测试AI阶段或完整流程：

```powershell
python main.py --stage ai
python main.py
```

## 6. 提示词判断内容

默认提示词要求AI区分：

- `true`：至少一个人明显意外倒卧、异常坐卧在地板或地面上。
- `false`：站立、行走、弯腰、下蹲、正常坐椅子、正常躺床或沙发。
- `uncertain`：人体严重遮挡、画面模糊或无法判断人是在地面还是家具上。

只允许三个短答案可减少输出Token、解析错误和响应时间。单张图片无法看到完整跌倒过程，所以它更适合确认“当前是否倒在地上”，不能单独证明之前是否发生了快速跌落。

## 7. AI与本地结果如何融合

| 当前情况 | 最终结果 |
|---|---|
| 没有API Key、断网、超时、接口报错 | 使用本地状态机 |
| AI回答`uncertain`或格式错误 | 使用本地状态机 |
| AI明确回答`true` | AI优先输出FALL并保持15秒 |
| AI明确回答`false` | AI优先输出NO_FALL并保持3秒 |
| AI结果过期且没有新回答 | 恢复使用本地状态机 |

配置`ai.fusion.ai_can_clear_local_fall: true`允许AI的`false`短暂覆盖本地FALL，符合当前“AI回答为主”的要求。如果部署时更重视漏报安全，可改成`false`，此时AI不能解除本地已经确认的FALL。

## 8. 后续语音接口

`decision_fusion.py`已经提供`ExternalConfirmation`和`submit_external_confirmation()`。后续语音识别模型只需要把老人回答转换成统一结论：

- “我没事”转换为`NO_FALL`；
- “救命、起不来”转换为`FALL`；
- 没听清或无回答转换为`UNCERTAIN`，不覆盖现有判断。

`alert_manager.py`已经支持注册新的处理器，可在FALL状态变化时触发机器人停车、语音询问、通知家属或上传护理平台，不需要修改本地五维算法。
