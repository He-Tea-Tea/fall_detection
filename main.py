# -*- coding: utf-8 -*-
"""跌倒检测主程序：Gemini 335Le(RGB-D) + YOLO Pose + FallDetector。

流程（对应设计文档）：
  RGB/Depth 帧 -> D2C 对齐 -> YOLO Pose/跟踪 -> 关键点深度反投影 3D
  -> 距离分级 -> FallDetector 五项评分/状态机 -> 可视化

关键点：
  - 内参不手写，从 335Le 当前激活 profile 自动读取（pipeline.get_camera_param()）
  - AlignFilter 做 D2C（Depth To Color），保证 YOLO 在 RGB 里给出的
    关键点坐标可以直接对应到对齐后的 Depth 图像像素
  - Depth 单位：原始 uint16 × depth_scale = 毫米；无效深度置 nan
"""
import yaml
import cv2
import numpy as np

from ultralytics import YOLO
from pyorbbecsdk import Pipeline, AlignFilter, OBFormat, OBStreamType

from fall_detector import FallDetector


def load_config(path="config.yaml"):
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def frame_to_bgr(color_frame):
    """彩色帧 -> BGR 图像；支持 RGB/BGR/YUYV/MJPG，不支持返回 None。"""
    h, w = color_frame.get_height(), color_frame.get_width()
    data = np.asanyarray(color_frame.get_data())
    fmt = color_frame.get_format()
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(np.reshape(data, (h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.reshape(data, (h, w, 3)).copy()
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(np.reshape(data, (h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None


def depth_to_mm(depth_frame):
    """深度帧 -> 毫米浮点图；无效深度（<=0 或 >20m）置 nan。"""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    scale = depth_frame.get_depth_scale()  # 当前帧深度比例
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
    depth_mm = data.astype(np.float32) * scale
    depth_mm[(depth_mm <= 0) | (depth_mm > 20000)] = np.nan
    return depth_mm


def depth_at(depth_mm, u, v, window=5):
    """在关键点邻域取深度中值，滤除单像素噪声；无效返回 None。

    注意：YOLO 给的是 RGB 坐标，因已做 D2C，直接在对齐后的 Depth 图取即可。
    """
    h, w = depth_mm.shape
    u, v = int(round(u)), int(round(v))
    if u < 0 or u >= w or v < 0 or v >= h:
        return None  # 关键点越界
    r = window // 2
    roi = depth_mm[max(0, v - r):min(h, v + r + 1),
                   max(0, u - r):min(w, u + r + 1)]
    valid = roi[np.isfinite(roi) & (roi > 100) & (roi < 6000)]
    if len(valid) < 3:
        return None  # 有效像素太少
    return float(np.median(valid))  # 中值比单像素稳定


def pixel_to_camera(u, v, z_mm, intrinsics):
    """像素(u,v) + 深度 z(mm) -> 相机系 3D 坐标(x,y,z)，单位 m。

    针孔模型：X=(u-cx)·Z/fx，Y=(v-cy)·Z/fy，Z 即深度。
    """
    if z_mm is None or z_mm <= 0:
        return None
    z = z_mm / 1000.0
    x = (u - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.array([x, y, z], dtype=np.float32)


def get_camera_calibration(pipeline):
    """从 335Le 当前激活的 profile 自动读取内参（避免手填占位值出错）。"""
    camera_param = pipeline.get_camera_param()
    rgb, depth = camera_param.rgb_intrinsic, camera_param.depth_intrinsic

    print("\n========== Camera Calibration ==========")
    print(f"\nRGB Intrinsic: {rgb.width}x{rgb.height}  fx={rgb.fx} fy={rgb.fy} "
          f"cx={rgb.cx} cy={rgb.cy}")
    print(f"RGB Distortion: {camera_param.rgb_distortion}")
    print(f"Depth Intrinsic: {depth.width}x{depth.height}  fx={depth.fx} "
          f"fy={depth.fy} cx={depth.cx} cy={depth.cy}")
    print(f"Depth Distortion: {camera_param.depth_distortion}")
    print(f"Depth -> RGB Extrinsic: {camera_param.transform}")

    return {"fx": float(rgb.fx), "fy": float(rgb.fy),
            "cx": float(rgb.cx), "cy": float(rgb.cy)}


# 状态 -> 框颜色（确认=红，跌倒=橙，疑似=黄，正常=绿）
STATE_COLORS = {
    "CONFIRMED": (0, 0, 255),
    "FALLING": (0, 165, 255),
    "SUSPECT": (0, 215, 255),
    "NORMAL": (0, 255, 0),
}


def draw_person(image, bbox, state, person_id, kp_xy, kp_conf,
                min_conf, distance_m, near_m):
    """绘制单人：检测框、状态文本、关键点、各项指标。"""
    x1, y1, x2, y2 = map(int, bbox)
    color = STATE_COLORS.get(state.state, (0, 255, 0))

    # 检测框 + 状态行（含距离，>near_m 时标注"远"）
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"ID {person_id} {state.state} {state.fall_score:.2f}"
    if distance_m is not None:
        text += f" {distance_m:.2f}m" + (" 远" if distance_m > near_m else "")
    cv2.putText(image, text, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 17 个人体关键点（只画置信度达标的）
    for (u, v), c in zip(kp_xy, kp_conf):
        if c >= min_conf:
            cv2.circle(image, (int(u), int(v)), 3, (255, 255, 255), -1)

    # 指标行（框下方逐行显示）
    info = [
        f"angle={state.body_angle:.1f}",
        f"H={state.body_height:.3f}" if state.body_height is not None else "H=N/A",
        f"V={state.vertical_velocity:.2f}", f"motion={state.motion_velocity:.2f}",
        f"pose={state.pose_score:.2f}", f"height={state.height_score:.2f}",
        f"vel={state.velocity_score:.2f}", f"static={state.static_score:.2f}",
    ]
    y_text = y2 + 18
    for item in info:
        cv2.putText(image, item, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y_text += 18


def wait_first_frames(pipeline, retries=10, timeout_ms=3000):
    """启动后等待首帧 RGB+Depth 同时到位，用于读取 profile 内参。"""
    for _ in range(retries):
        frames = pipeline.wait_for_frames(timeout_ms)
        if frames is not None and frames.get_color_frame() is not None \
                and frames.get_depth_frame() is not None:
            return frames
    raise RuntimeError("没有同时获取到 RGB 和 Depth")


def main():
    # ============ 1. 读取配置 ============
    cfg = load_config()

    # ============ 2. 加载 YOLO Pose 模型 ============
    print("加载 YOLO Pose ...")
    model = YOLO(cfg["model"])
    print("YOLO Pose 加载完成")

    # ============ 3. 启动 335Le Pipeline ============
    print("\n启动 Gemini 335Le ...")
    pipeline = Pipeline()

    try:
        pipeline.start()
        print("335Le Pipeline 启动成功")

        # 先等一帧，确保相机出流后才能读 profile 内参
        wait_first_frames(pipeline)

        # ============ 4. 自动读取相机内参 ============
        intrinsics = get_camera_calibration(pipeline)

        # ============ 5. D2C 对齐（Depth -> Color） ============
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        print("\nD2C 对齐：Depth -> Color")

        # ============ 6. 创建跌倒检测器与距离参数 ============
        detector = FallDetector(cfg)
        near_m = cfg["near_distance_m"]           # < 此距离为高精度区
        max_m = cfg["max_reliable_distance_m"]    # > 此距离只观察不计算

        print("\n开始跌倒检测，按 ESC 退出")
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            # ---- D2C 对齐：让 Depth 像素与 RGB 像素一一对应 ----
            try:
                aligned = align_filter.process(frames)
                if aligned is None:
                    continue
                frames = aligned
            except Exception as e:
                print(f"D2C 对齐失败: {e}")
                continue

            # ---- 取 RGB + Depth ----
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            image = frame_to_bgr(color_frame)
            if image is None:
                continue
            depth_mm = depth_to_mm(depth_frame)

            # ---- 校验对齐后分辨率一致（YOLO 关键点才能直接查 Depth）----
            img_h, img_w = image.shape[:2]
            depth_h, depth_w = depth_mm.shape[:2]
            if img_w != depth_w or img_h != depth_h:
                print(f"警告：对齐后尺寸不一致 RGB={img_w}x{img_h} "
                      f"Depth={depth_w}x{depth_h}")
                continue

            # ---- YOLO Pose + 跨帧跟踪（persist 保持同一人 ID）----
            results = model.track(image, persist=True, tracker=cfg["tracker"],
                                  conf=cfg["pose_conf"], verbose=False)
            if not results:
                cv2.imshow("Fall Detection", image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            result = results[0]
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()          # N x 4 检测框
            keypoints = result.keypoints.data.cpu().numpy()  # N x 17 x 3 (x,y,conf)
            track_ids = (result.boxes.id.int().cpu().numpy()
                         if result.boxes.id is not None
                         else np.arange(len(boxes)))         # 无 ID 时用序号兜底

            # ---- 遍历每一个人 ----
            for i, bbox in enumerate(boxes):
                person_id = int(track_ids[i])
                kp = keypoints[i]
                xy = kp[:, :2]  # 17 x 2 像素坐标
                kp_conf = kp[:, 2] if kp.shape[1] >= 3 else np.ones(len(kp), np.float32)

                # 2D 关键点 -> 3D：置信度达标 + 深度有效才反投影，否则留 nan
                points_3d = np.full((len(xy), 3), np.nan, dtype=np.float32)
                for j, (u, v) in enumerate(xy):
                    if kp_conf[j] < cfg["min_keypoint_conf"]:
                        continue
                    z_mm = depth_at(depth_mm, u, v, cfg["depth_window"])
                    point = pixel_to_camera(u, v, z_mm, intrinsics)
                    if point is not None:
                        points_3d[j] = point

                # 人体距离：有效 3D 点均值到相机的距离
                valid_xyz = points_3d[np.all(np.isfinite(points_3d), axis=1)]
                distance_m = (float(np.linalg.norm(np.mean(valid_xyz, axis=0)))
                              if len(valid_xyz) else None)

                # 距离策略：太远只画框标记，不做跌倒计算
                if distance_m is not None and distance_m > max_m:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(image, f"ID {person_id} FAR {distance_m:.2f}m",
                                (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
                    continue

                # 场景识别（床/沙发/椅子/地面）暂未接入，预留接口
                state = detector.update(
                    person_id=person_id, keypoints_2d=xy, keypoint_conf=kp_conf,
                    keypoints_3d=points_3d, bbox=bbox,
                    distance_m=distance_m, scene_name="unknown",
                )
                draw_person(image, bbox, state, person_id, xy, kp_conf,
                            cfg["min_keypoint_conf"], distance_m, near_m)

            # ---- 显示 ----
            cv2.imshow("Fall Detection", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n用户退出")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()
