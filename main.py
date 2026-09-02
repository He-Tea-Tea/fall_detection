# -*- coding: utf-8 -*-
"""Gemini 335Le跌倒检测完整执行流程。

main.py只负责系统编排，不重复三个功能模块的内部判断公式：

1. 启动RGB-D相机并执行Depth->Color对齐；
2. YOLO Pose + Track获得人员框、关键点和稳定ID；
3. YOLO26-Seg获得床、沙发和椅子的实例Mask；
4. RGB关键点与家具Mask结合对齐Depth，转换为相机3D几何；
5. body_angle.py计算人体方向角；
6. sense.py判断人与床/沙发/椅子/地面的关系；
7. fall_detector.py完成五项评分，只输出FALL/NO_FALL；
8. 显示每个人的测量值与最终二值结果。

所有在线可调参数来自config.yaml。ground_detector.py是独立标定工具，保持不变。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from body_angle import BodyAngleDetector, load_ground_plane, run_self_test as angle_self_test
from fall_detector import (FallDecision, FallDetector, FallObservation, run_self_test as fall_self_test,)
from sense import (SceneObjectDetection, SceneRelationDetector, SceneRelationResult, run_self_test as sense_self_test,)

BASE_DIR = Path(__file__).resolve().parent
LIVE_STAGES = ("body_angle", "sense", "fall_detector", "full")

# COCO Pose固定编号。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

@dataclass
class PersonMeasurement:
    """main.py从一帧RGB-D中整理出的单人基础测量。"""

    person_id: int
    tracking_valid: bool
    bbox: np.ndarray
    keypoints_2d: np.ndarray
    keypoint_conf: np.ndarray
    keypoints_3d: np.ndarray
    shoulder_center_3d: Optional[np.ndarray]
    hip_center_3d: Optional[np.ndarray]
    torso_center_3d: Optional[np.ndarray]
    hip_height_m: Optional[float]
    distance_m: Optional[float]
    data_quality: float
    reliable: bool
    issue: str

def load_config(path: str) -> dict:
    """加载统一配置并在启动阶段检查关键结构。"""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    required_sections = ("paths", "models", "camera", "depth", "pose", "body_angle", "sense", "fall_detector", "display",)
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"config.yaml缺少配置段：{', '.join(missing)}")

    window = int(config["depth"]["window_size"])
    if window <= 0 or window % 2 == 0:
        raise ValueError("depth.window_size必须为正奇数")
    quality_weights = config["pose"]["quality_weights"]
    if abs(sum(float(value) for value in quality_weights.values()) - 1.0) > 1e-6:
        raise ValueError("pose.quality_weights总和必须等于1")
    sense_weights = config["sense"]["relation"]["confidence_weights"]
    if abs(sum(float(value) for value in sense_weights.values()) - 1.0) > 1e-6:
        raise ValueError("sense.relation.confidence_weights总和必须等于1")
    floor_weights = config["sense"]["relation"]["floor_confidence_weights"]
    if abs(sum(float(value) for value in floor_weights.values()) - 1.0) > 1e-6:
        raise ValueError("sense.relation.floor_confidence_weights总和必须等于1")
    mask_threshold = float(config["sense"]["mask_threshold"])
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("sense.mask_threshold必须在0到1之间")
    if int(config["sense"]["min_mask_area_px"]) <= 0:
        raise ValueError("sense.min_mask_area_px必须大于0")
    mask_alpha = float(config["display"]["scene_mask_alpha"])
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("display.scene_mask_alpha必须在0到1之间")
    missing_windows = [stage for stage in LIVE_STAGES if stage not in config["display"].get("window_names", {})]
    if missing_windows:
        raise ValueError("display.window_names缺少：" + ", ".join(missing_windows))
    return config

def resolve_config_path(config_path: str, value: str) -> Path:
    """相对文件路径按config.yaml所在目录解析，而不是依赖启动目录。"""
    path = Path(value)
    return path if path.is_absolute() else Path(config_path).resolve().parent / path

def load_ground_metadata(path: str) -> dict:
    """读取ground.yaml元数据，用于验证当前相机仍与标定一致。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def validate_ground_calibration(ground_data: dict, intrinsics: dict, camera_cfg: dict) -> Sequence[str]:
    """比较地面标定内参与当前RGB profile，返回所有不一致项。"""
    issues: List[str] = []
    if ground_data.get("coordinate_system") != "RGB_COLOR_ALIGNED_DEPTH":
        issues.append("ground.yaml坐标系不是RGB_COLOR_ALIGNED_DEPTH")
    old = ground_data.get("camera_intrinsics")
    if not isinstance(old, dict):
        issues.append("ground.yaml缺少camera_intrinsics")
        return issues
    if (int(old["width"]), int(old["height"])) != (int(intrinsics["width"]), int(intrinsics["height"]),):
        issues.append("当前RGB分辨率与地面标定不一致")
    focal_tolerance = float(camera_cfg["focal_relative_tolerance"])
    for name in ("fx", "fy"):
        relative_error = abs(float(intrinsics[name]) - float(old[name])) / max(abs(float(old[name])), 1e-8)
        if relative_error > focal_tolerance:
            issues.append(f"当前{name}与地面标定不一致")
    center_tolerance = float(camera_cfg["principal_point_tolerance_px"])
    for name in ("cx", "cy"):
        if abs(float(intrinsics[name]) - float(old[name])) > center_tolerance:
            issues.append(f"当前{name}与地面标定不一致")
    return issues

def frame_to_bgr(color_frame):
    """Orbbec ColorFrame转OpenCV BGR，硬件依赖采用函数内延迟导入。"""
    import cv2
    from pyorbbecsdk import OBFormat

    height, width = color_frame.get_height(), color_frame.get_width()
    data = np.asanyarray(color_frame.get_data())
    fmt = color_frame.get_format()
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(np.reshape(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
    if hasattr(OBFormat, "BGR") and fmt == OBFormat.BGR:
        return np.reshape(data, (height, width, 3)).copy()
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(np.reshape(data, (height, width, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None

def depth_frame_to_meters(depth_frame) -> np.ndarray:
    """DepthFrame转米单位float32深度图，无效深度填NaN。"""
    height, width = depth_frame.get_height(), depth_frame.get_width()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
    # Orbbec depth_scale表示每个原始单位对应的毫米数。
    depth_m = raw.astype(np.float32) * float(depth_frame.get_depth_scale()) / 1000.0
    # 在线有效距离由config.yaml的depth.min/max_depth_m统一控制。
    depth_m[depth_m <= 0.0] = np.nan
    return depth_m

def sample_depth(depth_m: np.ndarray, u: float, v: float, depth_cfg: dict) -> Optional[float]:
    """在关键点邻域取有效深度中值，降低单像素飞点影响。"""
    height, width = depth_m.shape
    pixel_u, pixel_v = int(round(float(u))), int(round(float(v)))
    if pixel_u < 0 or pixel_u >= width or pixel_v < 0 or pixel_v >= height:
        return None
    radius = int(depth_cfg["window_size"]) // 2
    roi = depth_m[max(0, pixel_v - radius):min(height, pixel_v + radius + 1), max(0, pixel_u - radius):min(width, pixel_u + radius + 1),]
    valid = roi[np.isfinite(roi) & (roi >= float(depth_cfg["min_depth_m"])) & (roi <= float(depth_cfg["max_depth_m"]))]
    if valid.size < int(depth_cfg["min_valid_samples"]):
        return None
    return float(np.median(valid))

def pixel_to_camera(u: float, v: float, depth_m: Optional[float], intrinsics: dict) -> Optional[np.ndarray]:
    """针孔模型反投影：RGB像素+对齐深度 -> RGB相机系3D点，单位米。"""
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.0:
        return None
    x = (float(u) - float(intrinsics["cx"])) * depth_m / float(intrinsics["fx"])
    y = (float(v) - float(intrinsics["cy"])) * depth_m / float(intrinsics["fy"])
    return np.array([x, y, depth_m], dtype=np.float32)

def project_keypoints(
    keypoints_2d: np.ndarray,
    keypoint_conf: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict,
    config: dict,
) -> np.ndarray:
    """把一人的COCO 2D关键点批量转换为17x3，缺失位置填NaN。"""
    if depth_m.shape != (int(intrinsics["height"]), int(intrinsics["width"])):
        raise ValueError("D2C深度分辨率与RGB内参不一致")
    points_3d = np.full((len(keypoints_2d), 3), np.nan, dtype=np.float32)
    min_conf = float(config["pose"]["min_keypoint_conf"])
    for index, (u, v) in enumerate(keypoints_2d):
        if keypoint_conf[index] < min_conf:
            continue
        depth = sample_depth(depth_m, u, v, config["depth"])
        point = pixel_to_camera(u, v, depth, intrinsics)
        if point is not None:
            points_3d[index] = point
    return points_3d

def pair_center(points: np.ndarray, left: int, right: int) -> Optional[np.ndarray]:
    """左右3D点都有效时返回中心。"""
    if left >= len(points) or right >= len(points):
        return None
    if not np.all(np.isfinite(points[left])) or not np.all(np.isfinite(points[right])):
        return None
    return (points[left] + points[right]) / 2.0

def point_ground_height(point: Optional[np.ndarray], ground_plane: np.ndarray) -> Optional[float]:
    """计算3D点到已归一化地面平面的距离。"""
    if point is None or not np.all(np.isfinite(point)):
        return None
    return float(abs(np.dot(ground_plane[:3], point) + ground_plane[3]))

def compute_data_quality(
    keypoint_conf: np.ndarray,
    keypoints_3d: np.ndarray,
    bbox: np.ndarray,
    distance_m: Optional[float],
    config: dict,
) -> float:
    """把Pose、Depth、人体尺寸和距离融合为测量质量，不代表跌倒概率。"""
    pose_cfg = config["pose"]
    min_conf = float(pose_cfg["min_keypoint_conf"])
    confident = keypoint_conf >= min_conf
    pose_confidence = float(np.mean(np.clip(keypoint_conf, 0.0, 1.0)))
    pose_coverage = float(np.mean(confident))
    valid_3d = np.all(np.isfinite(keypoints_3d), axis=1)
    confident_count = int(np.count_nonzero(confident))
    depth_coverage = (float(np.count_nonzero(valid_3d & confident) / confident_count) if confident_count else 0.0)
    torso_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    torso_coverage = float(np.mean(valid_3d[torso_ids]))
    person_height = max(1.0, float(bbox[3] - bbox[1]))
    size_quality = float(np.clip(person_height / float(pose_cfg["person_size_reference_px"]), 0.0, 1.0))
    if distance_m is None:
        distance_quality = 0.0
    elif distance_m <= float(pose_cfg["near_distance_m"]):
        distance_quality = 1.0
    else:
        span = max(float(pose_cfg["max_reliable_distance_m"]) - float(pose_cfg["near_distance_m"]), 1e-8,)
        distance_quality = float(np.clip((float(pose_cfg["max_reliable_distance_m"]) - distance_m) / span, 0.0, 1.0,))
    weights = pose_cfg["quality_weights"]
    return float(
        np.clip(
            float(weights["pose_confidence"]) * pose_confidence
            + float(weights["pose_coverage"]) * pose_coverage
            + float(weights["depth_coverage"]) * depth_coverage
            + float(weights["torso_coverage"]) * torso_coverage
            + float(weights["person_size"]) * size_quality
            + float(weights["distance"]) * distance_quality,
            0.0,
            1.0,
        )
    )

def build_person_measurement(
    person_id: int,
    tracking_valid: bool,
    bbox: Sequence[float],
    keypoints_2d: np.ndarray,
    keypoint_conf: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict,
    ground_plane: np.ndarray,
    config: dict,
) -> PersonMeasurement:
    """将YOLO的一人结果整理为后续三个功能模块共享的基础测量。"""
    box = np.asarray(bbox, dtype=np.float32)
    points_3d = project_keypoints(keypoints_2d, keypoint_conf, depth_m, intrinsics, config)
    shoulder = pair_center(points_3d, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip = pair_center(points_3d, LEFT_HIP, RIGHT_HIP)
    torso_points = [
        points_3d[index]
        for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if np.all(np.isfinite(points_3d[index]))
    ]
    torso_center = (np.mean(np.asarray(torso_points), axis=0).astype(np.float32) if torso_points else None)
    distance_m = (float(np.linalg.norm(torso_center)) if torso_center is not None else None)
    hip_height = point_ground_height(hip, ground_plane)
    quality = compute_data_quality(keypoint_conf, points_3d, box, distance_m, config)
    valid_3d_count = int(np.count_nonzero(np.all(np.isfinite(points_3d), axis=1)))
    pose_cfg = config["pose"]

    reliable = True
    issue = "OK"
    if not tracking_valid:
        reliable, issue = False, "NO_STABLE_TRACK_ID"
    elif torso_center is None:
        reliable, issue = False, "NO_TORSO_DEPTH"
    elif distance_m is None:
        reliable, issue = False, "NO_DISTANCE"
    elif distance_m > float(pose_cfg["max_reliable_distance_m"]):
        reliable, issue = False, "OUT_OF_RANGE"
    elif valid_3d_count < int(pose_cfg["min_valid_3d_keypoints"]):
        reliable, issue = False, "INSUFFICIENT_3D_KEYPOINTS"
    elif quality < float(pose_cfg["min_data_quality"]):
        reliable, issue = False, "LOW_DATA_QUALITY"

    return PersonMeasurement(
        person_id=int(person_id),
        tracking_valid=bool(tracking_valid),
        bbox=box,
        keypoints_2d=np.asarray(keypoints_2d, dtype=np.float32),
        keypoint_conf=np.asarray(keypoint_conf, dtype=np.float32),
        keypoints_3d=points_3d,
        shoulder_center_3d=shoulder,
        hip_center_3d=hip,
        torso_center_3d=torso_center,
        hip_height_m=hip_height,
        distance_m=distance_m,
        data_quality=quality,
        reliable=reliable,
        issue=issue,
    )

def resize_segmentation_mask(mask_data: np.ndarray, image_shape: Sequence[int], threshold: float) -> np.ndarray:
    """把YOLO-Seg单个概率Mask缩放到RGB/Depth尺寸并转为布尔图。"""
    mask = np.asarray(mask_data, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"YOLO-Seg Mask必须是二维数组，实际形状为{mask.shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    if mask.shape != (height, width):
        # 最近邻缩放不制造新的概率值，并让--self-test无需安装OpenCV也能验证底层接口。
        source_y = np.minimum((np.arange(height) * mask.shape[0] / height).astype(np.int64), mask.shape[0] - 1)
        source_x = np.minimum((np.arange(width) * mask.shape[1] / width).astype(np.int64), mask.shape[1] - 1)
        mask = mask[np.ix_(source_y, source_x)]
    return mask >= float(threshold)

def parse_scene_detections(result, class_map: dict, image_shape: Sequence[int], sense_cfg: dict) -> List[SceneObjectDetection]:
    """提取YOLO26-Seg的家具框和Mask；普通检测模型没有Mask时明确报错。"""
    detections: List[SceneObjectDetection] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections
    if result.masks is None or result.masks.data is None:
        raise RuntimeError("场景模型返回了检测框但没有Mask，请确认models.sense_model使用*-seg.pt实例分割权重")
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    if len(masks) != len(boxes):
        raise RuntimeError(f"YOLO-Seg框数量{len(boxes)}与Mask数量{len(masks)}不一致")
    threshold = float(sense_cfg["mask_threshold"])
    min_area = int(sense_cfg["min_mask_area_px"])
    for box, class_id, confidence, mask_data in zip(boxes, classes, confidences, masks):
        raw_name = str(names[int(class_id)])
        canonical_name = class_map.get(raw_name)
        if canonical_name is None:
            continue
        mask = resize_segmentation_mask(mask_data, image_shape, threshold)
        if int(np.count_nonzero(mask)) < min_area:
            continue
        detections.append(
            SceneObjectDetection(
                name=str(canonical_name),
                confidence=float(confidence),
                bbox=np.asarray(box, dtype=np.float32),
                mask=mask,
            )
        )
    return detections

def camera_intrinsics(pipeline) -> dict:
    """读取当前实际RGB profile内参；D2C点云必须使用这一组参数。"""
    rgb = pipeline.get_camera_param().rgb_intrinsic
    return {
        "width": int(rgb.width),
        "height": int(rgb.height),
        "fx": float(rgb.fx),
        "fy": float(rgb.fy),
        "cx": float(rgb.cx),
        "cy": float(rgb.cy),
    }

def wait_first_frames(pipeline, camera_cfg: dict):
    """相机启动后等待RGB和Depth同时可用。"""
    for _ in range(int(camera_cfg["first_frame_retries"])):
        frames = pipeline.wait_for_frames(int(camera_cfg["first_frame_timeout_ms"]))
        if (frames is not None and frames.get_color_frame() is not None and frames.get_depth_frame() is not None):
            return frames
    raise RuntimeError("无法同时获取RGB和Depth首帧")

def draw_person(
    image,
    measurement: PersonMeasurement,
    angle_result,
    scene_result: Optional[SceneRelationResult],
    decision: Optional[FallDecision],
    config: dict,
    stage: str,
) -> None:
    """在主画面显示底层测量、场景关系、五项分数和最终二值结果。"""
    import cv2

    x1, y1, x2, y2 = map(int, measurement.bbox)
    if decision is not None and decision.is_fall:
        color = (0, 0, 255)
        label = f"ID {measurement.person_id} FALL {decision.fall_score:.2f}"
    elif decision is not None:
        color = (0, 255, 0)
        label = f"ID {measurement.person_id} NO_FALL {decision.fall_score:.2f}"
    elif measurement.reliable:
        color = (0, 255, 0)
        if stage == "body_angle":
            label = f"ID {measurement.person_id} ANGLE_OK"
        elif stage == "sense" and scene_result is not None:
            label = f"ID {measurement.person_id} {scene_result.relation}"
        else:
            # 完整评分所需的角度/场景证据还不完整，不能误显示NO_FALL。
            label = f"ID {measurement.person_id} NO_DECISION"
    else:
        color = (0, 215, 255)
        label = f"ID {measurement.person_id} {measurement.issue}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, float(config["display"]["font_scale"]), color, 2,)

    if bool(config["display"]["show_keypoints"]):
        min_conf = float(config["pose"]["min_keypoint_conf"])
        for (u, v), confidence in zip(measurement.keypoints_2d, measurement.keypoint_conf):
            if confidence >= min_conf:
                cv2.circle(image, (int(u), int(v)), 3, (255, 255, 255), -1)

    angle_text = ("N/A" if not angle_result.valid else f"{angle_result.filtered_angle_deg:.1f}")
    values = [
        f"angle={angle_text} mode={angle_result.mode}",
        f"hipH={measurement.hip_height_m if measurement.hip_height_m is not None else -1:.3f} "
        f"dist={measurement.distance_m if measurement.distance_m is not None else -1:.2f} Q={measurement.data_quality:.2f}",
    ]
    if scene_result is not None:
        values.append(f"relation={scene_result.relation} conf={scene_result.confidence:.2f}")
    if decision is not None:
        values.append(
            f"P={decision.pose_score:.2f} H={decision.height_score:.2f} "
            f"V={decision.velocity_score:.2f} S={decision.static_score:.2f} "
            f"C={decision.scene_score:.2f}"
        )
    line_height = int(config["display"]["line_height_px"])
    text_y = min(image.shape[0] - 8, y2 + line_height)
    for value in values:
        cv2.putText(image, value, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,)
        text_y = min(image.shape[0] - 8, text_y + line_height)

def draw_scene_detections(image, detections: Iterable[SceneObjectDetection], config: dict) -> None:
    """用半透明Mask和外框显示参与关系判断的家具实例。"""
    show_boxes = bool(config["display"]["show_scene_objects"])
    show_masks = bool(config["display"]["show_scene_masks"])
    if not show_boxes and not show_masks:
        return
    import cv2

    colors = {"bed": (255, 120, 0), "sofa": (180, 80, 255), "chair": (0, 200, 255)}
    alpha = float(config["display"]["scene_mask_alpha"])
    for detection in detections:
        color = colors.get(detection.name, (255, 160, 0))
        mask = np.asarray(detection.mask, dtype=bool)
        if show_masks and mask.shape == image.shape[:2] and np.any(mask):
            image[mask] = ((1.0 - alpha) * image[mask] + alpha * np.asarray(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1)
        if not show_boxes:
            continue
        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection.name} {detection.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

def run_live(config: dict, config_path: str, stage: str = "full") -> None:
    """启动真实相机并执行指定功能层的在线可视化测试。

    stage只控制主流程调用到哪一层，底层采集、D2C、Pose和3D反投影始终复用
    同一套实现：body_angle到角度为止；sense再加入场景关系；fall_detector和
    full均执行评分与二值状态机。main.py默认只传full。
    """
    if stage not in LIVE_STAGES:
        raise ValueError(f"未知测试阶段：{stage}，可选值为{LIVE_STAGES}")

    import cv2
    from pyorbbecsdk import AlignFilter, OBStreamType, Pipeline
    from ultralytics import YOLO

    needs_sense = stage in {"sense", "fall_detector", "full"}
    needs_fall = stage in {"fall_detector", "full"}
    ground_path = resolve_config_path(config_path, config["paths"]["ground_file"])
    ground_plane = load_ground_plane(str(ground_path))
    ground_data = load_ground_metadata(str(ground_path))

    print("加载YOLO26 Pose模型..." if not needs_sense else "加载YOLO26 Pose与Seg模型...")
    pose_model = YOLO(config["models"]["pose_model"])
    sense_model = (YOLO(config["models"]["sense_model"]) if needs_sense else None)
    angle_detector = BodyAngleDetector(config, ground_plane)
    sense_detector = (SceneRelationDetector(config, ground_plane) if needs_sense else None)
    fall_detector = FallDetector(config) if needs_fall else None

    pipeline = Pipeline()
    started = False
    try:
        pipeline.start()
        started = True
        wait_first_frames(pipeline, config["camera"])
        intrinsics = camera_intrinsics(pipeline)
        issues = validate_ground_calibration(ground_data, intrinsics, config["camera"])
        if issues:
            message = "；".join(issues)
            if bool(config["camera"]["strict_ground_calibration"]):
                raise RuntimeError(f"当前相机与ground.yaml不一致：{message}")
            print(f"警告：{message}")

        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        sense_detections: List[SceneObjectDetection] = []
        last_seen_s: Dict[int, float] = {}
        frame_index = 0
        display_enabled = bool(config["display"]["enabled"])
        window_name = str(config["display"]["window_names"][stage])
        print(f"{stage}相机测试已启动，按ESC退出")

        while True:
            frames = pipeline.wait_for_frames(int(config["camera"]["frame_timeout_ms"]))
            if frames is None:
                continue
            try:
                frames = align_filter.process(frames)
            except Exception as error:
                print(f"D2C对齐失败：{error}")
                continue
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            image = frame_to_bgr(color_frame)
            if image is None:
                continue
            depth_m = depth_frame_to_meters(depth_frame)
            if image.shape[:2] != depth_m.shape:
                print("D2C后RGB与Depth尺寸不一致，丢弃当前帧")
                continue

            # 家具变化慢，按配置间隔更新Seg；每个Mask会还原到D2C深度图的尺寸。
            if needs_sense:
                interval = int(config["models"]["sense_inference_interval_frames"])
                if frame_index % max(interval, 1) == 0:
                    sense_results = sense_model(
                        image,
                        conf=float(config["models"]["sense_conf"]),
                        retina_masks=bool(config["models"]["sense_retina_masks"]),
                        verbose=False,
                    )
                    sense_detections = parse_scene_detections(
                        sense_results[0] if sense_results else None,
                        config["sense"]["target_class_map"],
                        image.shape[:2],
                        config["sense"],
                    )
                draw_scene_detections(image, sense_detections, config)

            pose_results = pose_model.track(
                image,
                persist=True,
                tracker=config["models"]["tracker"],
                conf=float(config["models"]["pose_conf"]),
                verbose=False,
            )
            now = time.monotonic()
            if pose_results:
                result = pose_results[0]
                if result.boxes is not None and result.keypoints is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    keypoints = result.keypoints.data.cpu().numpy()
                    tracking_valid = result.boxes.id is not None
                    track_ids = (result.boxes.id.int().cpu().numpy() if tracking_valid else np.arange(len(boxes), dtype=np.int32))

                    for index, bbox in enumerate(boxes):
                        # 无稳定ID时使用本帧唯一负数显示，并在本帧结束立即清历史。
                        person_id = (int(track_ids[index]) if tracking_valid else -(frame_index * 1000 + index + 1))
                        if tracking_valid:
                            last_seen_s[person_id] = now
                        pose = keypoints[index]
                        confidence = (pose[:, 2] if pose.shape[1] >= 3 else np.ones(len(pose), dtype=np.float32))
                        measurement = build_person_measurement(
                            person_id=person_id,
                            tracking_valid=tracking_valid,
                            bbox=bbox,
                            keypoints_2d=pose[:, :2],
                            keypoint_conf=confidence,
                            depth_m=depth_m,
                            intrinsics=intrinsics,
                            ground_plane=ground_plane,
                            config=config,
                        )
                        angle_result = angle_detector.update(person_id, measurement.keypoints_3d, measurement.keypoint_conf,)

                        sense_result: Optional[SceneRelationResult] = None
                        if sense_detector is not None:
                            sense_result = sense_detector.analyze(
                                person_id=person_id,
                                person_bbox=measurement.bbox,
                                keypoints_3d=measurement.keypoints_3d,
                                body_angle_deg=(
                                    angle_result.filtered_angle_deg
                                    if angle_result.valid
                                    else None
                                ),
                                detections=sense_detections,
                                depth_m=depth_m,
                                intrinsics=intrinsics,
                            )

                        decision = None
                        # 只有完整评分测试才进入状态机；坏测量和临时ID不累计历史。
                        if (fall_detector is not None and sense_result is not None and measurement.reliable and angle_result.valid):
                            box_width = max(1.0, float(bbox[2] - bbox[0]))
                            box_height = max(1.0, float(bbox[3] - bbox[1]))
                            decision = fall_detector.update(
                                FallObservation(
                                    person_id=person_id,
                                    timestamp_s=now,
                                    body_angle_deg=angle_result.filtered_angle_deg,
                                    bbox_width_height_ratio=box_width / box_height,
                                    hip_height_m=measurement.hip_height_m,
                                    torso_center_3d=measurement.torso_center_3d,
                                    scene_name=sense_result.scene_name,
                                    data_quality=measurement.data_quality,
                                )
                            )
                        draw_person(image, measurement, angle_result, sense_result, decision, config, stage,)

                        if not tracking_valid:
                            angle_detector.reset_person(person_id)
                            if sense_detector is not None:
                                sense_detector.reset_person(person_id)

            # 稳定ID离开一定时间后三个功能模块同步清理，避免ID复用继承旧历史。
            stale_after = float(config["fall_detector"]["state_machine"]["stale_person_s"])
            stale_ids = [person_id for person_id, seen_at in last_seen_s.items() if now - seen_at > stale_after]
            for stale_id in stale_ids:
                angle_detector.reset_person(stale_id)
                if sense_detector is not None:
                    sense_detector.reset_person(stale_id)
                if fall_detector is not None:
                    fall_detector.reset_person(stale_id)
                del last_seen_s[stale_id]

            cv2.putText(image, f"MODE: {stage}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2,)
            if display_enabled:
                cv2.imshow(window_name, image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_index += 1

    except KeyboardInterrupt:
        print("用户退出")
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

def run_self_test(config: dict) -> None:
    """不导入相机/YOLO，验证三个模块和2D->3D基础接口。"""
    angle_self_test(config)
    sense_self_test(config)
    fall_self_test(config)

    intrinsics = {"width": 20, "height": 20, "fx": 20.0, "fy": 20.0, "cx": 10.0, "cy": 10.0,}
    depth_m = np.full((20, 20), 2.0, dtype=np.float32)
    points_2d = np.full((17, 2), [10.0, 10.0], dtype=np.float32)
    confidence = np.ones(17, dtype=np.float32)
    points_3d = project_keypoints(points_2d, confidence, depth_m, intrinsics, config)
    assert np.allclose(points_3d[:, 2], 2.0)
    raw_mask = np.zeros((10, 10), dtype=np.float32)
    raw_mask[2:8, 2:8] = 1.0
    resized_mask = resize_segmentation_mask(raw_mask, (20, 30), float(config["sense"]["mask_threshold"]))
    assert resized_mask.shape == (20, 30) and resized_mask.dtype == np.bool_ and np.any(resized_mask)
    print("main integration self-test: PASS")
    print("  config, projection, YOLO-Seg mask resize, body_angle, sense, fall_detector=PASS")

def main() -> None:
    """命令行入口：默认运行真实完整流程，--self-test运行无硬件测试。"""
    parser = argparse.ArgumentParser(description="Gemini 335Le完整跌倒检测流程")
    parser.add_argument("--config", default=str(BASE_DIR / "config.yaml"), help="统一配置文件路径",)
    parser.add_argument("--self-test", action="store_true", help="运行无硬件集成测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        run_live(config, args.config, stage="full")

if __name__ == "__main__":
    main()
