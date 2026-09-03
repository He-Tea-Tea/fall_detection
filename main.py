# -*- coding: utf-8 -*-
"""Gemini 335Le 跌倒检测系统完整执行流程。

main.py 只负责组织整个程序，不重复各个评分模块内部的计算公式。

完整流程：
1. 打开 Gemini 335Le，获取 RGB 和 Depth；
2. 将 Depth 对齐到 RGB 画面；
3. YOLO Pose + Track 识别人、17个关键点和稳定 Track ID；
4. YOLO Seg 识别床、沙发、椅子并输出实例 Mask；
5. 将可信的2D关键点结合Depth反投影成3D关键点；
6. pose_3D.py计算P3D，pose_2D.py计算P2D，并融合成姿态分P；
7. height.py、velocity.py、static.py分别计算H、V、S；
8. scene.py计算人物场景关系和场景分C；
9. fall_detector.py对有效维度加权并输出FALL或NO_FALL；
10. 在OpenCV窗口显示识别框、关键点、各维度分数和最终结果。

3D降级原则：
- 3D正常时使用P3D、H、V、S和C；
- Depth缺失、越界或质量不足时，不把缺失维度当成正常0分；
- 只要Track ID稳定、2D姿态角有效，仍可使用P2D更新状态机；
- 纯2D模式使用更高阈值和更长确认时间，降低误报；
- 已确认FALL后，不会因为Depth突然丢失而自动解除报警。

所有可调参数来自config.yaml。
ground_detector.py是独立地面标定工具，本文件只读取它生成的ground.yaml。
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from fall_detector import (
    FallDecision,
    FallDetector,
    FallEvidence,
    run_self_test as fall_self_test,
)
from height import (
    HeightScoreResult,
    HeightScorer,
    run_self_test as height_self_test,
)
from pose_2D import (
    Pose2DDetector,
    Pose2DResult,
    PoseDimensionResult,
    fuse_pose_dimension,
    run_self_test as pose_2d_self_test,
)
from pose_3D import (
    Pose3DDetector,
    Pose3DResult,
    load_ground_plane,
    run_self_test as pose_3d_self_test,
)
from scene import (
    SceneObjectDetection,
    SceneRelationResult,
    SceneScorer,
    run_self_test as scene_self_test,
)
from static import (
    StaticScoreResult,
    StaticScorer,
    run_self_test as static_self_test,
)
from velocity import (
    VelocityScoreResult,
    VelocityScorer,
    run_self_test as velocity_self_test,
)

BASE_DIR = Path(__file__).resolve().parent

# 每个功能文件不带--self-test运行时，都会调用main.py并选择对应stage。
LIVE_STAGES = (
    "pose_3d",
    "pose_2d",
    "height",
    "velocity",
    "static",
    "scene",
    "fall_detector",
    "full",
)

# YOLO Pose使用COCO 17点编号。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class PersonMeasurement:
    """一人在当前帧中的基础测量结果。

    reliable只表示3D测量是否可靠，不代表2D Pose是否不可用。
    当reliable=False时，只要tracking_valid=True且2D角度有效，
    系统仍然可以进入纯2D降级判断。
    """

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
    pose_2d_quality: float
    data_quality: float
    reliable: bool
    issue: str


# ----------------------------------------------------------------------
# 配置与地面标定
# ----------------------------------------------------------------------

def load_config(path: str) -> dict:
    """加载config.yaml，并在程序启动时检查必要配置。"""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required_sections = (
        "paths",
        "models",
        "camera",
        "depth",
        "pose",
        "pose_3d",
        "pose_2d",
        "pose_fusion",
        "height_score",
        "velocity_score",
        "static_score",
        "scene",
        "fall_detector",
        "display",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"config.yaml缺少配置段：{', '.join(missing)}")

    # 关键点Depth采用正方形邻域中值，因此窗口边长必须是正奇数。
    depth_window = int(config["depth"]["window_size"])
    if depth_window <= 0 or depth_window % 2 == 0:
        raise ValueError("depth.window_size必须为正奇数")

    quality_weights = config["pose"]["quality_weights"]
    quality_weight_sum = sum(float(value) for value in quality_weights.values())
    if abs(quality_weight_sum - 1.0) > 1e-6:
        raise ValueError("pose.quality_weights总和必须等于1")

    scene_weights = config["scene"]["relation"]["confidence_weights"]
    scene_weight_sum = sum(float(value) for value in scene_weights.values())
    if abs(scene_weight_sum - 1.0) > 1e-6:
        raise ValueError("scene.relation.confidence_weights总和必须等于1")

    floor_weights = config["scene"]["relation"]["floor_confidence_weights"]
    floor_weight_sum = sum(float(value) for value in floor_weights.values())
    if abs(floor_weight_sum - 1.0) > 1e-6:
        raise ValueError("scene.relation.floor_confidence_weights总和必须等于1")

    mask_threshold = float(config["scene"]["mask_threshold"])
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("scene.mask_threshold必须在0到1之间")

    if int(config["scene"]["min_mask_area_px"]) <= 0:
        raise ValueError("scene.min_mask_area_px必须大于0")

    mask_alpha = float(config["display"]["scene_mask_alpha"])
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("display.scene_mask_alpha必须在0到1之间")

    window_names = config["display"].get("window_names", {})
    missing_windows = [stage for stage in LIVE_STAGES if stage not in window_names]
    if missing_windows:
        raise ValueError(
            "display.window_names缺少：" + ", ".join(missing_windows)
        )

    return config


def resolve_config_path(config_path: str, value: str) -> Path:
    """把配置中的相对路径转换成相对于config.yaml的绝对路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent / path


def load_ground_metadata(path: str) -> dict:
    """读取ground.yaml中的相机内参和坐标系信息。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def validate_ground_calibration(
    ground_data: dict,
    intrinsics: dict,
    camera_cfg: dict,
) -> Sequence[str]:
    """检查当前相机内参是否仍与ground.yaml中的标定数据一致。"""
    issues: List[str] = []

    if ground_data.get("coordinate_system") != "RGB_COLOR_ALIGNED_DEPTH":
        issues.append("ground.yaml坐标系不是RGB_COLOR_ALIGNED_DEPTH")

    old_intrinsics = ground_data.get("camera_intrinsics")
    if not isinstance(old_intrinsics, dict):
        issues.append("ground.yaml缺少camera_intrinsics")
        return issues

    old_size = (
        int(old_intrinsics["width"]),
        int(old_intrinsics["height"]),
    )
    current_size = (
        int(intrinsics["width"]),
        int(intrinsics["height"]),
    )
    if old_size != current_size:
        issues.append("当前RGB分辨率与地面标定不一致")

    focal_tolerance = float(camera_cfg["focal_relative_tolerance"])
    for name in ("fx", "fy"):
        old_value = float(old_intrinsics[name])
        current_value = float(intrinsics[name])
        relative_error = abs(current_value - old_value) / max(
            abs(old_value), 1e-8
        )
        if relative_error > focal_tolerance:
            issues.append(f"当前{name}与地面标定不一致")

    center_tolerance = float(camera_cfg["principal_point_tolerance_px"])
    for name in ("cx", "cy"):
        difference = abs(
            float(intrinsics[name]) - float(old_intrinsics[name])
        )
        if difference > center_tolerance:
            issues.append(f"当前{name}与地面标定不一致")

    return issues


# ----------------------------------------------------------------------
# RGB、Depth与3D反投影
# ----------------------------------------------------------------------

def frame_to_bgr(color_frame):
    """将Orbbec彩色帧转换成OpenCV使用的BGR图像。"""
    import cv2
    from pyorbbecsdk import OBFormat

    height = color_frame.get_height()
    width = color_frame.get_width()
    data = np.asanyarray(color_frame.get_data())
    frame_format = color_frame.get_format()

    if frame_format == OBFormat.RGB:
        rgb_image = np.reshape(data, (height, width, 3))
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if hasattr(OBFormat, "BGR") and frame_format == OBFormat.BGR:
        return np.reshape(data, (height, width, 3)).copy()

    if frame_format == OBFormat.YUYV:
        yuyv_image = np.reshape(data, (height, width, 2))
        return cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2BGR_YUYV)

    if frame_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # 无法识别的图像格式交给主循环丢弃。
    return None


def depth_frame_to_meters(depth_frame) -> np.ndarray:
    """将Orbbec DepthFrame转换成单位为米的float32深度图。"""
    height = depth_frame.get_height()
    width = depth_frame.get_width()

    raw_depth = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape(height, width)

    # depth_scale表示一个原始深度单位对应多少毫米。
    depth_m = (
        raw_depth.astype(np.float32)
        * float(depth_frame.get_depth_scale())
        / 1000.0
    )

    # 原始值为0表示没有有效深度，使用NaN明确表示数据缺失。
    depth_m[depth_m <= 0.0] = np.nan
    return depth_m


def create_missing_depth(image_shape: Sequence[int]) -> np.ndarray:
    """创建全NaN深度图，使RGB流程在Depth整帧丢失时仍能继续运行。"""
    height = int(image_shape[0])
    width = int(image_shape[1])
    return np.full((height, width), np.nan, dtype=np.float32)


def sample_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    depth_cfg: dict,
) -> Optional[float]:
    """在关键点周围取Depth中值，降低单个噪声像素的影响。"""
    height, width = depth_m.shape
    pixel_u = int(round(float(u)))
    pixel_v = int(round(float(v)))

    if pixel_u < 0 or pixel_u >= width:
        return None
    if pixel_v < 0 or pixel_v >= height:
        return None

    radius = int(depth_cfg["window_size"]) // 2
    x1 = max(0, pixel_u - radius)
    x2 = min(width, pixel_u + radius + 1)
    y1 = max(0, pixel_v - radius)
    y2 = min(height, pixel_v + radius + 1)

    depth_roi = depth_m[y1:y2, x1:x2]
    min_depth = float(depth_cfg["min_depth_m"])
    max_depth = float(depth_cfg["max_depth_m"])

    valid_mask = (
        np.isfinite(depth_roi)
        & (depth_roi >= min_depth)
        & (depth_roi <= max_depth)
    )
    valid_depth = depth_roi[valid_mask]

    if valid_depth.size < int(depth_cfg["min_valid_samples"]):
        return None

    # 中值比平均值更不容易受到飞点和极端值影响。
    return float(np.median(valid_depth))


def pixel_to_camera(
    u: float,
    v: float,
    depth_m: Optional[float],
    intrinsics: dict,
) -> Optional[np.ndarray]:
    """使用针孔模型将RGB像素和Depth反投影成相机坐标系3D点。"""
    if depth_m is None:
        return None
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return None

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    z = depth_m

    return np.array([x, y, z], dtype=np.float32)


def project_keypoints(
    keypoints_2d: np.ndarray,
    keypoint_conf: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict,
    config: dict,
) -> np.ndarray:
    """将一人的COCO 2D关键点转换成Nx3的3D关键点数组。"""
    expected_shape = (
        int(intrinsics["height"]),
        int(intrinsics["width"]),
    )
    if depth_m.shape != expected_shape:
        raise ValueError("D2C深度分辨率与RGB内参不一致")

    keypoints_3d = np.full(
        (len(keypoints_2d), 3),
        np.nan,
        dtype=np.float32,
    )
    min_confidence = float(config["pose"]["min_keypoint_conf"])

    for index, (u, v) in enumerate(keypoints_2d):
        # 2D关键点本身不可信时，不读取该位置的Depth。
        if keypoint_conf[index] < min_confidence:
            continue

        depth_value = sample_depth(
            depth_m,
            u,
            v,
            config["depth"],
        )
        point_3d = pixel_to_camera(
            u,
            v,
            depth_value,
            intrinsics,
        )

        if point_3d is not None:
            keypoints_3d[index] = point_3d

    # 无法反投影的关键点保持NaN，不能使用[0,0,0]冒充真实坐标。
    return keypoints_3d


def pair_center(
    points: np.ndarray,
    left_index: int,
    right_index: int,
) -> Optional[np.ndarray]:
    """左右两个3D关键点都有效时，返回它们的中心点。"""
    if left_index >= len(points) or right_index >= len(points):
        return None

    left_point = points[left_index]
    right_point = points[right_index]

    if not np.all(np.isfinite(left_point)):
        return None
    if not np.all(np.isfinite(right_point)):
        return None

    return (left_point + right_point) / 2.0


def point_ground_height(
    point: Optional[np.ndarray],
    ground_plane: np.ndarray,
) -> Optional[float]:
    """计算一个3D点到地面平面的垂直距离。"""
    if point is None or not np.all(np.isfinite(point)):
        return None

    # ground_plane已经在pose_3D.py中归一化，因此结果单位为米。
    return float(
        abs(np.dot(ground_plane[:3], point) + ground_plane[3])
    )


# ----------------------------------------------------------------------
# 2D质量与RGB-D综合质量
# ----------------------------------------------------------------------

def compute_pose_2d_quality(
    keypoint_conf: np.ndarray,
    bbox: np.ndarray,
    config: dict,
) -> float:
    """只使用RGB信息计算2D Pose质量，结果范围为0～1。"""
    pose_cfg = config["pose"]
    confidence = np.asarray(keypoint_conf, dtype=np.float32)
    min_confidence = float(pose_cfg["min_keypoint_conf"])

    # 所有关键点置信度的平均值。
    pose_confidence = float(
        np.mean(np.clip(confidence, 0.0, 1.0))
    )

    # 超过最低置信度的关键点占全部关键点的比例。
    pose_coverage = float(
        np.mean(confidence >= min_confidence)
    )

    # 人框太小时关键点容易抖动，因此人框高度也参与质量判断。
    person_height = max(1.0, float(bbox[3] - bbox[1]))
    size_quality = float(
        np.clip(
            person_height
            / float(pose_cfg["person_size_reference_px"]),
            0.0,
            1.0,
        )
    )

    # 复用综合质量中与Depth无关的三个权重。
    weights = pose_cfg["quality_weights"]
    active_names = (
        "pose_confidence",
        "pose_coverage",
        "person_size",
    )
    active_weight = sum(
        float(weights[name]) for name in active_names
    )
    weighted_sum = (
        float(weights["pose_confidence"]) * pose_confidence
        + float(weights["pose_coverage"]) * pose_coverage
        + float(weights["person_size"]) * size_quality
    )

    # 因为这里只使用三个2D维度，所以要按有效权重重新归一化。
    quality = weighted_sum / max(active_weight, 1e-8)
    return float(np.clip(quality, 0.0, 1.0))


def compute_data_quality(
    keypoint_conf: np.ndarray,
    keypoints_3d: np.ndarray,
    bbox: np.ndarray,
    distance_m: Optional[float],
    config: dict,
) -> float:
    """计算当前人体RGB-D综合测量质量，结果范围为0～1。"""
    pose_cfg = config["pose"]
    min_confidence = float(pose_cfg["min_keypoint_conf"])

    confident_2d = keypoint_conf >= min_confidence
    valid_3d = np.all(np.isfinite(keypoints_3d), axis=1)

    pose_confidence = float(
        np.mean(np.clip(keypoint_conf, 0.0, 1.0))
    )
    pose_coverage = float(np.mean(confident_2d))

    confident_count = int(np.count_nonzero(confident_2d))
    if confident_count > 0:
        valid_depth_count = np.count_nonzero(
            valid_3d & confident_2d
        )
        depth_coverage = float(
            valid_depth_count / confident_count
        )
    else:
        depth_coverage = 0.0

    # 躯干覆盖率只检查双肩和双髋，因为它们决定角度、髋高和躯干中心。
    torso_indices = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    ]
    torso_coverage = float(
        np.mean(valid_3d[torso_indices])
    )

    person_height = max(1.0, float(bbox[3] - bbox[1]))
    size_quality = float(
        np.clip(
            person_height
            / float(pose_cfg["person_size_reference_px"]),
            0.0,
            1.0,
        )
    )

    # 距离越远，人体像素越少，Depth和关键点通常越不稳定。
    if distance_m is None:
        distance_quality = 0.0
    elif distance_m <= float(pose_cfg["near_distance_m"]):
        distance_quality = 1.0
    else:
        reliable_range = max(
            float(pose_cfg["max_reliable_distance_m"])
            - float(pose_cfg["near_distance_m"]),
            1e-8,
        )
        distance_quality = float(
            np.clip(
                (
                    float(pose_cfg["max_reliable_distance_m"])
                    - distance_m
                )
                / reliable_range,
                0.0,
                1.0,
            )
        )

    weights = pose_cfg["quality_weights"]
    quality = (
        float(weights["pose_confidence"]) * pose_confidence
        + float(weights["pose_coverage"]) * pose_coverage
        + float(weights["depth_coverage"]) * depth_coverage
        + float(weights["torso_coverage"]) * torso_coverage
        + float(weights["person_size"]) * size_quality
        + float(weights["distance"]) * distance_quality
    )
    return float(np.clip(quality, 0.0, 1.0))


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
    """把YOLO的一人结果整理成所有特征模块共享的基础测量。"""
    box = np.asarray(bbox, dtype=np.float32)
    points_2d = np.asarray(keypoints_2d, dtype=np.float32)
    confidence = np.asarray(keypoint_conf, dtype=np.float32)

    # 2D关键点结合对齐Depth得到3D关键点，失败的位置保持NaN。
    points_3d = project_keypoints(
        points_2d,
        confidence,
        depth_m,
        intrinsics,
        config,
    )

    shoulder_center = pair_center(
        points_3d,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
    )
    hip_center = pair_center(
        points_3d,
        LEFT_HIP,
        RIGHT_HIP,
    )

    # 躯干中心允许使用当前有效的肩部或髋部点，不强制四个点全部存在。
    torso_points = [
        points_3d[index]
        for index in (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        )
        if np.all(np.isfinite(points_3d[index]))
    ]
    torso_center = (
        np.mean(
            np.asarray(torso_points),
            axis=0,
        ).astype(np.float32)
        if torso_points
        else None
    )

    distance_m = (
        float(np.linalg.norm(torso_center))
        if torso_center is not None
        else None
    )
    hip_height_m = point_ground_height(
        hip_center,
        ground_plane,
    )

    pose_2d_quality = compute_pose_2d_quality(
        confidence,
        box,
        config,
    )
    data_quality = compute_data_quality(
        confidence,
        points_3d,
        box,
        distance_m,
        config,
    )

    valid_3d_count = int(
        np.count_nonzero(
            np.all(np.isfinite(points_3d), axis=1)
        )
    )
    pose_cfg = config["pose"]

    # reliable只描述3D是否可用，不能用它阻止P2D进入降级状态机。
    reliable = True
    issue = "OK"

    if not tracking_valid:
        reliable = False
        issue = "NO_STABLE_TRACK_ID"
    elif torso_center is None:
        reliable = False
        issue = "NO_TORSO_DEPTH"
    elif distance_m is None:
        reliable = False
        issue = "NO_DISTANCE"
    elif distance_m > float(pose_cfg["max_reliable_distance_m"]):
        reliable = False
        issue = "OUT_OF_RANGE"
    elif valid_3d_count < int(pose_cfg["min_valid_3d_keypoints"]):
        reliable = False
        issue = "INSUFFICIENT_3D_KEYPOINTS"
    elif data_quality < float(pose_cfg["min_data_quality"]):
        reliable = False
        issue = "LOW_DATA_QUALITY"

    return PersonMeasurement(
        person_id=int(person_id),
        tracking_valid=bool(tracking_valid),
        bbox=box,
        keypoints_2d=points_2d,
        keypoint_conf=confidence,
        keypoints_3d=points_3d,
        shoulder_center_3d=shoulder_center,
        hip_center_3d=hip_center,
        torso_center_3d=torso_center,
        hip_height_m=hip_height_m,
        distance_m=distance_m,
        pose_2d_quality=pose_2d_quality,
        data_quality=data_quality,
        reliable=reliable,
        issue=issue,
    )


# ----------------------------------------------------------------------
# 五维结果转换为状态机输入
# ----------------------------------------------------------------------

def build_fall_evidence(
    measurement: PersonMeasurement,
    timestamp_s: float,
    pose_3d_result: Pose3DResult,
    pose_2d_result: Optional[Pose2DResult],
    pose_result: Optional[PoseDimensionResult],
    height_result: Optional[HeightScoreResult],
    velocity_result: Optional[VelocityScoreResult],
    static_result: Optional[StaticScoreResult],
    scene_result: Optional[SceneRelationResult],
) -> Optional[FallEvidence]:
    """把当前帧各模块结果打包成FallEvidence。

    重要区别：
    - score=0表示该维度有效，并且没有发现跌倒风险；
    - valid=False表示该维度没有可靠数据；
    - 状态机会忽略无效维度，而不是把它当作正常0分。
    """
    # 没有稳定Track ID时无法连接前后帧，不能维护同一个人的状态机。
    if not measurement.tracking_valid:
        return None

    if pose_result is None:
        return None

    depth_reliable = bool(measurement.reliable)

    # P3D必须同时满足：基础3D可靠、融合结果声明P3D有效、
    # Pose3DDetector返回有效角度。
    pose_3d_valid = bool(
        depth_reliable
        and pose_result.valid_3d
        and pose_3d_result.valid
        and pose_3d_result.filtered_angle_deg is not None
    )

    # 纯2D降级不能只依靠人体框比例，必须真的测到肩—髋2D角度。
    pose_2d_valid = bool(
        pose_2d_result is not None
        and pose_2d_result.valid_angle
    )
    pose_valid = pose_3d_valid or pose_2d_valid

    if not pose_valid:
        return None

    # H、V和S都依赖3D，因此基础3D不可靠时必须标记为无效。
    height_valid = bool(
        depth_reliable
        and height_result is not None
        and height_result.valid
    )
    velocity_valid = bool(
        depth_reliable
        and velocity_result is not None
        and velocity_result.valid
    )
    static_valid = bool(
        depth_reliable
        and static_result is not None
        and static_result.valid_motion
    )

    # scene_result为unknown时仍然是一个有效的中性场景结果。
    scene_valid = scene_result is not None

    has_valid_3d_dimension = any(
        (
            pose_3d_valid,
            height_valid,
            velocity_valid,
            static_valid,
        )
    )

    # 有3D时使用RGB-D综合质量；完全没有3D时使用纯2D质量。
    decision_quality = (
        measurement.data_quality
        if has_valid_3d_dimension
        else measurement.pose_2d_quality
    )

    return FallEvidence(
        person_id=measurement.person_id,
        timestamp_s=float(timestamp_s),
        pose_score=pose_result.score,
        height_score=(
            height_result.score
            if height_result is not None
            else 0.0
        ),
        velocity_score=(
            velocity_result.score
            if velocity_result is not None
            else 0.0
        ),
        static_score=(
            static_result.score
            if static_result is not None
            else 0.0
        ),
        scene_score=(
            scene_result.scene_score
            if scene_result is not None
            else 0.0
        ),
        body_angle_3d_deg=(
            pose_3d_result.filtered_angle_deg
            if pose_3d_valid
            else None
        ),
        hip_height_m=(
            measurement.hip_height_m
            if height_valid
            else None
        ),
        data_quality=decision_quality,
        pose_valid=pose_valid,
        height_valid=height_valid,
        velocity_valid=velocity_valid,
        static_valid=static_valid,
        scene_valid=scene_valid,
        pose_3d_valid=pose_3d_valid,
    )


# ----------------------------------------------------------------------
# YOLO Seg结果转换
# ----------------------------------------------------------------------

def resize_segmentation_mask(
    mask_data: np.ndarray,
    image_shape: Sequence[int],
    threshold: float,
) -> np.ndarray:
    """把YOLO Seg概率Mask缩放到RGB图尺寸并转换成布尔Mask。"""
    mask = np.asarray(mask_data, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError(
            f"YOLO-Seg Mask必须是二维数组，实际形状为{mask.shape}"
        )

    target_height = int(image_shape[0])
    target_width = int(image_shape[1])

    if mask.shape != (target_height, target_width):
        # 使用最近邻方式映射，避免产生新的中间概率。
        source_y = np.minimum(
            (
                np.arange(target_height)
                * mask.shape[0]
                / target_height
            ).astype(np.int64),
            mask.shape[0] - 1,
        )
        source_x = np.minimum(
            (
                np.arange(target_width)
                * mask.shape[1]
                / target_width
            ).astype(np.int64),
            mask.shape[1] - 1,
        )
        mask = mask[np.ix_(source_y, source_x)]

    return mask >= float(threshold)


def parse_scene_detections(
    result,
    class_map: dict,
    image_shape: Sequence[int],
    scene_cfg: dict,
) -> List[SceneObjectDetection]:
    """提取床、沙发和椅子的检测框、置信度及实例Mask。"""
    detections: List[SceneObjectDetection] = []

    if result is None:
        return detections
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    # 普通detect模型只有框，没有Mask，不能用于当前场景几何逻辑。
    if result.masks is None or result.masks.data is None:
        raise RuntimeError(
            "场景模型返回了检测框但没有Mask，"
            "请确认models.scene_model使用*-seg.pt实例分割权重"
        )

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    if len(masks) != len(boxes):
        raise RuntimeError(
            f"YOLO-Seg框数量{len(boxes)}与Mask数量{len(masks)}不一致"
        )

    mask_threshold = float(scene_cfg["mask_threshold"])
    min_mask_area = int(scene_cfg["min_mask_area_px"])

    for box, class_id, confidence, mask_data in zip(
        boxes,
        classes,
        confidences,
        masks,
    ):
        raw_name = str(names[int(class_id)])
        canonical_name = class_map.get(raw_name)

        # 不是床、沙发或椅子的类别不进入场景关系判断。
        if canonical_name is None:
            continue

        mask = resize_segmentation_mask(
            mask_data,
            image_shape,
            mask_threshold,
        )
        mask_area = int(np.count_nonzero(mask))

        if mask_area < min_mask_area:
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


# ----------------------------------------------------------------------
# 相机辅助函数
# ----------------------------------------------------------------------

def camera_intrinsics(pipeline) -> dict:
    """读取当前RGB相机内参，供D2C后的Depth反投影使用。"""
    rgb_intrinsics = pipeline.get_camera_param().rgb_intrinsic
    return {
        "width": int(rgb_intrinsics.width),
        "height": int(rgb_intrinsics.height),
        "fx": float(rgb_intrinsics.fx),
        "fy": float(rgb_intrinsics.fy),
        "cx": float(rgb_intrinsics.cx),
        "cy": float(rgb_intrinsics.cy),
    }


def wait_first_frames(pipeline, camera_cfg: dict):
    """等待首帧，优先返回RGB和Depth同时存在的帧组。

    如果启动阶段Depth一直缺失，但RGB已经出现，则返回最后一组RGB帧，
    让系统进入纯2D降级模式，而不是直接无法启动。
    """
    last_color_frames = None

    for _ in range(int(camera_cfg["first_frame_retries"])):
        frames = pipeline.wait_for_frames(
            int(camera_cfg["first_frame_timeout_ms"])
        )
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is not None:
            last_color_frames = frames

        if color_frame is not None and depth_frame is not None:
            return frames

    if last_color_frames is not None:
        print("警告：启动时未获得Depth，将暂时使用纯2D降级模式")
        return last_color_frames

    raise RuntimeError("无法获取RGB首帧")


# ----------------------------------------------------------------------
# OpenCV画面显示
# ----------------------------------------------------------------------

def score_text(
    result,
    valid_attribute: str,
) -> str:
    """把评分结果转换为显示文字，无效结果显示N/A而不是0分。"""
    if result is None:
        return "N/A"
    if not bool(getattr(result, valid_attribute)):
        return "N/A"
    return f"{result.score:.2f}"


def draw_person(
    image,
    measurement: PersonMeasurement,
    pose_3d_result: Pose3DResult,
    pose_2d_result: Optional[Pose2DResult],
    pose_result: Optional[PoseDimensionResult],
    height_result: Optional[HeightScoreResult],
    velocity_result: Optional[VelocityScoreResult],
    static_result: Optional[StaticScoreResult],
    scene_result: Optional[SceneRelationResult],
    decision: Optional[FallDecision],
    config: dict,
    stage: str,
) -> None:
    """绘制人体框、关键点、五维数据和最终判断结果。"""
    import cv2

    x1, y1, x2, y2 = map(int, measurement.bbox)

    if decision is not None:
        color = (
            (0, 0, 255)
            if decision.is_fall
            else (0, 255, 0)
        )
        fallback_text = (
            " 2D-FALLBACK"
            if decision.degraded_mode
            else ""
        )
        label = (
            f"ID {measurement.person_id} "
            f"{decision.label} "
            f"{decision.fall_score:.2f}"
            f"{fallback_text}"
        )
    elif not measurement.tracking_valid:
        color = (0, 215, 255)
        label = (
            f"ID {measurement.person_id} "
            f"{measurement.issue}"
        )
    else:
        color = (0, 255, 0)

        pose_3d_score = (
            f"{pose_3d_result.score:.2f}"
            if pose_3d_result.valid
            else "N/A"
        )
        pose_2d_score = (
            f"{pose_2d_result.score:.2f}"
            if pose_2d_result is not None
            else "N/A"
        )
        height_score = score_text(
            height_result,
            "valid",
        )
        velocity_score = score_text(
            velocity_result,
            "valid",
        )
        static_score = score_text(
            static_result,
            "valid_motion",
        )

        stage_labels = {
            "pose_3d": f"POSE_3D P3D={pose_3d_score}",
            "pose_2d": f"POSE_2D P2D={pose_2d_score}",
            "height": f"HEIGHT H={height_score}",
            "velocity": f"VELOCITY V={velocity_score}",
            "static": f"STATIC S={static_score}",
            "scene": (
                scene_result.relation
                if scene_result is not None
                else "SCENE N/A"
            ),
        }
        label = (
            f"ID {measurement.person_id} "
            f"{stage_labels.get(stage, 'NO_DECISION')}"
        )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )
    cv2.putText(
        image,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(config["display"]["font_scale"]),
        color,
        2,
    )

    # 绘制置信度合格的COCO关键点。
    if bool(config["display"]["show_keypoints"]):
        min_confidence = float(
            config["pose"]["min_keypoint_conf"]
        )
        for (u, v), confidence in zip(
            measurement.keypoints_2d,
            measurement.keypoint_conf,
        ):
            if confidence >= min_confidence:
                cv2.circle(
                    image,
                    (int(u), int(v)),
                    3,
                    (255, 255, 255),
                    -1,
                )

    # 显示肩中心到髋中心的2D人体轴线，以及画面水平参考线。
    if (
        pose_2d_result is not None
        and pose_2d_result.valid_angle
        and bool(config["display"]["show_pose_2d_axis"])
    ):
        shoulder = tuple(
            np.rint(
                pose_2d_result.shoulder_center_2d
            ).astype(int)
        )
        hip = tuple(
            np.rint(
                pose_2d_result.hip_center_2d
            ).astype(int)
        )

        cv2.line(
            image,
            shoulder,
            hip,
            (0, 255, 255),
            3,
        )
        cv2.line(
            image,
            shoulder,
            (shoulder[0] + 70, shoulder[1]),
            (255, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"2D={pose_2d_result.image_angle_deg:.1f}deg",
            (shoulder[0] + 4, max(18, shoulder[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    angle_3d_text = (
        f"{pose_3d_result.filtered_angle_deg:.1f}"
        if pose_3d_result.valid
        else "N/A"
    )
    pose_3d_score_text = (
        f"{pose_3d_result.score:.2f}"
        if pose_3d_result.valid
        else "N/A"
    )
    hip_height_text = (
        f"{measurement.hip_height_m:.3f}"
        if measurement.hip_height_m is not None
        else "N/A"
    )
    distance_text = (
        f"{measurement.distance_m:.2f}"
        if measurement.distance_m is not None
        else "N/A"
    )

    values = [
        (
            f"P3D={pose_3d_score_text} "
            f"angle3D={angle_3d_text} "
            f"mode={pose_3d_result.mode}"
        ),
        (
            f"hipH={hip_height_text} "
            f"dist={distance_text} "
            f"Q3D={measurement.data_quality:.2f} "
            f"Q2D={measurement.pose_2d_quality:.2f}"
        ),
    ]

    if pose_2d_result is not None:
        angle_2d_text = (
            f"{pose_2d_result.image_angle_deg:.1f}"
            if pose_2d_result.image_angle_deg is not None
            else "N/A"
        )
        values.append(
            f"P2D={pose_2d_result.score:.2f} "
            f"angle2D={angle_2d_text} "
            f"ratio={pose_2d_result.bbox_width_height_ratio:.2f}"
        )

    if pose_result is not None:
        pose_sources = []
        if pose_result.valid_3d:
            pose_sources.append("P3D")
        if pose_result.valid_2d:
            pose_sources.append("P2D")
        source_text = "+".join(pose_sources) or "NONE"
        values.append(
            f"P={pose_result.score:.2f} source={source_text}"
        )

    if height_result is not None:
        height_score_text = (
            f"{height_result.score:.2f}"
            if height_result.valid
            else "N/A"
        )
        baseline_text = (
            f"{height_result.baseline_height_m:.2f}"
            if height_result.baseline_height_m is not None
            else "N/A"
        )
        drop_text = (
            f"{height_result.drop_m:.2f}"
            if height_result.drop_m is not None
            else "N/A"
        )
        values.append(
            f"H={height_score_text} "
            f"baseline={baseline_text} "
            f"drop={drop_text}"
        )

    if velocity_result is not None:
        velocity_score_text = (
            f"{velocity_result.score:.2f}"
            if velocity_result.valid
            else "N/A"
        )
        values.append(
            f"V={velocity_score_text} "
            f"vy={velocity_result.vertical_velocity_mps:.2f}m/s"
        )

    if static_result is not None:
        static_score_text = (
            f"{static_result.score:.2f}"
            if static_result.valid_motion
            else "N/A"
        )
        values.append(
            f"S={static_score_text} "
            f"speed={static_result.motion_velocity_mps:.2f}m/s "
            f"still={static_result.static_duration_s:.1f}s"
        )

    if scene_result is not None:
        values.append(
            f"C={scene_result.scene_score:.2f} "
            f"relation={scene_result.relation} "
            f"conf={scene_result.confidence:.2f}"
        )

    if decision is not None:
        valid_dimensions = (
            ",".join(decision.valid_dimensions)
            if decision.valid_dimensions
            else "NONE"
        )
        values.append(
            f"FallScore={decision.fall_score:.2f} "
            f"valid={valid_dimensions} "
            f"weight={decision.available_weight:.2f}"
        )

    line_height = int(config["display"]["line_height_px"])
    text_y = min(
        image.shape[0] - 8,
        y2 + line_height,
    )

    for value in values:
        cv2.putText(
            image,
            value,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        text_y = min(
            image.shape[0] - 8,
            text_y + line_height,
        )


def draw_scene_detections(
    image,
    detections: Iterable[SceneObjectDetection],
    config: dict,
) -> None:
    """在画面上绘制家具Mask、轮廓、检测框和置信度。"""
    show_boxes = bool(
        config["display"]["show_scene_objects"]
    )
    show_masks = bool(
        config["display"]["show_scene_masks"]
    )

    if not show_boxes and not show_masks:
        return

    import cv2

    colors = {
        "bed": (255, 120, 0),
        "sofa": (180, 80, 255),
        "chair": (0, 200, 255),
    }
    alpha = float(config["display"]["scene_mask_alpha"])

    for detection in detections:
        color = colors.get(
            detection.name,
            (255, 160, 0),
        )
        mask = np.asarray(
            detection.mask,
            dtype=bool,
        )

        if (
            show_masks
            and mask.shape == image.shape[:2]
            and np.any(mask)
        ):
            image[mask] = (
                (1.0 - alpha) * image[mask]
                + alpha * np.asarray(color)
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                image,
                contours,
                -1,
                color,
                1,
            )

        if not show_boxes:
            continue

        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )
        cv2.putText(
            image,
            f"{detection.name} {detection.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


# ----------------------------------------------------------------------
# 相机完整运行流程
# ----------------------------------------------------------------------

def run_live(
    config: dict,
    config_path: str,
    stage: str = "full",
) -> None:
    """打开真实相机并执行指定功能阶段或完整流程。"""
    if stage not in LIVE_STAGES:
        raise ValueError(
            f"未知测试阶段：{stage}，可选值为{LIVE_STAGES}"
        )

    # 硬件和图像库采用延迟导入，使--self-test不依赖真实相机。
    import cv2
    from pyorbbecsdk import AlignFilter, OBStreamType, Pipeline
    from ultralytics import YOLO

    needs_pose_score = stage in {
        "pose_2d",
        "static",
        "fall_detector",
        "full",
    }
    needs_height = stage in {
        "height",
        "static",
        "fall_detector",
        "full",
    }
    needs_velocity = stage in {
        "velocity",
        "fall_detector",
        "full",
    }
    needs_static = stage in {
        "static",
        "fall_detector",
        "full",
    }
    needs_scene = stage in {
        "scene",
        "fall_detector",
        "full",
    }
    needs_fall = stage in {
        "fall_detector",
        "full",
    }

    ground_path = resolve_config_path(
        config_path,
        config["paths"]["ground_file"],
    )
    pose_model_path = resolve_config_path(
        config_path,
        config["models"]["pose_model"],
    )
    scene_model_path = resolve_config_path(
        config_path,
        config["models"]["scene_model"],
    )

    ground_plane = load_ground_plane(str(ground_path))
    ground_data = load_ground_metadata(str(ground_path))

    if needs_scene:
        print("加载YOLO26 Pose与YOLO26 Seg模型...")
    else:
        print("加载YOLO26 Pose模型...")

    pose_model = YOLO(str(pose_model_path))
    scene_model = (
        YOLO(str(scene_model_path))
        if needs_scene
        else None
    )

    # 每个模块只创建一个实例，用Track ID分别保存每个人的历史。
    pose_3d_detector = Pose3DDetector(
        config,
        ground_plane,
    )
    pose_2d_detector = (
        Pose2DDetector(config)
        if needs_pose_score
        else None
    )
    height_scorer = (
        HeightScorer(config)
        if needs_height
        else None
    )
    velocity_scorer = (
        VelocityScorer(config)
        if needs_velocity
        else None
    )
    static_scorer = (
        StaticScorer(config)
        if needs_static
        else None
    )
    scene_scorer = (
        SceneScorer(config, ground_plane)
        if needs_scene
        else None
    )
    fall_detector = (
        FallDetector(config)
        if needs_fall
        else None
    )

    def reset_person(person_id: int) -> None:
        """一个Track ID离场后清除该人的全部历史。"""
        pose_3d_detector.reset_person(person_id)

        modules = (
            height_scorer,
            velocity_scorer,
            static_scorer,
            scene_scorer,
            fall_detector,
        )
        for module in modules:
            if module is not None:
                module.reset_person(person_id)

    pipeline = Pipeline()
    pipeline_started = False

    try:
        pipeline.start()
        pipeline_started = True

        wait_first_frames(
            pipeline,
            config["camera"],
        )
        intrinsics = camera_intrinsics(pipeline)

        calibration_issues = validate_ground_calibration(
            ground_data,
            intrinsics,
            config["camera"],
        )
        if calibration_issues:
            message = "；".join(calibration_issues)
            if bool(config["camera"]["strict_ground_calibration"]):
                raise RuntimeError(
                    f"当前相机与ground.yaml不一致：{message}"
                )
            print(f"警告：{message}")

        align_filter = AlignFilter(
            align_to_stream=OBStreamType.COLOR_STREAM
        )

        # 家具变化较慢，因此Seg不是每帧运行，帧间复用最近一次结果。
        scene_detections: List[SceneObjectDetection] = []
        last_seen_s: Dict[int, float] = {}
        frame_index = 0

        window_name = str(
            config["display"]["window_names"][stage]
        )
        print(f"{stage}相机测试已启动，按ESC退出")

        while True:
            raw_frames = pipeline.wait_for_frames(
                int(config["camera"]["frame_timeout_ms"])
            )
            if raw_frames is None:
                continue

            # D2C失败时仍保留原始RGB做纯2D判断，但不能使用未对齐Depth。
            frames = raw_frames
            depth_aligned = False

            try:
                aligned_frames = align_filter.process(raw_frames)
                if aligned_frames is not None:
                    frames = aligned_frames
                    depth_aligned = True
            except Exception as error:
                print(f"D2C对齐失败，本帧使用纯2D模式：{error}")

            color_frame = frames.get_color_frame()
            if color_frame is None and frames is not raw_frames:
                color_frame = raw_frames.get_color_frame()

            # 没有RGB就无法运行YOLO Pose，因此只能丢弃当前帧。
            if color_frame is None:
                continue

            image = frame_to_bgr(color_frame)
            if image is None:
                continue

            # 模型始终读取未绘制框和Mask的原始图像，避免可视化影响识别结果。
            inference_image = image.copy()

            depth_frame = (
                frames.get_depth_frame()
                if depth_aligned
                else None
            )

            if depth_frame is None:
                # 整帧Depth丢失时使用全NaN图，让后续模块进入2D降级模式。
                depth_m = create_missing_depth(image.shape[:2])
            else:
                depth_m = depth_frame_to_meters(depth_frame)

                if depth_m.shape != image.shape[:2]:
                    print(
                        "D2C后RGB与Depth尺寸不一致，"
                        "本帧使用纯2D模式"
                    )
                    depth_m = create_missing_depth(image.shape[:2])

            # 家具Seg按照配置的间隔运行，其余帧复用最近一次检测结果。
            if needs_scene and scene_model is not None:
                interval = max(
                    int(
                        config["models"][
                            "scene_inference_interval_frames"
                        ]
                    ),
                    1,
                )

                if frame_index % interval == 0:
                    scene_results = scene_model(
                        inference_image,
                        conf=float(config["models"]["scene_conf"]),
                        retina_masks=bool(
                            config["models"]["scene_retina_masks"]
                        ),
                        verbose=False,
                    )
                    first_scene_result = (
                        scene_results[0]
                        if scene_results
                        else None
                    )
                    scene_detections = parse_scene_detections(
                        first_scene_result,
                        config["scene"]["target_class_map"],
                        image.shape[:2],
                        config["scene"],
                    )

            # YOLO Pose每帧运行；persist=True让跟踪器保存跨帧状态。
            pose_results = pose_model.track(
                inference_image,
                persist=True,
                tracker=config["models"]["tracker"],
                conf=float(config["models"]["pose_conf"]),
                verbose=False,
            )
            now = time.monotonic()

            # 推理完成以后再画家具，防止Mask和框污染Pose模型输入。
            if needs_scene:
                draw_scene_detections(
                    image,
                    scene_detections,
                    config,
                )

            if pose_results:
                result = pose_results[0]

                if (
                    result.boxes is not None
                    and result.keypoints is not None
                ):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    keypoints = result.keypoints.data.cpu().numpy()

                    # boxes.id存在时才能认为Track ID稳定。
                    tracking_valid = result.boxes.id is not None
                    track_ids = (
                        result.boxes.id.int().cpu().numpy()
                        if tracking_valid
                        else np.arange(
                            len(boxes),
                            dtype=np.int32,
                        )
                    )

                    for index, bbox in enumerate(boxes):
                        if tracking_valid:
                            person_id = int(track_ids[index])
                            last_seen_s[person_id] = now
                        else:
                            # 临时负ID只用于当前帧显示，不能跨帧保存状态。
                            person_id = -(
                                frame_index * 1000
                                + index
                                + 1
                            )

                        keypoint_data = keypoints[index]
                        if keypoint_data.shape[1] >= 3:
                            confidence = keypoint_data[:, 2]
                        else:
                            confidence = np.ones(
                                len(keypoint_data),
                                dtype=np.float32,
                            )

                        measurement = build_person_measurement(
                            person_id=person_id,
                            tracking_valid=tracking_valid,
                            bbox=bbox,
                            keypoints_2d=keypoint_data[:, :2],
                            keypoint_conf=confidence,
                            depth_m=depth_m,
                            intrinsics=intrinsics,
                            ground_plane=ground_plane,
                            config=config,
                        )

                        # 3D不可靠时使用全NaN，防止错误Depth污染P3D历史。
                        if measurement.reliable:
                            depth_points = measurement.keypoints_3d
                        else:
                            depth_points = np.full_like(
                                measurement.keypoints_3d,
                                np.nan,
                            )

                        pose_3d_result = pose_3d_detector.update(
                            person_id,
                            depth_points,
                            measurement.keypoint_conf,
                        )
                        body_angle_3d = (
                            pose_3d_result.filtered_angle_deg
                            if pose_3d_result.valid
                            else None
                        )

                        # P2D完全不依赖Depth，只要YOLO Pose有效就可以计算。
                        pose_2d_result = (
                            pose_2d_detector.analyze(
                                measurement.keypoints_2d,
                                measurement.keypoint_conf,
                                measurement.bbox,
                            )
                            if pose_2d_detector is not None
                            else None
                        )

                        # P3D缺失时，fuse_pose_dimension会移除P3D权重，
                        # 此时姿态维度P等于有效的P2D。
                        pose_score_result = (
                            fuse_pose_dimension(
                                (
                                    pose_3d_result.score
                                    if pose_3d_result.valid
                                    else None
                                ),
                                pose_2d_result.score,
                                config,
                            )
                            if pose_2d_result is not None
                            else None
                        )

                        # H和V依赖髋高。3D不可靠时传None，
                        # 对应模块会返回valid=False且不写入错误历史。
                        hip_height_input = (
                            measurement.hip_height_m
                            if measurement.reliable
                            else None
                        )
                        height_result = (
                            height_scorer.update(
                                person_id,
                                now,
                                hip_height_input,
                            )
                            if height_scorer is not None
                            else None
                        )
                        velocity_result = (
                            velocity_scorer.update(
                                person_id,
                                now,
                                hip_height_input,
                            )
                            if velocity_scorer is not None
                            else None
                        )

                        # scene.py在3D缺失时通常返回unknown和中性场景分。
                        scene_result = (
                            scene_scorer.analyze(
                                person_id,
                                measurement.bbox,
                                depth_points,
                                body_angle_3d,
                                scene_detections,
                                depth_m,
                                intrinsics,
                            )
                            if scene_scorer is not None
                            else None
                        )

                        # S依赖躯干3D中心。中心缺失时仍调用update，
                        # 让static.py明确返回valid_motion=False。
                        static_result = None
                        if (
                            static_scorer is not None
                            and pose_score_result is not None
                            and height_result is not None
                        ):
                            torso_center_input = (
                                measurement.torso_center_3d
                                if measurement.reliable
                                else None
                            )
                            height_score_input = (
                                height_result.score
                                if height_result.valid
                                else 0.0
                            )
                            static_result = static_scorer.update(
                                person_id,
                                now,
                                torso_center_input,
                                pose_score_result.score,
                                height_score_input,
                            )

                        # 将有效性和分数一起交给状态机。
                        # 这里不再要求measurement.reliable或P3D必须有效。
                        evidence = build_fall_evidence(
                            measurement,
                            now,
                            pose_3d_result,
                            pose_2d_result,
                            pose_score_result,
                            height_result,
                            velocity_result,
                            static_result,
                            scene_result,
                        )

                        decision = None
                        if (
                            fall_detector is not None
                            and evidence is not None
                        ):
                            decision = fall_detector.update(evidence)

                        draw_person(
                            image,
                            measurement,
                            pose_3d_result,
                            pose_2d_result,
                            pose_score_result,
                            height_result,
                            velocity_result,
                            static_result,
                            scene_result,
                            decision,
                            config,
                            stage,
                        )

                        # 临时负ID不能保留历史，否则下一帧可能串到另一个人。
                        if not tracking_valid:
                            reset_person(person_id)

            # 长时间没有重新出现的Track ID需要清除历史。
            stale_after = float(
                config["fall_detector"]["state_machine"][
                    "stale_person_s"
                ]
            )
            stale_ids = [
                person_id
                for person_id, seen_at in last_seen_s.items()
                if now - seen_at > stale_after
            ]
            for stale_id in stale_ids:
                reset_person(stale_id)
                del last_seen_s[stale_id]

            cv2.putText(
                image,
                f"MODE: {stage}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )

            if bool(config["display"]["enabled"]):
                cv2.imshow(window_name, image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_index += 1

    except KeyboardInterrupt:
        print("用户退出")
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


# ----------------------------------------------------------------------
# 无相机集成测试
# ----------------------------------------------------------------------

def run_self_test(config: dict) -> None:
    """运行所有模块测试，并验证Depth缺失时仍能生成跌倒证据。"""
    pose_3d_self_test(config)
    pose_2d_self_test(config)
    height_self_test(config)
    velocity_self_test(config)
    static_self_test(config)
    scene_self_test(config)
    fall_self_test(config)

    intrinsics = {
        "width": 20,
        "height": 20,
        "fx": 20.0,
        "fy": 20.0,
        "cx": 10.0,
        "cy": 10.0,
    }
    depth_m = np.full(
        (20, 20),
        2.0,
        dtype=np.float32,
    )
    points_2d = np.full(
        (17, 2),
        [10.0, 10.0],
        dtype=np.float32,
    )
    confidence = np.ones(
        17,
        dtype=np.float32,
    )

    # 验证2D像素可以结合Depth正确反投影到2米位置。
    points_3d = project_keypoints(
        points_2d,
        confidence,
        depth_m,
        intrinsics,
        config,
    )
    assert np.allclose(points_3d[:, 2], 2.0)

    # 验证Seg Mask缩放后的尺寸和类型正确。
    raw_mask = np.zeros(
        (10, 10),
        dtype=np.float32,
    )
    raw_mask[2:8, 2:8] = 1.0

    resized_mask = resize_segmentation_mask(
        raw_mask,
        (20, 30),
        float(config["scene"]["mask_threshold"]),
    )
    assert resized_mask.shape == (20, 30)
    assert resized_mask.dtype == np.bool_
    assert np.any(resized_mask)

    # 模拟Depth完全丢失，但2D人体接近水平。
    fallback_intrinsics = {
        "width": 100,
        "height": 100,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
    }
    fallback_points_2d = np.full(
        (17, 2),
        [50.0, 50.0],
        dtype=np.float32,
    )

    # 肩中心在左侧、髋中心在右侧，形成接近水平的2D人体轴线。
    fallback_points_2d[LEFT_SHOULDER] = [20.0, 45.0]
    fallback_points_2d[RIGHT_SHOULDER] = [20.0, 55.0]
    fallback_points_2d[LEFT_HIP] = [80.0, 45.0]
    fallback_points_2d[RIGHT_HIP] = [80.0, 55.0]

    missing_depth = create_missing_depth((100, 100))
    ground_plane = np.array(
        [0.0, -1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    measurement = build_person_measurement(
        person_id=99,
        tracking_valid=True,
        bbox=[10.0, 35.0, 90.0, 65.0],
        keypoints_2d=fallback_points_2d,
        keypoint_conf=confidence,
        depth_m=missing_depth,
        intrinsics=fallback_intrinsics,
        ground_plane=ground_plane,
        config=config,
    )

    pose_3d_result = Pose3DDetector(
        config,
        ground_plane,
    ).update(
        99,
        measurement.keypoints_3d,
        confidence,
    )
    pose_2d_result = Pose2DDetector(config).analyze(
        fallback_points_2d,
        confidence,
        measurement.bbox,
    )
    pose_result = fuse_pose_dimension(
        None,
        pose_2d_result.score,
        config,
    )

    height_result = HeightScorer(config).update(
        99,
        0.0,
        None,
    )
    velocity_result = VelocityScorer(config).update(
        99,
        0.0,
        None,
    )
    static_result = StaticScorer(config).update(
        99,
        0.0,
        None,
        pose_result.score,
        0.0,
    )

    evidence = build_fall_evidence(
        measurement,
        0.0,
        pose_3d_result,
        pose_2d_result,
        pose_result,
        height_result,
        velocity_result,
        static_result,
        None,
    )

    # 关键验证：3D完全缺失时仍然成功生成状态机输入。
    assert evidence is not None
    assert evidence.pose_valid
    assert not evidence.pose_3d_valid
    assert not evidence.height_valid
    assert not evidence.velocity_valid
    assert not evidence.static_valid

    decision = FallDetector(config).update(evidence)
    assert decision.valid_dimensions == ("P",)
    assert decision.degraded_mode

    print("main integration self-test: PASS")
    print(
        "  projection, Seg mask, P/H/V/S/C, "
        "missing-3D fallback, state machine=PASS"
    )


def main() -> None:
    """程序入口：默认运行完整流程，--self-test运行无硬件测试。"""
    parser = argparse.ArgumentParser(
        description="Gemini 335Le完整跌倒检测流程"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行全部无硬件测试",
    )
    parser.add_argument(
        "--stage",
        choices=LIVE_STAGES,
        default="full",
        help="选择相机可视化功能层，默认full",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.self_test:
        run_self_test(config)
    else:
        run_live(
            config,
            args.config,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()