# -*- coding: utf-8 -*-
"""
Gemini 335Le + YOLO26s Pose + YOLO26s + Depth + Ground Plane
养老机器人场景状态综合测试。

功能：
  1. YOLO26s Pose检测人体；
  2. YOLO26s检测bed/couch/chair；
  3. RGB-D对齐；
  4. 获取人体躯干3D位置；
  5. 获取人体髋部3D位置；
  6. 获取人体肩部3D位置；
  7. 获取人体离地高度；
  8. 获取家具表面Depth点；
  9. 获取家具代表3D位置；
  10. 获取家具离地高度；
  11. 计算人体与家具3D最近距离；
  12. 计算人体与家具2D IoU；
  13. 计算躯干与地面法向量夹角；
  14. 根据人体姿态 + 家具 + Depth判断：
        standing_near
        sitting_on
        lying_on
        lying_floor
        unknown
  15. 计算SceneConfidence；
  16. 多帧场景稳定；
  17. 人体3D异常跳变过滤；
  18. 人体Depth短时丢失保护；
  19. 画面实时显示当前状态；
  20. 终端输出当前状态和关键测量值。

最终显示：
  当前状态：站立-靠近椅子
  当前状态：坐在椅子上
  当前状态：躺在床上
  当前状态：躺在地面
  当前状态：未知

注意：
  ground.yaml必须与当前3D坐标系一致。
  如果ground.yaml是在原始Depth坐标系下生成，而当前3D坐标
  使用的是D2C后的RGB坐标系，则HipH不能直接用于最终判断。
"""

import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import yaml

from ultralytics import YOLO
from pyorbbecsdk import Pipeline, AlignFilter, OBFormat, OBStreamType


# ============================================================
# 模型
# ============================================================
POSE_MODEL_PATH = "yolo26s-pose.pt"
SCENE_MODEL_PATH = "yolo26s.pt"
GROUND_FILE = "ground.yaml"


# ============================================================
# YOLO参数
# ============================================================
POSE_CONF = 0.45
SCENE_CONF = 0.35
KEYPOINT_CONF = 0.50


# ============================================================
# 场景类别
# ============================================================
TARGET_SCENE_CLASSES = {
    "bed",
    "couch",
    "chair",
}


# ============================================================
# Depth参数
# ============================================================
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 6.00
DEPTH_WINDOW = 5


# ============================================================
# 人体3D保护
# ============================================================
MAX_PERSON_JUMP_M = 0.60
PERSON_HOLD_FRAMES = 5


# ============================================================
# 家具Depth采样
# ============================================================
OBJECT_SAMPLE_STEP = 5
MIN_OBJECT_POINTS = 15
OBJECT_X_MARGIN = 0.15


# ============================================================
# 家具表面区域
#
# 注意：
# 这里只是对检测框进行区域缩减。
# 最终真正SeatHeight仍然应该通过更严格的点云平面估计得到。
# ============================================================
OBJECT_SURFACE_BANDS = {
    "chair": (0.35, 0.62),
    "couch": (0.35, 0.62),
    "bed": (0.25, 0.60),
}


# ============================================================
# 2D关系参数
# ============================================================
IOU_FULL = 0.20
MIN_RELATION_IOU = 0.03


# ============================================================
# 3D关系参数
# ============================================================
OBJECT_CLOSE_DISTANCE_M = 0.40
OBJECT_NEAR_DISTANCE_M = 0.80


# ============================================================
# Ground Plane辅助判断
# ============================================================
FLOOR_HIP_HEIGHT_M = 0.22


# ============================================================
# 姿态参数
# ============================================================
STANDING_ANGLE_MAX_DEG = 35.0
LYING_ANGLE_MIN_DEG = 55.0


# ============================================================
# 场景置信度
#
# Ground Plane不作为主要场景识别依据。
# ============================================================
SCENE_CONF_THRESHOLD = 0.55


# ============================================================
# 多帧稳定
# ============================================================
SCENE_HISTORY_LEN = 10


# ============================================================
# COCO Pose关键点
# ============================================================
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


# ============================================================
# 状态中文名称
# ============================================================
STATE_TEXT = {
    "standing_near_chair": "站立-靠近椅子",
    "sitting_on_chair": "坐在椅子上",
    "standing_near_couch": "站立-靠近沙发",
    "sitting_on_couch": "坐在沙发上",
    "lying_on_couch": "躺在沙发上",
    "standing_near_bed": "站立-靠近床",
    "sitting_on_bed": "坐在床上",
    "lying_on_bed": "躺在床上",
    "lying_floor": "躺在地面",
    "unknown": "未知状态",
}


# ============================================================
# 状态颜色
# ============================================================
STATE_COLORS = {
    "standing_near_chair": (0, 255, 0),
    "sitting_on_chair": (255, 255, 0),
    "standing_near_couch": (0, 255, 0),
    "sitting_on_couch": (255, 255, 0),
    "lying_on_couch": (0, 165, 255),
    "standing_near_bed": (0, 255, 0),
    "sitting_on_bed": (255, 255, 0),
    "lying_on_bed": (0, 165, 255),
    "lying_floor": (0, 0, 255),
    "unknown": (255, 255, 255),
}


# ============================================================
# RGB转换
# ============================================================
def frame_to_bgr(color_frame):
    """Gemini 335Le ColorFrame -> OpenCV BGR图像。"""

    h = color_frame.get_height()
    w = color_frame.get_width()

    data = color_frame.get_data()
    fmt = color_frame.get_format()

    if fmt == OBFormat.RGB:

        image = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(
            h,
            w,
            3
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    if fmt == OBFormat.MJPG:

        image = np.frombuffer(
            data,
            dtype=np.uint8
        )

        return cv2.imdecode(
            image,
            cv2.IMREAD_COLOR
        )

    return None


# ============================================================
# Depth转换
# ============================================================
def depth_frame_to_meters(depth_frame):
    """DepthFrame -> 米。"""

    h = depth_frame.get_height()
    w = depth_frame.get_width()

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16
    ).reshape(
        h,
        w
    )

    scale = depth_frame.get_depth_scale()

    depth = (
        raw.astype(np.float32)
        * scale
        / 1000.0
    )

    depth[
        (depth <= 0.0)
        |
        (depth > 20.0)
    ] = np.nan

    return depth


# ============================================================
# Ground Plane
# ============================================================
def load_ground_plane(path):
    """读取ground.yaml中的地面平面。"""

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"找不到 {path}，请先运行ground_detector.py"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = yaml.safe_load(f)

    if "plane" not in data:

        raise ValueError(
            "ground.yaml中不存在plane"
        )

    p = data["plane"]

    plane = np.array(
        [
            float(p["A"]),
            float(p["B"]),
            float(p["C"]),
            float(p["D"]),
        ],
        dtype=np.float64
    )

    norm = np.linalg.norm(
        plane[:3]
    )

    if norm < 1e-8:

        raise ValueError(
            "Ground Plane无效"
        )

    plane = plane / norm

    if plane[1] > 0:

        plane = -plane

    return plane


# ============================================================
# 点到地面
# ============================================================
def point_to_ground_height(
    point,
    plane
):
    """计算3D点到地面平面的垂直距离。"""

    if point is None:

        return None

    point = np.asarray(
        point,
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(point)
    ):

        return None

    A, B, C, D = plane

    return float(
        abs(
            A * point[0]
            + B * point[1]
            + C * point[2]
            + D
        )
    )


# ============================================================
# Depth读取
# ============================================================
def depth_at(
    depth,
    u,
    v,
    window=5
):
    """读取像素邻域Depth中值。"""

    h, w = depth.shape

    u = int(
        round(u)
    )

    v = int(
        round(v)
    )

    if (
        u < 0
        or u >= w
        or v < 0
        or v >= h
    ):

        return None

    r = window // 2

    roi = depth[
        max(0, v-r):
        min(h, v+r+1),
        max(0, u-r):
        min(w, u+r+1)
    ]

    valid = roi[
        np.isfinite(roi)
        &
        (roi >= MIN_DEPTH_M)
        &
        (roi <= MAX_DEPTH_M)
    ]

    if len(valid) < 3:

        return None

    return float(
        np.median(valid)
    )


# ============================================================
# 像素转3D
# ============================================================
def pixel_to_3d(
    u,
    v,
    z,
    intrinsics
):
    """像素+Depth -> 相机坐标系XYZ。"""

    if z is None:

        return None

    if not np.isfinite(z):

        return None

    if (
        z < MIN_DEPTH_M
        or
        z > MAX_DEPTH_M
    ):

        return None

    x = (
        (u - intrinsics["cx"])
        * z
        /
        intrinsics["fx"]
    )

    y = (
        (v - intrinsics["cy"])
        * z
        /
        intrinsics["fy"]
    )

    return np.array(
        [x, y, z],
        dtype=np.float32
    )


# ============================================================
# 鲁棒中值
# ============================================================
def robust_median(points):
    """一组三维点分别取中值。"""

    if not points:

        return None

    points = np.asarray(
        points,
        dtype=np.float32
    )

    valid = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[
        valid
    ]

    if len(points) == 0:

        return None

    return np.median(
        points,
        axis=0
    ).astype(
        np.float32
    )


# ============================================================
# 3D距离
# ============================================================
def distance3d(
    p1,
    p2
):
    """计算两个3D点欧氏距离。"""

    if (
        p1 is None
        or p2 is None
    ):

        return None

    if (
        not np.all(np.isfinite(p1))
        or
        not np.all(np.isfinite(p2))
    ):

        return None

    return float(
        np.linalg.norm(
            np.asarray(p1)
            -
            np.asarray(p2)
        )
    )


# ============================================================
# 人体3D关键点
# ============================================================
def get_person_points_3d(
    keypoint_xy,
    keypoint_conf,
    depth,
    intrinsics
):
    """获取左右肩、左右髋3D坐标。"""

    points = {}

    for idx in [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    ]:

        if (
            keypoint_conf[idx]
            < KEYPOINT_CONF
        ):

            continue

        u, v = keypoint_xy[idx]

        z = depth_at(
            depth,
            u,
            v,
            DEPTH_WINDOW
        )

        point = pixel_to_3d(
            u,
            v,
            z,
            intrinsics
        )

        if point is not None:

            points[idx] = point

    return points


# ============================================================
# 人体几何
# ============================================================
def get_person_geometry(
    points
):
    """计算人体躯干、髋部、肩部3D位置。"""

    hip_points = []
    shoulder_points = []
    all_points = []

    for idx in [
        LEFT_HIP,
        RIGHT_HIP,
    ]:

        if idx in points:

            hip_points.append(
                points[idx]
            )

            all_points.append(
                points[idx]
            )

    for idx in [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
    ]:

        if idx in points:

            shoulder_points.append(
                points[idx]
            )

            all_points.append(
                points[idx]
            )

    hip = robust_median(
        hip_points
    )

    shoulder = robust_median(
        shoulder_points
    )

    torso = robust_median(
        all_points
    )

    return (
        torso,
        hip,
        shoulder
    )


# ============================================================
# 人体姿态角
# ============================================================
def compute_body_angle(
    shoulder_3d,
    hip_3d,
    ground_plane
):
    """
    计算人体躯干和地面法向量夹角。

    0°：
      基本竖直

    90°：
      基本水平
    """

    if (
        shoulder_3d is None
        or hip_3d is None
    ):

        return None

    vector = (
        hip_3d
        -
        shoulder_3d
    )

    vector_norm = np.linalg.norm(
        vector
    )

    normal = (
        ground_plane[:3]
    )

    normal_norm = np.linalg.norm(
        normal
    )

    if (
        vector_norm < 1e-8
        or
        normal_norm < 1e-8
    ):

        return None

    cosine = (
        abs(
            np.dot(
                vector,
                normal
            )
        )
        /
        (
            vector_norm
            *
            normal_norm
        )
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


# ============================================================
# 3D跳变保护
# ============================================================
def valid_person_jump(
    current,
    previous
):
    """过滤人体3D异常跳变。"""

    if current is None:

        return False

    if previous is None:

        return True

    jump = distance3d(
        current,
        previous
    )

    if jump is None:

        return False

    return (
        jump
        <= MAX_PERSON_JUMP_M
    )


# ============================================================
# 家具3D采样
# ============================================================
def get_object_points_3d(
    bbox,
    depth,
    intrinsics,
    object_name,
    person_box=None
):
    """
    从家具检测框提取Depth点。

    注意：
      当前是工程测试版。
      只用于获得家具近似3D位置。
    """

    x1, y1, x2, y2 = bbox

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    band = (
        OBJECT_SURFACE_BANDS.get(
            object_name,
            (0.30, 0.60)
        )
    )

    sx1 = int(
        x1
        +
        width
        *
        OBJECT_X_MARGIN
    )

    sx2 = int(
        x2
        -
        width
        *
        OBJECT_X_MARGIN
    )

    sy1 = int(
        y1
        +
        height
        *
        band[0]
    )

    sy2 = int(
        y1
        +
        height
        *
        band[1]
    )

    h, w = depth.shape

    sx1 = max(
        0,
        min(
            w - 1,
            sx1
        )
    )

    sx2 = max(
        0,
        min(
            w,
            sx2
        )
    )

    sy1 = max(
        0,
        min(
            h - 1,
            sy1
        )
    )

    sy2 = max(
        0,
        min(
            h,
            sy2
        )
    )

    if (
        sx2 <= sx1
        or
        sy2 <= sy1
    ):

        return []

    points = []

    for v in range(
        sy1,
        sy2,
        OBJECT_SAMPLE_STEP
    ):

        for u in range(
            sx1,
            sx2,
            OBJECT_SAMPLE_STEP
        ):

            # --------------------------------------------
            # 排除人体框
            # --------------------------------------------
            if person_box is not None:

                if (
                    person_box[0]
                    <= u
                    <= person_box[2]
                    and
                    person_box[1]
                    <= v
                    <= person_box[3]
                ):

                    continue

            z = depth_at(
                depth,
                u,
                v,
                DEPTH_WINDOW
            )

            point = pixel_to_3d(
                u,
                v,
                z,
                intrinsics
            )

            if point is not None:

                points.append(
                    point
                )

    return points


# ============================================================
# 家具几何
# ============================================================
def get_object_geometry(
    points,
    ground_plane
):
    """计算家具代表3D位置和离地高度。"""

    if len(points) < MIN_OBJECT_POINTS:

        return (
            None,
            None,
            None
        )

    points = np.asarray(
        points,
        dtype=np.float32
    )

    heights = []
    valid_points = []

    for point in points:

        height = (
            point_to_ground_height(
                point,
                ground_plane
            )
        )

        if height is None:

            continue

        # 排除明显异常点
        if (
            height < 0.02
            or
            height > 2.00
        ):

            continue

        heights.append(
            height
        )

        valid_points.append(
            point
        )

    if len(valid_points) < MIN_OBJECT_POINTS:

        return (
            None,
            None,
            None
        )

    heights = np.asarray(
        heights,
        dtype=np.float32
    )

    valid_points = np.asarray(
        valid_points,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 家具代表高度
    # --------------------------------------------------------
    object_height = float(
        np.median(
            heights
        )
    )

    # --------------------------------------------------------
    # 排除高度极端点
    # --------------------------------------------------------
    low = np.percentile(
        heights,
        20
    )

    high = np.percentile(
        heights,
        80
    )

    mask = (
        (heights >= low)
        &
        (heights <= high)
    )

    stable_points = (
        valid_points[mask]
    )

    if len(stable_points) == 0:

        stable_points = (
            valid_points
        )

    object_3d = np.median(
        stable_points,
        axis=0
    ).astype(
        np.float32
    )

    return (
        object_3d,
        object_height,
        stable_points
    )


# ============================================================
# 最近3D距离
# ============================================================
def nearest_point_distance(
    point,
    object_points
):
    """计算人体点到家具表面点云的最近距离。"""

    if (
        point is None
        or
        object_points is None
    ):

        return None

    if len(
        object_points
    ) == 0:

        return None

    distances = np.linalg.norm(
        object_points
        -
        point,
        axis=1
    )

    return float(
        np.min(
            distances
        )
    )


# ============================================================
# IoU
# ============================================================
def box_iou(
    box1,
    box2
):
    """计算二维IoU。"""

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    iw = max(
        0.0,
        x2 - x1
    )

    ih = max(
        0.0,
        y2 - y1
    )

    inter = (
        iw * ih
    )

    area1 = max(
        0.0,
        box1[2] - box1[0]
    ) * max(
        0.0,
        box1[3] - box1[1]
    )

    area2 = max(
        0.0,
        box2[2] - box2[0]
    ) * max(
        0.0,
        box2[3] - box2[1]
    )

    union = (
        area1
        +
        area2
        -
        inter
    )

    if union <= 0:

        return 0.0

    return float(
        inter / union
    )


# ============================================================
# IoU评分
# ============================================================
def iou_score(
    iou
):
    """IoU -> 0~1。"""

    return float(
        np.clip(
            iou / IOU_FULL,
            0.0,
            1.0
        )
    )


# ============================================================
# 3D距离评分
# ============================================================
def depth_score(
    distance
):
    """人体与家具越近，关系分数越高。"""

    if distance is None:

        return 0.0

    if (
        distance
        <= OBJECT_CLOSE_DISTANCE_M
    ):

        return 1.0

    if (
        distance
        >= OBJECT_NEAR_DISTANCE_M
    ):

        return 0.0

    return float(
        (
            OBJECT_NEAR_DISTANCE_M
            -
            distance
        )
        /
        (
            OBJECT_NEAR_DISTANCE_M
            -
            OBJECT_CLOSE_DISTANCE_M
        )
    )


# ============================================================
# 高度关系评分
# ============================================================
def height_score(
    hip_height,
    object_height
):
    """人体髋部高度和家具代表高度越接近，分数越高。"""

    if (
        hip_height is None
        or
        object_height is None
    ):

        return 0.0

    diff = abs(
        hip_height
        -
        object_height
    )

    if diff <= 0.18:

        return 1.0

    if diff >= 0.80:

        return 0.0

    return float(
        (
            0.80
            -
            diff
        )
        /
        (
            0.80
            -
            0.18
        )
    )


# ============================================================
# 人体中心是否进入家具框
# ============================================================
def center_inside(
    person_box,
    object_box
):
    """判断人体检测框中心是否进入家具框。"""

    px = (
        person_box[0]
        +
        person_box[2]
    ) / 2.0

    py = (
        person_box[1]
        +
        person_box[3]
    ) / 2.0

    return (
        object_box[0]
        <= px
        <= object_box[2]
        and
        object_box[1]
        <= py
        <= object_box[3]
    )


# ============================================================
# 场景关系判断
# ============================================================
def calculate_scene_relation(
    object_name,
    object_confidence,
    person_box,
    object_box,
    body_angle,
    hip_height,
    person_3d,
    object_3d,
    object_height,
    object_points
):
    """
    综合判断人与家具关系。

    核心依据：
      1. YOLO家具置信度；
      2. 2D IoU；
      3. 人体中心；
      4. 人体-家具3D最近距离；
      5. 髋部高度；
      6. 躯干姿态。
    """

    iou = box_iou(
        person_box,
        object_box
    )

    overlap = iou_score(
        iou
    )

    inside = center_inside(
        person_box,
        object_box
    )

    center_score = (
        1.0
        if inside
        else 0.0
    )

    distance = nearest_point_distance(
        hip_height_point=None,
        object_points=None
    ) if False else None

    # --------------------------------------------------------
    # 优先使用髋部与家具表面最近距离
    # --------------------------------------------------------
    distance = nearest_point_distance(
        person_3d,
        object_points
    )

    if (
        distance is None
        and
        person_3d is not None
        and
        object_3d is not None
    ):

        distance = distance3d(
            person_3d,
            object_3d
        )

    d_score = depth_score(
        distance
    )

    h_score = height_score(
        hip_height,
        object_height
    )

    # --------------------------------------------------------
    # SceneConfidence
    #
    # 这里降低高度作用：
    # 主要依靠YOLO + 2D + 3D
    # --------------------------------------------------------
    confidence = (
        0.30 * object_confidence
        +
        0.20 * overlap
        +
        0.10 * center_score
        +
        0.35 * d_score
        +
        0.05 * h_score
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # 人体姿态关系
    # --------------------------------------------------------

    if body_angle is None:

        relation = "unknown"

    elif object_name == "chair":

        if (
            body_angle
            <= STANDING_ANGLE_MAX_DEG
        ):

            relation = (
                "standing_near_chair"
            )

        elif (
            body_angle
            < LYING_ANGLE_MIN_DEG
        ):

            if (
                d_score >= 0.35
                and
                h_score >= 0.35
            ):

                relation = (
                    "sitting_on_chair"
                )

            else:

                relation = (
                    "standing_near_chair"
                )

        else:

            if d_score >= 0.35:

                relation = (
                    "sitting_on_chair"
                )

            else:

                relation = (
                    "unknown"
                )

    elif object_name == "couch":

        if (
            body_angle
            <= STANDING_ANGLE_MAX_DEG
        ):

            relation = (
                "standing_near_couch"
            )

        elif (
            body_angle
            < LYING_ANGLE_MIN_DEG
        ):

            if (
                d_score >= 0.35
                and
                h_score >= 0.30
            ):

                relation = (
                    "sitting_on_couch"
                )

            else:

                relation = (
                    "standing_near_couch"
                )

        else:

            if d_score >= 0.30:

                relation = (
                    "lying_on_couch"
                )

            else:

                relation = (
                    "unknown"
                )

    elif object_name == "bed":

        if (
            body_angle
            >= LYING_ANGLE_MIN_DEG
        ):

            if d_score >= 0.25:

                relation = (
                    "lying_on_bed"
                )

            else:

                relation = (
                    "unknown"
                )

        elif (
            body_angle
            < LYING_ANGLE_MIN_DEG
            and
            h_score >= 0.35
            and
            d_score >= 0.30
        ):

            relation = (
                "sitting_on_bed"
            )

        else:

            relation = (
                "standing_near_bed"
            )

    else:

        relation = "unknown"

    return {
        "scene": object_name,
        "relation": relation,
        "confidence": confidence,
        "distance": distance,
        "iou": iou,
        "iou_score": overlap,
        "depth_score": d_score,
        "height_score": h_score,
        "object_3d": object_3d,
        "object_height": object_height,
    }


# ============================================================
# 地面关系
# ============================================================
def calculate_floor_relation(
    body_angle,
    hip_height
):
    """根据姿态+Ground Plane判断是否躺在地面。"""

    if (
        body_angle is not None
        and
        body_angle >= LYING_ANGLE_MIN_DEG
        and
        hip_height is not None
        and
        hip_height <= FLOOR_HIP_HEIGHT_M
    ):

        return (
            "lying_floor",
            0.95
        )

    return (
        "unknown",
        0.30
    )


# ============================================================
# 多帧状态稳定器
# ============================================================
class StateStabilizer:
    """最近N帧投票，稳定最终状态。"""

    def __init__(
        self,
        maxlen=SCENE_HISTORY_LEN
    ):

        self.history = defaultdict(
            lambda: deque(
                maxlen=maxlen
            )
        )

    def update(
        self,
        person_id,
        result
    ):
        """保存当前结果。"""

        self.history[
            person_id
        ].append(
            result
        )

    def get_stable(
        self,
        person_id
    ):
        """获取最近多帧稳定状态。"""

        history = (
            self.history[
                person_id
            ]
        )

        if not history:

            return None

        groups = defaultdict(
            list
        )

        for item in history:

            groups[
                item["relation"]
            ].append(
                item
            )

        best_relation = None
        best_value = -1.0
        best_item = None

        for relation, items in groups.items():

            ratio = (
                len(items)
                /
                len(history)
            )

            mean_conf = float(
                np.mean(
                    [
                        x["confidence"]
                        for x in items
                    ]
                )
            )

            value = (
                0.65 * ratio
                +
                0.35 * mean_conf
            )

            if value > best_value:

                best_value = value
                best_relation = relation
                best_item = items[-1]

        if best_item is None:

            return None

        stable = dict(
            best_item
        )

        stable[
            "relation"
        ] = best_relation

        stable[
            "stable_confidence"
        ] = float(
            np.clip(
                best_value,
                0.0,
                1.0
            )
        )

        return stable


# ============================================================
# 绘制文字
# ============================================================
def put_text(
    image,
    text,
    x,
    y,
    scale=0.43,
    color=(255, 255, 255),
    thickness=1
):
    """绘制英文数字信息。"""

    cv2.putText(
        image,
        str(text),
        (
            int(x),
            int(y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 主程序
# ============================================================
def main():

    print("=" * 80)
    print(
        "Gemini 335Le + YOLO26s Pose + YOLO26s "
        "+ Depth + Ground Plane"
    )
    print(
        "养老机器人实时场景状态检测"
    )
    print("=" * 80)

    # ========================================================
    # 1. Ground Plane
    # ========================================================
    print(
        "\n[1] 加载 Ground Plane..."
    )

    ground_plane = load_ground_plane(
        GROUND_FILE
    )

    print(
        f"A={ground_plane[0]:.8f}"
    )

    print(
        f"B={ground_plane[1]:.8f}"
    )

    print(
        f"C={ground_plane[2]:.8f}"
    )

    print(
        f"D={ground_plane[3]:.8f}"
    )

    # ========================================================
    # 2. YOLO Pose
    # ========================================================
    print(
        "\n[2] 加载 YOLO26s Pose..."
    )

    pose_model = YOLO(
        POSE_MODEL_PATH
    )

    print(
        "Pose模型加载完成"
    )

    # ========================================================
    # 3. YOLO Scene
    # ========================================================
    print(
        "\n[3] 加载 YOLO26s..."
    )

    scene_model = YOLO(
        SCENE_MODEL_PATH
    )

    print(
        "Scene模型加载完成"
    )

    # ========================================================
    # 4. 状态稳定器
    # ========================================================
    stabilizer = StateStabilizer(
        maxlen=SCENE_HISTORY_LEN
    )

    # ========================================================
    # 5. 启动相机
    # ========================================================
    print(
        "\n[4] 启动 Gemini 335Le..."
    )

    pipeline = Pipeline()

    align_filter = AlignFilter(
        align_to_stream=
        OBStreamType.COLOR_STREAM
    )

    # ========================================================
    # 人体历史
    # ========================================================
    person_previous_3d = {}

    person_hold_3d = {}

    person_hold_count = defaultdict(
        int
    )

    # ========================================================
    # 启动
    # ========================================================
    try:

        pipeline.start()

        print(
            "Pipeline 启动成功"
        )

        # ====================================================
        # 6. 相机参数
        # ====================================================
        camera_param = (
            pipeline
            .get_camera_param()
        )

        rgb = (
            camera_param
            .rgb_intrinsic
        )

        depth_intrinsic = (
            camera_param
            .depth_intrinsic
        )

        rgb_intrinsics = {
            "width": int(
                rgb.width
            ),
            "height": int(
                rgb.height
            ),
            "fx": float(
                rgb.fx
            ),
            "fy": float(
                rgb.fy
            ),
            "cx": float(
                rgb.cx
            ),
            "cy": float(
                rgb.cy
            ),
        }

        print(
            "\nRGB内参："
        )

        print(
            f"{rgb_intrinsics['width']}x"
            f"{rgb_intrinsics['height']}"
        )

        print(
            f"fx={rgb_intrinsics['fx']:.6f} "
            f"fy={rgb_intrinsics['fy']:.6f}"
        )

        print(
            f"cx={rgb_intrinsics['cx']:.6f} "
            f"cy={rgb_intrinsics['cy']:.6f}"
        )

        print(
            "\n[5] 开始实时场景识别"
        )

        print(
            "ESC退出"
        )

        # ====================================================
        # 7. 主循环
        # ====================================================
        while True:

            frames = (
                pipeline
                .wait_for_frames(
                    1000
                )
            )

            if frames is None:

                continue

            # ------------------------------------------------
            # D2C
            # ------------------------------------------------
            try:

                aligned = (
                    align_filter
                    .process(
                        frames
                    )
                )

                if aligned is not None:

                    frames = aligned

            except Exception as e:

                print(
                    f"\nD2C失败：{e}"
                )

                continue

            # ------------------------------------------------
            # 获取RGB和Depth
            # ------------------------------------------------
            color_frame = (
                frames
                .get_color_frame()
            )

            depth_frame = (
                frames
                .get_depth_frame()
            )

            if (
                color_frame is None
                or
                depth_frame is None
            ):

                continue

            image = frame_to_bgr(
                color_frame
            )

            if image is None:

                continue

            depth = (
                depth_frame_to_meters(
                    depth_frame
                )
            )

            # =================================================
            # 8. YOLO Scene
            # =================================================
            scene_results = (
                scene_model(
                    image,
                    conf=SCENE_CONF,
                    verbose=False
                )
            )

            scene_objects = []

            if scene_results:

                scene_result = (
                    scene_results[0]
                )

                if (
                    scene_result.boxes
                    is not None
                ):

                    boxes = (
                        scene_result
                        .boxes
                        .xyxy
                        .cpu()
                        .numpy()
                    )

                    confs = (
                        scene_result
                        .boxes
                        .conf
                        .cpu()
                        .numpy()
                    )

                    classes = (
                        scene_result
                        .boxes
                        .cls
                        .int()
                        .cpu()
                        .numpy()
                    )

                    names = (
                        scene_result
                        .names
                    )

                    for i in range(
                        len(boxes)
                    ):

                        name = names[
                            int(
                                classes[i]
                            )
                        ]

                        if name not in TARGET_SCENE_CLASSES:

                            continue

                        scene_objects.append({
                            "name": name,
                            "bbox": boxes[i],
                            "confidence": float(
                                confs[i]
                            ),
                        })

            # =================================================
            # 9. YOLO Pose
            # =================================================
            pose_results = (
                pose_model.track(
                    image,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=POSE_CONF,
                    verbose=False
                )
            )

            if not pose_results:

                continue

            pose_result = (
                pose_results[0]
            )

            if (
                pose_result.boxes is None
                or
                pose_result.keypoints is None
            ):

                continue

            person_boxes = (
                pose_result
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            keypoints = (
                pose_result
                .keypoints
                .data
                .cpu()
                .numpy()
            )

            if pose_result.boxes.id is not None:

                track_ids = (
                    pose_result
                    .boxes
                    .id
                    .int()
                    .cpu()
                    .numpy()
                )

            else:

                track_ids = np.arange(
                    len(person_boxes)
                )

            # =================================================
            # 10. 处理每个人
            # =================================================
            for i, person_box in enumerate(
                person_boxes
            ):

                person_id = int(
                    track_ids[i]
                )

                kp = keypoints[i]

                keypoint_xy = kp[
                    :,
                    :2
                ]

                keypoint_conf = (
                    kp[:, 2]
                    if kp.shape[1] >= 3
                    else np.ones(
                        len(kp),
                        dtype=np.float32
                    )
                )

                # =============================================
                # 10.1 人体3D
                # =============================================
                person_points = (
                    get_person_points_3d(
                        keypoint_xy,
                        keypoint_conf,
                        depth,
                        rgb_intrinsics
                    )
                )

                (
                    torso_3d,
                    hip_3d,
                    shoulder_3d
                ) = get_person_geometry(
                    person_points
                )

                # =============================================
                # 10.2 3D跳变保护
                # =============================================
                previous_3d = (
                    person_previous_3d.get(
                        person_id
                    )
                )

                if (
                    torso_3d is not None
                    and
                    not valid_person_jump(
                        torso_3d,
                        previous_3d
                    )
                ):

                    torso_3d = None

                # =============================================
                # 10.3 短时保持
                # =============================================
                if torso_3d is not None:

                    person_previous_3d[
                        person_id
                    ] = torso_3d

                    person_hold_3d[
                        person_id
                    ] = torso_3d

                    person_hold_count[
                        person_id
                    ] = 0

                elif person_id in person_hold_3d:

                    if (
                        person_hold_count[
                            person_id
                        ]
                        <
                        PERSON_HOLD_FRAMES
                    ):

                        torso_3d = (
                            person_hold_3d[
                                person_id
                            ]
                        )

                        person_hold_count[
                            person_id
                        ] += 1

                # =============================================
                # 10.4 髋部高度
                # =============================================
                hip_height = (
                    point_to_ground_height(
                        hip_3d,
                        ground_plane
                    )
                )

                torso_height = (
                    point_to_ground_height(
                        torso_3d,
                        ground_plane
                    )
                )

                # =============================================
                # 10.5 姿态角
                # =============================================
                body_angle = (
                    compute_body_angle(
                        shoulder_3d,
                        hip_3d,
                        ground_plane
                    )
                )

                # =============================================
                # 10.6 人体框
                # =============================================
                px1, py1, px2, py2 = map(
                    int,
                    person_box
                )

                cv2.rectangle(
                    image,
                    (
                        px1,
                        py1
                    ),
                    (
                        px2,
                        py2
                    ),
                    (0, 255, 0),
                    2
                )

                # =============================================
                # 10.7 关键点
                # =============================================
                for j, (
                    u,
                    v
                ) in enumerate(
                    keypoint_xy
                ):

                    if (
                        keypoint_conf[j]
                        <
                        KEYPOINT_CONF
                    ):

                        continue

                    cv2.circle(
                        image,
                        (
                            int(u),
                            int(v)
                        ),
                        3,
                        (0, 255, 255),
                        -1
                    )

                # =============================================
                # 10.8 寻找最佳家具
                # =============================================
                best = None

                for obj in scene_objects:

                    object_name = (
                        obj["name"]
                    )

                    object_box = (
                        obj["bbox"]
                    )

                    # -----------------------------------------
                    # 家具Depth点
                    # -----------------------------------------
                    object_points = (
                        get_object_points_3d(
                            object_box,
                            depth,
                            rgb_intrinsics,
                            object_name,
                            person_box
                        )
                    )

                    # -----------------------------------------
                    # 家具3D
                    # -----------------------------------------
                    (
                        object_3d,
                        object_height,
                        stable_object_points
                    ) = get_object_geometry(
                        object_points,
                        ground_plane
                    )

                    # -----------------------------------------
                    # 场景关系
                    # -----------------------------------------
                    result = (
                        calculate_scene_relation(
                            object_name=object_name,
                            object_confidence=
                                obj["confidence"],
                            person_box=person_box,
                            object_box=object_box,
                            body_angle=body_angle,
                            hip_height=hip_height,
                            person_3d=torso_3d,
                            object_3d=object_3d,
                            object_height=object_height,
                            object_points=
                                stable_object_points
                        )
                    )

                    result[
                        "object"
                    ] = obj

                    if (
                        best is None
                        or
                        result["confidence"]
                        >
                        best["confidence"]
                    ):

                        best = result

                # =============================================
                # 10.9 无可靠家具
                # =============================================
                if (
                    best is None
                    or
                    best["confidence"]
                    <
                    SCENE_CONF_THRESHOLD
                ):

                    (
                        floor_relation,
                        floor_confidence
                    ) = calculate_floor_relation(
                        body_angle,
                        hip_height
                    )

                    if (
                        floor_relation
                        == "lying_floor"
                    ):

                        current_result = {
                            "scene": "floor",
                            "relation":
                                "lying_floor",
                            "confidence":
                                floor_confidence,
                            "distance": None,
                            "iou": 0.0,
                            "iou_score": 0.0,
                            "depth_score": 0.0,
                            "height_score": 1.0,
                            "object_3d": None,
                            "object_height": None,
                            "object": None,
                        }

                    else:

                        current_result = {
                            "scene": "unknown",
                            "relation":
                                "unknown",
                            "confidence":
                                0.30,
                            "distance": None,
                            "iou": 0.0,
                            "iou_score": 0.0,
                            "depth_score": 0.0,
                            "height_score": 0.0,
                            "object_3d": None,
                            "object_height": None,
                            "object": None,
                        }

                else:

                    current_result = best

                # =============================================
                # 10.10 多帧稳定
                # =============================================
                stabilizer.update(
                    person_id,
                    current_result
                )

                stable = (
                    stabilizer.get_stable(
                        person_id
                    )
                )

                if stable is None:

                    stable = (
                        current_result
                    )

                stable_relation = (
                    stable["relation"]
                )

                stable_confidence = (
                    stable.get(
                        "stable_confidence",
                        stable["confidence"]
                    )
                )

                stable_text = (
                    STATE_TEXT.get(
                        stable_relation,
                        "未知状态"
                    )
                )

                # =============================================
                # 10.11 家具框
                # =============================================
                obj = (
                    current_result.get(
                        "object"
                    )
                )

                if obj is not None:

                    bx1, by1, bx2, by2 = map(
                        int,
                        obj["bbox"]
                    )

                    cv2.rectangle(
                        image,
                        (
                            bx1,
                            by1
                        ),
                        (
                            bx2,
                            by2
                        ),
                        (255, 0, 0),
                        2
                    )

                    put_text(
                        image,
                        (
                            f"{obj['name']} "
                            f"{obj['confidence']:.2f}"
                        ),
                        bx1,
                        max(
                            20,
                            by1 - 8
                        ),
                        0.50,
                        (255, 0, 0),
                        2
                    )

                # =============================================
                # 10.12 人体状态颜色
                # =============================================
                state_color = STATE_COLORS.get(
                    stable_relation,
                    (255, 255, 255)
                )

                # =============================================
                # 10.13 顶部状态
                # =============================================
                put_text(
                    image,
                    (
                        f"ID={person_id} "
                        f"STATE={stable_relation} "
                        f"{stable_confidence:.2f}"
                    ),
                    px1,
                    max(
                        20,
                        py1 - 10
                    ),
                    0.55,
                    state_color,
                    2
                )

                # =============================================
                # 10.14 左上角大状态
                # =============================================
                panel_x = 10
                panel_y = 25

                cv2.rectangle(
                    image,
                    (
                        panel_x - 5,
                        panel_y - 22
                    ),
                    (
                        panel_x + 300,
                        panel_y + 15
                    ),
                    (0, 0, 0),
                    -1
                )

                put_text(
                    image,
                    (
                        f"当前状态: {stable_text}"
                    ),
                    panel_x,
                    panel_y,
                    0.55,
                    state_color,
                    2
                )

                # =============================================
                # 10.15 人体3D数据
                # =============================================
                info_x = px1
                info_y = py2 + 18

                if torso_3d is not None:

                    put_text(
                        image,
                        (
                            f"Torso3D="
                            f"({torso_3d[0]:.2f},"
                            f"{torso_3d[1]:.2f},"
                            f"{torso_3d[2]:.2f})"
                        ),
                        info_x,
                        info_y
                    )

                else:

                    put_text(
                        image,
                        "Torso3D=None",
                        info_x,
                        info_y
                    )

                info_y += 17

                if hip_3d is not None:

                    put_text(
                        image,
                        (
                            f"Hip3D="
                            f"({hip_3d[0]:.2f},"
                            f"{hip_3d[1]:.2f},"
                            f"{hip_3d[2]:.2f})"
                        ),
                        info_x,
                        info_y
                    )

                else:

                    put_text(
                        image,
                        "Hip3D=None",
                        info_x,
                        info_y
                    )

                info_y += 17

                if hip_height is not None:

                    put_text(
                        image,
                        (
                            f"HipH="
                            f"{hip_height:.2f}m"
                        ),
                        info_x,
                        info_y
                    )

                else:

                    put_text(
                        image,
                        "HipH=None",
                        info_x,
                        info_y
                    )

                info_y += 17

                if body_angle is not None:

                    put_text(
                        image,
                        (
                            f"Angle="
                            f"{body_angle:.1f}deg"
                        ),
                        info_x,
                        info_y
                    )

                else:

                    put_text(
                        image,
                        "Angle=None",
                        info_x,
                        info_y
                    )

                info_y += 17

                # =============================================
                # 10.16 家具数据
                # =============================================
                if best is not None:

                    object_3d = (
                        best.get(
                            "object_3d"
                        )
                    )

                    object_height = (
                        best.get(
                            "object_height"
                        )
                    )

                    object_distance = (
                        best.get(
                            "distance"
                        )
                    )

                    d_score = (
                        best.get(
                            "depth_score",
                            0.0
                        )
                    )

                    h_score = (
                        best.get(
                            "height_score",
                            0.0
                        )
                    )

                    iou_s = (
                        best.get(
                            "iou_score",
                            0.0
                        )
                    )

                    if object_3d is not None:

                        put_text(
                            image,
                            (
                                f"Obj3D="
                                f"({object_3d[0]:.2f},"
                                f"{object_3d[1]:.2f},"
                                f"{object_3d[2]:.2f})"
                            ),
                            info_x,
                            info_y
                        )

                    else:

                        put_text(
                            image,
                            "Obj3D=None",
                            info_x,
                            info_y
                        )

                    info_y += 17

                    if object_height is not None:

                        put_text(
                            image,
                            (
                                f"ObjH="
                                f"{object_height:.2f}m"
                            ),
                            info_x,
                            info_y
                        )

                    else:

                        put_text(
                            image,
                            "ObjH=None",
                            info_x,
                            info_y
                        )

                    info_y += 17

                    if object_distance is not None:

                        put_text(
                            image,
                            (
                                f"3DDist="
                                f"{object_distance:.2f}m"
                            ),
                            info_x,
                            info_y
                        )

                    else:

                        put_text(
                            image,
                            "3DDist=None",
                            info_x,
                            info_y
                        )

                    info_y += 17

                    put_text(
                        image,
                        (
                            f"IoUScore="
                            f"{iou_s:.2f}"
                        ),
                        info_x,
                        info_y
                    )

                    info_y += 17

                    put_text(
                        image,
                        (
                            f"DepthScore="
                            f"{d_score:.2f}"
                        ),
                        info_x,
                        info_y
                    )

                    info_y += 17

                    put_text(
                        image,
                        (
                            f"HeightScore="
                            f"{h_score:.2f}"
                        ),
                        info_x,
                        info_y
                    )

                    info_y += 17

                    put_text(
                        image,
                        (
                            f"SceneConf="
                            f"{best['confidence']:.2f}"
                        ),
                        info_x,
                        info_y
                    )

                # =============================================
                # 10.17 终端输出
                # =============================================
                terminal_parts = [
                    f"ID={person_id}",
                    f"STATE={stable_relation}",
                    f"Conf={stable_confidence:.2f}",
                ]

                if body_angle is not None:

                    terminal_parts.append(
                        f"Angle={body_angle:.1f}"
                    )

                if hip_height is not None:

                    terminal_parts.append(
                        f"HipH={hip_height:.2f}m"
                    )

                if torso_3d is not None:

                    terminal_parts.append(
                        "Torso3D="
                        f"({torso_3d[0]:.2f},"
                        f"{torso_3d[1]:.2f},"
                        f"{torso_3d[2]:.2f})"
                    )

                if best is not None:

                    if object_3d is not None:

                        terminal_parts.append(
                            "Obj3D="
                            f"({object_3d[0]:.2f},"
                            f"{object_3d[1]:.2f},"
                            f"{object_3d[2]:.2f})"
                        )

                    if object_height is not None:

                        terminal_parts.append(
                            f"ObjH={object_height:.2f}m"
                        )

                    if object_distance is not None:

                        terminal_parts.append(
                            f"3DDist={object_distance:.2f}m"
                        )

                    terminal_parts.append(
                        f"SceneConf="
                        f"{best['confidence']:.2f}"
                    )

                # 每个人独立一行，避免原来输出互相粘连
                print(
                    "\n"
                    +
                    " | ".join(
                        terminal_parts
                    ),
                    end=""
                )

            # =================================================
            # 11. 显示FPS提示
            # =================================================
            put_text(
                image,
                "ESC: exit",
                10,
                image.shape[0] - 15,
                0.45,
                (255, 255, 255),
                1
            )

            # =================================================
            # 12. 显示
            # =================================================
            cv2.imshow(
                "Gemini 335Le Scene State",
                image
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

            if key == 27:

                break

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()

        print(
            "\n\nPipeline stopped."
        )


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":

    main()