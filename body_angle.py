# -*- coding: utf-8 -*-
"""
Gemini 335Le + YOLO26s Pose
人体3D躯干角度独立测试模块。

本模块只负责：
  1. 获取RGB + Depth；
  2. RGB-D对齐；
  3. YOLO26s Pose获取人体关键点；
  4. 获取左右肩、左右髋、左右膝、左右踝Depth；
  5. 将2D关键点转换为3D坐标；
  6. 根据当前可见人体区域选择角度计算模式；
  7. 完整人体：使用肩部中心->髋部中心计算躯干向量；
  8. 只有下半身：使用髋部中心->膝部/下肢中心计算下半身方向向量；
  9. 只有上半身且无法形成可靠地面方向：不计算角度；
  10. 根据Ground Plane法向量计算人体方向角度；
  11. 对角度进行多帧中值滤波；
  12. 对3D关键点进行异常跳变过滤；
  13. 实时显示角度、计算模式和3D数据。

最终输出：
  person_id
  measurement_mode
  shoulder_3d
  hip_3d
  lower_body_3d
  body_vector
  body_angle

角度计算模式：
  FULL_BODY
      左右肩 + 左右髋有效
      -> ShoulderCenter -> HipCenter

  LOWER_BODY
      肩部不可用
      但髋部 + 膝部/踝部足够有效
      -> HipCenter -> LowerBodyCenter

  INVALID
      只有上半身或有效关键点不足
      -> Angle=None

角度定义：
  0°   -> 身体方向与地面法向量平行
  90°  -> 身体方向与地面平面平行

公式：
  v = P2 - P1

  n = Ground Plane Normal

  cos(theta) =
      |v · n| / (|v| * |n|)

  theta = arccos(cos(theta))

注意：
  1. ground.yaml必须与当前人体3D坐标系一致；
  2. 本模块不进行跌倒判断；
  3. 本模块不进行姿态分类；
  4. INVALID状态不输出角度；
  5. LOWER_BODY角度表示下半身方向，不等同于完整躯干角度。

运行：
  python test_body_angle.py
"""

import os
import math
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
GROUND_FILE = "ground.yaml"


# ============================================================
# YOLO参数
# ============================================================
POSE_CONF = 0.45

# Pose关键点最低置信度
KEYPOINT_CONF = 0.50


# ============================================================
# Depth参数
# ============================================================
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 6.00

# Depth邻域窗口
DEPTH_WINDOW = 5


# ============================================================
# 角度滤波
# ============================================================
ANGLE_HISTORY_LEN = 7


# ============================================================
# 3D跳变保护
# ============================================================
MAX_SHOULDER_JUMP_M = 0.40
MAX_HIP_JUMP_M = 0.40
MAX_KNEE_JUMP_M = 0.40
MAX_ANKLE_JUMP_M = 0.50


# ============================================================
# COCO Pose关键点
# ============================================================
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# ============================================================
# RGB转换
# ============================================================
def frame_to_bgr(color_frame):
    """
    将Gemini 335Le ColorFrame转换为OpenCV BGR图像。
    """

    h = color_frame.get_height()
    w = color_frame.get_width()

    data = color_frame.get_data()
    fmt = color_frame.get_format()

    # --------------------------------------------------------
    # RGB格式
    # --------------------------------------------------------
    if fmt == OBFormat.RGB:

        image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------
    # MJPG格式
    # --------------------------------------------------------
    if fmt == OBFormat.MJPG:

        image = np.frombuffer(data, dtype=np.uint8)

        return cv2.imdecode(image, cv2.IMREAD_COLOR)

    # --------------------------------------------------------
    # BGR格式
    # --------------------------------------------------------
    if hasattr(OBFormat, "BGR"):

        if fmt == OBFormat.BGR:

            image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

            return image.copy()

    return None


# ============================================================
# Depth转换
# ============================================================
def depth_frame_to_meters(depth_frame):
    """
    将DepthFrame转换为米单位浮点Depth图。
    """

    h = depth_frame.get_height()
    w = depth_frame.get_width()

    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)

    scale = depth_frame.get_depth_scale()

    depth = raw.astype(np.float32) * scale / 1000.0

    # --------------------------------------------------------
    # 去掉明显无效Depth
    # --------------------------------------------------------
    depth[(depth <= 0.0) | (depth > 20.0)] = np.nan

    return depth


# ============================================================
# Ground Plane读取
# ============================================================
def load_ground_plane(path):
    """
    从ground.yaml读取地面平面：

        A*x + B*y + C*z + D = 0
    """

    if not os.path.exists(path):

        raise FileNotFoundError(f"找不到Ground Plane文件：{path}")

    with open(path, "r", encoding="utf-8") as f:

        data = yaml.safe_load(f)

    if data is None or "plane" not in data:

        raise ValueError("ground.yaml中不存在plane字段")

    p = data["plane"]

    plane = np.array([float(p["A"]), float(p["B"]), float(p["C"]), float(p["D"])], dtype=np.float64)

    # --------------------------------------------------------
    # 法向量长度
    # --------------------------------------------------------
    norm = np.linalg.norm(plane[:3])

    if norm < 1e-8:

        raise ValueError("Ground Plane法向量无效")

    # --------------------------------------------------------
    # 单位化
    # --------------------------------------------------------
    plane = plane / norm

    # --------------------------------------------------------
    # 与之前Ground Plane程序保持一致
    # --------------------------------------------------------
    if plane[1] > 0:

        plane = -plane

    return plane


# ============================================================
# Depth邻域中值
# ============================================================
def depth_at(depth, u, v, window=5):
    """
    获取像素附近窗口的Depth中值。

    不直接读取单个Depth像素，
    用于减小335Le深度噪声。
    """

    h, w = depth.shape

    u = int(round(u))
    v = int(round(v))

    # --------------------------------------------------------
    # 像素范围检查
    # --------------------------------------------------------
    if u < 0 or u >= w or v < 0 or v >= h:

        return None

    radius = window // 2

    # --------------------------------------------------------
    # 提取邻域
    # --------------------------------------------------------
    roi = depth[max(0, v - radius):min(h, v + radius + 1), max(0, u - radius):min(w, u + radius + 1)]

    # --------------------------------------------------------
    # 只保留有效Depth
    # --------------------------------------------------------
    valid = roi[np.isfinite(roi) & (roi >= MIN_DEPTH_M) & (roi <= MAX_DEPTH_M)]

    # --------------------------------------------------------
    # 有效点太少
    # --------------------------------------------------------
    if len(valid) < 3:

        return None

    # --------------------------------------------------------
    # 中值比单点更稳定
    # --------------------------------------------------------
    return float(np.median(valid))


# ============================================================
# 像素 + Depth -> 3D
# ============================================================
def pixel_to_3d(u, v, z, intrinsics):
    """
    使用针孔模型计算相机坐标系3D坐标。

    X = (u-cx) * Z / fx
    Y = (v-cy) * Z / fy
    Z = Z
    """

    if z is None:

        return None

    if not np.isfinite(z):

        return None

    if z < MIN_DEPTH_M or z > MAX_DEPTH_M:

        return None

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------
    x = (u - intrinsics["cx"]) * z / intrinsics["fx"]

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------
    y = (v - intrinsics["cy"]) * z / intrinsics["fy"]

    return np.array([x, y, z], dtype=np.float32)


# ============================================================
# 获取单个关键点3D
# ============================================================
def get_keypoint_3d(keypoint_xy, keypoint_conf, index, depth, intrinsics):
    """
    获取指定Pose关键点的3D坐标。
    """

    # --------------------------------------------------------
    # Pose置信度不足
    # --------------------------------------------------------
    if keypoint_conf[index] < KEYPOINT_CONF:

        return None

    u, v = keypoint_xy[index]

    # --------------------------------------------------------
    # 读取Depth
    # --------------------------------------------------------
    z = depth_at(depth, u, v, DEPTH_WINDOW)

    # --------------------------------------------------------
    # 反投影
    # --------------------------------------------------------
    return pixel_to_3d(u, v, z, intrinsics)


# ============================================================
# 两个3D点求中心
# ============================================================
def center_of_two(p1, p2):
    """
    计算左右肩或左右髋的中心点。
    """

    if p1 is None or p2 is None:

        return None

    return (p1 + p2) / 2.0


# ============================================================
# 多个3D点求中心
# ============================================================
def center_of_points(points):
    """
    计算多个有效3D点的中心。
    """

    valid = []

    for point in points:

        if point is None:

            continue

        if not np.all(np.isfinite(point)):

            continue

        valid.append(point)

    if len(valid) == 0:

        return None

    return np.mean(
        np.asarray(valid, dtype=np.float32),
        axis=0
    )


# ============================================================
# 3D距离
# ============================================================
def distance3d(a, b):
    """
    计算两个3D点的欧氏距离。
    """

    if a is None or b is None:

        return None

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):

        return None

    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ============================================================
# 3D跳变保护
# ============================================================
def valid_jump(current, previous, max_jump):
    """
    判断当前3D点是否发生异常跳变。

    例如：
      当前Z = 0.8m
      下一帧Z = 4.5m

    这种情况认为Depth异常。
    """

    if current is None:

        return False

    if previous is None:

        return True

    jump = distance3d(current, previous)

    if jump is None:

        return False

    return jump <= max_jump


# ============================================================
# 判断关键点是否有效
# ============================================================
def is_valid_point(point):
    """
    判断3D关键点是否有效。
    """

    if point is None:

        return False

    return bool(
        np.all(
            np.isfinite(point)
        )
    )


# ============================================================
# 选择人体角度计算模式
# ============================================================
def select_measurement_mode(
    shoulder_center,
    hip_center,
    left_knee,
    right_knee,
    left_ankle,
    right_ankle
):
    """
    根据当前可见关键点选择角度计算方式。

    FULL_BODY：
      肩 + 髋完整可见。
      使用肩中心 -> 髋中心。

    LOWER_BODY：
      上半身不可用，
      但下半身关键点足够。
      使用髋部 -> 下肢中心。

    INVALID：
      只有上半身，
      或者有效下半身关键点不足。
    """

    # --------------------------------------------------------
    # 完整人体
    # --------------------------------------------------------
    if (
        is_valid_point(shoulder_center)
        and
        is_valid_point(hip_center)
    ):

        return "FULL_BODY"

    # --------------------------------------------------------
    # 下半身：
    # 髋 + 左右膝至少两个有效点
    # 或髋 + 左右踝至少两个有效点
    # --------------------------------------------------------
    knee_count = int(is_valid_point(left_knee)) + int(is_valid_point(right_knee))
    ankle_count = int(is_valid_point(left_ankle)) + int(is_valid_point(right_ankle))

    if is_valid_point(hip_center) and (knee_count >= 2 or ankle_count >= 2):

        return "LOWER_BODY"

    # --------------------------------------------------------
    # 只有上半身或有效关键点不足
    # --------------------------------------------------------
    return "INVALID"


# ============================================================
# 计算完整人体方向向量
# ============================================================
def compute_full_body_vector(
    shoulder_center,
    hip_center
):
    """
    完整人体模式：

        ShoulderCenter -> HipCenter

    得到人体躯干方向向量。
    """

    if (
        shoulder_center is None
        or
        hip_center is None
    ):

        return None

    return (
        hip_center
        -
        shoulder_center
    )


# ============================================================
# 计算下半身方向向量
# ============================================================
def compute_lower_body_vector(
    hip_center,
    left_knee,
    right_knee,
    left_ankle,
    right_ankle
):
    """
    下半身模式：

      第一优先：
        HipCenter -> KneeCenter

      如果膝盖不可用：
        HipCenter -> AnkleCenter

    注意：
      这个向量表示“下半身方向”，
      不等同于完整人体躯干方向。
    """

    if not is_valid_point(hip_center):

        return None

    knee_points = []

    if is_valid_point(left_knee):

        knee_points.append(
            left_knee
        )

    if is_valid_point(right_knee):

        knee_points.append(
            right_knee
        )

    # --------------------------------------------------------
    # 优先使用膝部中心
    # --------------------------------------------------------
    if len(knee_points) >= 2:

        knee_center = center_of_points(
            knee_points
        )

        if knee_center is not None:

            return (
                knee_center
                -
                hip_center
            )

    # --------------------------------------------------------
    # 膝盖不足时使用踝部中心
    # --------------------------------------------------------
    ankle_points = []

    if is_valid_point(left_ankle):

        ankle_points.append(
            left_ankle
        )

    if is_valid_point(right_ankle):

        ankle_points.append(
            right_ankle
        )

    if len(ankle_points) >= 2:

        ankle_center = center_of_points(
            ankle_points
        )

        if ankle_center is not None:

            return (
                ankle_center
                -
                hip_center
            )

    return None


# ============================================================
# 人体角度
# ============================================================
def compute_angle_from_vector(
    body_vector,
    ground_plane
):
    """
    计算任意人体方向向量与Ground Plane法向量的夹角。

    body_vector：
      人体当前测量方向。

    ground_plane：
      A*x+B*y+C*z+D=0。

    法向量：
      n=[A,B,C]

    公式：

      cos(theta)
      =
      |v·n| / (|v||n|)

      theta
      =
      arccos(cos(theta))

    角度：
      0°  -> 当前人体方向接近竖直
      90° -> 当前人体方向接近水平
    """

    if body_vector is None:

        return None

    body_norm = np.linalg.norm(
        body_vector
    )

    if body_norm < 1e-8:

        return None

    normal = (
        ground_plane[:3]
    )

    normal_norm = np.linalg.norm(
        normal
    )

    if normal_norm < 1e-8:

        return None

    cosine = (
        abs(
            np.dot(
                body_vector,
                normal
            )
        )
        /
        (
            body_norm
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

    angle = math.degrees(
        math.acos(
            cosine
        )
    )

    return float(
        angle
    )


# ============================================================
# Angle多帧滤波
# ============================================================
class AngleFilter:
    """
    对最近N帧角度取中值。

    作用：
      减少Depth和Pose抖动。

    注意：
      这里只是“测量滤波”，
      不参与跌倒判断。
    """

    def __init__(
        self,
        maxlen=ANGLE_HISTORY_LEN
    ):

        self.history = defaultdict(
            lambda:
            deque(
                maxlen=maxlen
            )
        )

    def update(
        self,
        person_id,
        mode,
        angle
    ):

        if angle is None:

            return None

        key = (
            person_id,
            mode
        )

        self.history[
            key
        ].append(
            angle
        )

        return float(
            np.median(
                self.history[
                    key
                ]
            )
        )


# ============================================================
# 绘制文字
# ============================================================
def put_text(
    image,
    text,
    x,
    y,
    scale=0.50,
    color=(255, 255, 255),
    thickness=2
):
    """
    在图像上绘制文字。
    """

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
# 获取显示颜色
# ============================================================
def get_mode_color(mode):
    """
    根据测量模式返回显示颜色。

    FULL_BODY：
      绿色

    LOWER_BODY：
      蓝色

    INVALID：
      红色
    """

    if mode == "FULL_BODY":

        return (
            0,
            255,
            0
        )

    if mode == "LOWER_BODY":

        return (
            255,
            180,
            0
        )

    return (
        0,
        0,
        255
    )


# ============================================================
# 主程序
# ============================================================
def main():

    print(
        "=" * 80
    )

    print(
        "Gemini 335Le + YOLO26s Pose"
    )

    print(
        "人体3D人体方向角度独立测试模块"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # 1. Ground Plane
    # ========================================================
    print(
        "\n[1] 加载Ground Plane..."
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
        "\n[2] 加载YOLO26s Pose..."
    )

    model = YOLO(
        POSE_MODEL_PATH
    )

    print(
        "Pose模型加载完成"
    )

    # ========================================================
    # 3. Angle Filter
    # ========================================================
    angle_filter = AngleFilter()

    # ========================================================
    # 4. Pipeline
    # ========================================================
    print(
        "\n[3] 启动Gemini 335Le..."
    )

    pipeline = Pipeline()

    # --------------------------------------------------------
    # Depth对齐到Color
    # --------------------------------------------------------
    align_filter = AlignFilter(
        align_to_stream=
        OBStreamType.COLOR_STREAM
    )

    # ========================================================
    # 历史3D数据
    # ========================================================
    previous_points = {}

    try:

        # ====================================================
        # 5. 启动相机
        # ====================================================
        pipeline.start()

        print(
            "Pipeline启动成功"
        )

        # ====================================================
        # 6. 获取相机内参
        # ====================================================
        camera_param = (
            pipeline
            .get_camera_param()
        )

        rgb = (
            camera_param
            .rgb_intrinsic
        )

        intrinsics = {
            "width":
                int(
                    rgb.width
                ),

            "height":
                int(
                    rgb.height
                ),

            "fx":
                float(
                    rgb.fx
                ),

            "fy":
                float(
                    rgb.fy
                ),

            "cx":
                float(
                    rgb.cx
                ),

            "cy":
                float(
                    rgb.cy
                )
        }

        print(
            "\nRGB内参："
        )

        print(
            f"分辨率："
            f"{intrinsics['width']}x"
            f"{intrinsics['height']}"
        )

        print(
            f"fx={intrinsics['fx']:.6f}"
            f"  fy={intrinsics['fy']:.6f}"
        )

        print(
            f"cx={intrinsics['cx']:.6f}"
            f"  cy={intrinsics['cy']:.6f}"
        )

        # ====================================================
        # 7. 开始测试
        # ====================================================
        print(
            "\n[4] 开始人体3D方向角度测试"
        )

        print(
            "FULL_BODY：肩髋完整可见"
        )

        print(
            "LOWER_BODY：只有下半身可用"
        )

        print(
            "INVALID：无法形成可靠人体方向"
        )

        print(
            "本模块只计算Angle，不判断跌倒"
        )

        print(
            "ESC退出"
        )

        # ====================================================
        # 主循环
        # ====================================================
        while True:

            # =================================================
            # 8. 获取RGB-D帧
            # =================================================
            frames = (
                pipeline
                .wait_for_frames(
                    1000
                )
            )

            if frames is None:

                continue

            # =================================================
            # 9. RGB-D对齐
            # =================================================
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

            # =================================================
            # 10. RGB
            # =================================================
            color_frame = (
                frames
                .get_color_frame()
            )

            if color_frame is None:

                continue

            image = frame_to_bgr(
                color_frame
            )

            if image is None:

                continue

            # =================================================
            # 11. Depth
            # =================================================
            depth_frame = (
                frames
                .get_depth_frame()
            )

            if depth_frame is None:

                put_text(
                    image,
                    "DEPTH NONE",
                    15,
                    30,
                    0.60,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

                cv2.imshow(
                    "Gemini 335Le Body Angle",
                    image
                )

                if (
                    cv2.waitKey(1)
                    &
                    0xFF
                ) == 27:

                    break

                continue

            depth = (
                depth_frame_to_meters(
                    depth_frame
                )
            )

            # =================================================
            # 12. YOLO Pose
            # =================================================
            results = model.track(
                image,
                persist=True,
                tracker="botsort.yaml",
                conf=POSE_CONF,
                verbose=False
            )

            # =================================================
            # 13. 无人体时仍显示画面
            # =================================================
            if (
                not results
                or
                results[0].boxes is None
                or
                results[0].keypoints is None
            ):

                put_text(
                    image,
                    "NO PERSON",
                    15,
                    30,
                    0.60,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

                put_text(
                    image,
                    "ESC: EXIT",
                    15,
                    image.shape[0] - 15,
                    0.45,
                    (
                        255,
                        255,
                        255
                    ),
                    1
                )

                cv2.imshow(
                    "Gemini 335Le Body Angle",
                    image
                )

                if (
                    cv2.waitKey(1)
                    &
                    0xFF
                ) == 27:

                    break

                continue

            # =================================================
            # 14. 获取YOLO结果
            # =================================================
            result = (
                results[0]
            )

            boxes = (
                result
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            keypoints = (
                result
                .keypoints
                .data
                .cpu()
                .numpy()
            )

            # =================================================
            # 15. 获取Track ID
            # =================================================
            if (
                result.boxes.id
                is not None
            ):

                track_ids = (
                    result
                    .boxes
                    .id
                    .int()
                    .cpu()
                    .numpy()
                )

            else:

                track_ids = np.arange(
                    len(boxes)
                )

            # =================================================
            # 16. 遍历人体
            # =================================================
            for i, bbox in enumerate(
                boxes
            ):

                person_id = int(
                    track_ids[i]
                )

                kp = keypoints[i]

                keypoint_xy = (
                    kp[
                        :,
                        :2
                    ]
                )

                if (
                    kp.shape[1]
                    >= 3
                ):

                    keypoint_conf = (
                        kp[
                            :,
                            2
                        ]
                    )

                else:

                    keypoint_conf = (
                        np.ones(
                            len(kp),
                            dtype=np.float32
                        )
                    )

                # =================================================
                # 17. 获取左右肩
                # =================================================
                left_shoulder = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        LEFT_SHOULDER,
                        depth,
                        intrinsics
                    )
                )

                right_shoulder = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        RIGHT_SHOULDER,
                        depth,
                        intrinsics
                    )
                )

                # =================================================
                # 18. 获取左右髋
                # =================================================
                left_hip = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        LEFT_HIP,
                        depth,
                        intrinsics
                    )
                )

                right_hip = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        RIGHT_HIP,
                        depth,
                        intrinsics
                    )
                )

                # =================================================
                # 19. 获取左右膝
                # =================================================
                left_knee = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        LEFT_KNEE,
                        depth,
                        intrinsics
                    )
                )

                right_knee = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        RIGHT_KNEE,
                        depth,
                        intrinsics
                    )
                )

                # =================================================
                # 20. 获取左右踝
                # =================================================
                left_ankle = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        LEFT_ANKLE,
                        depth,
                        intrinsics
                    )
                )

                right_ankle = (
                    get_keypoint_3d(
                        keypoint_xy,
                        keypoint_conf,
                        RIGHT_ANKLE,
                        depth,
                        intrinsics
                    )
                )

                # =================================================
                # 21. 计算肩部中心
                # =================================================
                shoulder_center = (
                    center_of_two(
                        left_shoulder,
                        right_shoulder
                    )
                )

                # =================================================
                # 22. 计算髋部中心
                # =================================================
                hip_center = (
                    center_of_two(
                        left_hip,
                        right_hip
                    )
                )

                # =================================================
                # 23. 3D跳变保护
                # =================================================
                previous = (
                    previous_points.get(
                        person_id
                    )
                )

                if (
                    shoulder_center is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "shoulder"
                    ) is not None
                ):

                    if not valid_jump(
                        shoulder_center,
                        previous["shoulder"],
                        MAX_SHOULDER_JUMP_M
                    ):

                        shoulder_center = None

                if (
                    hip_center is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "hip"
                    ) is not None
                ):

                    if not valid_jump(
                        hip_center,
                        previous["hip"],
                        MAX_HIP_JUMP_M
                    ):

                        hip_center = None

                if (
                    left_knee is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "left_knee"
                    ) is not None
                ):

                    if not valid_jump(
                        left_knee,
                        previous["left_knee"],
                        MAX_KNEE_JUMP_M
                    ):

                        left_knee = None

                if (
                    right_knee is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "right_knee"
                    ) is not None
                ):

                    if not valid_jump(
                        right_knee,
                        previous["right_knee"],
                        MAX_KNEE_JUMP_M
                    ):

                        right_knee = None

                if (
                    left_ankle is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "left_ankle"
                    ) is not None
                ):

                    if not valid_jump(
                        left_ankle,
                        previous["left_ankle"],
                        MAX_ANKLE_JUMP_M
                    ):

                        left_ankle = None

                if (
                    right_ankle is not None
                    and
                    previous is not None
                    and
                    previous.get(
                        "right_ankle"
                    ) is not None
                ):

                    if not valid_jump(
                        right_ankle,
                        previous["right_ankle"],
                        MAX_ANKLE_JUMP_M
                    ):

                        right_ankle = None

                # =================================================
                # 24. 保存当前3D点
                # =================================================
                previous_points[
                    person_id
                ] = {
                    "shoulder":
                        shoulder_center,

                    "hip":
                        hip_center,

                    "left_knee":
                        left_knee,

                    "right_knee":
                        right_knee,

                    "left_ankle":
                        left_ankle,

                    "right_ankle":
                        right_ankle,
                }

                # =================================================
                # 25. 选择计算模式
                # =================================================
                measurement_mode = (
                    select_measurement_mode(
                        shoulder_center,
                        hip_center,
                        left_knee,
                        right_knee,
                        left_ankle,
                        right_ankle
                    )
                )

                # =================================================
                # 26. 初始化
                # =================================================
                body_vector = None
                lower_body_center = None

                # =================================================
                # 27. FULL_BODY
                # =================================================
                if (
                    measurement_mode
                    ==
                    "FULL_BODY"
                ):

                    body_vector = (
                        compute_full_body_vector(
                            shoulder_center,
                            hip_center
                        )
                    )

                # =================================================
                # 28. LOWER_BODY
                # =================================================
                elif (
                    measurement_mode
                    ==
                    "LOWER_BODY"
                ):

                    # ------------------------------------------------
                    # 膝部优先
                    # ------------------------------------------------
                    knee_points = []

                    if is_valid_point(
                        left_knee
                    ):

                        knee_points.append(
                            left_knee
                        )

                    if is_valid_point(
                        right_knee
                    ):

                        knee_points.append(
                            right_knee
                        )

                    if len(knee_points) >= 2:

                        lower_body_center = (
                            center_of_points(
                                knee_points
                            )
                        )

                    else:

                        # ------------------------------------------------
                        # 膝部不足，改用踝部
                        # ------------------------------------------------
                        ankle_points = []

                        if is_valid_point(
                            left_ankle
                        ):

                            ankle_points.append(
                                left_ankle
                            )

                        if is_valid_point(
                            right_ankle
                        ):

                            ankle_points.append(
                                right_ankle
                            )

                        if len(ankle_points) >= 2:

                            lower_body_center = (
                                center_of_points(
                                    ankle_points
                                )
                            )

                    if lower_body_center is not None:

                        body_vector = (
                            lower_body_center
                            -
                            hip_center
                        )

                    else:

                        measurement_mode = (
                            "INVALID"
                        )

                # =================================================
                # 29. INVALID
                # =================================================
                else:

                    body_vector = None

                # =================================================
                # 30. 计算Raw Angle
                # =================================================
                raw_angle = (
                    compute_angle_from_vector(
                        body_vector,
                        ground_plane
                    )
                )

                # =================================================
                # 31. Angle多帧滤波
                # =================================================
                filtered_angle = (
                    angle_filter.update(
                        person_id,
                        measurement_mode,
                        raw_angle
                    )
                )

                # =================================================
                # 32. 当前显示颜色
                # =================================================
                mode_color = (
                    get_mode_color(
                        measurement_mode
                    )
                )

                # =================================================
                # 33. 人体框
                # =================================================
                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                cv2.rectangle(
                    image,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    mode_color,
                    2
                )

                # =================================================
                # 34. Pose关键点
                # =================================================
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
                        (
                            0,
                            255,
                            255
                        ),
                        -1
                    )

                # =================================================
                # 35. 获取2D肩中心
                # =================================================
                shoulder_2d = None

                if (
                    keypoint_conf[
                        LEFT_SHOULDER
                    ]
                    >= KEYPOINT_CONF
                    and
                    keypoint_conf[
                        RIGHT_SHOULDER
                    ]
                    >= KEYPOINT_CONF
                ):

                    shoulder_2d = (
                        int(
                            (
                                keypoint_xy[
                                    LEFT_SHOULDER,
                                    0
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_SHOULDER,
                                    0
                                ]
                            )
                            /
                            2.0
                        ),
                        int(
                            (
                                keypoint_xy[
                                    LEFT_SHOULDER,
                                    1
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_SHOULDER,
                                    1
                                ]
                            )
                            /
                            2.0
                        )
                    )

                # =================================================
                # 36. 获取2D髋中心
                # =================================================
                hip_2d = None

                if (
                    keypoint_conf[
                        LEFT_HIP
                    ]
                    >= KEYPOINT_CONF
                    and
                    keypoint_conf[
                        RIGHT_HIP
                    ]
                    >= KEYPOINT_CONF
                ):

                    hip_2d = (
                        int(
                            (
                                keypoint_xy[
                                    LEFT_HIP,
                                    0
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_HIP,
                                    0
                                ]
                            )
                            /
                            2.0
                        ),
                        int(
                            (
                                keypoint_xy[
                                    LEFT_HIP,
                                    1
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_HIP,
                                    1
                                ]
                            )
                            /
                            2.0
                        )
                    )

                # =================================================
                # 37. 获取2D膝中心
                # =================================================
                knee_2d = None

                if (
                    keypoint_conf[
                        LEFT_KNEE
                    ]
                    >= KEYPOINT_CONF
                    and
                    keypoint_conf[
                        RIGHT_KNEE
                    ]
                    >= KEYPOINT_CONF
                ):

                    knee_2d = (
                        int(
                            (
                                keypoint_xy[
                                    LEFT_KNEE,
                                    0
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_KNEE,
                                    0
                                ]
                            )
                            /
                            2.0
                        ),
                        int(
                            (
                                keypoint_xy[
                                    LEFT_KNEE,
                                    1
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_KNEE,
                                    1
                                ]
                            )
                            /
                            2.0
                        )
                    )

                # =================================================
                # 38. 获取2D踝中心
                # =================================================
                ankle_2d = None

                if (
                    keypoint_conf[
                        LEFT_ANKLE
                    ]
                    >= KEYPOINT_CONF
                    and
                    keypoint_conf[
                        RIGHT_ANKLE
                    ]
                    >= KEYPOINT_CONF
                ):

                    ankle_2d = (
                        int(
                            (
                                keypoint_xy[
                                    LEFT_ANKLE,
                                    0
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_ANKLE,
                                    0
                                ]
                            )
                            /
                            2.0
                        ),
                        int(
                            (
                                keypoint_xy[
                                    LEFT_ANKLE,
                                    1
                                ]
                                +
                                keypoint_xy[
                                    RIGHT_ANKLE,
                                    1
                                ]
                            )
                            /
                            2.0
                        )
                    )

                # =================================================
                # 39. 画关键方向线
                # =================================================
                if (
                    measurement_mode
                    ==
                    "FULL_BODY"
                    and
                    shoulder_2d is not None
                    and
                    hip_2d is not None
                ):

                    cv2.line(
                        image,
                        shoulder_2d,
                        hip_2d,
                        (
                            255,
                            0,
                            255
                        ),
                        3
                    )

                elif (
                    measurement_mode
                    ==
                    "LOWER_BODY"
                    and
                    hip_2d is not None
                ):

                    if (
                        knee_2d is not None
                    ):

                        cv2.line(
                            image,
                            hip_2d,
                            knee_2d,
                            (
                                255,
                                180,
                                0
                            ),
                            3
                        )

                    elif (
                        ankle_2d is not None
                    ):

                        cv2.line(
                            image,
                            hip_2d,
                            ankle_2d,
                            (
                                255,
                                180,
                                0
                            ),
                            3
                        )

                # =================================================
                # 40. 肩部中心
                # =================================================
                if (
                    shoulder_2d
                    is not None
                ):

                    cv2.circle(
                        image,
                        shoulder_2d,
                        6,
                        (
                            255,
                            0,
                            255
                        ),
                        -1
                    )

                # =================================================
                # 41. 髋部中心
                # =================================================
                if (
                    hip_2d
                    is not None
                ):

                    cv2.circle(
                        image,
                        hip_2d,
                        6,
                        (
                            255,
                            0,
                            255
                        ),
                        -1
                    )

                # =================================================
                # 42. 下肢中心
                # =================================================
                if (
                    lower_body_center is not None
                    and
                    measurement_mode
                    ==
                    "LOWER_BODY"
                ):

                    if knee_2d is not None:

                        cv2.circle(
                            image,
                            knee_2d,
                            6,
                            (
                                255,
                                180,
                                0
                            ),
                            -1
                        )

                    elif ankle_2d is not None:

                        cv2.circle(
                            image,
                            ankle_2d,
                            6,
                            (
                                255,
                                180,
                                0
                            ),
                            -1
                        )

                # =================================================
                # 43. Mode文字
                # =================================================
                put_text(
                    image,
                    (
                        f"ID={person_id} "
                        f"Mode={measurement_mode}"
                    ),
                    x1,
                    max(
                        25,
                        y1 - 30
                    ),
                    0.52,
                    mode_color,
                    2
                )

                # =================================================
                # 44. Angle文字
                # =================================================
                if filtered_angle is not None:

                    put_text(
                        image,
                        (
                            f"Angle="
                            f"{filtered_angle:.1f} deg"
                        ),
                        x1,
                        max(
                            50,
                            y1 - 8
                        ),
                        0.50,
                        (
                            255,
                            255,
                            255
                        ),
                        2
                    )

                else:

                    put_text(
                        image,
                        "Angle=None",
                        x1,
                        max(
                            50,
                            y1 - 8
                        ),
                        0.50,
                        (
                            0,
                            0,
                            255
                        ),
                        2
                    )

                # =================================================
                # 45. 右侧数据显示
                # =================================================
                info_x = (
                    x2 + 10
                )

                if (
                    info_x
                    >
                    image.shape[1] - 260
                ):

                    info_x = max(
                        5,
                        x1 - 255
                    )

                info_y = max(
                    25,
                    y1
                )

                # -------------------------------------------------
                # Measurement Mode
                # -------------------------------------------------
                put_text(
                    image,
                    (
                        f"Mode="
                        f"{measurement_mode}"
                    ),
                    info_x,
                    info_y,
                    0.38,
                    mode_color,
                    1
                )

                info_y += 17

                # -------------------------------------------------
                # Shoulder3D
                # -------------------------------------------------
                if (
                    shoulder_center
                    is not None
                ):

                    put_text(
                        image,
                        (
                            f"Shoulder3D="
                            f"({shoulder_center[0]:.2f},"
                            f"{shoulder_center[1]:.2f},"
                            f"{shoulder_center[2]:.2f})"
                        ),
                        info_x,
                        info_y,
                        0.36
                    )

                else:

                    put_text(
                        image,
                        "Shoulder3D=None",
                        info_x,
                        info_y,
                        0.36
                    )

                info_y += 17

                # -------------------------------------------------
                # Hip3D
                # -------------------------------------------------
                if (
                    hip_center
                    is not None
                ):

                    put_text(
                        image,
                        (
                            f"Hip3D="
                            f"({hip_center[0]:.2f},"
                            f"{hip_center[1]:.2f},"
                            f"{hip_center[2]:.2f})"
                        ),
                        info_x,
                        info_y,
                        0.36
                    )

                else:

                    put_text(
                        image,
                        "Hip3D=None",
                        info_x,
                        info_y,
                        0.36
                    )

                info_y += 17

                # -------------------------------------------------
                # LowerBody3D
                # -------------------------------------------------
                if (
                    lower_body_center
                    is not None
                ):

                    put_text(
                        image,
                        (
                            f"Lower3D="
                            f"({lower_body_center[0]:.2f},"
                            f"{lower_body_center[1]:.2f},"
                            f"{lower_body_center[2]:.2f})"
                        ),
                        info_x,
                        info_y,
                        0.36,
                        (
                            255,
                            180,
                            0
                        )
                    )

                else:

                    put_text(
                        image,
                        "Lower3D=None",
                        info_x,
                        info_y,
                        0.36
                    )

                info_y += 17

                # -------------------------------------------------
                # Body Vector
                # -------------------------------------------------
                if (
                    body_vector
                    is not None
                ):

                    put_text(
                        image,
                        (
                            f"Vector="
                            f"({body_vector[0]:.2f},"
                            f"{body_vector[1]:.2f},"
                            f"{body_vector[2]:.2f})"
                        ),
                        info_x,
                        info_y,
                        0.36
                    )

                else:

                    put_text(
                        image,
                        "Vector=None",
                        info_x,
                        info_y,
                        0.36
                    )

                info_y += 17

                # -------------------------------------------------
                # Raw Angle
                # -------------------------------------------------
                if raw_angle is not None:

                    put_text(
                        image,
                        (
                            f"RawAngle="
                            f"{raw_angle:.1f}deg"
                        ),
                        info_x,
                        info_y,
                        0.36
                    )

                else:

                    put_text(
                        image,
                        "RawAngle=None",
                        info_x,
                        info_y,
                        0.36
                    )

                info_y += 17

                # -------------------------------------------------
                # Filtered Angle
                # -------------------------------------------------
                if filtered_angle is not None:

                    put_text(
                        image,
                        (
                            f"FilteredAngle="
                            f"{filtered_angle:.1f}deg"
                        ),
                        info_x,
                        info_y,
                        0.36
                    )

                else:

                    put_text(
                        image,
                        "FilteredAngle=None",
                        info_x,
                        info_y,
                        0.36
                    )

                # =================================================
                # 46. 左上角状态面板
                # =================================================
                cv2.rectangle(
                    image,
                    (
                        5,
                        5
                    ),
                    (
                        335,
                        105
                    ),
                    (
                        0,
                        0,
                        0
                    ),
                    -1
                )

                put_text(
                    image,
                    "BODY DIRECTION TEST",
                    15,
                    28,
                    0.50,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

                put_text(
                    image,
                    (
                        f"MODE: "
                        f"{measurement_mode}"
                    ),
                    15,
                    52,
                    0.48,
                    mode_color,
                    2
                )

                if filtered_angle is not None:

                    put_text(
                        image,
                        (
                            f"ANGLE: "
                            f"{filtered_angle:.1f} deg"
                        ),
                        15,
                        78,
                        0.55,
                        (
                            0,
                            255,
                            255
                        ),
                        2
                    )

                else:

                    put_text(
                        image,
                        "ANGLE: None",
                        15,
                        78,
                        0.55,
                        (
                            0,
                            0,
                            255
                        ),
                        2
                    )

                # =================================================
                # 47. 图像底部提示
                # =================================================
                put_text(
                    image,
                    (
                        "FULL_BODY=Green  "
                        "LOWER_BODY=Blue  "
                        "INVALID=Red"
                    ),
                    10,
                    image.shape[0] - 32,
                    0.40,
                    (
                        255,
                        255,
                        255
                    ),
                    1
                )

                put_text(
                    image,
                    "ESC: EXIT",
                    10,
                    image.shape[0] - 12,
                    0.40,
                    (
                        255,
                        255,
                        255
                    ),
                    1
                )

                # =================================================
                # 48. 终端输出
                # =================================================
                terminal = (
                    f"ID={person_id} | "
                    f"Mode={measurement_mode} | "
                )

                if filtered_angle is not None:

                    terminal += (
                        f"Angle={filtered_angle:.1f}deg | "
                    )

                else:

                    terminal += (
                        "Angle=None | "
                    )

                if shoulder_center is not None:

                    terminal += (
                        "Shoulder3D=("
                        f"{shoulder_center[0]:.2f},"
                        f"{shoulder_center[1]:.2f},"
                        f"{shoulder_center[2]:.2f}) | "
                    )

                if hip_center is not None:

                    terminal += (
                        "Hip3D=("
                        f"{hip_center[0]:.2f},"
                        f"{hip_center[1]:.2f},"
                        f"{hip_center[2]:.2f})"
                    )

                print(
                    "\r"
                    + terminal,
                    end=""
                )

            # =================================================
            # 49. 显示画面
            #
            # 无论有没有人体，都持续刷新摄像头画面。
            # =================================================
            cv2.imshow(
                "Gemini 335Le Body Angle",
                image
            )

            # =================================================
            # 50. ESC退出
            # =================================================
            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == 27:

                break

    finally:

        # =====================================================
        # 51. 关闭Pipeline
        # =====================================================
        pipeline.stop()

        # =====================================================
        # 52. 关闭OpenCV窗口
        # =====================================================
        cv2.destroyAllWindows()

        print(
            "\n\nPipeline stopped."
        )


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":

    main()