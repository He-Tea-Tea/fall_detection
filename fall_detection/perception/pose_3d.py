# -*- coding: utf-8 -*-
"""三维人体姿态检测模块。

本文件只负责计算三维人体姿态，不负责相机采集、跌倒状态机或场景判断。

输入：
    1. person_id：人员跟踪ID。
    2. keypoints_3d：人体3D关键点，单位为米。
    3. keypoint_conf：每个关键点的置信度。
    4. ground_plane：ground.yaml中的地面平面[A, B, C, D]。

输出：
    1. 人体3D方向角。
    2. 经过多帧中值滤波后的角度。
    3. 三维姿态风险子分P3D。
    4. 本次角度测量质量。

角度定义：
    0°：人体轴线接近地面法向量，通常表示站立。
    90°：人体轴线接近地面平面，通常表示躺卧。

优先使用“肩中心→髋中心”作为人体轴线。
肩部关键点不可用时，可以回退使用“髋中心→膝中心”或“髋中心→踝中心”。
"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]

# COCO 17个人体关键点编号。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class Pose3DResult:
    """单个人员的一次三维姿态检测结果。

    valid：
        True表示成功获得可靠的三维角度。
        False表示关键点不足、深度无效或人体轴线太短。

    mode：
        FULL_BODY：使用肩中心到髋中心。
        LOWER_BODY_KNEE：使用髋中心到膝中心。
        LOWER_BODY_ANKLE：使用髋中心到踝中心。
        INVALID：无法形成可靠人体轴线。

    score：
        P3D三维姿态风险分，范围为0～1。
        越接近0越像站立，越接近1越像水平躺卧。
    """

    person_id: int
    valid: bool
    mode: str
    raw_angle_deg: Optional[float]
    filtered_angle_deg: Optional[float]
    score: float
    measurement_quality: float
    shoulder_center_3d: Optional[np.ndarray]
    hip_center_3d: Optional[np.ndarray]
    lower_center_3d: Optional[np.ndarray]
    body_vector_3d: Optional[np.ndarray]
    reason: str


def load_config(path: str) -> dict:
    """读取统一配置文件config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_ground_plane(path: str) -> np.ndarray:
    """读取并归一化ground.yaml中的地面平面[A, B, C, D]。

    地面平面公式为：
        A×X + B×Y + C×Z + D = 0

    其中[A, B, C]是地面法向量。归一化以后，法向量长度等于1。
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    plane_data = data.get("plane")
    if not isinstance(plane_data, dict):
        raise ValueError("ground.yaml中不存在有效plane字段")

    plane = np.array(
        [
            plane_data["A"],
            plane_data["B"],
            plane_data["C"],
            plane_data["D"],
        ],
        dtype=np.float64,
    )

    normal_length = float(np.linalg.norm(plane[:3]))
    if normal_length < 1e-8:
        raise ValueError("ground.yaml地面法向量无效")

    # A、B、C、D必须同时除以法向量长度，保持平面方程不变。
    return plane / normal_length


def _angle_score(angle_deg: float, normal_deg: float, fall_deg: float) -> float:
    """把三维角度转换为0～1的P3D风险分。

    angle <= normal：返回0，表示接近正常直立。
    angle >= fall：返回1，表示接近水平躺卧。
    两个阈值之间：按照角度线性增加。
    """
    if fall_deg <= normal_deg:
        raise ValueError("pose_3d.angle_fall_deg必须大于angle_normal_deg")

    score = (angle_deg - normal_deg) / (fall_deg - normal_deg)
    return float(np.clip(score, 0.0, 1.0))


class Pose3DDetector:
    """按人员Track ID计算并滤波三维人体角度。

    本类只使用3D关键点、关键点置信度和地面法向量，不读取二维角度、
    人体检测框、高度分、速度分、场景分或最终跌倒状态。
    """

    def __init__(self, config: dict, ground_plane: Sequence[float]):
        self.cfg = config["pose_3d"]

        # 从config.yaml读取所有可调参数。
        self.min_keypoint_conf = float(self.cfg["min_keypoint_conf"])
        self.history_length = int(self.cfg["history_length"])
        self.min_vector_length_m = float(self.cfg["min_vector_length_m"])
        self.allow_lower_body = bool(self.cfg["allow_lower_body_fallback"])
        self.angle_normal_deg = float(self.cfg["angle_normal_deg"])
        self.angle_fall_deg = float(self.cfg["angle_fall_deg"])

        # 启动时检查配置，避免运行过程中才出现错误。
        if not 0.0 <= self.min_keypoint_conf <= 1.0:
            raise ValueError("pose_3d.min_keypoint_conf必须在0到1之间")
        if self.history_length <= 0:
            raise ValueError("pose_3d.history_length必须大于0")
        if self.min_vector_length_m <= 0.0:
            raise ValueError("pose_3d.min_vector_length_m必须大于0")
        if self.angle_fall_deg <= self.angle_normal_deg:
            raise ValueError("pose_3d.angle_fall_deg必须大于angle_normal_deg")

        # 地面可以由GroundManager在线更新，因此通过统一方法保存法向量。
        self.ground_normal = np.zeros(3, dtype=np.float64)
        self.set_ground_plane(ground_plane)

        # 每个人、每种测量模式分别保存角度历史，避免不同人员或模式混用。
        self.histories: Dict[Tuple[int, str], deque] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

    def set_ground_plane(
        self,
        ground_plane: Sequence[float],
        reset_histories: bool = False,
    ) -> None:
        """更新地面法向量；接受新地面时可同时清除旧角度历史。"""
        plane = np.asarray(ground_plane, dtype=np.float64)
        if plane.shape != (4,):
            raise ValueError("ground_plane必须是[A, B, C, D]")
        normal_length = float(np.linalg.norm(plane[:3]))
        if normal_length < 1e-8:
            raise ValueError("地面法向量无效")
        self.ground_normal = plane[:3] / normal_length
        if reset_histories and hasattr(self, "histories"):
            self.histories.clear()

    def _point_valid(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        index: int,
    ) -> bool:
        """判断一个3D关键点能否参与角度计算。

        必须同时满足：
            1. 关键点编号没有超出数组范围。
            2. X、Y、Z都不是NaN或无穷大。
            3. 关键点置信度达到配置阈值。
        """
        if index >= len(points) or index >= len(confidence):
            return False
        if not np.all(np.isfinite(points[index])):
            return False
        return bool(confidence[index] >= self.min_keypoint_conf)

    def _pair_center(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        left: int,
        right: int,
    ) -> Optional[np.ndarray]:
        """计算左右两个关键点的三维中心。

        左右点必须同时有效，避免只使用单侧关键点导致人体轴线偏斜。
        """
        left_valid = self._point_valid(points, confidence, left)
        right_valid = self._point_valid(points, confidence, right)

        if not left_valid or not right_valid:
            return None

        return (points[left] + points[right]) / 2.0

    def _select_vector(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple:
        """根据有效关键点选择人体三维轴线。

        返回内容依次为：
            测量模式、肩中心、髋中心、下肢中心、人体轴线、使用的关键点编号。
        """
        shoulder_center = self._pair_center(
            points,
            confidence,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
        )
        hip_center = self._pair_center(
            points,
            confidence,
            LEFT_HIP,
            RIGHT_HIP,
        )

        # 第一优先级：肩中心到髋中心，最能代表完整躯干方向。
        if shoulder_center is not None and hip_center is not None:
            body_vector = hip_center - shoulder_center
            used_indices = (
                LEFT_SHOULDER,
                RIGHT_SHOULDER,
                LEFT_HIP,
                RIGHT_HIP,
            )
            return (
                "FULL_BODY",
                shoulder_center,
                hip_center,
                None,
                body_vector,
                used_indices,
            )

        # 没有髋中心，或者配置禁止下半身回退时，无法继续计算。
        if not self.allow_lower_body or hip_center is None:
            return (
                "INVALID",
                shoulder_center,
                hip_center,
                None,
                None,
                (),
            )

        # 第二优先级：髋中心到膝中心。
        knee_center = self._pair_center(
            points,
            confidence,
            LEFT_KNEE,
            RIGHT_KNEE,
        )
        if knee_center is not None:
            body_vector = knee_center - hip_center
            used_indices = (
                LEFT_HIP,
                RIGHT_HIP,
                LEFT_KNEE,
                RIGHT_KNEE,
            )
            return (
                "LOWER_BODY_KNEE",
                shoulder_center,
                hip_center,
                knee_center,
                body_vector,
                used_indices,
            )

        # 第三优先级：髋中心到踝中心。
        ankle_center = self._pair_center(
            points,
            confidence,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        )
        if ankle_center is not None:
            body_vector = ankle_center - hip_center
            used_indices = (
                LEFT_HIP,
                RIGHT_HIP,
                LEFT_ANKLE,
                RIGHT_ANKLE,
            )
            return (
                "LOWER_BODY_ANKLE",
                shoulder_center,
                hip_center,
                ankle_center,
                body_vector,
                used_indices,
            )

        return (
            "INVALID",
            shoulder_center,
            hip_center,
            None,
            None,
            (),
        )

    def update(
        self,
        person_id: int,
        keypoints_3d: np.ndarray,
        keypoint_conf: Optional[np.ndarray] = None,
    ) -> Pose3DResult:
        """计算一名人员当前帧的三维姿态结果。

        参数：
            person_id：人员Track ID，用于保存各自的角度历史。
            keypoints_3d：Nx3数组，每行为[X, Y, Z]，单位为米。
            keypoint_conf：长度为N的关键点置信度数组。

        返回：
            Pose3DResult对象。
        """
        points = np.asarray(keypoints_3d, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("keypoints_3d必须是Nx3数组")

        if keypoint_conf is None:
            confidence = np.ones(len(points), dtype=np.float64)
        else:
            confidence = np.asarray(keypoint_conf, dtype=np.float64)

        if confidence.shape != (len(points),):
            raise ValueError("keypoint_conf数量必须与keypoints_3d一致")

        (
            mode,
            shoulder_center,
            hip_center,
            lower_center,
            body_vector,
            used_indices,
        ) = self._select_vector(points, confidence)

        # 没有形成有效人体轴线时，明确返回无效结果，不编造角度。
        if body_vector is None:
            return Pose3DResult(
                person_id=int(person_id),
                valid=False,
                mode="INVALID",
                raw_angle_deg=None,
                filtered_angle_deg=None,
                score=0.0,
                measurement_quality=0.0,
                shoulder_center_3d=shoulder_center,
                hip_center_3d=hip_center,
                lower_center_3d=lower_center,
                body_vector_3d=None,
                reason="关键点不足，无法形成可靠3D人体轴线",
            )

        vector_length = float(np.linalg.norm(body_vector))
        if vector_length < self.min_vector_length_m:
            return Pose3DResult(
                person_id=int(person_id),
                valid=False,
                mode=mode,
                raw_angle_deg=None,
                filtered_angle_deg=None,
                score=0.0,
                measurement_quality=0.0,
                shoulder_center_3d=shoulder_center,
                hip_center_3d=hip_center,
                lower_center_3d=lower_center,
                body_vector_3d=body_vector,
                reason="3D人体轴线过短",
            )

        # 点积公式：cosθ = |人体轴线·地面法向量| / |人体轴线|。
        # 使用绝对值是因为轴线正方向和反方向不影响人体是竖直还是水平。
        cosine = abs(
            float(np.dot(body_vector, self.ground_normal))
        ) / vector_length
        cosine = float(np.clip(cosine, -1.0, 1.0))
        raw_angle = float(np.degrees(np.arccos(cosine)))

        # 对同一ID、同一模式的最近多帧角度取中值，减少Depth抖动。
        history_key = (int(person_id), mode)
        history = self.histories[history_key]
        history.append(raw_angle)
        filtered_angle = float(np.median(np.asarray(history, dtype=np.float64)))

        # 测量质量同时考虑关键点置信度和人体轴线长度。
        used_confidence = confidence[list(used_indices)]
        confidence_quality = float(np.mean(used_confidence))
        expected_length = max(self.min_vector_length_m * 2.0, 1e-8)
        length_quality = float(
            np.clip(vector_length / expected_length, 0.0, 1.0)
        )
        measurement_quality = float(
            np.clip(confidence_quality * length_quality, 0.0, 1.0)
        )

        # 将滤波后的三维角度转换为P3D风险分。
        score = _angle_score(
            filtered_angle,
            self.angle_normal_deg,
            self.angle_fall_deg,
        )

        return Pose3DResult(
            person_id=int(person_id),
            valid=True,
            mode=mode,
            raw_angle_deg=raw_angle,
            filtered_angle_deg=filtered_angle,
            score=score,
            measurement_quality=measurement_quality,
            shoulder_center_3d=shoulder_center,
            hip_center_3d=hip_center,
            lower_center_3d=lower_center,
            body_vector_3d=body_vector,
            reason="OK",
        )

    def reset_person(self, person_id: int) -> None:
        """清除指定人员的角度历史。

        人员离开画面或Track ID失效后必须清除，避免以后复用相同ID时继承旧角度。
        """
        person_id = int(person_id)
        keys_to_remove = [
            key
            for key in self.histories
            if key[0] == person_id
        ]

        for key in keys_to_remove:
            del self.histories[key]


def run_self_test(config: dict) -> None:
    """使用合成3D关键点验证主要判断逻辑，不需要相机。"""
    # 测试地面法向量指向Y轴负方向，D不会参与角度计算。
    detector = Pose3DDetector(
        config,
        ground_plane=[0.0, -1.0, 0.0, 1.0],
    )
    confidence = np.ones(17, dtype=np.float64)

    # 站立：肩到髋沿Y轴，人体轴线与地面法向量平行，角度应为0°。
    vertical = np.full((17, 3), np.nan, dtype=np.float64)
    vertical[LEFT_SHOULDER] = [-0.2, 0.0, 2.0]
    vertical[RIGHT_SHOULDER] = [0.2, 0.0, 2.0]
    vertical[LEFT_HIP] = [-0.2, 0.5, 2.0]
    vertical[RIGHT_HIP] = [0.2, 0.5, 2.0]
    standing = detector.update(1, vertical, confidence)

    # 躺卧：肩到髋沿X轴，人体轴线与地面法向量垂直，角度应为90°。
    horizontal = vertical.copy()
    horizontal[LEFT_SHOULDER] = [-0.4, 0.2, 2.0]
    horizontal[RIGHT_SHOULDER] = [-0.2, 0.2, 2.0]
    horizontal[LEFT_HIP] = [0.2, 0.2, 2.0]
    horizontal[RIGHT_HIP] = [0.4, 0.2, 2.0]
    lying = detector.update(2, horizontal, confidence)

    # 肩部缺失：应回退使用髋中心到膝中心。
    lower_body = vertical.copy()
    lower_body[LEFT_SHOULDER] = np.nan
    lower_body[RIGHT_SHOULDER] = np.nan
    lower_body[LEFT_KNEE] = [-0.2, 1.0, 2.0]
    lower_body[RIGHT_KNEE] = [0.2, 1.0, 2.0]
    fallback = detector.update(3, lower_body, confidence)

    # 全部关键点无效：必须返回valid=False和0分。
    invalid_points = np.full((17, 3), np.nan, dtype=np.float64)
    invalid = detector.update(4, invalid_points, confidence)

    assert standing.valid
    assert abs(standing.filtered_angle_deg) < 1e-6
    assert standing.score == 0.0

    assert lying.valid
    assert abs(lying.filtered_angle_deg - 90.0) < 1e-6
    assert lying.score == 1.0
    assert fallback.valid
    assert fallback.mode == "LOWER_BODY_KNEE"
    assert not invalid.valid
    assert invalid.score == 0.0

    print("pose_3D self-test: PASS")
    print(
        "  vertical, horizontal, P3D score, fallback, "
        "invalid-input guard=PASS"
    )


def main() -> None:
    """运行合成测试或打开相机测试三维姿态模块。"""
    parser = argparse.ArgumentParser(
        description="独立3D姿态角与P3D子分"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行无相机合成测试",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        # 相机采集和可视化统一由main.py负责，本文件只提供计算模块。
        from ..app.main import run_live
        run_live(
            config,
            args.config,
            stage="pose_3d",
        )

if __name__ == "__main__":
    main()
