# -*- coding: utf-8 -*-
"""二维人体姿态检测模块。

本文件只负责计算二维人体姿态，不负责相机采集、深度计算、场景判断或跌倒状态机。

输入：
    1. keypoints_2d：人体2D关键点，单位为像素。
    2. keypoint_conf：每个关键点的置信度。
    3. bbox：人体检测框[x1, y1, x2, y2]。

输出：
    1. 肩中心到髋中心连线与画面x轴的夹角。
    2. 人体框宽高比。
    3. 二维姿态风险子分P2D。
    4. P3D和P2D融合后的姿态维度分P。

二维角度定义：
    90°：人体轴线接近画面竖直方向，通常表示站立。
    0°：人体轴线接近画面水平方向，通常表示躺卧。

注意：
    二维角度只表示人体在画面中的方向，不代表人体相对于真实地面的方向。
    摄像头倾斜时，必须结合pose_3D.py的三维角度一起判断。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent

# COCO 17个人体关键点编号。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class Pose2DResult:
    """一次二维姿态检测结果。

    score：
        二维姿态风险子分P2D，范围为0～1。
        越接近0越像正常站立，越接近1越像水平躺卧。

    image_angle_deg：
        人体轴线与画面x轴的夹角。
        90°表示接近竖直，0°表示接近水平。

    bbox_width_height_ratio：
        人体框宽度除以高度。
        站立时通常较小，躺卧时通常较大。

    valid_angle：
        True表示成功计算二维人体角度。
        False表示肩部、髋部关键点不足或人体轴线太短。
    """

    score: float
    image_angle_deg: Optional[float]
    bbox_width_height_ratio: float
    angle_score: float
    bbox_ratio_score: float
    shoulder_center_2d: Optional[np.ndarray]
    hip_center_2d: Optional[np.ndarray]
    valid_angle: bool
    reason: str


@dataclass
class PoseDimensionResult:
    """姿态维度P的融合结果。

    本结果只融合pose_3D.py输出的P3D和本文件输出的P2D，
    不会在这里重新计算二维角度或三维角度。
    """

    score: float
    pose_3d_score: Optional[float]
    pose_2d_score: Optional[float]
    valid_3d: bool
    valid_2d: bool


def load_config(path: str) -> dict:
    """读取统一配置文件config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(
    value: float,
    start: float,
    full: float,
) -> float:
    """计算数值越大、风险越高的线性分数。

    value <= start：风险分为0。
    value >= full：风险分为1。
    两个阈值之间：风险分从0线性增加到1。
    """
    if full <= start:
        raise ValueError("递增评分满分阈值必须大于起始阈值")

    score = (value - start) / (full - start)
    return float(np.clip(score, 0.0, 1.0))


def _decreasing_score(
    value: float,
    full: float,
    normal: float,
) -> float:
    """计算数值越小、风险越高的线性分数。

    value <= full：风险分为1。
    value >= normal：风险分为0。
    两个阈值之间：风险分从1线性降低到0。
    """
    if normal <= full:
        raise ValueError("二维正常角度阈值必须大于跌倒角度阈值")

    score = (normal - value) / (normal - full)
    return float(np.clip(score, 0.0, 1.0))


class Pose2DDetector:
    """根据二维人体角度和人体框宽高比计算P2D。

    本类只使用RGB画面中的2D关键点和人体框，不读取Depth、
    ground.yaml、三维角度、高度分、速度分或最终跌倒状态。
    """

    def __init__(self, config: dict):
        self.cfg = config["pose_2d"]
        self.weights = self.cfg["weights"]

        # 保存常用配置，避免后续重复读取字典。
        self.min_keypoint_conf = float(
            self.cfg["min_keypoint_conf"]
        )
        self.min_vector_length_px = float(
            self.cfg["min_image_vector_length_px"]
        )
        self.angle_fall_deg = float(
            self.cfg["angle_fall_deg"]
        )
        self.angle_normal_deg = float(
            self.cfg["angle_normal_deg"]
        )
        self.bbox_ratio_normal = float(
            self.cfg["bbox_ratio_normal"]
        )
        self.bbox_ratio_fall = float(
            self.cfg["bbox_ratio_fall"]
        )

        # 二维角度分和人体框比例分的权重必须非负且总和等于1。
        weight_values = [
            float(self.weights["image_angle"]),
            float(self.weights["bbox_ratio"]),
        ]
        if any(value < 0.0 for value in weight_values):
            raise ValueError("pose_2d.weights不能为负数")
        if abs(sum(weight_values) - 1.0) > 1e-6:
            raise ValueError("pose_2d.weights总和必须等于1")

        # 检查关键点和人体轴线配置。
        if not 0.0 <= self.min_keypoint_conf <= 1.0:
            raise ValueError(
                "pose_2d.min_keypoint_conf必须在0到1之间"
            )
        if self.min_vector_length_px <= 0.0:
            raise ValueError(
                "pose_2d.min_image_vector_length_px必须大于0"
            )

        # 提前检查评分阈值，避免运行过程中才发现配置错误。
        if self.angle_normal_deg <= self.angle_fall_deg:
            raise ValueError(
                "pose_2d.angle_normal_deg必须大于angle_fall_deg"
            )
        if self.bbox_ratio_fall <= self.bbox_ratio_normal:
            raise ValueError(
                "pose_2d.bbox_ratio_fall必须大于bbox_ratio_normal"
            )

    def _center(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        indices: Sequence[int],
    ) -> Optional[np.ndarray]:
        """计算一组可信二维关键点的平均中心。

        二维角度允许只检测到一侧肩部或一侧髋部。例如左肩被遮挡时，
        如果右肩可信，仍然可以使用右肩位置作为肩部中心。
        """
        valid_points = []

        for index in indices:
            if index >= len(keypoints) or index >= len(confidence):
                continue
            if confidence[index] < self.min_keypoint_conf:
                continue
            if not np.all(np.isfinite(keypoints[index])):
                continue

            valid_points.append(keypoints[index])

        if not valid_points:
            return None

        return np.mean(
            np.asarray(valid_points, dtype=np.float32),
            axis=0,
        )

    def image_axis_angle(
        self,
        keypoints_2d: np.ndarray,
        keypoint_conf: np.ndarray,
    ) -> tuple:
        """计算人体二维轴线与画面x轴的夹角。

        人体二维轴线定义为：
            body_vector = hip_center - shoulder_center

        返回：
            角度、肩中心、髋中心、结果说明。

        角度范围为0°～90°：
            0°表示接近水平。
            90°表示接近竖直。
        """
        points = np.asarray(
            keypoints_2d,
            dtype=np.float32,
        )
        confidence = np.asarray(
            keypoint_conf,
            dtype=np.float32,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("keypoints_2d必须是Nx2数组")
        if confidence.shape != (len(points),):
            raise ValueError(
                "keypoint_conf数量必须与keypoints_2d一致"
            )

        shoulder_center = self._center(
            points,
            confidence,
            (LEFT_SHOULDER, RIGHT_SHOULDER),
        )
        hip_center = self._center(
            points,
            confidence,
            (LEFT_HIP, RIGHT_HIP),
        )

        if shoulder_center is None or hip_center is None:
            return (
                None,
                shoulder_center,
                hip_center,
                "肩部或髋部2D关键点不足",
            )

        body_vector = hip_center - shoulder_center
        vector_length = float(
            np.linalg.norm(body_vector)
        )

        if vector_length < self.min_vector_length_px:
            return (
                None,
                shoulder_center,
                hip_center,
                "2D人体轴线过短",
            )

        # abs(dx)和abs(dy)将角度限制在0°～90°。
        # 人体轴线朝左或朝右，不影响水平与竖直姿态判断。
        dx = abs(float(body_vector[0]))
        dy = abs(float(body_vector[1]))
        image_angle = float(
            np.degrees(np.arctan2(dy, dx))
        )

        return (
            image_angle,
            shoulder_center,
            hip_center,
            "OK",
        )

    def analyze(
        self,
        keypoints_2d: np.ndarray,
        keypoint_conf: np.ndarray,
        bbox: Sequence[float],
    ) -> Pose2DResult:
        """根据二维人体角度和人体框宽高比计算P2D。

        P2D由两项证据组成：
            1. 二维角度分：人体越接近水平，风险越高。
            2. 人框比例分：人体框越宽、越矮，风险越高。

        二维角度无效时不会直接记0分，而是从本帧权重中移除，
        然后只根据有效的人体框比例重新计算P2D。
        """
        if len(bbox) != 4:
            raise ValueError(
                "bbox必须是[x1, y1, x2, y2]"
            )

        x1, y1, x2, y2 = map(float, bbox)

        # 使用至少1像素的宽和高，防止检测框异常时发生除零。
        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)
        bbox_ratio = bbox_width / bbox_height

        (
            image_angle,
            shoulder_center,
            hip_center,
            angle_reason,
        ) = self.image_axis_angle(
            keypoints_2d,
            keypoint_conf,
        )

        # 二维角度越小，说明人体越接近画面水平方向。
        if image_angle is None:
            angle_score = 0.0
        else:
            angle_score = _decreasing_score(
                image_angle,
                self.angle_fall_deg,
                self.angle_normal_deg,
            )

        # 人体框宽高比越大，说明人体框越宽、越矮。
        bbox_ratio_score = _increasing_score(
            bbox_ratio,
            self.bbox_ratio_normal,
            self.bbox_ratio_fall,
        )

        # 每项内容依次为：分数、配置权重、当前证据是否有效。
        evidence = [
            (
                angle_score,
                float(self.weights["image_angle"]),
                image_angle is not None,
            ),
            (
                bbox_ratio_score,
                float(self.weights["bbox_ratio"]),
                True,
            ),
        ]

        # 只统计有效证据的权重。
        active_weight = sum(
            weight
            for _, weight, valid in evidence
            if valid
        )

        weighted_score = sum(
            value * weight
            for value, weight, valid in evidence
            if valid
        )
        score = weighted_score / max(
            active_weight,
            1e-8,
        )

        valid_angle = image_angle is not None
        reason = "OK" if valid_angle else angle_reason

        return Pose2DResult(
            score=float(np.clip(score, 0.0, 1.0)),
            image_angle_deg=image_angle,
            bbox_width_height_ratio=bbox_ratio,
            angle_score=angle_score,
            bbox_ratio_score=bbox_ratio_score,
            shoulder_center_2d=shoulder_center,
            hip_center_2d=hip_center,
            valid_angle=valid_angle,
            reason=reason,
        )


def fuse_pose_dimension(
    pose_3d_score: Optional[float],
    pose_2d_score: Optional[float],
    config: dict,
) -> PoseDimensionResult:
    """将P3D和P2D融合为最终姿态维度分P。

    本函数不会重新计算二维或三维角度，只融合两个模块已经算好的分数。

    如果其中一项无效：
        1. 无效项不会按0分处理。
        2. 无效项的权重会从本帧中移除。
        3. 剩余有效证据会重新归一化。

    例如P3D无效、P2D有效时，最终P直接等于P2D。
    """
    weights = config["pose_fusion"]["weights"]

    weight_values = [
        float(weights["pose_3d"]),
        float(weights["pose_2d"]),
    ]
    if any(value < 0.0 for value in weight_values):
        raise ValueError(
            "pose_fusion.weights不能为负数"
        )
    if abs(sum(weight_values) - 1.0) > 1e-6:
        raise ValueError(
            "pose_fusion.weights总和必须等于1"
        )

    valid_3d = (
        pose_3d_score is not None
        and bool(np.isfinite(pose_3d_score))
    )
    valid_2d = (
        pose_2d_score is not None
        and bool(np.isfinite(pose_2d_score))
    )

    clipped_3d_score = (
        float(np.clip(pose_3d_score, 0.0, 1.0))
        if valid_3d
        else 0.0
    )
    clipped_2d_score = (
        float(np.clip(pose_2d_score, 0.0, 1.0))
        if valid_2d
        else 0.0
    )

    # 每项内容依次为：子分、配置权重、当前子分是否有效。
    evidence = [
        (
            clipped_3d_score,
            float(weights["pose_3d"]),
            valid_3d,
        ),
        (
            clipped_2d_score,
            float(weights["pose_2d"]),
            valid_2d,
        ),
    ]

    active_weight = sum(
        weight
        for _, weight, valid in evidence
        if valid
    )

    if active_weight <= 0.0:
        score = 0.0
    else:
        weighted_score = sum(
            value * weight
            for value, weight, valid in evidence
            if valid
        )
        score = weighted_score / active_weight

    return PoseDimensionResult(
        score=float(np.clip(score, 0.0, 1.0)),
        pose_3d_score=(
            float(pose_3d_score)
            if valid_3d
            else None
        ),
        pose_2d_score=(
            float(pose_2d_score)
            if valid_2d
            else None
        ),
        valid_3d=valid_3d,
        valid_2d=valid_2d,
    )


def run_self_test(config: dict) -> None:
    """使用合成关键点验证二维姿态和P融合，不需要相机。"""
    detector = Pose2DDetector(config)
    confidence = np.ones(17, dtype=np.float32)

    # 站立：肩到髋连线沿画面竖直方向，二维角度应为90°。
    standing = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    standing[LEFT_SHOULDER] = [40, 20]
    standing[RIGHT_SHOULDER] = [60, 20]
    standing[LEFT_HIP] = [40, 80]
    standing[RIGHT_HIP] = [60, 80]

    normal = detector.analyze(
        standing,
        confidence,
        bbox=[30, 10, 70, 190],
    )

    # 躺卧：肩到髋连线沿画面水平方向，二维角度应为0°。
    lying = standing.copy()
    lying[LEFT_SHOULDER] = [20, 45]
    lying[RIGHT_SHOULDER] = [20, 55]
    lying[LEFT_HIP] = [80, 45]
    lying[RIGHT_HIP] = [80, 55]

    abnormal = detector.analyze(
        lying,
        confidence,
        bbox=[10, 30, 190, 90],
    )

    # 所有关键点置信度为0，二维角度应无效，但人框比例仍可参与评分。
    invalid = detector.analyze(
        standing,
        np.zeros(17, dtype=np.float32),
        bbox=[30, 10, 70, 190],
    )

    # 验证P3D和P2D融合。
    fused_normal = fuse_pose_dimension(
        0.0,
        normal.score,
        config,
    )
    fused_fall = fuse_pose_dimension(
        1.0,
        abnormal.score,
        config,
    )
    fused_without_3d = fuse_pose_dimension(
        None,
        abnormal.score,
        config,
    )

    assert normal.valid_angle
    assert abs(normal.image_angle_deg - 90.0) < 1e-4
    assert normal.score < 0.1

    assert abnormal.valid_angle
    assert abnormal.image_angle_deg < 1e-4
    assert abnormal.score > 0.9

    assert not invalid.valid_angle
    assert fused_normal.score < 0.1
    assert fused_fall.score > 0.9

    # P3D缺失时，最终P应该等于有效的P2D。
    assert abs(
        fused_without_3d.score - abnormal.score
    ) < 1e-6

    print("pose_2D self-test: PASS")
    print(
        "  2D angle, bbox ratio, invalid-input guard, "
        "P3D/P2D fusion=PASS"
    )


def main() -> None:
    """运行合成测试或打开相机测试二维姿态模块。"""
    parser = argparse.ArgumentParser(
        description="独立2D姿态角、人框比例与P2D子分"
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
        # 相机采集和可视化由main.py统一负责，本文件只提供二维姿态计算。
        from main import run_live

        run_live(
            config,
            args.config,
            stage="pose_2d",
        )


if __name__ == "__main__":
    main()