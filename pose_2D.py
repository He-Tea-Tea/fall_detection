# -*- coding: utf-8 -*-
"""二维姿态模块：只计算RGB画面人体角度、人框比例和独立子分P2D。"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12

@dataclass
class Pose2DResult:
    """二维姿态输出；angle为90°竖直、0°水平，score范围为0～1。"""
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
    """姿态维度P输出；只融合P3D和P2D，不重新计算任何角度。"""
    score: float
    pose_3d_score: Optional[float]
    pose_2d_score: Optional[float]
    valid_3d: bool
    valid_2d: bool

def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def _increasing_score(value: float, start: float, full: float) -> float:
    """数值越大风险越高：start以下为0，full以上为1。"""
    if full <= start:
        raise ValueError("递增评分满分阈值必须大于起始阈值")
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))

def _decreasing_score(value: float, full: float, normal: float) -> float:
    """数值越小风险越高：full以下为1，normal以上为0。"""
    if normal <= full:
        raise ValueError("二维正常角度阈值必须大于跌倒角度阈值")
    return float(np.clip((normal - value) / (normal - full), 0.0, 1.0))

class Pose2DDetector:
    """计算独立P2D；不读取Depth、ground.yaml或3D人体角度。"""

    def __init__(self, config: dict):
        self.cfg = config["pose_2d"]
        self.weights = self.cfg["weights"]
        values = [float(self.weights[name]) for name in ("image_angle", "bbox_ratio")]
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("pose_2d.weights必须非负且总和等于1")
        min_conf = float(self.cfg["min_keypoint_conf"])
        if not 0.0 <= min_conf <= 1.0 or float(self.cfg["min_image_vector_length_px"]) <= 0.0:
            raise ValueError("pose_2d关键点置信度或最小轴线长度无效")
        _decreasing_score(0.0, float(self.cfg["angle_fall_deg"]), float(self.cfg["angle_normal_deg"]))
        _increasing_score(0.0, float(self.cfg["bbox_ratio_normal"]), float(self.cfg["bbox_ratio_fall"]))

    def _center(self, keypoints: np.ndarray, confidence: np.ndarray, indices: Sequence[int]) -> Optional[np.ndarray]:
        """返回可信左右点的平均中心；二维角度允许只看到一侧肩或髋。"""
        valid = [keypoints[index] for index in indices if index < len(keypoints) and index < len(confidence) and confidence[index] >= float(self.cfg["min_keypoint_conf"]) and np.all(np.isfinite(keypoints[index]))]
        return None if not valid else np.mean(np.asarray(valid, dtype=np.float32), axis=0)

    def image_axis_angle(self, keypoints_2d: np.ndarray, keypoint_conf: np.ndarray) -> tuple:
        """计算肩中心到髋中心连线与画面x轴夹角：0°水平，90°竖直。"""
        points = np.asarray(keypoints_2d, dtype=np.float32)
        confidence = np.asarray(keypoint_conf, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or confidence.shape != (len(points),):
            raise ValueError("二维关键点必须为Nx2，且置信度数量一致")
        shoulder = self._center(points, confidence, (LEFT_SHOULDER, RIGHT_SHOULDER))
        hip = self._center(points, confidence, (LEFT_HIP, RIGHT_HIP))
        if shoulder is None or hip is None:
            return None, shoulder, hip, "肩部或髋部2D关键点不足"
        vector = hip - shoulder
        if float(np.linalg.norm(vector)) < float(self.cfg["min_image_vector_length_px"]):
            return None, shoulder, hip, "2D人体轴线过短"
        angle = float(np.degrees(np.arctan2(abs(float(vector[1])), abs(float(vector[0])))))
        return angle, shoulder, hip, "OK"

    def analyze(self, keypoints_2d: np.ndarray, keypoint_conf: np.ndarray, bbox: Sequence[float]) -> Pose2DResult:
        """只根据2D角度和人体框宽高比计算P2D。"""
        if len(bbox) != 4:
            raise ValueError("bbox必须是[x1,y1,x2,y2]")
        x1, y1, x2, y2 = map(float, bbox)
        ratio = max(1.0, x2 - x1) / max(1.0, y2 - y1)
        image_angle, shoulder, hip, reason = self.image_axis_angle(keypoints_2d, keypoint_conf)
        angle_score = 0.0 if image_angle is None else _decreasing_score(image_angle, float(self.cfg["angle_fall_deg"]), float(self.cfg["angle_normal_deg"]))
        bbox_score = _increasing_score(ratio, float(self.cfg["bbox_ratio_normal"]), float(self.cfg["bbox_ratio_fall"]))
        weighted = [(angle_score, float(self.weights["image_angle"]), image_angle is not None), (bbox_score, float(self.weights["bbox_ratio"]), True)]
        active_weight = sum(weight for _, weight, valid in weighted if valid)
        score = sum(value * weight for value, weight, valid in weighted if valid) / max(active_weight, 1e-8)
        return Pose2DResult(float(np.clip(score, 0.0, 1.0)), image_angle, ratio, angle_score, bbox_score, shoulder, hip, image_angle is not None, "OK" if image_angle is not None else reason)

def fuse_pose_dimension(pose_3d_score: Optional[float], pose_2d_score: Optional[float], config: dict) -> PoseDimensionResult:
    """把两个已经独立算好的子分融合为P；缺失项不参与并重新归一化权重。"""
    weights = config["pose_fusion"]["weights"]
    values = [float(weights[name]) for name in ("pose_3d", "pose_2d")]
    if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("pose_fusion.weights必须非负且总和等于1")
    valid_3d = pose_3d_score is not None and np.isfinite(pose_3d_score)
    valid_2d = pose_2d_score is not None and np.isfinite(pose_2d_score)
    evidence = [(0.0 if not valid_3d else float(np.clip(pose_3d_score, 0.0, 1.0)), float(weights["pose_3d"]), valid_3d), (0.0 if not valid_2d else float(np.clip(pose_2d_score, 0.0, 1.0)), float(weights["pose_2d"]), valid_2d)]
    active_weight = sum(weight for _, weight, valid in evidence if valid)
    score = 0.0 if active_weight <= 0.0 else sum(value * weight for value, weight, valid in evidence if valid) / active_weight
    return PoseDimensionResult(float(np.clip(score, 0.0, 1.0)), None if not valid_3d else float(pose_3d_score), None if not valid_2d else float(pose_2d_score), valid_3d, valid_2d)

def run_self_test(config: dict) -> None:
    """验证二维角度、人框比例、无效关键点保护和P融合。"""
    detector = Pose2DDetector(config)
    confidence = np.ones(17, dtype=np.float32)
    standing = np.zeros((17, 2), dtype=np.float32)
    standing[5], standing[6], standing[11], standing[12] = [40, 20], [60, 20], [40, 80], [60, 80]
    normal = detector.analyze(standing, confidence, [30, 10, 70, 190])
    lying = standing.copy()
    lying[5], lying[6], lying[11], lying[12] = [20, 45], [20, 55], [80, 45], [80, 55]
    abnormal = detector.analyze(lying, confidence, [10, 30, 190, 90])
    invalid = detector.analyze(standing, np.zeros(17, dtype=np.float32), [30, 10, 70, 190])
    fused_normal = fuse_pose_dimension(0.0, normal.score, config)
    fused_fall = fuse_pose_dimension(1.0, abnormal.score, config)
    fused_without_3d = fuse_pose_dimension(None, abnormal.score, config)
    assert normal.valid_angle and abs(normal.image_angle_deg - 90.0) < 1e-4 and normal.score < 0.1
    assert abnormal.valid_angle and abnormal.image_angle_deg < 1e-4 and abnormal.score > 0.9
    assert not invalid.valid_angle and fused_normal.score < 0.1 and fused_fall.score > 0.9
    assert abs(fused_without_3d.score - abnormal.score) < 1e-6
    print("pose_2D self-test: PASS")
    print("  2D angle, bbox ratio, invalid-input guard, P3D/P2D fusion=PASS")

def main() -> None:
    """--self-test使用合成数据；不带参数时复用main.py打开2D姿态窗口。"""
    parser = argparse.ArgumentParser(description="独立2D姿态角、人框比例与P2D子分")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"), help="统一配置文件路径")
    parser.add_argument("--self-test", action="store_true", help="运行无相机合成测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        from main import run_live
        run_live(config, args.config, stage="pose_2d")

if __name__ == "__main__":
    main()
