# -*- coding: utf-8 -*-
"""三维姿态模块：只计算人体3D轴线与地面法向量的角度和风险子分P3D。"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12
LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 13, 14, 15, 16

@dataclass
class Pose3DResult:
    """三维姿态输出；0°表示竖直，90°表示水平，score范围为0～1。"""
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
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def load_ground_plane(path: str) -> np.ndarray:
    """读取并归一化ground.yaml中的地面平面[A,B,C,D]。"""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    plane_data = data.get("plane")
    if not isinstance(plane_data, dict):
        raise ValueError("ground.yaml中不存在有效plane字段")
    plane = np.array([plane_data["A"], plane_data["B"], plane_data["C"], plane_data["D"]], dtype=np.float64)
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm < 1e-8:
        raise ValueError("ground.yaml地面法向量无效")
    return plane / normal_norm

def _angle_score(angle_deg: float, normal_deg: float, fall_deg: float) -> float:
    """3D角度越大风险越高：normal以下为0，fall以上为1。"""
    if fall_deg <= normal_deg:
        raise ValueError("pose_3d.angle_fall_deg必须大于angle_normal_deg")
    return float(np.clip((angle_deg - normal_deg) / (fall_deg - normal_deg), 0.0, 1.0))

class Pose3DDetector:
    """按Track ID计算并滤波三维人体角度，不读取2D角度或人框比例。"""

    def __init__(self, config: dict, ground_plane: Sequence[float]):
        self.cfg = config["pose_3d"]
        self.min_keypoint_conf = float(self.cfg["min_keypoint_conf"])
        self.history_length = int(self.cfg["history_length"])
        self.min_vector_length_m = float(self.cfg["min_vector_length_m"])
        self.allow_lower_body = bool(self.cfg["allow_lower_body_fallback"])
        if not 0.0 <= self.min_keypoint_conf <= 1.0:
            raise ValueError("pose_3d.min_keypoint_conf必须在0到1之间")
        if self.history_length <= 0 or self.min_vector_length_m <= 0.0:
            raise ValueError("pose_3d历史长度和最小轴线长度必须大于0")
        _angle_score(0.0, float(self.cfg["angle_normal_deg"]), float(self.cfg["angle_fall_deg"]))
        plane = np.asarray(ground_plane, dtype=np.float64)
        if plane.shape != (4,):
            raise ValueError("ground_plane必须是[A,B,C,D]")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm < 1e-8:
            raise ValueError("地面法向量无效")
        self.ground_normal = plane[:3] / normal_norm
        self.histories: Dict[Tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _point_valid(self, points: np.ndarray, confidence: np.ndarray, index: int) -> bool:
        """关键点必须同时具备有限3D坐标和足够置信度。"""
        return index < len(points) and index < len(confidence) and bool(np.all(np.isfinite(points[index]))) and confidence[index] >= self.min_keypoint_conf

    def _pair_center(self, points: np.ndarray, confidence: np.ndarray, left: int, right: int) -> Optional[np.ndarray]:
        """左右关键点都有效时返回中心，避免单侧点改变三维轴线方向。"""
        if not self._point_valid(points, confidence, left) or not self._point_valid(points, confidence, right):
            return None
        return (points[left] + points[right]) / 2.0

    def _select_vector(self, points: np.ndarray, confidence: np.ndarray) -> tuple:
        """优先肩到髋；肩缺失时按配置回退为髋到膝或髋到踝。"""
        shoulder = self._pair_center(points, confidence, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip = self._pair_center(points, confidence, LEFT_HIP, RIGHT_HIP)
        if shoulder is not None and hip is not None:
            return "FULL_BODY", shoulder, hip, None, hip - shoulder, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if not self.allow_lower_body or hip is None:
            return "INVALID", shoulder, hip, None, None, ()
        knee = self._pair_center(points, confidence, LEFT_KNEE, RIGHT_KNEE)
        if knee is not None:
            return "LOWER_BODY_KNEE", shoulder, hip, knee, knee - hip, (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE)
        ankle = self._pair_center(points, confidence, LEFT_ANKLE, RIGHT_ANKLE)
        if ankle is not None:
            return "LOWER_BODY_ANKLE", shoulder, hip, ankle, ankle - hip, (LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)
        return "INVALID", shoulder, hip, None, None, ()

    def update(self, person_id: int, keypoints_3d: np.ndarray, keypoint_conf: Optional[np.ndarray] = None) -> Pose3DResult:
        """输入Nx3米制关键点，输出独立的3D角度、P3D子分和测量质量。"""
        points = np.asarray(keypoints_3d, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("keypoints_3d必须是Nx3数组")
        confidence = np.ones(len(points), dtype=np.float64) if keypoint_conf is None else np.asarray(keypoint_conf, dtype=np.float64)
        if confidence.shape != (len(points),):
            raise ValueError("keypoint_conf数量必须与keypoints_3d一致")
        mode, shoulder, hip, lower, vector, used_indices = self._select_vector(points, confidence)
        if vector is None:
            return Pose3DResult(int(person_id), False, "INVALID", None, None, 0.0, 0.0, shoulder, hip, lower, None, "关键点不足，无法形成可靠3D人体轴线")
        vector_length = float(np.linalg.norm(vector))
        if vector_length < self.min_vector_length_m:
            return Pose3DResult(int(person_id), False, mode, None, None, 0.0, 0.0, shoulder, hip, lower, vector, "3D人体轴线过短")
        cosine = abs(float(np.dot(vector, self.ground_normal))) / vector_length
        raw_angle = float(np.degrees(np.arccos(float(np.clip(cosine, -1.0, 1.0)))))
        history = self.histories[(int(person_id), mode)]
        history.append(raw_angle)
        filtered_angle = float(np.median(np.asarray(history, dtype=np.float64)))
        confidence_quality = float(np.mean(confidence[list(used_indices)]))
        length_quality = float(np.clip(vector_length / max(self.min_vector_length_m * 2.0, 1e-8), 0.0, 1.0))
        quality = float(np.clip(confidence_quality * length_quality, 0.0, 1.0))
        score = _angle_score(filtered_angle, float(self.cfg["angle_normal_deg"]), float(self.cfg["angle_fall_deg"]))
        return Pose3DResult(int(person_id), True, mode, raw_angle, filtered_angle, score, quality, shoulder, hip, lower, vector, "OK")

    def reset_person(self, person_id: int) -> None:
        """清除离场人员历史，避免Track ID复用后继承旧角度。"""
        for key in [key for key in self.histories if key[0] == int(person_id)]:
            del self.histories[key]

def run_self_test(config: dict) -> None:
    """验证竖直、水平、下半身回退及无效关键点保护。"""
    detector = Pose3DDetector(config, [0.0, -1.0, 0.0, 1.0])
    confidence = np.ones(17, dtype=np.float64)
    vertical = np.full((17, 3), np.nan, dtype=np.float64)
    vertical[5], vertical[6], vertical[11], vertical[12] = [-0.2, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.5, 2.0], [0.2, 0.5, 2.0]
    standing = detector.update(1, vertical, confidence)
    horizontal = vertical.copy()
    horizontal[5], horizontal[6], horizontal[11], horizontal[12] = [-0.4, 0.2, 2.0], [-0.2, 0.2, 2.0], [0.2, 0.2, 2.0], [0.4, 0.2, 2.0]
    lying = detector.update(2, horizontal, confidence)
    lower = vertical.copy()
    lower[5] = lower[6] = np.nan
    lower[13], lower[14] = [-0.2, 1.0, 2.0], [0.2, 1.0, 2.0]
    fallback = detector.update(3, lower, confidence)
    invalid = detector.update(4, np.full((17, 3), np.nan), confidence)
    assert standing.valid and abs(standing.filtered_angle_deg) < 1e-6 and standing.score == 0.0
    assert lying.valid and abs(lying.filtered_angle_deg - 90.0) < 1e-6 and lying.score == 1.0
    assert fallback.valid and fallback.mode == "LOWER_BODY_KNEE"
    assert not invalid.valid and invalid.score == 0.0
    print("pose_3D self-test: PASS")
    print("  vertical, horizontal, P3D score, fallback, invalid-input guard=PASS")

def main() -> None:
    """--self-test使用合成数据；不带参数时复用main.py打开3D姿态窗口。"""
    parser = argparse.ArgumentParser(description="独立3D姿态角与P3D子分")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"), help="统一配置文件路径")
    parser.add_argument("--self-test", action="store_true", help="运行无相机合成测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        from main import run_live
        run_live(config, args.config, stage="pose_3d")

if __name__ == "__main__":
    main()
