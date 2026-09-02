# -*- coding: utf-8 -*-
"""人体3D方向角检测模块。

职责边界
--------
本文件只负责根据一人的3D关键点和地面法向量计算人体方向角：

* 0°：人体方向与地面法向量平行，接近竖直；
* 90°：人体方向与地面平面平行，接近水平；
* 优先使用肩中心 -> 髋中心；
* 肩部遮挡时可回退为髋中心 -> 膝中心/踝中心；
* 对同一个 Track ID、同一种测量模式做多帧中值滤波。

本模块算法不获取相机、不运行YOLO、不识别场景，也不判断跌倒。main.py负责
把关键点传入本模块，fall_detector.py只消费这里输出的角度。直接执行本文件
时，会调用main.py提供的公共相机测试流程显示角度结果。
"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

# COCO Pose 17点固定编号，不属于可调参数。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

@dataclass
class BodyAngleResult:
    """一次人体角度测量结果。

    filtered_angle_deg 是主流程应使用的值；raw_angle_deg 用于调试。
    mode明确角度来自完整躯干还是下半身回退，避免把不同物理含义混在一起。
    """

    person_id: int
    valid: bool
    mode: str
    raw_angle_deg: Optional[float]
    filtered_angle_deg: Optional[float]
    shoulder_center_3d: Optional[np.ndarray]
    hip_center_3d: Optional[np.ndarray]
    lower_center_3d: Optional[np.ndarray]
    body_vector_3d: Optional[np.ndarray]
    measurement_quality: float
    reason: str

def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def load_ground_plane(path: str) -> np.ndarray:
    """读取并归一化地面平面[A,B,C,D]。

    ground.yaml必须来自D2C后的RGB坐标系；归一化后点面距离可以直接使用米。
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    plane_data = data.get("plane")
    if not isinstance(plane_data, dict):
        raise ValueError("ground.yaml中不存在有效plane字段")
    plane = np.array([plane_data["A"], plane_data["B"], plane_data["C"], plane_data["D"]], dtype=np.float64,)
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm < 1e-8:
        raise ValueError("ground.yaml地面法向量无效")
    return plane / normal_norm

class BodyAngleDetector:
    """按Track ID计算并滤波人体角度。"""

    def __init__(self, config: dict, ground_plane: Sequence[float]):
        """读取角度与关键点阈值，归一化地面法向量并创建按人员、模式隔离的历史。"""
        angle_cfg = config["body_angle"]
        pose_cfg = config["pose"]
        self.min_keypoint_conf = float(pose_cfg["min_keypoint_conf"])
        self.history_length = int(angle_cfg["history_length"])
        self.min_vector_length_m = float(angle_cfg["min_vector_length_m"])
        self.allow_lower_body = bool(angle_cfg["allow_lower_body_fallback"])

        plane = np.asarray(ground_plane, dtype=np.float64)
        if plane.shape != (4,):
            raise ValueError("ground_plane必须是[A,B,C,D]")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm < 1e-8:
            raise ValueError("地面法向量无效")
        self.ground_normal = plane[:3] / normal_norm

        # 不同模式的角度物理含义略有不同，因此分别维护历史，切换模式时不会混值。
        self.histories: Dict[Tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=self.history_length))

    @staticmethod
    def _finite_point(points: np.ndarray, index: int) -> bool:
        """检查指定关键点是否存在且为有限3D坐标。"""
        return index < len(points) and bool(np.all(np.isfinite(points[index])))

    def _point_valid(self, points: np.ndarray, confidence: np.ndarray, index: int) -> bool:
        """3D坐标和Pose置信度必须同时有效。"""
        return self._finite_point(points, index) and (index < len(confidence) and confidence[index] >= self.min_keypoint_conf)

    def _pair_center(self, points: np.ndarray, confidence: np.ndarray, left_index: int, right_index: int,) -> Optional[np.ndarray]:
        """左右点都可靠时才计算中心，避免单侧关键点造成系统偏移。"""
        if not self._point_valid(points, confidence, left_index):
            return None
        if not self._point_valid(points, confidence, right_index):
            return None
        return (points[left_index] + points[right_index]) / 2.0

    def _select_vector(
        self, points: np.ndarray, confidence: np.ndarray
    ) -> Tuple[
        str,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Tuple[int, ...],
    ]:
        """按优先级选择人体方向向量。

        返回：模式、肩中心、髋中心、下肢中心、方向向量、参与测量的关键点编号。
        """
        shoulder = self._pair_center(points, confidence, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip = self._pair_center(points, confidence, LEFT_HIP, RIGHT_HIP)

        if shoulder is not None and hip is not None:
            return ("FULL_BODY", shoulder, hip, None, hip - shoulder, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),)

        if not self.allow_lower_body or hip is None:
            return "INVALID", shoulder, hip, None, None, ()

        knee = self._pair_center(points, confidence, LEFT_KNEE, RIGHT_KNEE)
        if knee is not None:
            return ("LOWER_BODY_KNEE", shoulder, hip, knee, knee - hip, (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE),)

        ankle = self._pair_center(points, confidence, LEFT_ANKLE, RIGHT_ANKLE)
        if ankle is not None:
            return ("LOWER_BODY_ANKLE", shoulder, hip, ankle, ankle - hip, (LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE),)

        return "INVALID", shoulder, hip, None, None, ()

    def update(self, person_id: int, keypoints_3d: np.ndarray, keypoint_conf: Optional[np.ndarray] = None,) -> BodyAngleResult:
        """计算一人的当前角度并返回滤波结果。

        keypoints_3d形状必须是Nx3，单位米；无效关键点用NaN表示。
        keypoint_conf为空时视为所有关键点置信度为1，便于合成测试。
        """
        points = np.asarray(keypoints_3d, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("keypoints_3d必须是Nx3数组")
        confidence = (np.ones(len(points), dtype=np.float64) if keypoint_conf is None else np.asarray(keypoint_conf, dtype=np.float64))
        if confidence.shape != (len(points),):
            raise ValueError("keypoint_conf数量必须与keypoints_3d一致")

        mode, shoulder, hip, lower, vector, used_indices = self._select_vector(points, confidence)
        if vector is None:
            return BodyAngleResult(
                person_id=int(person_id),
                valid=False,
                mode="INVALID",
                raw_angle_deg=None,
                filtered_angle_deg=None,
                shoulder_center_3d=shoulder,
                hip_center_3d=hip,
                lower_center_3d=lower,
                body_vector_3d=None,
                measurement_quality=0.0,
                reason="关键点不足，无法形成可靠人体方向向量",
            )

        vector_length = float(np.linalg.norm(vector))
        if vector_length < self.min_vector_length_m:
            return BodyAngleResult(
                person_id=int(person_id),
                valid=False,
                mode=mode,
                raw_angle_deg=None,
                filtered_angle_deg=None,
                shoulder_center_3d=shoulder,
                hip_center_3d=hip,
                lower_center_3d=lower,
                body_vector_3d=vector,
                measurement_quality=0.0,
                reason="人体方向向量过短，角度数值不稳定",
            )

        cosine = abs(float(np.dot(vector, self.ground_normal))) / vector_length
        raw_angle = float(np.degrees(np.arccos(float(np.clip(cosine, -1.0, 1.0)))))
        history = self.histories[(int(person_id), mode)]
        history.append(raw_angle)
        filtered_angle = float(np.median(np.asarray(history, dtype=np.float64)))

        confidence_quality = float(np.mean(confidence[list(used_indices)]))
        length_quality = float(np.clip(vector_length / max(self.min_vector_length_m * 2.0, 1e-8), 0.0, 1.0))
        quality = float(np.clip(confidence_quality * length_quality, 0.0, 1.0))

        return BodyAngleResult(
            person_id=int(person_id),
            valid=True,
            mode=mode,
            raw_angle_deg=raw_angle,
            filtered_angle_deg=filtered_angle,
            shoulder_center_3d=shoulder,
            hip_center_3d=hip,
            lower_center_3d=lower,
            body_vector_3d=vector,
            measurement_quality=quality,
            reason="OK",
        )

    def reset_person(self, person_id: int) -> None:
        """清除离开画面的人员历史，防止Track ID复用后继承旧角度。"""
        keys = [key for key in self.histories if key[0] == int(person_id)]
        for key in keys:
            del self.histories[key]

def run_self_test(config: dict) -> None:
    """用可精确计算的合成3D点验证竖直、水平和下半身回退。"""
    plane = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float64)
    detector = BodyAngleDetector(config, plane)
    confidence = np.ones(17, dtype=np.float64)

    vertical = np.full((17, 3), np.nan, dtype=np.float64)
    vertical[LEFT_SHOULDER] = [-0.2, 0.0, 2.0]
    vertical[RIGHT_SHOULDER] = [0.2, 0.0, 2.0]
    vertical[LEFT_HIP] = [-0.2, 0.5, 2.0]
    vertical[RIGHT_HIP] = [0.2, 0.5, 2.0]
    result = detector.update(1, vertical, confidence)
    assert result.valid and abs(result.filtered_angle_deg - 0.0) < 1e-6

    horizontal = vertical.copy()
    horizontal[LEFT_SHOULDER] = [-0.4, 0.2, 2.0]
    horizontal[RIGHT_SHOULDER] = [-0.2, 0.2, 2.0]
    horizontal[LEFT_HIP] = [0.2, 0.2, 2.0]
    horizontal[RIGHT_HIP] = [0.4, 0.2, 2.0]
    result = detector.update(2, horizontal, confidence)
    assert result.valid and abs(result.filtered_angle_deg - 90.0) < 1e-6

    lower = vertical.copy()
    lower[LEFT_SHOULDER] = np.nan
    lower[RIGHT_SHOULDER] = np.nan
    lower[LEFT_KNEE] = [-0.2, 1.0, 2.0]
    lower[RIGHT_KNEE] = [0.2, 1.0, 2.0]
    result = detector.update(3, lower, confidence)
    assert result.valid and result.mode == "LOWER_BODY_KNEE"
    assert abs(result.filtered_angle_deg - 0.0) < 1e-6

    print("body_angle self-test: PASS")
    print("  vertical=0deg, horizontal=90deg, lower-body fallback=PASS")

def main() -> None:
    """命令行独立测试入口；默认打开相机，--self-test无需硬件。"""
    parser = argparse.ArgumentParser(description="人体3D方向角模块")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"), help="统一配置文件路径",)
    parser.add_argument("--self-test", action="store_true", help="运行无需相机的合成数据测试",)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        # 相机采集仍由main.py统一实现，避免本文件复制SDK和2D->3D代码。
        from main import run_live

        run_live(config, args.config, stage="body_angle")

if __name__ == "__main__":
    main()
