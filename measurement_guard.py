# -*- coding: utf-8 -*-
"""人体3D几何与2D/3D姿态一致性保护。

本模块不计算跌倒分数，只判断当前3D证据能否信任。典型异常包括左右髋深度
相差过大、躯干3D长度不合理，以及2D明显站立但3D却显示水平。冲突持续达到
配置帧数后禁用当前P3D/H/V/S，主流程仍保留P2D和AI安全复核能力。
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12


@dataclass
class MeasurementGuardResult:
    """一帧保护结果；depth_valid=False时所有依赖3D的维度必须降级。"""

    depth_valid: bool
    pose_conflict: bool
    conflict_frames: int
    angle_disagreement_deg: Optional[float]
    torso_length_m: Optional[float]
    hip_depth_difference_m: Optional[float]
    reason: str


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class MeasurementGuard:
    """按Track ID累计姿态冲突，同时执行单帧人体3D几何检查。"""

    def __init__(self, config: dict):
        self.cfg = config["measurement_guard"]
        self.conflict_counts: Dict[int, int] = defaultdict(int)

    @staticmethod
    def _valid_point(points: np.ndarray, index: int) -> bool:
        return index < len(points) and bool(np.all(np.isfinite(points[index])))

    def update(
        self,
        person_id: int,
        keypoints_3d: np.ndarray,
        angle_2d_deg: Optional[float],
        angle_3d_deg: Optional[float],
        base_depth_reliable: bool,
        ground_usable: bool,
    ) -> MeasurementGuardResult:
        """检查一人的Depth几何，并比较统一为“0°竖直/90°水平”的姿态角。"""
        person_id = int(person_id)
        points = np.asarray(keypoints_3d, dtype=np.float64)
        reason = "OK"
        depth_valid = bool(base_depth_reliable and ground_usable)
        hip_difference = None
        torso_length = None

        if depth_valid and self._valid_point(points, LEFT_HIP) and self._valid_point(points, RIGHT_HIP):
            hip_difference = abs(float(points[LEFT_HIP, 2] - points[RIGHT_HIP, 2]))
            if hip_difference > float(self.cfg["max_left_right_depth_difference_m"]):
                depth_valid, reason = False, "左右髋深度差过大"

        shoulders_valid = self._valid_point(points, LEFT_SHOULDER) and self._valid_point(points, RIGHT_SHOULDER)
        hips_valid = self._valid_point(points, LEFT_HIP) and self._valid_point(points, RIGHT_HIP)
        if depth_valid and shoulders_valid and hips_valid:
            shoulder_center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
            hip_center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
            torso_length = float(np.linalg.norm(hip_center - shoulder_center))
            if not float(self.cfg["min_torso_length_m"]) <= torso_length <= float(self.cfg["max_torso_length_m"]):
                depth_valid, reason = False, "3D躯干长度不合理"

        disagreement = None
        raw_conflict = False
        if angle_2d_deg is not None and angle_3d_deg is not None:
            # 2D定义为90°竖直/0°水平，因此用90-angle2D转换后再与3D比较。
            comparable_2d = 90.0 - float(angle_2d_deg)
            disagreement = abs(float(angle_3d_deg) - comparable_2d)
            raw_conflict = disagreement >= float(self.cfg["pose_conflict_angle_deg"])

        if raw_conflict:
            self.conflict_counts[person_id] += 1
        else:
            self.conflict_counts[person_id] = 0
        conflict_frames = self.conflict_counts[person_id]
        pose_conflict = conflict_frames >= int(self.cfg["pose_conflict_confirm_frames"])
        if pose_conflict:
            depth_valid = False
            reason = "2D与3D姿态持续冲突"

        return MeasurementGuardResult(
            depth_valid,
            pose_conflict,
            conflict_frames,
            disagreement,
            torso_length,
            hip_difference,
            reason,
        )

    def reset_person(self, person_id: int) -> None:
        self.conflict_counts.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """验证正常站立、姿态冲突和左右髋错误Depth。"""
    guard = MeasurementGuard(config)
    points = np.full((17, 3), np.nan, dtype=np.float64)
    points[5], points[6] = [-0.2, 0.0, 2.0], [0.2, 0.0, 2.0]
    points[11], points[12] = [-0.2, 0.5, 2.0], [0.2, 0.5, 2.0]
    normal = guard.update(1, points, 90.0, 0.0, True, True)
    assert normal.depth_valid and not normal.pose_conflict
    result = None
    for _ in range(int(config["measurement_guard"]["pose_conflict_confirm_frames"])):
        result = guard.update(2, points, 85.0, 82.0, True, True)
    assert result is not None and result.pose_conflict and not result.depth_valid
    points[12, 2] = 3.0
    bad_depth = guard.update(3, points, 90.0, 0.0, True, True)
    assert not bad_depth.depth_valid
    print("measurement_guard self-test: PASS")
    print("  geometry validation and 2D/3D conflict protection=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="人体3D测量一致性保护")
    parser.add_argument("--config", default=str(BASE_DIR / "config.yaml"), help="统一配置文件路径")
    parser.add_argument("--self-test", action="store_true", help="运行无相机合成测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        from main import run_live
        run_live(config, args.config, stage="full")


if __name__ == "__main__":
    main()
