# -*- coding: utf-8 -*-
"""在线地面质量评估与自动重估。

ground_detector.py仍然是人工标定工具，本模块不会修改它。程序启动时先读取
ground.yaml作为初始平面，随后从每帧D2C深度的下部区域重新寻找大面积平面。
只有候选平面连续多次稳定、内点比例和覆盖范围均达标时，才允许P3D、H、V、S
和地面场景关系使用它。重估期间本地流程自动退化到P2D，避免错误地面制造误报。
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class GroundEstimate:
    """当前地面状态；plane为归一化[A,B,C,D]，距离单位为米。"""

    state: str
    plane: Optional[np.ndarray]
    quality: float
    usable: bool
    warmup_active: bool
    changed: bool
    inlier_ratio: float
    rmse_m: float
    coverage_ratio: float
    camera_height_m: Optional[float]
    reason: str


def load_config(path: str) -> dict:
    """读取统一配置文件。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_plane(plane: Sequence[float]) -> np.ndarray:
    """把平面[A,B,C,D]归一化，并固定D为非负，便于多帧比较。"""
    result = np.asarray(plane, dtype=np.float64)
    if result.shape != (4,):
        raise ValueError("ground plane必须是[A,B,C,D]")
    normal_length = float(np.linalg.norm(result[:3]))
    if normal_length < 1e-8:
        raise ValueError("地面法向量无效")
    result = result / normal_length
    return -result if result[3] < 0.0 else result


def plane_difference(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float]:
    """返回两个平面的法向量夹角和相机到平面距离差。"""
    first_plane = normalize_plane(first)
    second_plane = normalize_plane(second)
    cosine = abs(float(np.dot(first_plane[:3], second_plane[:3])))
    angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    offset_m = abs(abs(float(first_plane[3])) - abs(float(second_plane[3])))
    return angle_deg, offset_m


class GroundManager:
    """用轻量RANSAC持续检查地面，并在相机移动后自动建立新平面。"""

    def __init__(self, config: dict, initial_plane: Optional[Sequence[float]] = None):
        self.cfg = config["ground_manager"]
        self.depth_cfg = config["depth"]
        self.enabled = bool(self.cfg["enabled"])
        self.current_plane = normalize_plane(initial_plane) if initial_plane is not None else None
        self.state = "STARTUP"
        self.quality = 0.0
        self.good_count = 0
        self.bad_count = 0
        self.warmup_until_s = 0.0
        self.last_result: Optional[GroundEstimate] = None
        self.pending_planes = deque(maxlen=int(self.cfg["stable_frames"]))
        self.rng = np.random.default_rng(int(self.cfg.get("random_seed", 2026)))

    def _unchanged_result(self, now_s: float) -> GroundEstimate:
        """非估计帧复用最近结果，但changed只在接受新平面的当帧为True。"""
        if self.last_result is None:
            return GroundEstimate(
                "STARTUP", self.current_plane, 0.0, False, False, False,
                0.0, float("inf"), 0.0, None, "等待在线地面验证",
            )
        warmup = now_s < self.warmup_until_s
        return GroundEstimate(
            self.last_result.state,
            None if self.last_result.plane is None else self.last_result.plane.copy(),
            self.last_result.quality,
            self.last_result.usable,
            warmup,
            False,
            self.last_result.inlier_ratio,
            self.last_result.rmse_m,
            self.last_result.coverage_ratio,
            self.last_result.camera_height_m,
            self.last_result.reason,
        )

    def current_estimate(self, now_s: float) -> GroundEstimate:
        """公开读取当前地面状态，不执行新一轮RANSAC。"""
        return self._unchanged_result(float(now_s))

    def _collect_points(
        self,
        depth_m: np.ndarray,
        intrinsics: dict,
        exclusion_masks: Iterable[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从画面下部稀疏采样3D点，并排除人体框和家具Mask。"""
        height, width = depth_m.shape
        start_y = int(np.clip(float(self.cfg["roi_start_ratio"]), 0.0, 0.95) * height)
        step = max(1, int(self.cfg["sample_step_px"]))
        rows = np.arange(start_y, height, step, dtype=np.int32)
        cols = np.arange(0, width, step, dtype=np.int32)
        grid_v, grid_u = np.meshgrid(rows, cols, indexing="ij")
        valid = np.ones(grid_u.shape, dtype=bool)
        for exclusion in exclusion_masks:
            mask = np.asarray(exclusion, dtype=bool)
            if mask.shape == depth_m.shape:
                valid &= ~mask[grid_v, grid_u]

        z = depth_m[grid_v, grid_u]
        valid &= np.isfinite(z)
        valid &= z >= float(self.depth_cfg["min_depth_m"])
        valid &= z <= float(self.depth_cfg["max_depth_m"])
        u = grid_u[valid].astype(np.float64)
        v = grid_v[valid].astype(np.float64)
        z = z[valid].astype(np.float64)
        if z.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
        y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
        return np.column_stack((x, y, z)), np.column_stack((u, v))

    @staticmethod
    def _plane_from_three(points: np.ndarray) -> Optional[np.ndarray]:
        """由三个3D点生成平面；三点近似共线时返回None。"""
        first, second, third = points
        normal = np.cross(second - first, third - first)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            return None
        normal /= length
        return normalize_plane(np.append(normal, -float(np.dot(normal, first))))

    @staticmethod
    def _refine_plane(points: np.ndarray) -> np.ndarray:
        """用全部RANSAC内点做SVD最小二乘精修。"""
        center = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        normal = vh[-1]
        return normalize_plane(np.append(normal, -float(np.dot(normal, center))))

    def _coverage(self, pixels: np.ndarray, inliers: np.ndarray, width: int, height: int) -> float:
        """计算内点覆盖多少网格，防止把局部桌面或小块物体当成地面。"""
        grid_x = max(1, int(self.cfg["coverage_grid_x"]))
        grid_y = max(1, int(self.cfg["coverage_grid_y"]))
        selected = pixels[inliers]
        if selected.size == 0:
            return 0.0
        x_index = np.clip((selected[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
        y_index = np.clip((selected[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
        occupied = len(set(zip(x_index.tolist(), y_index.tolist())))
        return float(occupied / (grid_x * grid_y))

    def _estimate_plane(
        self,
        depth_m: np.ndarray,
        intrinsics: dict,
        exclusion_masks: Iterable[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float, float, float, Optional[float], str]:
        """返回候选平面、内点比例、RMSE、覆盖率、相机高度和失败原因。"""
        points, pixels = self._collect_points(depth_m, intrinsics, exclusion_masks)
        min_points = int(self.cfg["min_points"])
        if len(points) < min_points:
            return None, 0.0, float("inf"), 0.0, None, "有效地面候选点不足"

        threshold = float(self.cfg["ransac_distance_threshold_m"])
        best_inliers = None
        best_count = 0
        for _ in range(int(self.cfg["ransac_iterations"])):
            indices = self.rng.choice(len(points), size=3, replace=False)
            plane = self._plane_from_three(points[indices])
            if plane is None:
                continue
            distances = np.abs(points @ plane[:3] + plane[3])
            inliers = distances <= threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count, best_inliers = count, inliers

        if best_inliers is None or best_count < min_points:
            return None, 0.0, float("inf"), 0.0, None, "RANSAC没有找到足够大的平面"

        plane = self._refine_plane(points[best_inliers])
        # Gemini头部通常只在有限俯仰范围内转动。地面法向量在相机Y轴上应
        # 保留一定分量，且不应在一次更新中跳到与旧地面近似垂直的墙面。
        if abs(float(plane[1])) < float(self.cfg["min_abs_normal_y"]):
            return None, 0.0, float("inf"), 0.0, None, "候选平面更像墙面而不是地面"
        if self.current_plane is not None:
            change_angle, _ = plane_difference(self.current_plane, plane)
            if change_angle > float(self.cfg["max_recalibration_angle_deg"]):
                return None, 0.0, float("inf"), 0.0, None, "候选平面与原地面夹角过大"
        distances = np.abs(points @ plane[:3] + plane[3])
        inliers = distances <= threshold
        inlier_ratio = float(np.mean(inliers))
        rmse = float(np.sqrt(np.mean(np.square(distances[inliers]))))
        coverage = self._coverage(pixels, inliers, depth_m.shape[1], depth_m.shape[0])
        camera_height = abs(float(plane[3]))
        return plane, inlier_ratio, rmse, coverage, camera_height, "OK"

    def _quality(
        self,
        inlier_ratio: float,
        rmse_m: float,
        coverage_ratio: float,
        camera_height_m: Optional[float],
    ) -> float:
        """把拟合质量、覆盖范围和合理相机高度合成为0～1。"""
        inlier_score = np.clip(inlier_ratio / float(self.cfg["target_inlier_ratio"]), 0.0, 1.0)
        residual_score = np.clip(1.0 - rmse_m / float(self.cfg["max_rmse_m"]), 0.0, 1.0)
        coverage_score = np.clip(coverage_ratio / float(self.cfg["target_coverage_ratio"]), 0.0, 1.0)
        min_height = float(self.cfg["camera_height_min_m"])
        max_height = float(self.cfg["camera_height_max_m"])
        height_score = 1.0 if camera_height_m is not None and min_height <= camera_height_m <= max_height else 0.0
        weights = self.cfg["quality_weights"]
        value = (
            float(weights["inlier_ratio"]) * inlier_score
            + float(weights["residual"]) * residual_score
            + float(weights["coverage"]) * coverage_score
            + float(weights["camera_height"]) * height_score
        )
        return float(np.clip(value, 0.0, 1.0))

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        depth_m: np.ndarray,
        intrinsics: dict,
        exclusion_masks: Iterable[np.ndarray] = (),
    ) -> GroundEstimate:
        """处理一帧Depth；非采样帧直接返回最近状态。"""
        now = float(timestamp_s)
        if not self.enabled:
            usable = self.current_plane is not None
            result = GroundEstimate(
                "FIXED", self.current_plane, 1.0 if usable else 0.0, usable,
                False, False, 1.0 if usable else 0.0, 0.0, 1.0 if usable else 0.0,
                abs(float(self.current_plane[3])) if usable else None,
                "在线地面管理已关闭",
            )
            self.last_result = result
            return result
        if int(frame_index) % max(1, int(self.cfg["update_interval_frames"])) != 0:
            return self._unchanged_result(now)

        plane, inlier_ratio, rmse, coverage, camera_height, reason = self._estimate_plane(
            np.asarray(depth_m, dtype=np.float32), intrinsics, exclusion_masks,
        )
        quality = self._quality(inlier_ratio, rmse, coverage, camera_height) if plane is not None else 0.0
        good = bool(
            plane is not None
            and quality >= float(self.cfg["min_quality"])
            and inlier_ratio >= float(self.cfg["min_inlier_ratio"])
            and coverage >= float(self.cfg["min_coverage_ratio"])
        )

        changed = False
        if good:
            if self.pending_planes:
                angle, offset = plane_difference(self.pending_planes[-1], plane)
                stable = angle <= float(self.cfg["stable_angle_deg"]) and offset <= float(self.cfg["stable_offset_m"])
            else:
                stable = True
            if not stable:
                self.pending_planes.clear()
            self.pending_planes.append(plane)
            self.good_count = len(self.pending_planes)
            self.bad_count = 0

            if self.good_count >= int(self.cfg["stable_frames"]):
                candidate = normalize_plane(np.median(np.asarray(self.pending_planes), axis=0))
                if self.current_plane is None:
                    changed = True
                else:
                    angle, offset = plane_difference(self.current_plane, candidate)
                    changed = angle >= float(self.cfg["plane_change_angle_deg"]) or offset >= float(self.cfg["plane_change_offset_m"])
                self.current_plane = candidate
                self.state = "VALID"
                self.quality = quality
                if changed:
                    self.warmup_until_s = now + float(self.cfg["history_warmup_s"])
                reason = "在线地面有效" if not changed else "已接受新的在线地面，时序特征正在预热"
            else:
                self.state = "RECALIBRATING"
                self.quality = quality
                reason = f"候选地面稳定确认{self.good_count}/{int(self.cfg['stable_frames'])}"
        else:
            self.pending_planes.clear()
            self.good_count = 0
            self.bad_count += 1
            self.quality = quality
            grace = int(self.cfg["bad_frame_grace"])
            self.state = "SUSPECT" if self.current_plane is not None and self.bad_count <= grace else "INVALID"

        warmup = now < self.warmup_until_s
        usable = bool(self.current_plane is not None and self.state == "VALID" and self.quality >= float(self.cfg["min_quality"]))
        result = GroundEstimate(
            self.state,
            None if self.current_plane is None else self.current_plane.copy(),
            float(self.quality),
            usable,
            warmup,
            changed,
            inlier_ratio,
            rmse,
            coverage,
            camera_height,
            reason,
        )
        self.last_result = result
        return result


def run_self_test(config: dict) -> None:
    """生成理想平面Depth，验证自动估计、质量和相机移动后重估。"""
    test_config = yaml.safe_load(yaml.safe_dump(config))
    test_config["ground_manager"]["update_interval_frames"] = 1
    test_config["ground_manager"]["stable_frames"] = 2
    test_config["ground_manager"]["min_points"] = 40
    intrinsics = {"width": 160, "height": 100, "fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 40.0}

    def plane_depth(camera_height: float) -> np.ndarray:
        rows = np.arange(100, dtype=np.float64)[:, None]
        ray_y = (rows - intrinsics["cy"]) / intrinsics["fy"]
        depth = np.full((100, 160), np.nan, dtype=np.float32)
        valid_rows = ray_y[:, 0] > 0.05
        depth[valid_rows, :] = (camera_height / ray_y[valid_rows]).astype(np.float32)
        return depth

    manager = GroundManager(test_config, [0.0, -1.0, 0.0, 1.0])
    result = manager.update(0, 0.0, plane_depth(1.0), intrinsics)
    result = manager.update(1, 0.1, plane_depth(1.0), intrinsics)
    assert result.usable and result.quality >= float(test_config["ground_manager"]["min_quality"])
    result = manager.update(2, 1.0, plane_depth(1.3), intrinsics)
    result = manager.update(3, 1.1, plane_depth(1.3), intrinsics)
    assert result.usable and result.changed and abs(float(result.plane[3]) - 1.3) < 0.05
    print("ground_manager self-test: PASS")
    print("  online plane quality, stable confirmation, automatic recalibration=PASS")


def main() -> None:
    """运行无相机测试；在线流程由main.py统一调用。"""
    parser = argparse.ArgumentParser(description="在线地面质量与自动重估")
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
