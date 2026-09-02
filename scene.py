# -*- coding: utf-8 -*-
"""场景关系与场景风险评分模块（scene）。

职责边界
--------
本文件只回答“人和环境是什么关系”，例如：

* lying_on_bed / lying_on_sofa：躺在床或沙发上，通常不是跌倒；
* sitting_on_chair：坐在椅子上；
* standing_near_bed：站在家具附近；
* lying_on_floor：人体水平且髋部接近地面，属于高风险场景；
* unknown：证据不足。

输入来自main.py：人体框、人体3D关键点、pose_3D.py输出的角度、YOLO-Seg
检测到的家具框和Mask、D2C深度及RGB内参。家具3D点只从Mask与承载表面区域
的交集采样，减少背景Depth干扰；最后把关系映射为0～1的scene风险分。
"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

# COCO Pose固定编号。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class SceneObjectDetection:
    """YOLO-Seg家具结果；bbox和mask均已映射到RGB原图坐标。"""

    name: str
    confidence: float
    bbox: np.ndarray
    mask: np.ndarray


@dataclass
class SceneObjectGeometry:
    """家具Mask内有效Depth反投影得到的3D几何。"""

    name: str
    confidence: float
    bbox: np.ndarray
    points_3d: np.ndarray
    center_3d: np.ndarray
    surface_height_m: float
    mask_area_px: int
    sampled_point_count: int


@dataclass
class SceneRelationResult:
    """一人与场景的最终关系。"""

    person_id: int
    relation: str
    scene_name: str
    confidence: float
    scene_score: float
    object_name: Optional[str]
    iou: float
    nearest_distance_m: Optional[float]
    hip_height_m: Optional[float]
    object_surface_height_m: Optional[float]
    reason: str


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_ground_plane(path: str) -> np.ndarray:
    """读取并归一化ground.yaml中的[A,B,C,D]。"""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    item = data.get("plane")
    if not isinstance(item, dict):
        raise ValueError("ground.yaml中不存在有效plane字段")
    plane = np.array([item["A"], item["B"], item["C"], item["D"]], dtype=np.float64)
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm < 1e-8:
        raise ValueError("地面法向量无效")
    return plane / normal_norm


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """计算两个[x1,y1,x2,y2]框的二维交并比。"""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 1e-8 else float(intersection / union)


class SceneScorer:
    """将人体姿态、家具3D几何和地面高度融合为稳定场景关系。"""

    def __init__(self, config: dict, ground_plane: Sequence[float]):
        """读取关系阈值和Depth范围，归一化地面平面并建立每个人的关系历史。"""
        scene_cfg = config["scene"]
        depth_cfg = config["depth"]
        self.class_map = dict(scene_cfg["target_class_map"])
        self.sample_step = int(scene_cfg["object_sample_step"])
        self.min_object_points = int(scene_cfg["min_object_points"])
        self.mask_threshold = float(scene_cfg["mask_threshold"])
        self.min_mask_area_px = int(scene_cfg["min_mask_area_px"])
        self.x_margin_ratio = float(scene_cfg["object_x_margin_ratio"])
        self.surface_bands = dict(scene_cfg["surface_bands"])
        self.relation_cfg = dict(scene_cfg["relation"])
        self.confidence_threshold = float(scene_cfg["confidence_threshold"])
        self.history_length = int(scene_cfg["history_length"])
        self.risk_scores = dict(scene_cfg["risk_scores"])
        self.min_depth_m = float(depth_cfg["min_depth_m"])
        self.max_depth_m = float(depth_cfg["max_depth_m"])

        plane = np.asarray(ground_plane, dtype=np.float64)
        if plane.shape != (4,):
            raise ValueError("ground_plane必须是[A,B,C,D]")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm < 1e-8:
            raise ValueError("地面法向量无效")
        self.ground_plane = plane / normal_norm
        self.histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_length))

    def point_ground_height(self, point: Optional[np.ndarray]) -> Optional[float]:
        """计算3D点到地面的绝对距离，单位米。"""
        if point is None or not np.all(np.isfinite(point)):
            return None
        return float(abs(np.dot(self.ground_plane[:3], point) + self.ground_plane[3]))

    @staticmethod
    def _pair_center(points: np.ndarray, left: int, right: int) -> Optional[np.ndarray]:
        """左右关键点都有效时返回中心。"""
        if left >= len(points) or right >= len(points):
            return None
        if not np.all(np.isfinite(points[left])) or not np.all(np.isfinite(points[right])):
            return None
        return (points[left] + points[right]) / 2.0

    def _object_geometry(
        self,
        detection: SceneObjectDetection,
        depth_m: np.ndarray,
        intrinsics: dict,
    ) -> Optional[SceneObjectGeometry]:
        """只在家具Mask与承载表面区域的交集中采样Depth并转换为3D点集。"""
        name = detection.name
        if name not in self.surface_bands:
            return None
        height, width = depth_m.shape
        mask = np.asarray(detection.mask)
        if mask.shape != depth_m.shape:
            raise ValueError(f"{name} Mask尺寸{mask.shape}与D2C Depth尺寸{depth_m.shape}不一致")
        mask = mask.astype(bool) if mask.dtype == np.bool_ else mask >= self.mask_threshold
        mask_area = int(np.count_nonzero(mask))
        if mask_area < self.min_mask_area_px:
            return None
        x1, y1, x2, y2 = map(float, detection.bbox)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width < 2.0 or box_height < 2.0:
            return None

        band_start, band_end = self.surface_bands[name]
        sample_x1 = int(np.clip(x1 + box_width * self.x_margin_ratio, 0, width - 1))
        sample_x2 = int(np.clip(x2 - box_width * self.x_margin_ratio, 0, width - 1))
        sample_y1 = int(np.clip(y1 + box_height * float(band_start), 0, height - 1))
        sample_y2 = int(np.clip(y1 + box_height * float(band_end), 0, height - 1))
        if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
            return None

        points: List[List[float]] = []
        for v in range(sample_y1, sample_y2 + 1, self.sample_step):
            for u in range(sample_x1, sample_x2 + 1, self.sample_step):
                if not mask[v, u]:
                    continue
                z = float(depth_m[v, u])
                if not np.isfinite(z) or z < self.min_depth_m or z > self.max_depth_m:
                    continue
                x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
                y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
                points.append([x, y, z])

        if len(points) < self.min_object_points:
            return None
        points_array = np.asarray(points, dtype=np.float32)
        center = np.median(points_array, axis=0).astype(np.float32)
        heights = np.abs(points_array @ self.ground_plane[:3] + self.ground_plane[3])
        surface_height = float(np.median(heights))
        return SceneObjectGeometry(
            name=name,
            confidence=float(detection.confidence),
            bbox=np.asarray(detection.bbox, dtype=np.float32),
            points_3d=points_array,
            center_3d=center,
            surface_height_m=surface_height,
            mask_area_px=mask_area,
            sampled_point_count=len(points),
        )

    @staticmethod
    def _nearest_distance(person_points: np.ndarray, object_points: np.ndarray) -> Optional[float]:
        """计算少量人体点到家具采样点的最小3D距离。"""
        if len(person_points) == 0 or len(object_points) == 0:
            return None
        differences = person_points[:, None, :] - object_points[None, :, :]
        return float(np.min(np.linalg.norm(differences, axis=2)))

    def _relation_confidence(
        self, iou: float, distance_m: Optional[float], height_gap_m: Optional[float]
    ) -> float:
        """把2D重叠、3D距离和表面高度差归一化为关系可信度。"""
        cfg = self.relation_cfg
        min_iou = float(cfg["min_iou"])
        full_iou = float(cfg["full_iou"])
        iou_score = float(np.clip((iou - min_iou) / max(full_iou - min_iou, 1e-8), 0.0, 1.0))
        distance_score = (
            0.0
            if distance_m is None
            else float(
                np.clip(1.0 - distance_m / max(float(cfg["near_distance_m"]), 1e-8), 0.0, 1.0)
            )
        )
        height_score = (
            0.0
            if height_gap_m is None
            else float(
                np.clip(
                    1.0 - height_gap_m / max(float(cfg["surface_height_tolerance_m"]), 1e-8),
                    0.0,
                    1.0,
                )
            )
        )
        weights = cfg["confidence_weights"]
        return float(
            np.clip(
                float(weights["iou"]) * iou_score
                + float(weights["distance"]) * distance_score
                + float(weights["height"]) * height_score,
                0.0,
                1.0,
            )
        )

    def classify_measurements(
        self,
        person_id: int,
        person_bbox: Sequence[float],
        person_points_3d: np.ndarray,
        body_angle_deg: Optional[float],
        hip_height_m: Optional[float],
        objects: Iterable[SceneObjectGeometry],
    ) -> SceneRelationResult:
        """基于已完成的3D测量判断人与家具或地面的关系。"""
        cfg = self.relation_cfg
        best: Optional[SceneRelationResult] = None
        for obj in objects:
            iou = bbox_iou(person_bbox, obj.bbox)
            nearest = self._nearest_distance(person_points_3d, obj.points_3d)
            height_gap = (
                None
                if hip_height_m is None
                else abs(float(hip_height_m) - float(obj.surface_height_m))
            )
            # 几何关系置信度再乘家具Seg置信度，低可信Mask不能产生高可信场景关系。
            confidence = self._relation_confidence(iou, nearest, height_gap) * float(
                np.clip(obj.confidence, 0.0, 1.0)
            )
            has_contact = iou >= float(cfg["min_iou"]) or (
                nearest is not None and nearest <= float(cfg["on_distance_m"])
            )
            near_object = nearest is not None and nearest <= float(cfg["near_distance_m"])
            surface_match = height_gap is not None and height_gap <= float(
                cfg["surface_height_tolerance_m"]
            )

            relation: Optional[str] = None
            if (
                body_angle_deg is not None
                and body_angle_deg >= float(cfg["lying_angle_min_deg"])
                and obj.name in {"bed", "sofa"}
                and has_contact
                and surface_match
            ):
                relation = f"lying_on_{obj.name}"
            elif (
                body_angle_deg is not None
                and body_angle_deg <= float(cfg["standing_angle_max_deg"])
                and has_contact
                and surface_match
            ):
                relation = f"sitting_on_{obj.name}"
            elif (
                body_angle_deg is not None
                and body_angle_deg <= float(cfg["standing_angle_max_deg"])
                and near_object
            ):
                relation = f"standing_near_{obj.name}"

            if relation is None:
                continue
            candidate = SceneRelationResult(
                person_id=int(person_id),
                relation=relation,
                scene_name=obj.name,
                confidence=confidence,
                scene_score=float(self.risk_scores.get(obj.name, self.risk_scores["unknown"])),
                object_name=obj.name,
                iou=iou,
                nearest_distance_m=nearest,
                hip_height_m=hip_height_m,
                object_surface_height_m=obj.surface_height_m,
                reason="人体姿态、2D重叠、3D距离和表面高度关系一致",
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        # 家具关系优先；只有没有可靠家具承载关系时才判断躺在地面。
        if best is None and (
            body_angle_deg is not None
            and body_angle_deg >= float(cfg["lying_angle_min_deg"])
            and hip_height_m is not None
            and hip_height_m <= float(cfg["floor_hip_height_m"])
        ):
            angle_score = float(
                np.clip(
                    (body_angle_deg - float(cfg["lying_angle_min_deg"]))
                    / max(90.0 - float(cfg["lying_angle_min_deg"]), 1e-8),
                    0.0,
                    1.0,
                )
            )
            height_score = float(
                np.clip(
                    1.0 - hip_height_m / max(float(cfg["floor_hip_height_m"]), 1e-8),
                    0.0,
                    1.0,
                )
            )
            floor_weights = cfg["floor_confidence_weights"]
            best = SceneRelationResult(
                person_id=int(person_id),
                relation="lying_on_floor",
                scene_name="floor",
                confidence=float(
                    float(floor_weights["angle"]) * angle_score
                    + float(floor_weights["height"]) * height_score
                ),
                scene_score=float(self.risk_scores["floor"]),
                object_name=None,
                iou=0.0,
                nearest_distance_m=None,
                hip_height_m=hip_height_m,
                object_surface_height_m=0.0,
                reason="人体接近水平且髋部接近地面",
            )

        if best is None:
            best = SceneRelationResult(
                person_id=int(person_id),
                relation="unknown",
                scene_name="unknown",
                confidence=0.0,
                scene_score=float(self.risk_scores["unknown"]),
                object_name=None,
                iou=0.0,
                nearest_distance_m=None,
                hip_height_m=hip_height_m,
                object_surface_height_m=None,
                reason="当前证据不足以确认人与家具或地面的关系",
            )
        return best

    def _stabilize(self, result: SceneRelationResult) -> SceneRelationResult:
        """按Track ID对最近多帧关系做置信度加权投票。"""
        history = self.histories[result.person_id]
        history.append(result)
        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        latest: Dict[str, SceneRelationResult] = {}
        for item in history:
            totals[item.relation] += max(item.confidence, 0.01)
            counts[item.relation] += 1
            latest[item.relation] = item
        relation = max(totals, key=totals.get)
        selected = latest[relation]
        stable_confidence = totals[relation] / max(counts[relation], 1)
        if stable_confidence < self.confidence_threshold and relation != "unknown":
            return replace(
                selected,
                relation="unknown",
                scene_name="unknown",
                confidence=stable_confidence,
                scene_score=float(self.risk_scores["unknown"]),
                reason="场景关系尚未达到稳定置信度",
            )
        return replace(selected, confidence=float(np.clip(stable_confidence, 0.0, 1.0)))

    def analyze(
        self,
        person_id: int,
        person_bbox: Sequence[float],
        keypoints_3d: np.ndarray,
        body_angle_deg: Optional[float],
        detections: Iterable[SceneObjectDetection],
        depth_m: np.ndarray,
        intrinsics: dict,
    ) -> SceneRelationResult:
        """主接口：从原始人体/家具测量输出多帧稳定关系。"""
        points = np.asarray(keypoints_3d, dtype=np.float32)
        hip = self._pair_center(points, LEFT_HIP, RIGHT_HIP)
        hip_height = self.point_ground_height(hip)
        person_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        person_points = np.asarray(
            [points[index] for index in person_indices if np.all(np.isfinite(points[index]))],
            dtype=np.float32,
        )
        if person_points.size == 0:
            person_points = np.empty((0, 3), dtype=np.float32)

        object_geometries = []
        for detection in detections:
            geometry = self._object_geometry(detection, depth_m, intrinsics)
            if geometry is not None:
                object_geometries.append(geometry)
        raw = self.classify_measurements(
            person_id=person_id,
            person_bbox=person_bbox,
            person_points_3d=person_points,
            body_angle_deg=body_angle_deg,
            hip_height_m=hip_height,
            objects=object_geometries,
        )
        return self._stabilize(raw)

    def reset_person(self, person_id: int) -> None:
        """清理离开画面的Track ID场景历史。"""
        self.histories.pop(int(person_id), None)


SceneRelationDetector = SceneScorer  # 兼容旧代码中的类名，新主流程使用SceneScorer。


def run_self_test(config: dict) -> None:
    """用合成Mask和测量验证家具取深度及三类人物关系。"""
    detector = SceneScorer(config, [0.0, -1.0, 0.0, 1.0])
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:70, 15:85] = True
    depth_m = np.full((100, 100), 5.0, dtype=np.float32)
    depth_m[mask] = 2.0
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    detection = SceneObjectDetection("bed", 0.9, np.array([10, 10, 90, 90], dtype=np.float32), mask)
    geometry = detector._object_geometry(detection, depth_m, intrinsics)
    assert geometry is not None and np.allclose(geometry.points_3d[:, 2], 2.0)
    assert (
        geometry.mask_area_px == int(np.count_nonzero(mask))
        and geometry.sampled_point_count >= detector.min_object_points
    )

    person_points = np.array([[0.0, 0.4, 2.0]], dtype=np.float32)
    bed = SceneObjectGeometry(
        name="bed",
        confidence=0.9,
        bbox=np.array([20, 40, 180, 190], dtype=np.float32),
        points_3d=np.array([[0.1, 0.45, 2.0]], dtype=np.float32),
        center_3d=np.array([0.1, 0.45, 2.0], dtype=np.float32),
        surface_height_m=0.55,
        mask_area_px=1000,
        sampled_point_count=80,
    )
    result = detector.classify_measurements(1, [30, 30, 170, 180], person_points, 80.0, 0.55, [bed])
    assert (
        result.relation == "lying_on_bed"
        and result.scene_name == "bed"
        and result.scene_score == 0.0
    )

    result = detector.classify_measurements(2, [30, 30, 170, 180], person_points, 80.0, 0.10, [])
    assert (
        result.relation == "lying_on_floor"
        and result.scene_name == "floor"
        and result.scene_score == 1.0
    )

    chair = SceneObjectGeometry(
        name="chair",
        confidence=0.9,
        bbox=np.array([180, 40, 260, 190], dtype=np.float32),
        points_3d=np.array([[0.5, 0.5, 2.0]], dtype=np.float32),
        center_3d=np.array([0.5, 0.5, 2.0], dtype=np.float32),
        surface_height_m=0.50,
        mask_area_px=500,
        sampled_point_count=40,
    )
    result = detector.classify_measurements(
        3, [30, 30, 170, 180], person_points, 10.0, 0.90, [chair]
    )
    assert result.relation == "standing_near_chair"

    print("scene self-test: PASS")
    print("  mask depth, relation recognition, scene risk score=PASS")


def main() -> None:
    """命令行独立测试入口；默认打开相机，--self-test无需硬件。"""
    parser = argparse.ArgumentParser(description="人与场景关系识别模块")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument("--self-test", action="store_true", help="运行合成数据测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        # 延迟导入可避免算法模块依赖相机SDK；只有在线测试时才加载主流程。
        from main import run_live

        run_live(config, args.config, stage="scene")


if __name__ == "__main__":
    main()
