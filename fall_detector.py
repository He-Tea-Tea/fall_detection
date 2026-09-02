# -*- coding: utf-8 -*-
"""跌倒检测核心算法：五项评分 + 时序状态机。

指标对应设计文档：
  1. 姿态异常   : 3D躯干倾角（肩中心-髋中心连线与地面法向量夹角）+ 人体框宽高比
  2. 高度下降   : 相对约 0.5s 前基准高度的下降量（基于地面平面）
  3. 下降速度   : 基于滤波后人体离地高度的向下速度
  4. 倒地后静止 : 仅在姿态/高度已异常（倒地候选）后计时的静止时长
  5. 场景语义   : 床/沙发/椅子为低风险，地面为高风险，用于排除正常躺卧/坐姿

所有指标归一化到 0~1，加权得到 FallScore，送入状态机：
  NORMAL -> SUSPECT -> FALLING -> CONFIRMED -> RECOVERED -> NORMAL

本版本改进：
  - 使用 ground.yaml 中的地面平面计算真实离地高度；
  - 使用地面法向量计算3D人体躯干倾角；
  - 高度、速度、人体运动均采用多帧历史进行滤波，降低 640x400 RGB-D 抖动；
  - 下降速度只对“向下”运动计分；
  - 增加地面模型与数据有效性检查。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import time
import os
import yaml
import numpy as np

# COCO 17 关键点索引
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class PersonState:
    """单个人的时序状态与各项指标分数。"""
    person_id: int
    last_time: Optional[float] = None
    # 历史缓冲（maxlen=60，条目为 (时间戳, 数值)）
    height_history: deque = field(default_factory=lambda: deque(maxlen=60))  # (t, 高度m)
    center_history: deque = field(default_factory=lambda: deque(maxlen=60))  # (t, 3D中心)
    angle_history: deque = field(default_factory=lambda: deque(maxlen=60))   # (t, 躯干角°)
    score_history: deque = field(default_factory=lambda: deque(maxlen=60))   # (t, FallScore)
    # 当前指标
    body_angle: float = 0.0
    body_height: Optional[float] = None      # 髋部中心到地面的真实距离(m)
    vertical_velocity: float = 0.0           # 离地高度变化速度(m/s)，向下为负
    motion_velocity: float = 0.0             # 3D 人体中心运动速度(m/s)
    # 五项评分（0~1）
    pose_score: float = 0.0
    height_score: float = 0.0
    velocity_score: float = 0.0
    static_score: float = 0.0
    scene_score: float = 0.5
    fall_score: float = 0.0
    # 状态机
    state: str = "NORMAL"
    suspect_count: int = 0
    falling_count: int = 0
    confirmed_time: Optional[float] = None
    static_start_time: Optional[float] = None
    # 数据质量
    depth_valid_ratio: float = 0.0
    data_quality: float = 0.0


class FallDetector:
    """跌倒检测器：每帧喂入一个人的 2D/3D 关键点与检测框，输出 PersonState。"""

    def __init__(self, cfg, ground_file="ground.yaml"):
        self.cfg = cfg
        self.people: Dict[int, PersonState] = {}
        w = cfg["weights"]
        self.w_pose = w["pose"]
        self.w_height = w["height"]
        self.w_velocity = w["velocity"]
        self.w_static = w["static"]
        self.w_scene = w["scene"]
        self.ground_plane = self.load_ground_plane(ground_file)
        self.ground_normal = self.ground_plane[:3]
        self.ground_normal = self.ground_normal / max(np.linalg.norm(self.ground_normal), 1e-8)

    # ---------------- 地面模型 ----------------
    def load_ground_plane(self, ground_file):
        """读取 ground.yaml 中的地面平面 Ax+By+Cz+D=0。"""
        if not os.path.exists(ground_file):
            raise FileNotFoundError(f"找不到地面模型文件: {ground_file}，请先运行 ground_detector.py")
        with open(ground_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "plane" not in data:
            raise ValueError("ground.yaml 中不存在 plane")
        p = data["plane"]
        plane = np.array([
            float(p["A"]),
            float(p["B"]),
            float(p["C"]),
            float(p["D"]),
        ], dtype=np.float64)
        if np.linalg.norm(plane[:3]) < 1e-8:
            raise ValueError("ground.yaml 中的地面平面法向量无效")
        # 统一法向量方向，保证与地面法向量方向一致
        if plane[1] > 0:
            plane = -plane
        return plane

    def point_to_ground_distance(self, point):
        """计算3D点到地面平面的垂直距离，单位m。"""
        if point is None or not np.all(np.isfinite(point)):
            return None
        A, B, C, D = self.ground_plane
        denominator = math.sqrt(A * A + B * B + C * C)
        return float(abs(A * point[0] + B * point[1] + C * point[2] + D) / denominator)

    def point_signed_ground_height(self, point):
        """计算3D点相对地面的有符号距离，用于判断上下方向。"""
        if point is None or not np.all(np.isfinite(point)):
            return None
        A, B, C, D = self.ground_plane
        denominator = math.sqrt(A * A + B * B + C * C)
        return float((A * point[0] + B * point[1] + C * point[2] + D) / denominator)

    # ---------------- 基础工具 ----------------
    @staticmethod
    def distance3d(a, b):
        """两点 3D 距离（任一为 None 返回 None）。"""
        if a is None or b is None:
            return None
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return None
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @staticmethod
    def point_confident(kp, threshold):
        """关键点有效：存在且置信度达标（kp=[x, y, conf]）。"""
        return kp is not None and len(kp) >= 3 and kp[2] >= threshold

    def get_person(self, person_id):
        if person_id not in self.people:
            self.people[person_id] = PersonState(person_id)
        return self.people[person_id]

    def median_history(self, history, now, window_s=0.4):
        """获取最近一段时间的历史中值，用于抑制 RGB-D 瞬时抖动。"""
        values = [v for t, v in reversed(history) if now - t <= window_s]
        if not values:
            return None
        return float(np.median(values))

    # ---------------- 数据质量 ----------------
    def compute_data_quality(self, keypoint_conf, keypoints_3d, bbox, distance_m):
        """根据Pose置信度、Depth有效率、人体尺寸和距离估计当前数据可信度。"""
        valid_conf = keypoint_conf[keypoint_conf >= self.cfg["min_keypoint_conf"]]
        pose_quality = float(np.mean(valid_conf)) if len(valid_conf) > 0 else 0.0
        valid_3d = np.all(np.isfinite(keypoints_3d), axis=1)
        depth_quality = float(np.mean(valid_3d))
        x1, y1, x2, y2 = bbox
        person_h = max(1.0, y2 - y1)
        size_quality = float(np.clip(person_h / 250.0, 0.0, 1.0))
        if distance_m is None:
            distance_quality = 0.0
        else:
            max_m = self.cfg["max_reliable_distance_m"]
            distance_quality = float(np.clip(1.0 - distance_m / max(max_m, 1e-6), 0.0, 1.0))
        quality = 0.35 * pose_quality + 0.35 * depth_quality + 0.20 * size_quality + 0.10 * distance_quality
        return float(np.clip(quality, 0.0, 1.0)), float(depth_quality)

    # ---------------- 指标 1：姿态异常 ----------------
    def compute_body_angle(self, pts, conf):
        """肩部中心->髋部中心连线与地面法向量的夹角（0°竖直 / 90°水平）。"""
        required = [
            (pts[LEFT_SHOULDER], conf[LEFT_SHOULDER]),
            (pts[RIGHT_SHOULDER], conf[RIGHT_SHOULDER]),
            (pts[LEFT_HIP], conf[LEFT_HIP]),
            (pts[RIGHT_HIP], conf[RIGHT_HIP]),
        ]
        if not all(c >= self.cfg["min_keypoint_conf"] for _, c in required):
            return None
        ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        if not np.all(np.isfinite(pts[ids])):
            return None
        shoulder = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2.0
        hip = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2.0
        v = hip - shoulder
        v_norm = np.linalg.norm(v)
        n_norm = np.linalg.norm(self.ground_normal)
        if v_norm < 1e-8 or n_norm < 1e-8:
            return None
        # 计算肩->髋方向与地面法向量的夹角
        cos_theta = abs(float(np.dot(v, self.ground_normal))) / (v_norm * n_norm)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))
        return float(angle)

    def score_pose(self, angle, bbox):
        """姿态分数：躯干倾角（70%）+ 人体框宽高比（30%）。"""
        if angle is None:
            return 0.0
        normal, fall = self.cfg["pose_angle_normal"], self.cfg["pose_angle_fall"]
        angle_score = (
            0.0 if angle <= normal else
            1.0 if angle >= fall else
            (angle - normal) / (fall - normal)
        )
        x1, y1, x2, y2 = bbox
        ratio = max(1.0, x2 - x1) / max(1.0, y2 - y1)
        shape_score = (
            0.0 if ratio < 0.5 else
            1.0 if ratio >= 0.9 else
            (ratio - 0.5) / 0.4
        )
        return float(0.7 * angle_score + 0.3 * shape_score)

    # ---------------- 指标 2：高度下降 ----------------
    def score_height(self, state, now):
        """取约 0.5s 前的基准高度，计算真实离地高度下降量并归一化到 0~1。"""
        if state.body_height is None or len(state.height_history) < 2:
            return 0.0
        current_h = self.median_history(state.height_history, now, window_s=0.15)
        previous_h = None
        for ts, h in reversed(state.height_history):
            if now - ts >= 0.45:
                previous_h = h
                break
        if current_h is None or previous_h is None:
            return 0.0
        drop = previous_h - current_h
        min_drop = self.cfg["height_drop_min_m"]
        full_drop = self.cfg["height_drop_full_m"]
        if drop <= min_drop:
            return 0.0
        if drop >= full_drop:
            return 1.0
        return float((drop - min_drop) / (full_drop - min_drop))

    # ---------------- 指标 3：下降速度 ----------------
    def score_velocity(self, state):
        """基于滤波后的真实离地高度变化，只有向下运动才增加跌倒风险。"""
        if state.vertical_velocity >= 0.0:
            return 0.0
        speed = -state.vertical_velocity
        normal = self.cfg["velocity_normal_mps"]
        fall = self.cfg["velocity_fall_mps"]
        if speed <= normal:
            return 0.0
        if speed >= fall:
            return 1.0
        return float((speed - normal) / (fall - normal))

    # ---------------- 速度/历史更新 ----------------
    def update_motion(self, state, center3d, now):
        """更新3D人体中心运动速度，使用多帧历史降低关键点抖动影响。"""
        if center3d is None or not np.all(np.isfinite(center3d)):
            return
        state.center_history.append((now, np.asarray(center3d, dtype=np.float32)))
        if len(state.center_history) < 4:
            state.motion_velocity = 0.0
            return
        current_t, current_center = state.center_history[-1]
        previous = None
        for ts, center in reversed(state.center_history):
            if current_t - ts >= 0.25:
                previous = (ts, center)
                break
        if previous is None:
            return
        old_t, old_center = previous
        dt = current_t - old_t
        if dt <= 1e-3:
            return
        raw_velocity = float(np.linalg.norm(current_center - old_center) / dt)
        state.motion_velocity = float(0.7 * state.motion_velocity + 0.3 * raw_velocity)

    def update_vertical_velocity(self, state, now):
        """更新真实离地高度的滤波值与下降速度。"""
        if state.body_height is None:
            return
        state.height_history.append((now, state.body_height))
        if len(state.height_history) < 4:
            state.vertical_velocity = 0.0
            return
        current_h = self.median_history(state.height_history, now, window_s=0.15)
        previous = None
        for ts, h in reversed(state.height_history):
            if now - ts >= 0.25:
                previous = (ts, h)
                break
        if current_h is None or previous is None:
            return
        old_t, old_h = previous
        dt = now - old_t
        if dt <= 1e-3:
            return
        raw_velocity = float((current_h - old_h) / dt)
        state.vertical_velocity = float(0.7 * state.vertical_velocity + 0.3 * raw_velocity)

    # ---------------- 指标 4：倒地后静止 ----------------
    def score_static(self, state, now):
        """仅当姿态/高度已异常（倒地候选）才计时；人体持续低速 -> 0~1。"""
        if state.pose_score < 0.5 and state.height_score < 0.5:
            state.static_start_time = None
            return 0.0
        motion_threshold = self.cfg.get("static_motion_threshold_mps", 0.12)
        if state.motion_velocity >= motion_threshold:
            state.static_start_time = None
            return 0.0
        if state.static_start_time is None:
            state.static_start_time = now
        static_time = now - state.static_start_time
        start = self.cfg["static_start_s"]
        full = self.cfg["static_full_s"]
        if static_time <= start:
            return 0.0
        if static_time >= full:
            return 1.0
        return float((static_time - start) / (full - start))

    # ---------------- 指标 5：场景语义 ----------------
    def get_scene_score(self, scene_name):
        """根据场景类别返回跌倒风险分数。"""
        return self.cfg["scene"].get(scene_name, 0.5)

    # ---------------- 主更新 ----------------
    def update(self, person_id, keypoints_2d, keypoint_conf, keypoints_3d, bbox, distance_m, scene_name="unknown"):
        """每帧更新一个人的全部指标与状态，返回 PersonState。

        参数：
          keypoints_2d：RGB图像中的2D关键点；
          keypoint_conf：17个关键点置信度；
          keypoints_3d：17×3相机系3D坐标，深度缺失处为 nan；
          bbox：人体检测框；
          distance_m：人体距离相机的距离；
          scene_name：当前场景类别。
        """
        now = time.monotonic()
        state = self.get_person(person_id)

        # 0) 数据质量
        state.data_quality, state.depth_valid_ratio = self.compute_data_quality(
            keypoint_conf, keypoints_3d, bbox, distance_m
        )

        # 1) 姿态
        angle = self.compute_body_angle(keypoints_3d, keypoint_conf)
        if angle is not None:
            state.body_angle = angle
            state.angle_history.append((now, angle))
        state.pose_score = self.score_pose(state.body_angle, bbox)

        # 2) 人体高度：左右髋中心到地面平面的真实距离
        lh, rh = keypoints_3d[LEFT_HIP], keypoints_3d[RIGHT_HIP]
        if np.all(np.isfinite(lh)) and np.all(np.isfinite(rh)):
            hip = (lh + rh) / 2.0
            height = self.point_to_ground_distance(hip)
            if height is not None:
                state.body_height = height
                self.update_vertical_velocity(state, now)

        # 3) 高度下降
        state.height_score = self.score_height(state, now)

        # 4) 垂直下降速度
        state.velocity_score = self.score_velocity(state)

        # 5) 3D 运动：优先使用躯干/髋部区域中心，减少手脚抖动
        torso_points = []
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
            if np.all(np.isfinite(keypoints_3d[idx])):
                torso_points.append(keypoints_3d[idx])
        if torso_points:
            center3d = np.mean(np.asarray(torso_points), axis=0)
            self.update_motion(state, center3d, now)

        # 6) 倒地后静止
        state.static_score = self.score_static(state, now)

        # 7) 场景
        state.scene_score = self.get_scene_score(scene_name)

        # 8) FallScore = 加权求和
        state.fall_score = (
            self.w_pose * state.pose_score
            + self.w_height * state.height_score
            + self.w_velocity * state.velocity_score
            + self.w_static * state.static_score
            + self.w_scene * state.scene_score
        )

        # 数据质量过低时，降低当前分数的可信程度
        quality_floor = self.cfg.get("quality_floor", 0.35)
        if state.data_quality < quality_floor:
            state.fall_score *= state.data_quality / max(quality_floor, 1e-6)

        state.fall_score = float(np.clip(state.fall_score, 0.0, 1.0))
        state.score_history.append((now, state.fall_score))

        # 9) 状态机
        self.update_state(state, now)
        state.last_time = now
        return state

    def update_state(self, state, now):
        """时序状态机：NORMAL / SUSPECT / FALLING / CONFIRMED / RECOVERED。"""
        score = state.fall_score
        min_confirm_quality = self.cfg.get("min_confirm_quality", 0.45)

        # ---- CONFIRMED：已经确认跌倒后，不因为单帧分数下降马上解除 ----
        if state.state == "CONFIRMED":
            recovered = (
                state.body_angle < self.cfg.get("recover_angle_deg", 30.0)
                and state.body_height is not None
                and state.static_score < 0.2
            )
            if recovered:
                if state.confirmed_time is None:
                    state.confirmed_time = now
                elif now - state.confirmed_time >= self.cfg["recover_s"]:
                    state.state = "RECOVERED"
                    state.confirmed_time = None
                    state.suspect_count = 0
                    state.falling_count = 0
            else:
                state.confirmed_time = None
            return

        # ---- NORMAL：低分时清空异常计数 ----
        if score < self.cfg["score_suspect"]:
            state.state = "NORMAL"
            state.suspect_count = 0
            state.falling_count = 0
            state.static_start_time = None
            return

        # ---- SUSPECT：连续达到门槛 ----
        if score >= self.cfg["score_suspect"]:
            state.suspect_count += 1
        else:
            state.suspect_count = 0

        if state.suspect_count >= 2 and score < self.cfg["score_falling"]:
            state.state = "SUSPECT"

        # ---- FALLING：连续多帧且存在明显下降/姿态异常 ----
        if score >= self.cfg["score_falling"]:
            abnormal = (
                state.pose_score >= 0.5
                or state.height_score >= 0.5
                or state.velocity_score >= 0.5
            )
            if abnormal:
                state.falling_count += 1
            else:
                state.falling_count = 0
            if state.falling_count >= 3:
                state.state = "FALLING"

        # ---- CONFIRMED：高分 + 关键条件 + 数据质量达标 ----
        if score >= self.cfg["score_confirmed"] and state.data_quality >= min_confirm_quality:
            confirmed = (
                state.pose_score >= 0.6
                and state.height_score >= 0.5
                and state.static_score >= 0.4
                and state.scene_score >= 0.5
            )
            if confirmed:
                state.state = "CONFIRMED"
                state.confirmed_time = now
                state.suspect_count = 0
                state.falling_count = 0