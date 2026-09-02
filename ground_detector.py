# -*- coding: utf-8 -*-
"""Gemini 335Le 地面检测与地面平面标定（RGB-D 对齐坐标系）。

流程：
  启动 335Le RGB+Depth -> D2C 深度对齐到 RGB -> 自动读 RGB 内参
  -> 连续采集多帧（逐帧有效率检查）-> 中值融合（有效帧>=2 才保留）
  -> RGB 内参转 XYZ 点云 -> ROI/距离过滤 -> RANSAC（法向约束排除墙面）
  -> SVD 精化 -> 质量检查（内点比例/误差/法向/有效率）
  -> 保存 ground.yaml（A*x+B*y+C*z+D=0）-> 可视化

用法：python ground_detector.py

注意：
  - 标定时相机必须固定，标定期间不要移动相机，否则 ground.yaml 失效；
  - 视野中的地面区域尽量无人，避免大面积桌面/床面进入 ROI；
  - 本版本地面模型使用 D2C 对齐后的 RGB 坐标系，
    后续人体 3D 与角度计算也必须使用同一 RGB 坐标系。
"""
import os
import time
import yaml
import cv2
import numpy as np

from pyorbbecsdk import Pipeline, AlignFilter, OBStreamType


# ============================================================
# 配置参数
# ============================================================
NUM_FRAMES = 30                          # 连续采集帧数（用于中值融合）
MIN_DEPTH_M = 0.30                       # 有效深度范围（米）
MAX_DEPTH_M = 6.00

ROI_Y_START = 0.35                       # 候选地面区域（图像下半部分，比例坐标）
ROI_Y_END = 0.98
ROI_X_START = 0.05
ROI_X_END = 0.95

RANSAC_ITERATIONS = 500                  # RANSAC 迭代次数
RANSAC_SAMPLE_POINTS = 20000             # RANSAC 最多采样点数（提速）
RANSAC_DISTANCE_THRESHOLD = 0.025        # 2.5cm 以内视为同一平面

MIN_POINTS = 500                         # 最少有效点数量
MIN_INLIER_RATIO = 0.70                  # 地面内点比例最低要求
MAX_MEAN_ERROR_M = 0.020                 # 平面平均误差上限 20mm
MAX_MAX_ERROR_M = 0.080                  # 最大误差上限 80mm

# 地面法向约束：RGB 坐标系 X右/Y下/Z前，水平地面法向量应主要沿 Y 方向
GROUND_NORMAL_Y_MIN = 0.70

# 地面点高度范围（预留，未启用）：仅作宽松过滤
MIN_GROUND_HEIGHT_M = -0.30
MAX_GROUND_HEIGHT_M = 2.00

GROUND_FILE = "ground.yaml"              # 输出文件
MAX_DRAW_INLIERS = 3000                  # 可视化最多绘制内点数（防 OpenCV 卡顿）
MIN_FRAME_VALID_RATIO = 0.10             # 单帧至少 10% 像素有有效深度
MAX_INVALID_FRAMES = 10                  # 连续采集时允许的无效帧上限


# ============================================================
# Depth 数据转换
# ============================================================
def depth_frame_to_meters(depth_frame):
    """DepthFrame -> 米单位浮点深度图；无效深度（<=0 或 >20m）置 nan。"""
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
    scale = depth_frame.get_depth_scale()
    depth_m = raw.astype(np.float32) * scale / 1000.0
    depth_m[(depth_m <= 0.0) | (depth_m > 20.0)] = np.nan
    return depth_m


# ============================================================
# Depth -> XYZ 点云
# ============================================================
def depth_to_points(depth_m, intrinsics):
    """深度图 -> RGB 坐标系 XYZ 点云 (H,W,3)；depth_m 须已 D2C 对齐到 RGB。

    针孔模型：X=(u-cx)·Z/fx，Y=(v-cy)·Z/fy，Z 即深度。
    """
    h, w = depth_m.shape
    fx, fy, cx, cy = (intrinsics["fx"], intrinsics["fy"],
                      intrinsics["cx"], intrinsics["cy"])
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return np.stack([x, y, depth_m], axis=-1)


# ============================================================
# ROI + 深度过滤
# ============================================================
def select_ground_candidates(points):
    """筛出可能属于地面的点：图像下半 ROI + 距离 0.3~6m + 去 nan。

    返回：
      xyz：候选 3D 点（N,3）
      uv ：每个候选点对应的原图像素坐标（N,2），供可视化回投
    """
    h, w, _ = points.shape
    x1, x2 = int(w * ROI_X_START), int(w * ROI_X_END)
    y1, y2 = int(h * ROI_Y_START), int(h * ROI_Y_END)
    roi = points[y1:y2, x1:x2]
    xyz = roi.reshape(-1, 3)

    # 与 3D 点一一对应的原图像素坐标
    u_grid, v_grid = np.meshgrid(np.arange(x1, x2, dtype=np.int32),
                                 np.arange(y1, y2, dtype=np.int32))
    uv = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)

    # 删除 NaN/Inf
    valid = np.all(np.isfinite(xyz), axis=1)
    xyz, uv = xyz[valid], uv[valid]

    # 按到相机的欧氏距离过滤
    distance = np.linalg.norm(xyz, axis=1)
    mask = (distance >= MIN_DEPTH_M) & (distance <= MAX_DEPTH_M)
    return xyz[mask], uv[mask]


# ============================================================
# 平面拟合
# ============================================================
def plane_from_points(p1, p2, p3):
    """三点拟合平面，返回 [A,B,C,D]（A*x+B*y+C*z+D=0）；共线返回 None。"""
    v1, v2 = p2 - p1, p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    D = -np.dot(normal, p1)
    return np.array([normal[0], normal[1], normal[2], D], dtype=np.float64)


def plane_distance(points, plane):
    """点到平面的距离；平面退化（法向模长≈0）时返回 inf。"""
    A, B, C, D = plane
    numerator = np.abs(A * points[:, 0] + B * points[:, 1]
                       + C * points[:, 2] + D)
    denominator = np.sqrt(A * A + B * B + C * C)
    if denominator < 1e-8:
        return np.full(len(points), np.inf, dtype=np.float64)
    return numerator / denominator


def orient_ground_plane(plane):
    """统一地面法向量方向：RGB 坐标 Y 向下，地面法向应朝上（-Y），令 B<=0。"""
    plane = plane.copy()
    if plane[1] > 0:
        plane = -plane
    norm = np.linalg.norm(plane[:3])
    if norm > 1e-8:
        plane = plane / norm  # 再归一化一次
    return plane


def ground_normal_tilt_deg(plane):
    """地面法向量与相机竖直向上方向 [0,-1,0] 的夹角（度）。"""
    A, B, C, _ = plane
    norm = np.sqrt(A * A + B * B + C * C)
    if norm < 1e-8:
        return 90.0
    cos_theta = np.clip(-B / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def refine_plane(points):
    """最小二乘精化：对 RANSAC 内点做 SVD，取最小奇异向量为法向。"""
    if len(points) < 3:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    D = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], D], dtype=np.float64)


# ============================================================
# RANSAC
# ============================================================
def ransac_plane(points):
    """RANSAC 寻找最大地面平面，返回 (plane, inlier_mask)。

    策略：点多时采样提速；每次拟合用法向约束快速剔除墙面；
    找到候选后用全部点重算内点 -> SVD 精化 -> 再重算内点。
    """
    if len(points) < MIN_POINTS:
        raise RuntimeError(f"有效点太少：{len(points)}")

    rng = np.random.default_rng()
    num_points = len(points)

    # 采样：控制参与迭代的点数
    if num_points > RANSAC_SAMPLE_POINTS:
        sample_points = points[rng.choice(num_points, size=RANSAC_SAMPLE_POINTS,
                                          replace=False)]
    else:
        sample_points = points
    sample_num = len(sample_points)

    best_plane, best_count = None, 0
    for _ in range(RANSAC_ITERATIONS):
        ids = rng.choice(sample_num, size=3, replace=False)
        plane = plane_from_points(sample_points[ids[0]], sample_points[ids[1]],
                                  sample_points[ids[2]])
        if plane is None:
            continue
        # 法向约束：先定向再检查 |normal_y|，快速排除墙面
        plane = orient_ground_plane(plane)
        if abs(plane[1]) < GROUND_NORMAL_Y_MIN:
            continue
        count = int(np.sum(plane_distance(sample_points, plane)
                           < RANSAC_DISTANCE_THRESHOLD))
        if count > best_count:
            best_count, best_plane = count, plane

    if best_plane is None:
        raise RuntimeError("RANSAC 没有找到符合地面方向约束的平面。\n"
                           "请检查 ROI、相机视角和地面区域。")

    # 用全部点重算内点
    best_plane = orient_ground_plane(best_plane)
    best_mask = plane_distance(points, best_plane) < RANSAC_DISTANCE_THRESHOLD
    if np.sum(best_mask) < MIN_POINTS:
        raise RuntimeError("RANSAC 最终地面内点太少。")

    # SVD 精化后再重算内点
    refined = refine_plane(points[best_mask])
    if refined is not None:
        best_plane = orient_ground_plane(refined)
    best_mask = plane_distance(points, best_plane) < RANSAC_DISTANCE_THRESHOLD
    return best_plane, best_mask


# ============================================================
# 地面质量检查
# ============================================================
def check_ground_quality(plane, points, inlier_mask):
    """校验地面平面是否满足工程要求，不达标抛 RuntimeError。

    返回：inlier_ratio / mean_error / max_error / normal_y / tilt_deg / valid_ratio
    """
    inliers = points[inlier_mask]
    if len(inliers) < MIN_POINTS:
        raise RuntimeError(f"地面内点不足：{len(inliers)}")

    inlier_ratio = len(inliers) / len(points)
    distances = plane_distance(inliers, plane)
    mean_error = float(np.mean(distances))
    max_error = float(np.max(distances))
    normal_y = abs(plane[1])
    tilt_deg = ground_normal_tilt_deg(plane)
    valid_ratio = len(inliers) / len(points)

    if inlier_ratio < MIN_INLIER_RATIO:
        raise RuntimeError(f"地面内点比例过低：{inlier_ratio * 100:.2f}%\n"
                           "建议扩大 ROI 或移除桌面/床面/墙面等干扰物。")
    if mean_error > MAX_MEAN_ERROR_M:
        raise RuntimeError(f"地面平均误差过大：{mean_error * 1000:.2f} mm\n"
                           "建议检查 Depth 质量、地面平整度或 RANSAC 阈值。")
    if max_error > MAX_MAX_ERROR_M:
        raise RuntimeError(f"地面最大误差过大：{max_error * 1000:.2f} mm\n"
                           "说明存在较明显的局部深度异常或非平面点。")
    if normal_y < GROUND_NORMAL_Y_MIN:
        raise RuntimeError(f"地面法向量方向异常：|normal_y|={normal_y:.3f}")

    return {"inlier_ratio": float(inlier_ratio), "mean_error": mean_error,
            "max_error": max_error, "normal_y": float(normal_y),
            "tilt_deg": float(tilt_deg), "valid_ratio": float(valid_ratio)}


# ============================================================
# 保存地面参数
# ============================================================
def save_ground_plane(plane, intrinsics, inlier_count, total_count, quality):
    """把地面平面、RGB 内参、坐标系与质量指标保存到 ground.yaml。"""
    A, B, C, D = plane
    data = {
        "coordinate_system": "RGB_COLOR_ALIGNED_DEPTH",
        "description": "Ground plane fitted from Depth aligned to RGB color coordinates",
        "plane": {"A": float(A), "B": float(B), "C": float(C), "D": float(D)},
        "normal": {"x": float(A), "y": float(B), "z": float(C)},
        "camera_intrinsics": {
            "width": int(intrinsics["width"]), "height": int(intrinsics["height"]),
            "fx": float(intrinsics["fx"]), "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]), "cy": float(intrinsics["cy"]),
        },
        "ransac": {
            "distance_threshold_m": float(RANSAC_DISTANCE_THRESHOLD),
            "inlier_count": int(inlier_count),
            "total_point_count": int(total_count),
            "inlier_ratio": float(quality["inlier_ratio"]),
        },
        "quality": {
            "mean_error_m": float(quality["mean_error"]),
            "max_error_m": float(quality["max_error"]),
            "normal_y": float(quality["normal_y"]),
            "normal_tilt_deg": float(quality["tilt_deg"]),
            "valid_ratio": float(quality["valid_ratio"]),
        },
    }
    with open(GROUND_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"\n地面参数已保存：{GROUND_FILE}")


def wait_first_frames(pipeline, retries=10, timeout_ms=3000):
    """启动后等待首帧 RGB+Depth 同时到位。"""
    for _ in range(retries):
        frames = pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            continue
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is not None and depth is not None:
            return frames
    raise RuntimeError("没有获取到 RGB + Depth 帧")


# ============================================================
# 深度可视化
# ============================================================
def depth_to_display(depth_m, h, w):
    """深度图 -> JET 伪彩色显示图（MAX_DEPTH_M 内归一化，无效区域为黑色）。"""
    display = depth_m.copy()
    valid = np.isfinite(display)
    normalized = np.clip(display / MAX_DEPTH_M, 0, 1)
    normalized[~valid] = 0
    depth_img = cv2.applyColorMap((normalized * 255).astype(np.uint8),
                                  cv2.COLORMAP_JET)
    if not np.any(valid):
        depth_img = np.zeros((h, w, 3), dtype=np.uint8)
    return depth_img


def draw_ground_overlay(image, candidate_uv, inlier_mask):
    """在深度图上用绿色点绘制 RANSAC 最终认定的地面内点（抽样防卡顿）。"""
    inlier_uv = candidate_uv[inlier_mask]
    if len(inlier_uv) == 0:
        return
    if len(inlier_uv) > MAX_DRAW_INLIERS:
        ids = np.random.default_rng().choice(len(inlier_uv),
                                             size=MAX_DRAW_INLIERS, replace=False)
        inlier_uv = inlier_uv[ids]
    for u, v in inlier_uv:
        cv2.circle(image, (int(u), int(v)), 1, (0, 255, 0), -1)


def draw_roi(image):
    """绘制当前地面候选 ROI 框。"""
    h, w = image.shape[:2]
    x1, x2 = int(w * ROI_X_START), int(w * ROI_X_END)
    y1, y2 = int(h * ROI_Y_START), int(h * ROI_Y_END)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("Gemini 335Le 地面平面检测")
    print("RGB-D 对齐坐标系版本")
    print("=" * 60)

    pipeline = Pipeline()
    # D2C：深度对齐到 RGB，保证点云与人体关键点使用同一坐标系
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    try:
        # ---- 1. 启动相机 ----
        print("\n[1] 启动 Gemini 335Le")
        pipeline.start()
        print("Pipeline 启动成功")

        # ---- 2. 等首帧 RGB+Depth ----
        print("\n[2] 等待 RGB + Depth...")
        first_frames = wait_first_frames(pipeline)
        print("RGB + Depth 首帧获取成功")

        # ---- 3. 验证 D2C 对齐可用 ----
        print("\n[3] 测试 Depth -> RGB 对齐...")
        try:
            aligned = align_filter.process(first_frames)
        except Exception as e:
            raise RuntimeError(f"D2C 对齐失败：{e}")
        if aligned is None:
            raise RuntimeError("D2C 对齐返回 None")
        if aligned.get_color_frame() is None or aligned.get_depth_frame() is None:
            raise RuntimeError("D2C 后 RGB 或 Depth 为空")
        print("D2C 对齐成功")

        # ---- 4. 自动读取 RGB 内参（点云/人体 3D 都用它）----
        rgb = pipeline.get_camera_param().rgb_intrinsic
        intrinsics = {"width": int(rgb.width), "height": int(rgb.height),
                      "fx": float(rgb.fx), "fy": float(rgb.fy),
                      "cx": float(rgb.cx), "cy": float(rgb.cy)}
        print("\n[4] RGB 内参")
        print(f"分辨率：{intrinsics['width']} x {intrinsics['height']}")
        print(f"fx = {intrinsics['fx']:.6f}  fy = {intrinsics['fy']:.6f}")
        print(f"cx = {intrinsics['cx']:.6f}  cy = {intrinsics['cy']:.6f}")

        # ---- 5. 连续采集（逐帧 D2C + 有效率检查）----
        print("\n[5] 开始采集地面数据")
        print("请保证视野中的主要地面区域无人")
        print("同时尽量避免桌面、床面等大面积平面进入 ROI")
        print(f"正在采集 {NUM_FRAMES} 帧...")
        depth_frames, invalid_count = [], 0
        while len(depth_frames) < NUM_FRAMES:
            frames = pipeline.wait_for_frames(3000)
            if frames is None:
                invalid_count += 1
                if invalid_count > MAX_INVALID_FRAMES:
                    raise RuntimeError("连续多次无法获取有效 RGB+Depth 帧")
                continue
            try:
                aligned = align_filter.process(frames)
            except Exception:
                invalid_count += 1
                continue
            if aligned is None:
                invalid_count += 1
                continue
            depth = aligned.get_depth_frame()
            if depth is None:
                invalid_count += 1
                continue

            depth_m = depth_frame_to_meters(depth)
            valid_ratio = float(np.mean(np.isfinite(depth_m)))
            if valid_ratio < MIN_FRAME_VALID_RATIO:  # 有效深度太少，弃帧
                invalid_count += 1
                print(f"\r当前帧 Depth 有效率过低：{valid_ratio * 100:.1f}%", end="")
                continue

            depth_frames.append(depth_m)
            invalid_count = 0
            print(f"\r采集进度：{len(depth_frames)}/{NUM_FRAMES}"
                  f"  Depth 有效率：{valid_ratio * 100:.1f}%", end="")
        print()

        # ---- 6. 多帧中值融合（至少 2 帧有效的像素才保留）----
        print("\n[6] 正在进行多帧深度中值融合...")
        depth_stack = np.stack(depth_frames, axis=0)
        valid_count_map = np.sum(np.isfinite(depth_stack), axis=0)
        with np.errstate(invalid="ignore"):
            depth_median = np.nanmedian(depth_stack, axis=0)
        depth_median[valid_count_map < 2] = np.nan

        final_valid_ratio = float(np.mean(np.isfinite(depth_median)))
        print(f"融合后 Depth 有效率：{final_valid_ratio * 100:.2f}%")
        if final_valid_ratio < MIN_FRAME_VALID_RATIO:
            raise RuntimeError("多帧融合后有效 Depth 比例过低")

        # ---- 7. 点云转换（RGB 内参）----
        print("[7] 转换为 RGB 坐标系 3D 点云...")
        points = depth_to_points(depth_median, intrinsics)

        # ---- 8. 筛选地面候选点 ----
        print("[8] 筛选地面候选区域...")
        ground_points, candidate_uv = select_ground_candidates(points)
        print(f"候选点数量：{len(ground_points)}")
        if len(ground_points) < MIN_POINTS:
            raise RuntimeError("候选地面点太少，请检查：\n"
                               "1. 相机是否能看到地面\n"
                               "2. Depth 是否正常\n"
                               "3. ROI 是否合适")

        # ---- 9. RANSAC 拟合 ----
        print("\n[9] RANSAC 拟合地面...")
        plane, inlier_mask = ransac_plane(ground_points)
        A, B, C, D = plane
        inliers = ground_points[inlier_mask]

        # ---- 10. 质量检查 ----
        print("\n[10] 地面质量检查...")
        quality = check_ground_quality(plane, ground_points, inlier_mask)

        print("\n========== 地面检测结果 ==========")
        print("坐标系：RGB_COLOR_ALIGNED_DEPTH")
        print(f"平面方程：{A:.6f}x + {B:.6f}y + {C:.6f}z + {D:.6f} = 0")
        print(f"A = {A:.8f}  B = {B:.8f}  C = {C:.8f}  D = {D:.8f}")
        print(f"总候选点：{len(ground_points)}")
        print(f"地面内点：{len(inliers)}")
        print(f"内点比例：{quality['inlier_ratio'] * 100:.2f}%")
        print(f"平均平面误差：{quality['mean_error'] * 1000:.2f} mm")
        print(f"最大平面误差：{quality['max_error'] * 1000:.2f} mm")
        print(f"|Normal Y|：{quality['normal_y']:.4f}")
        print(f"法向量倾角：{quality['tilt_deg']:.2f}°")
        print(f"融合后有效率：{quality['valid_ratio'] * 100:.2f}%")

        # ---- 11. 保存 ----
        save_ground_plane(plane, intrinsics, len(inliers), len(ground_points), quality)

        # ---- 12. 可视化 ----
        print("\n[11] 显示地面检测结果")
        depth_img = depth_to_display(depth_median, intrinsics["height"], intrinsics["width"])
        draw_ground_overlay(depth_img, candidate_uv, inlier_mask)  # 绿色内点
        draw_roi(depth_img)                                        # ROI 框

        for text, y in ((f"Inlier: {quality['inlier_ratio'] * 100:.1f}%", 30),
                        (f"Mean Error: {quality['mean_error'] * 1000:.1f}mm", 60),
                        (f"Max Error: {quality['max_error'] * 1000:.1f}mm", 90),
                        (f"NormalY: {quality['normal_y']:.2f}", 120),
                        (f"Tilt: {quality['tilt_deg']:.1f}deg", 150),
                        ("GREEN = GROUND", 180)):
            cv2.putText(depth_img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

        print("\n可视化说明：")
        print("绿色点 = RANSAC 最终认定的地面点")
        print("白色框 = 当前地面候选 ROI")
        print("伪彩色 = Depth 距离")
        print("当前 ground.yaml = RGB 坐标系地面模型")
        print("按 ESC 退出...")

        # ---- 13. 显示循环 ----
        while True:
            cv2.imshow("Ground Detection - RGB Coordinate", depth_img)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    finally:
        # ---- 14. 关闭 ----
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nPipeline stopped")


if __name__ == "__main__":
    main()
