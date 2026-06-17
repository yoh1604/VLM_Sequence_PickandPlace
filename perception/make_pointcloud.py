import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ============================================================
# PROJECT IMPORT SETUP
# ============================================================
# File ini berada di:
# /home/b401/Documents/pick_place_occlusion_noetic/perception/make_pointcloud.py
#
# Agar bisa import capture_config.py dari root project, tambahkan root project ke sys.path.

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import capture_config as cfg  # noqa: E402


# ============================================================
# PATH CONFIG FROM capture_config.py
# ============================================================
# Semua path mengikuti TEST_NAME dan config aktif di capture_config.py.
# Jadi tidak hardcoded lagi ke test_01_try_milk.

RGB_PATH = Path(cfg.IMAGE_PATH)
DEPTH_PATH = Path(cfg.DEPTH_PATH)
INTRINSICS_PATH = Path(cfg.INTRINSICS_PATH)
MASK_PATH = Path(cfg.FASTSAM_MASK_PATH)

VISION_OUTPUT_DIR = Path(cfg.VISION_OUTPUT_DIR)

# Output utama
OUTPUT_PLY = VISION_OUTPUT_DIR / "masked_object_pointcloud.ply"
OUTPUT_INFO_JSON = VISION_OUTPUT_DIR / "masked_object_pointcloud_info.json"
OUTPUT_CLEAN_MASK = VISION_OUTPUT_DIR / "fastsam_mask_clean_debug.png"

# Output tambahan: valid-only dan median-fill
OUTPUT_VALID_ONLY_PLY = VISION_OUTPUT_DIR / "masked_object_pointcloud_valid_only.ply"
OUTPUT_VALID_ONLY_INFO_JSON = VISION_OUTPUT_DIR / "masked_object_pointcloud_valid_only_info.json"

OUTPUT_MEDIAN_FILL_PLY = VISION_OUTPUT_DIR / "masked_object_pointcloud_median_fill.ply"
OUTPUT_MEDIAN_FILL_INFO_JSON = VISION_OUTPUT_DIR / "masked_object_pointcloud_median_fill_info.json"


# ============================================================
# SETTINGS
# ============================================================

MIN_DEPTH = 0.1
MAX_DEPTH = 2.0

# Tolerance besar karena depth objek kamu banyak kosong/noisy.
# Ini hanya berpengaruh ke titik depth valid, bukan pixel depth 0.
DEPTH_TOLERANCE = 0.5

# Untuk standalone debugging, visualisasi boleh True.
# Kalau dipanggil dari pipeline utama, sebaiknya False supaya tidak pause.
VISUALIZE = True

# Untuk GraspNet/AnyGrasp, lebih aman pakai valid-only.
# Median-fill dibuat sebagai fallback/visualisasi/centroid.
CREATE_VALID_ONLY = True
CREATE_MEDIAN_FILL = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path: Path, name: str):
    print(f"{name}: {path}")
    print(f"{name} exists:", path.exists())

    if not path.exists():
        raise FileNotFoundError(f"{name} tidak ditemukan: {path}")


def load_rgb(rgb_path: Path):
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if rgb_bgr is None:
        raise RuntimeError(f"Gagal membaca RGB image: {rgb_path}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth(depth_path: Path):
    depth_raw = np.load(str(depth_path)).astype(np.float32)
    return depth_raw


def load_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"Gagal membaca mask: {mask_path}")

    return mask


def load_intrinsics(intrinsics_path: Path):
    with open(intrinsics_path, "r") as f:
        intr = json.load(f)

    fx = float(intr["fx"])
    fy = float(intr["fy"])

    # RealSense memakai ppx/ppy.
    # Fallback cx/cy disediakan kalau format intrinsics berubah.
    if "ppx" in intr and "ppy" in intr:
        cx = float(intr["ppx"])
        cy = float(intr["ppy"])
    elif "cx" in intr and "cy" in intr:
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    else:
        raise KeyError("Intrinsics harus memiliki ppx/ppy atau cx/cy.")

    depth_scale = float(intr.get("depth_scale", 0.001))

    return intr, fx, fy, cx, cy, depth_scale


def prepare_mask(mask, target_shape, output_clean_mask_path=None):
    """
    Menyiapkan mask binary.

    Catatan:
    - Tidak pakai erode karena mask kamu cenderung kecil.
    - Hanya threshold + morphology close ringan.
    """

    h, w = target_shape

    if mask.shape[:2] != (h, w):
        print("[PointCloud] Resize mask:", mask.shape[:2], "->", (h, w))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binary mask
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Cleaning ringan: close untuk menutup lubang kecil.
    kernel3 = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel3, iterations=1)

    if output_clean_mask_path is not None:
        output_clean_mask_path = Path(output_clean_mask_path)
        output_clean_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_clean_mask_path), mask_clean)
        print("[PointCloud] Clean mask saved to:", output_clean_mask_path)

    return mask_clean


def visualize_pointcloud(pcd: o3d.geometry.PointCloud, title="Masked Object Point Cloud"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()


def create_masked_pointcloud(
    rgb_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    mask_path: Path,
    output_ply_path: Path,
    output_info_json_path: Path,
    output_clean_mask_path: Path = None,
    min_depth: float = 0.1,
    max_depth: float = 2.0,
    depth_tolerance: float = 0.5,
    use_median_fill: bool = False,
    visualize: bool = False,
):
    """
    Membuat point cloud objek dari RGB + depth + intrinsics + mask.

    use_median_fill=False:
        Hanya memakai depth valid asli.
        Lebih aman untuk input awal GraspNet/AnyGrasp.

    use_median_fill=True:
        Pixel mask yang depth-nya invalid/0 diisi median depth.
        Point cloud lebih penuh, tetapi bentuk 3D menjadi lebih flat.
        Cocok untuk visualisasi/centroid/fallback, bukan input utama GraspNet.
    """

    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    intrinsics_path = Path(intrinsics_path)
    mask_path = Path(mask_path)
    output_ply_path = Path(output_ply_path)
    output_info_json_path = Path(output_info_json_path)

    print("\n========== POINT CLOUD INPUT CHECK ==========")
    check_file(rgb_path, "RGB_PATH")
    check_file(depth_path, "DEPTH_PATH")
    check_file(intrinsics_path, "INTRINSICS_PATH")
    check_file(mask_path, "MASK_PATH")

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(rgb_path)
    depth_raw = load_depth(depth_path)
    mask = load_mask(mask_path)

    intr, fx, fy, cx, cy, depth_scale = load_intrinsics(intrinsics_path)

    print("\n========== CAMERA INTRINSICS ==========")
    print(json.dumps(intr, indent=2))
    print("fx:", fx)
    print("fy:", fy)
    print("cx/ppx:", cx)
    print("cy/ppy:", cy)
    print("depth_scale:", depth_scale)

    depth_m = depth_raw * depth_scale

    print("\n========== RAW DATA INFO ==========")
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth_m.shape)
    print("Mask shape:", mask.shape)
    print("Depth raw dtype:", depth_raw.dtype)
    print("Depth raw min:", float(np.nanmin(depth_raw)))
    print("Depth raw max:", float(np.nanmax(depth_raw)))
    print("Depth meter min:", float(np.nanmin(depth_m)))
    print("Depth meter max:", float(np.nanmax(depth_m)))

    h, w = depth_m.shape[:2]

    if rgb.shape[:2] != (h, w):
        print("[PointCloud] Resize RGB:", rgb.shape[:2], "->", (h, w))
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_clean = prepare_mask(
        mask=mask,
        target_shape=(h, w),
        output_clean_mask_path=output_clean_mask_path,
    )

    mask_bool = mask_clean > 0
    mask_pixels = int(np.sum(mask_bool))

    print("\n========== MASK INFO ==========")
    print("Mask pixels:", mask_pixels)

    if mask_pixels == 0:
        raise RuntimeError("Mask kosong setelah binary/cleaning.")

    ys_all, xs_all = np.where(mask_bool)
    zs_all = depth_m[ys_all, xs_all]

    valid_depth = (
        np.isfinite(zs_all)
        & (zs_all > min_depth)
        & (zs_all < max_depth)
    )

    valid_depth_count = int(np.sum(valid_depth))
    zero_depth_count = int(np.sum(zs_all == 0))

    print("Depth values in mask:", int(zs_all.size))
    print("Valid depth in mask:", valid_depth_count)
    print("Zero depth in mask:", zero_depth_count)

    if valid_depth_count == 0:
        raise RuntimeError(
            "Tidak ada depth valid di area mask. "
            "Cek depth alignment, pencahayaan, jarak kamera, atau mask."
        )

    valid_depth_values = zs_all[valid_depth]

    depth_min = float(np.min(valid_depth_values))
    depth_max = float(np.max(valid_depth_values))
    depth_median = float(np.median(valid_depth_values))
    depth_mean = float(np.mean(valid_depth_values))

    print("Depth in mask min:", depth_min)
    print("Depth in mask max:", depth_max)
    print("Depth in mask median:", depth_median)
    print("Depth in mask mean:", depth_mean)

    # ============================================================
    # SELECT POINTS
    # ============================================================

    if use_median_fill:
        # Pakai semua pixel mask.
        # Depth invalid/0 diisi median depth.
        xs = xs_all.copy()
        ys = ys_all.copy()
        zs = zs_all.copy()

        zs[~valid_depth] = depth_median

        final_valid = (
            np.isfinite(zs)
            & (zs > min_depth)
            & (zs < max_depth)
        )

        xs = xs[final_valid]
        ys = ys[final_valid]
        zs = zs[final_valid]

        method = "median_fill_all_mask_pixels"

    else:
        # Pakai hanya depth valid asli.
        xs = xs_all[valid_depth]
        ys = ys_all[valid_depth]
        zs = zs_all[valid_depth]

        # Filter outlier berdasarkan median.
        # Dengan tolerance 0.5, ini cukup longgar.
        near_median = np.abs(zs - depth_median) < depth_tolerance

        xs = xs[near_median]
        ys = ys[near_median]
        zs = zs[near_median]

        method = "valid_depth_only_near_median"

    if len(zs) == 0:
        raise RuntimeError("Point cloud kosong setelah filtering.")

    print("\n========== POINT CLOUD FILTER ==========")
    print("Method:", method)
    print("Median object depth:", depth_median)
    print("Depth tolerance:", depth_tolerance)
    print("Final valid points before outlier removal:", int(len(zs)))

    # ============================================================
    # 2D PIXEL + DEPTH -> 3D CAMERA FRAME
    # ============================================================

    X = (xs.astype(np.float32) - cx) * zs / fx
    Y = -(ys.astype(np.float32) - cy) * zs / fy
    Z = zs

    points = np.stack([X, Y, Z], axis=1)
    colors = rgb[ys, xs].astype(np.float32) / 255.0

    # ============================================================
    # CREATE OPEN3D POINT CLOUD
    # ============================================================

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    points_before_outlier = len(points)

    # Outlier removal ringan.
    # Jangan terlalu agresif karena depth D455 pada objek kamu sudah banyak bolong.
    if points_before_outlier >= 50:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

    pcd_points = np.asarray(pcd.points)

    if pcd_points.shape[0] == 0:
        raise RuntimeError("Point cloud kosong setelah outlier removal.")

    center_mean = np.mean(pcd_points, axis=0)
    center_median = np.median(pcd_points, axis=0)

    print("\n========== OBJECT CENTER CAMERA FRAME ==========")
    print("Mean center  X Y Z:", center_mean)
    print("Median center X Y Z:", center_median)

    print("\nUntuk pick-and-place, biasanya pakai median center:")
    print("object_x:", float(center_median[0]))
    print("object_y:", float(center_median[1]))
    print("object_z:", float(center_median[2]))

    # ============================================================
    # SAVE POINT CLOUD + INFO
    # ============================================================

    o3d.io.write_point_cloud(str(output_ply_path), pcd)

    final_points = int(np.asarray(pcd.points).shape[0])

    print("\nPoint cloud saved to:", output_ply_path)
    print("Total points after filtering:", final_points)

    info = {
        "pointcloud_path": str(output_ply_path),
        "clean_mask_path": str(output_clean_mask_path) if output_clean_mask_path else None,
        "method": method,
        "use_median_fill": bool(use_median_fill),
        "mask_pixels": int(mask_pixels),
        "valid_depth_pixels": int(valid_depth_count),
        "zero_depth_pixels": int(zero_depth_count),
        "points_before_outlier_removal": int(points_before_outlier),
        "final_points": int(final_points),
        "min_depth_m": float(min_depth),
        "max_depth_m": float(max_depth),
        "depth_tolerance_m": float(depth_tolerance),
        "depth_min_m": float(depth_min),
        "depth_max_m": float(depth_max),
        "depth_median_m": float(depth_median),
        "depth_mean_m": float(depth_mean),
        "center_mean_camera_m": [
            float(center_mean[0]),
            float(center_mean[1]),
            float(center_mean[2]),
        ],
        "center_median_camera_m": [
            float(center_median[0]),
            float(center_median[1]),
            float(center_median[2]),
        ],
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx_or_ppx": float(cx),
            "cy_or_ppy": float(cy),
            "depth_scale": float(depth_scale),
        },
        "frame": "RealSense D455 aligned color camera frame",
        "note": (
            "valid_only lebih aman untuk GraspNet/AnyGrasp. "
            "median_fill lebih penuh untuk visualisasi/centroid, tetapi bentuk 3D lebih flat."
        ),
    }

    output_info_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_info_json_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("Point cloud info saved to:", output_info_json_path)

    if visualize:
        visualize_pointcloud(pcd, title=f"Masked Object Point Cloud - {method}")

    return info


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== MAKE POINTCLOUD FROM CONFIG ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("TEST_NAME:", cfg.TEST_NAME)
    print("VISION_OUTPUT_DIR:", VISION_OUTPUT_DIR)
    print("RGB_PATH:", RGB_PATH)
    print("DEPTH_PATH:", DEPTH_PATH)
    print("INTRINSICS_PATH:", INTRINSICS_PATH)
    print("MASK_PATH:", MASK_PATH)
    print("================================================\n")

    result = {}

    # Output utama: valid-only untuk GraspNet/AnyGrasp
    if CREATE_VALID_ONLY:
        valid_only_info = create_masked_pointcloud(
            rgb_path=RGB_PATH,
            depth_path=DEPTH_PATH,
            intrinsics_path=INTRINSICS_PATH,
            mask_path=MASK_PATH,
            output_ply_path=OUTPUT_VALID_ONLY_PLY,
            output_info_json_path=OUTPUT_VALID_ONLY_INFO_JSON,
            output_clean_mask_path=OUTPUT_CLEAN_MASK,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            depth_tolerance=DEPTH_TOLERANCE,
            use_median_fill=False,
            visualize=VISUALIZE,
        )

        result["valid_only"] = valid_only_info

        # Copy juga ke nama standar masked_object_pointcloud.ply
        # supaya script lama tetap bisa membaca file yang sama.
        o3d.io.write_point_cloud(
            str(OUTPUT_PLY),
            o3d.io.read_point_cloud(str(OUTPUT_VALID_ONLY_PLY))
        )

        with open(OUTPUT_INFO_JSON, "w") as f:
            json.dump(valid_only_info, f, indent=2, ensure_ascii=False)

        print("\nStandard point cloud also saved to:", OUTPUT_PLY)
        print("Standard point cloud info also saved to:", OUTPUT_INFO_JSON)

    # Output tambahan: median-fill untuk visualisasi/centroid fallback
    if CREATE_MEDIAN_FILL:
        median_fill_info = create_masked_pointcloud(
            rgb_path=RGB_PATH,
            depth_path=DEPTH_PATH,
            intrinsics_path=INTRINSICS_PATH,
            mask_path=MASK_PATH,
            output_ply_path=OUTPUT_MEDIAN_FILL_PLY,
            output_info_json_path=OUTPUT_MEDIAN_FILL_INFO_JSON,
            output_clean_mask_path=None,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            depth_tolerance=DEPTH_TOLERANCE,
            use_median_fill=True,
            visualize=False,
        )

        result["median_fill"] = median_fill_info

    print("\n========== POINT CLOUD GENERATION DONE ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()