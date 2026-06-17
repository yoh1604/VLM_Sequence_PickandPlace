import os
import sys
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT SAFETY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOCAL CONFIG
# ============================================================

import capture_config as cfg

from planning.vlm_engine import VLMEngine
from planning.validator_engine import LogicValidator
from perception.yolo_world_engine import YoloWorldEngine
from perception.fastsam_engine import FastSAMEngine
from perception.depth_engine import DepthEngine
from perception.grasp_runner import GraspNetRunner

from capture_config import (
    PROJECT_DIR,
    ENV_PATH,
    BASE_DIR,
    TEST_NAME,
    USER_QUERY,
    IMAGE_PATH,
    DEPTH_PATH,
    INTRINSICS_PATH,
    DEPTH_VIS_PATH,
    VLM_OUTPUT_DIR,
    VISION_OUTPUT_DIR,
    ACTION_PLAN_JSON,
    VLM_DETECTIONS_JSON,
    VALIDATION_JSON,
    YOLO_DETECTIONS_JSON,
    YOLO_RESULT_IMAGE,
    FASTSAM_MASK_PATH,
    FASTSAM_RESULT_IMAGE,
    OBJECT_POSITION_JSON,
    SNAPSHOT_CURRENT_RGB,
    SNAPSHOT_CURRENT_DEPTH,
    YOLO_WORLD_MODEL_PATH,
    FASTSAM_MODEL_PATH,
    print_config,
)


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(path_like):
    path = Path(str(path_like)).expanduser()

    if not path.is_absolute():
        path = Path(PROJECT_DIR).expanduser().resolve() / path

    return path.resolve()


def load_json_file(path, name="JSON"):
    path = resolve_project_path(path)

    print(f"[CHECK] {name}: {path}")

    if not path.exists():
        raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    with open(path, "r") as f:
        return json.load(f), path


def save_json_file(path, data):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


# ============================================================
# ENV / PROVIDER
# ============================================================

def get_provider_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    print("ENV path:", ENV_PATH)
    print("OPENAI_API_KEY detected:", bool(openai_key))
    print("GEMINI_API_KEY detected:", bool(gemini_key))

    return openai_key, gemini_key


def save_input_snapshots():
    if os.path.exists(IMAGE_PATH):
        shutil.copyfile(IMAGE_PATH, SNAPSHOT_CURRENT_RGB)
        print("Saved RGB snapshot:", SNAPSHOT_CURRENT_RGB)

    if os.path.exists(DEPTH_VIS_PATH):
        shutil.copyfile(DEPTH_VIS_PATH, SNAPSHOT_CURRENT_DEPTH)
        print("Saved depth snapshot:", SNAPSHOT_CURRENT_DEPTH)


def create_planner(openai_key, gemini_key):
    if openai_key:
        print("Menggunakan OpenAI Cloud VLM Planner...")
        return VLMEngine(
            api_key=openai_key,
            provider="openai"
        )

    if gemini_key:
        print("Menggunakan Gemini Cloud VLM Planner...")
        return VLMEngine(
            api_key=gemini_key,
            provider="gemini"
        )

    raise RuntimeError("Tidak ada API key untuk planner.")


def create_validator(openai_key, gemini_key):
    if openai_key:
        print("Menggunakan OpenAI Cloud LogicValidator...")
        return LogicValidator(
            api_key=openai_key,
            provider="openai"
        )

    if gemini_key:
        print("Menggunakan Gemini Cloud LogicValidator...")
        return LogicValidator(
            api_key=gemini_key,
            provider="gemini"
        )

    raise RuntimeError("Tidak ada API key untuk validator.")


# ============================================================
# VLM DETECTION FORMATTER
# ============================================================

def save_vlm_detections(planner_result, output_path):
    visual_analysis = planner_result.get("visual_analysis", [])

    detections = []

    for item in visual_analysis:
        if not isinstance(item, dict):
            continue

        obj = item.get("object", "")
        bbox_yxyx = item.get("bbox", [0, 0, 0, 0])

        if not obj:
            continue

        if not isinstance(bbox_yxyx, list) or len(bbox_yxyx) != 4:
            bbox_yxyx = [0, 0, 0, 0]

        ymin, xmin, ymax, xmax = bbox_yxyx

        detections.append({
            "label": obj,
            "confidence": None,
            "bbox_format": "normalized_0_1000_xyxy",
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_original_format": "normalized_0_1000_yxyx",
            "bbox_original_yxyx": [ymin, xmin, ymax, xmax],
            "source": "vlm_visual_analysis",
            "description": item.get("description", ""),
            "status": item.get("status", ""),
            "target_role": item.get("target_role", "")
        })

    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)

    return detections


def normalize_repeated_words(text):
    words = str(text).lower().strip().split()
    clean = []

    for w in words:
        if len(clean) == 0 or clean[-1] != w:
            clean.append(w)

    return " ".join(clean)


def get_first_target_from_validation(validation_result):
    status = str(validation_result.get("validation_status", "FAIL")).upper()

    if status != "PASS":
        raise RuntimeError("Plan belum PASS. Tidak boleh lanjut ke YOLO/FastSAM.")

    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list) or len(final_plan) == 0:
        raise RuntimeError("final_action_plan kosong.")

    first_step = final_plan[0]
    target = first_step.get("target")
    target = normalize_repeated_words(target)

    if not target:
        raise RuntimeError("Target pada step pertama kosong.")

    print("\nTarget pertama untuk YOLO-World:", target)

    return target, first_step, final_plan


# ============================================================
# POINT CLOUD GENERATION
# ============================================================

def create_masked_pointcloud(
    rgb_path,
    depth_path,
    intrinsics_path,
    mask_path,
    output_ply_path,
    output_info_json_path=None,
    output_clean_mask_path=None,
    min_depth=0.1,
    max_depth=2.0,
    depth_tolerance=0.25,
    use_median_fill=False,
    visualize=False
):
    """
    Membuat masked point cloud dari RGB + depth + intrinsics + mask.

    Mode:
    - use_median_fill=False:
      hanya memakai pixel yang punya depth valid asli.
      Lebih aman untuk GraspNet.

    - use_median_fill=True:
      pixel depth invalid di dalam mask diisi median depth.
      Lebih penuh untuk visualisasi/centroid, tapi kurang ideal untuk GraspNet.
    """

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image tidak ditemukan: {rgb_path}")

    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file tidak ditemukan: {depth_path}")

    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics file tidak ditemukan: {intrinsics_path}")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")

    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Gagal membaca RGB image: {rgb_path}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = np.load(depth_path).astype(np.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Gagal membaca mask: {mask_path}")

    with open(intrinsics_path, "r") as f:
        intr = json.load(f)

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])
    depth_scale = float(intr.get("depth_scale", 0.001))

    depth_m = depth_raw * depth_scale

    print("\n========== POINT CLOUD RAW DATA INFO ==========")
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

    if mask.shape[:2] != (h, w):
        print("[PointCloud] Resize mask:", mask.shape[:2], "->", (h, w))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel3 = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel3, iterations=1)

    if output_clean_mask_path is not None:
        os.makedirs(os.path.dirname(output_clean_mask_path), exist_ok=True)
        cv2.imwrite(output_clean_mask_path, mask_clean)
        print("[PointCloud] Clean mask saved to:", output_clean_mask_path)

    mask_bool = mask_clean > 0
    mask_pixels = int(np.sum(mask_bool))

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

    print("\n========== POINT CLOUD MASK INFO ==========")
    print("Mask pixels:", mask_pixels)
    print("Depth values in mask:", len(zs_all))
    print("Valid depth in mask:", valid_depth_count)
    print("Zero depth in mask:", zero_depth_count)

    if valid_depth_count == 0:
        raise RuntimeError("Tidak ada depth valid di area mask.")

    valid_depth_values = zs_all[valid_depth]
    median_z = float(np.median(valid_depth_values))
    mean_z = float(np.mean(valid_depth_values))

    print("Depth in mask min:", float(np.min(valid_depth_values)))
    print("Depth in mask max:", float(np.max(valid_depth_values)))
    print("Depth in mask median:", median_z)
    print("Depth in mask mean:", mean_z)

    if use_median_fill:
        xs = xs_all.copy()
        ys = ys_all.copy()
        zs = zs_all.copy()
        zs[~valid_depth] = median_z

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
        xs = xs_all[valid_depth]
        ys = ys_all[valid_depth]
        zs = zs_all[valid_depth]

        near_median = np.abs(zs - median_z) < depth_tolerance

        xs = xs[near_median]
        ys = ys[near_median]
        zs = zs[near_median]

        method = "valid_depth_only_near_median"

    if len(zs) == 0:
        raise RuntimeError("Point cloud kosong setelah filtering.")

    print("\n========== POINT CLOUD FILTER ==========")
    print("Method:", method)
    print("Median object depth:", median_z)
    print("Depth tolerance:", depth_tolerance)
    print("Final valid points:", len(zs))

    X = (xs.astype(np.float32) - cx) * zs / fx
    Y = (ys.astype(np.float32) - cy) * zs / fy
    Z = zs

    points = np.stack([X, Y, Z], axis=1)
    colors = rgb[ys, xs].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if len(points) >= 50:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

    pcd_points = np.asarray(pcd.points)

    if pcd_points.shape[0] == 0:
        raise RuntimeError("Point cloud kosong setelah outlier removal.")

    center_mean = np.mean(pcd_points, axis=0)
    center_median = np.median(pcd_points, axis=0)

    print("\n========== POINT CLOUD OBJECT CENTER CAMERA FRAME ==========")
    print("Mean center  X Y Z:", center_mean)
    print("Median center X Y Z:", center_median)

    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(output_ply_path, pcd)

    print("\n[PointCloud] Point cloud saved to:", output_ply_path)
    print("[PointCloud] Total points after filtering:", np.asarray(pcd.points).shape[0])

    info = {
        "pointcloud_path": output_ply_path,
        "clean_mask_path": output_clean_mask_path,
        "method": method,
        "use_median_fill": bool(use_median_fill),
        "mask_pixels": int(mask_pixels),
        "valid_depth_pixels": int(valid_depth_count),
        "zero_depth_pixels": int(zero_depth_count),
        "final_points": int(np.asarray(pcd.points).shape[0]),
        "median_depth_m": float(median_z),
        "mean_depth_m": float(mean_z),
        "center_mean_camera_m": [
            float(center_mean[0]),
            float(center_mean[1]),
            float(center_mean[2])
        ],
        "center_median_camera_m": [
            float(center_median[0]),
            float(center_median[1]),
            float(center_median[2])
        ],
        "frame": "RealSense D455 aligned color camera frame",
        "note": "Point cloud masih camera frame, belum base frame UR5."
    }

    if output_info_json_path is not None:
        os.makedirs(os.path.dirname(output_info_json_path), exist_ok=True)
        with open(output_info_json_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print("[PointCloud] Info saved to:", output_info_json_path)

    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Masked Object Point Cloud",
            width=1280,
            height=720
        )
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        vis.run()
        vis.destroy_window()

    return info


# ============================================================
# GRASPNET STAGE
# ============================================================

def run_graspnet_stage():
    project_dir = Path(PROJECT_DIR).expanduser().resolve()

    runner = GraspNetRunner(
        project_dir=project_dir,
        conda_env_name="anygrasp_py310",
    )

    result = runner.run(
        rgb_path=cfg.IMAGE_PATH,
        depth_path=cfg.DEPTH_PATH,
        intrinsics_path=cfg.INTRINSICS_PATH,
        mask_path=cfg.FASTSAM_MASK_PATH,
        output_dir=cfg.VISION_OUTPUT_DIR,
        num_point=10000,
        num_view=300,
        collision_thresh=-1,
        voxel_size=0.01,
        max_center_dist=0.10,
        mask_dilate_iter=1,
        no_vis=True,
    )

    return result


# ============================================================
# CAMERA GRASP TO BASE GRASP TRANSFORM
# ============================================================

def validate_rotation_matrix(R):
    R = np.asarray(R, dtype=np.float64)

    det = np.linalg.det(R)
    orth_error = np.linalg.norm(R.T @ R - np.eye(3))

    if abs(det - 1.0) > 0.05:
        print(f"[WARN] Determinant rotation matrix tidak dekat 1: det={det}")

    if orth_error > 0.05:
        print(f"[WARN] Rotation matrix kurang orthonormal: error={orth_error}")

    return float(det), float(orth_error)


def transform_grasp_camera_to_base(
    tf_json_path=None,
    best_grasp_camera_json=None,
    best_grasp_base_json=None,
    pregrasp_offset_z=0.10
):
    if tf_json_path is None:
        tf_json_path = Path(PROJECT_DIR) / "configs" / "T_base_camera.json"

    if best_grasp_camera_json is None:
        best_grasp_camera_json = Path(VISION_OUTPUT_DIR) / "best_grasp_camera.json"

    if best_grasp_base_json is None:
        best_grasp_base_json = Path(VISION_OUTPUT_DIR) / "best_grasp_base.json"

    tf_data, tf_path = load_json_file(tf_json_path, "T_BASE_CAMERA_JSON")
    grasp_data, grasp_camera_path = load_json_file(
        best_grasp_camera_json,
        "BEST_GRASP_CAMERA_JSON"
    )

    if not tf_data.get("success", True):
        raise RuntimeError("T_base_camera.json success=False")

    if not grasp_data.get("success", False):
        raise RuntimeError("best_grasp_camera.json success=False")

    if "T_base_camera" not in tf_data:
        raise KeyError("T_base_camera tidak ditemukan di file transform.")

    if "best_grasp" not in grasp_data:
        raise KeyError("best_grasp tidak ditemukan di best_grasp_camera.json.")

    T_base_camera = np.array(tf_data["T_base_camera"], dtype=np.float64)

    if T_base_camera.shape != (4, 4):
        raise ValueError(f"T_base_camera harus 4x4, sekarang: {T_base_camera.shape}")

    R_base_camera = T_base_camera[:3, :3]
    t_base_camera = T_base_camera[:3, 3]

    best_grasp = grasp_data["best_grasp"]

    p_camera = np.array(best_grasp["translation_camera"], dtype=np.float64)
    R_camera_grasp = np.array(best_grasp["rotation_matrix_camera"], dtype=np.float64)

    if p_camera.shape != (3,):
        raise ValueError(f"translation_camera harus 3D, sekarang: {p_camera.shape}")

    if R_camera_grasp.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix_camera harus 3x3, sekarang: {R_camera_grasp.shape}"
        )

    p_base = R_base_camera @ p_camera + t_base_camera
    R_base_grasp = R_base_camera @ R_camera_grasp

    det, orth_error = validate_rotation_matrix(R_base_grasp)

    p_pregrasp = p_base.copy()
    p_pregrasp[2] += float(pregrasp_offset_z)

    result = {
        "success": True,
        "method": "camera_to_base_transform",
        "translation_base": p_base.tolist(),
        "rotation_matrix_base": R_base_grasp.tolist(),
        "score": best_grasp.get("score"),
        "width": best_grasp.get("width"),
        "height": best_grasp.get("height"),
        "depth": best_grasp.get("depth"),
        "object_id": best_grasp.get("object_id"),
        "frame": "UR5 base frame",
        "source_frame": "RealSense D455 aligned color camera frame",
        "source_file": str(grasp_camera_path),
        "source_transform": str(tf_path),
        "rotation_validation": {
            "determinant": det,
            "orthonormal_error": orth_error,
        },
        "camera_grasp_used": {
            "translation_camera": best_grasp.get("translation_camera"),
            "rotation_matrix_camera": best_grasp.get("rotation_matrix_camera"),
        },
        "pre_grasp_preview": {
            "translation_base": p_pregrasp.tolist(),
            "offset_z_m": float(pregrasp_offset_z),
            "note": "Gunakan pose ini untuk test pertama agar robot hanya bergerak ke atas objek."
        },
        "note": (
            "Pose grasp sudah ditransformasikan dari camera frame ke UR5 base frame. "
            "Untuk test pertama, gunakan pre_grasp_preview dan jangan langsung descend."
        ),
    }

    output_path = save_json_file(best_grasp_base_json, result)

    print("\n========== GRASP CAMERA TO BASE ==========")
    print("Input grasp camera:", grasp_camera_path)
    print("Transform:", tf_path)
    print("Output grasp base:", output_path)
    print("-----------------------------------------")
    print("translation_base:", result["translation_base"])
    print("pre_grasp_translation_base:", result["pre_grasp_preview"]["translation_base"])
    print("score:", result["score"])
    print("width:", result["width"])
    print("=========================================\n")

    return result


# ============================================================
# VLM + VALIDATOR
# ============================================================

def run_vlm_and_validator(user_query):
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    openai_key, gemini_key = get_provider_keys()

    planner = create_planner(openai_key, gemini_key)

    planner_result = planner.get_strategy(
        img_path=IMAGE_PATH,
        user_query=user_query
    )

    if planner_result is None:
        raise RuntimeError("VLM Planner tidak menghasilkan action plan.")

    if isinstance(planner_result, dict) and "error" in planner_result:
        raise RuntimeError(f"VLM Planner error: {planner_result['error']}")

    with open(ACTION_PLAN_JSON, "w") as f:
        json.dump(planner_result, f, indent=2, ensure_ascii=False)

    vlm_detections = save_vlm_detections(
        planner_result,
        VLM_DETECTIONS_JSON
    )

    print("\nAction plan berhasil dibuat.")
    print(f"Saved to: {ACTION_PLAN_JSON}")
    print(json.dumps(planner_result, indent=2, ensure_ascii=False))

    print("\nVLM detections berhasil dibuat.")
    print(f"Saved to: {VLM_DETECTIONS_JSON}")
    print(json.dumps(vlm_detections, indent=2, ensure_ascii=False))

    validator = create_validator(openai_key, gemini_key)

    validation_result = validator.validate_strategy(
        image_path=IMAGE_PATH,
        user_query=user_query,
        planner_json=planner_result
    )

    with open(VALIDATION_JSON, "w") as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)

    print("\nValidation berhasil dibuat.")
    print(f"Saved to: {VALIDATION_JSON}")
    print(json.dumps(validation_result, indent=2, ensure_ascii=False))

    return planner_result, validation_result


# ============================================================
# SPATIAL PIPELINE
# ============================================================

def run_spatial_pipeline(validation_result):
    target, first_step, final_plan = get_first_target_from_validation(
        validation_result
    )

    yolo = YoloWorldEngine(
        model_name=YOLO_WORLD_MODEL_PATH,
        conf=0.50,
        output_dir=VISION_OUTPUT_DIR
    )

    best_detection, all_detections = yolo.detect_target(
        image_path=IMAGE_PATH,
        target=target,
        output_json=YOLO_DETECTIONS_JSON,
        output_image=YOLO_RESULT_IMAGE,
        conf=0.50,
        use_generic_fallback=True
    )

    bbox = best_detection["bbox"]

    fastsam = FastSAMEngine(
        model_name=FASTSAM_MODEL_PATH,
        device="cpu",
        imgsz=640,
        conf=0.4,
        iou=0.9,
        output_dir=VISION_OUTPUT_DIR
    )

    mask_path, fastsam_result = fastsam.segment_bbox(
        image_path=IMAGE_PATH,
        bbox=bbox,
        mask_path=FASTSAM_MASK_PATH,
        result_image_path=FASTSAM_RESULT_IMAGE
    )

    depth = DepthEngine(
        depth_path=DEPTH_PATH,
        intrinsics_path=INTRINSICS_PATH,
        min_depth=0.1,
        max_depth=2.0
    )

    object_position = depth.extract_from_mask(
        target=target,
        mask_path=mask_path,
        output_path=OBJECT_POSITION_JSON
    )

    pointcloud_valid_only_path = os.path.join(
        VISION_OUTPUT_DIR,
        "masked_object_pointcloud_valid_only.ply"
    )

    pointcloud_valid_only_info_path = os.path.join(
        VISION_OUTPUT_DIR,
        "masked_object_pointcloud_valid_only_info.json"
    )

    pointcloud_median_fill_path = os.path.join(
        VISION_OUTPUT_DIR,
        "masked_object_pointcloud_median_fill.ply"
    )

    pointcloud_median_fill_info_path = os.path.join(
        VISION_OUTPUT_DIR,
        "masked_object_pointcloud_median_fill_info.json"
    )

    clean_mask_debug_path = os.path.join(
        VISION_OUTPUT_DIR,
        "fastsam_mask_clean_debug.png"
    )

    pointcloud_valid_only = create_masked_pointcloud(
        rgb_path=IMAGE_PATH,
        depth_path=DEPTH_PATH,
        intrinsics_path=INTRINSICS_PATH,
        mask_path=mask_path,
        output_ply_path=pointcloud_valid_only_path,
        output_info_json_path=pointcloud_valid_only_info_path,
        output_clean_mask_path=clean_mask_debug_path,
        min_depth=0.1,
        max_depth=2.0,
        depth_tolerance=0.25,
        use_median_fill=False,
        visualize=False
    )

    pointcloud_median_fill = create_masked_pointcloud(
        rgb_path=IMAGE_PATH,
        depth_path=DEPTH_PATH,
        intrinsics_path=INTRINSICS_PATH,
        mask_path=mask_path,
        output_ply_path=pointcloud_median_fill_path,
        output_info_json_path=pointcloud_median_fill_info_path,
        output_clean_mask_path=None,
        min_depth=0.1,
        max_depth=2.0,
        depth_tolerance=0.25,
        use_median_fill=True,
        visualize=False
    )

    return {
        "target": target,
        "first_step": first_step,
        "final_plan": final_plan,
        "best_detection": best_detection,
        "all_detections": all_detections,
        "fastsam_result": fastsam_result,
        "object_position": object_position,
        "pointcloud_valid_only": pointcloud_valid_only,
        "pointcloud_median_fill": pointcloud_median_fill
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print_config()
    load_dotenv(ENV_PATH)
    save_input_snapshots()

    planner_result, validation_result = run_vlm_and_validator(USER_QUERY)

    spatial_result = run_spatial_pipeline(validation_result)

    grasp_camera_result = run_graspnet_stage()

    if not grasp_camera_result.get("success", False):
        raise RuntimeError("GraspNet gagal menghasilkan grasp valid.")

    best_grasp = grasp_camera_result["best_grasp"]

    print("\n========== BEST GRASP CAMERA ==========")
    print(json.dumps(best_grasp, indent=2, ensure_ascii=False))
    print("======================================\n")

    grasp_base_result = transform_grasp_camera_to_base(
        tf_json_path=Path(PROJECT_DIR) / "configs" / "T_base_camera.json",
        best_grasp_camera_json=Path(VISION_OUTPUT_DIR) / "best_grasp_camera.json",
        best_grasp_base_json=Path(VISION_OUTPUT_DIR) / "best_grasp_base.json",
        pregrasp_offset_z=0.10
    )

    final_result = {
        "test_name": TEST_NAME,
        "user_query": USER_QUERY,
        "spatial_result": spatial_result,
        "grasp_camera_result": grasp_camera_result,
        "grasp_base_result": grasp_base_result,
        "outputs": {
            "best_grasp_camera_json": str(Path(VISION_OUTPUT_DIR) / "best_grasp_camera.json"),
            "best_grasp_base_json": str(Path(VISION_OUTPUT_DIR) / "best_grasp_base.json"),
            "grasp_candidates_camera_json": str(Path(VISION_OUTPUT_DIR) / "grasp_candidates_camera.json"),
            "object_position_camera_json": OBJECT_POSITION_JSON,
            "fastsam_mask": FASTSAM_MASK_PATH,
        }
    }

    print("\nFULL D455 PIPELINE SELESAI.")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()