import json
import os
import shutil
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# IMPORT CAPTURE CONFIG SAFELY
# ============================================================

import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import capture_config as cfg


# ============================================================
# BASIC HELPERS
# ============================================================

def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def as_path(value):
    if value is None:
        return None
    return Path(str(value)).expanduser()


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src, dst, label="file"):
    src = as_path(src)
    dst = as_path(dst)

    if src is None or dst is None:
        print(f"[STEP COPY][SKIP] {label}: src/dst is None")
        return False

    if not src.exists():
        print(f"[STEP COPY][SKIP] {label}: source not found: {src}")
        return False

    ensure_parent(dst)
    shutil.copy2(src, dst)
    print(f"[STEP COPY][OK] {label}: {src} -> {dst}")
    return True


def save_json(path, data):
    path = as_path(path)
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[STEP COPY][OK] report: {path}")


def get_step_index(default=1):
    """
    Priority:
    1. STEP_INDEX from environment variable, passed by Gradio
    2. STEP_INDEX from capture_config.py
    3. default = 1
    """

    env_value = os.getenv("STEP_INDEX")
    if env_value is not None:
        try:
            return int(env_value)
        except Exception:
            pass

    cfg_value = getattr(cfg, "STEP_INDEX", default)
    try:
        return int(cfg_value)
    except Exception:
        return default


def get_test_name(default="ui_test_01"):
    """
    Priority:
    1. TEST_NAME from environment variable, passed by Gradio
    2. TEST_NAME from capture_config.py
    3. default
    """

    env_value = os.getenv("TEST_NAME")
    if env_value:
        return env_value

    cfg_value = getattr(cfg, "TEST_NAME", default)
    return str(cfg_value)


def get_vision_output_dir():
    if hasattr(cfg, "VISION_OUTPUT_DIR"):
        return as_path(cfg.VISION_OUTPUT_DIR)

    test_name = get_test_name()
    return PROJECT_DIR / "data" / "d455_capture" / "tests" / test_name / "vision_output"


def get_vlm_output_dir():
    if hasattr(cfg, "VLM_OUTPUT_DIR"):
        return as_path(cfg.VLM_OUTPUT_DIR)

    test_name = get_test_name()
    return PROJECT_DIR / "data" / "d455_capture" / "tests" / test_name / "vlm_output"


def get_post_output_dir():
    if hasattr(cfg, "POST_OUTPUT_DIR"):
        return as_path(cfg.POST_OUTPUT_DIR)

    test_name = get_test_name()
    return PROJECT_DIR / "data" / "d455_capture" / "tests" / test_name / "post_check_output"


# ============================================================
# COPY VISION OUTPUTS PER STEP
# ============================================================

def copy_vision_outputs_for_step(step_index=None):
    """
    This copies the latest normal pipeline outputs into step-specific names.

    Example:
    yolo_world_result.jpg
      -> STEP_1_yolo_world_result.jpg

    fastsam_result.jpg
      -> STEP_1_fastsam_result.jpg

    detections_yolo.json
      -> STEP_1_detections_yolo.json
    """

    if step_index is None:
        step_index = get_step_index()

    step_index = int(step_index)
    step_prefix = f"STEP_{step_index}"

    test_name = get_test_name()
    vision_dir = get_vision_output_dir()
    vlm_dir = get_vlm_output_dir()

    vision_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== STEP VISION OUTPUT COPY ==========")
    print("time:", now_text())
    print("test_name:", test_name)
    print("step_index:", step_index)
    print("vision_dir:", vision_dir)
    print("vlm_dir:", vlm_dir)
    print("============================================")

    copied = {}

    # Main vision images
    copied["yolo_world_result"] = copy_if_exists(
        getattr(cfg, "YOLO_RESULT_IMAGE", vision_dir / "yolo_world_result.jpg"),
        vision_dir / f"{step_prefix}_yolo_world_result.jpg",
        "YOLO result image",
    )

    copied["fastsam_result"] = copy_if_exists(
        getattr(cfg, "FASTSAM_RESULT_IMAGE", vision_dir / "fastsam_result.jpg"),
        vision_dir / f"{step_prefix}_fastsam_result.jpg",
        "FastSAM result image",
    )

    copied["fastsam_mask"] = copy_if_exists(
        getattr(cfg, "FASTSAM_MASK_PATH", vision_dir / "fastsam_mask.png"),
        vision_dir / f"{step_prefix}_fastsam_mask.png",
        "FastSAM mask",
    )

    # Main vision JSON
    copied["detections_yolo"] = copy_if_exists(
        getattr(cfg, "YOLO_DETECTIONS_JSON", vision_dir / "detections_yolo.json"),
        vision_dir / f"{step_prefix}_detections_yolo.json",
        "YOLO detections JSON",
    )

    copied["object_position_camera"] = copy_if_exists(
        getattr(cfg, "OBJECT_POSITION_JSON", vision_dir / "object_position_camera.json"),
        vision_dir / f"{step_prefix}_object_position_camera.json",
        "Object position camera JSON",
    )

    # VLM JSON
    copied["action_plan_real"] = copy_if_exists(
        getattr(cfg, "ACTION_PLAN_JSON", vlm_dir / "action_plan_real.json"),
        vlm_dir / f"{step_prefix}_action_plan_real.json",
        "Action plan JSON",
    )

    copied["validation_result_real"] = copy_if_exists(
        getattr(cfg, "VALIDATION_JSON", vlm_dir / "validation_result_real.json"),
        vlm_dir / f"{step_prefix}_validation_result_real.json",
        "Validation JSON",
    )

    copied["detections_from_vlm"] = copy_if_exists(
        getattr(cfg, "VLM_DETECTIONS_JSON", vlm_dir / "detections_from_vlm.json"),
        vlm_dir / f"{step_prefix}_detections_from_vlm.json",
        "VLM detections JSON",
    )

    # Grasp JSON files usually generated inside vision_output
    copied["best_grasp_camera"] = copy_if_exists(
        vision_dir / "best_grasp_camera.json",
        vision_dir / f"{step_prefix}_best_grasp_camera.json",
        "Best grasp camera JSON",
    )

    copied["best_grasp_base"] = copy_if_exists(
        vision_dir / "best_grasp_base.json",
        vision_dir / f"{step_prefix}_best_grasp_base.json",
        "Best grasp base JSON",
    )

    copied["grasp_candidates_camera"] = copy_if_exists(
        vision_dir / "grasp_candidates_camera.json",
        vision_dir / f"{step_prefix}_grasp_candidates_camera.json",
        "Grasp candidates camera JSON",
    )

    copied["tool0_pregrasp_target"] = copy_if_exists(
        vision_dir / "tool0_pregrasp_target.json",
        vision_dir / f"{step_prefix}_tool0_pregrasp_target.json",
        "Tool0 pregrasp target JSON",
    )

    # Optional debug files
    copied["pointcloud_valid_info"] = copy_if_exists(
        vision_dir / "masked_object_pointcloud_valid_only_info.json",
        vision_dir / f"{step_prefix}_masked_object_pointcloud_valid_only_info.json",
        "Pointcloud valid info JSON",
    )

    copied["pointcloud_median_info"] = copy_if_exists(
        vision_dir / "masked_object_pointcloud_median_fill_info.json",
        vision_dir / f"{step_prefix}_masked_object_pointcloud_median_fill_info.json",
        "Pointcloud median info JSON",
    )

    report = {
        "test_name": test_name,
        "step_index": step_index,
        "step_prefix": step_prefix,
        "time": now_text(),
        "vision_dir": str(vision_dir),
        "vlm_dir": str(vlm_dir),
        "copied": copied,
    }

    save_json(vision_dir / f"{step_prefix}_copy_report.json", report)

    print("========== STEP VISION OUTPUT COPY DONE ==========\n")
    return report


# ============================================================
# COPY POST-CHECK OUTPUTS PER STEP
# ============================================================

def copy_post_outputs_for_step(step_index=None):
    """
    This copies post-check outputs into consistent step-specific names.

    Example:
    STEP_1_post_check_result.json
    STEP_1_post_check_yolo_result.jpg
    STEP_1_remaining_plan.json
    """

    if step_index is None:
        step_index = get_step_index()

    step_index = int(step_index)
    step_prefix = f"STEP_{step_index}"

    test_name = get_test_name()
    post_dir = get_post_output_dir()

    post_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== STEP POST OUTPUT COPY ==========")
    print("time:", now_text())
    print("test_name:", test_name)
    print("step_index:", step_index)
    print("post_dir:", post_dir)
    print("==========================================")

    copied = {}

    copied["post_check_json"] = copy_if_exists(
        getattr(cfg, "POST_CHECK_JSON", post_dir / f"{step_prefix}_post_check_result.json"),
        post_dir / f"{step_prefix}_post_check_result.json",
        "Post-check result JSON",
    )

    copied["post_yolo_json"] = copy_if_exists(
        getattr(cfg, "POST_YOLO_JSON", post_dir / f"{step_prefix}_post_yolo_detections.json"),
        post_dir / f"{step_prefix}_post_yolo_detections.json",
        "Post YOLO detections JSON",
    )

    copied["post_yolo_image"] = copy_if_exists(
        getattr(cfg, "POST_YOLO_IMAGE", post_dir / f"{step_prefix}_post_check_yolo_result.jpg"),
        post_dir / f"{step_prefix}_post_check_yolo_result.jpg",
        "Post-check YOLO image",
    )

    copied["post_fastsam_mask"] = copy_if_exists(
        getattr(cfg, "POST_FASTSAM_MASK", post_dir / f"{step_prefix}_post_fastsam_mask.png"),
        post_dir / f"{step_prefix}_post_fastsam_mask.png",
        "Post FastSAM mask",
    )

    copied["post_fastsam_image"] = copy_if_exists(
        getattr(cfg, "POST_FASTSAM_IMAGE", post_dir / f"{step_prefix}_post_fastsam_result.jpg"),
        post_dir / f"{step_prefix}_post_fastsam_result.jpg",
        "Post FastSAM image",
    )

    copied["remaining_plan"] = copy_if_exists(
        getattr(cfg, "REMAINING_PLAN_JSON", post_dir / f"{step_prefix}_remaining_plan.json"),
        post_dir / f"{step_prefix}_remaining_plan.json",
        "Remaining plan JSON",
    )

    copied["sync_report"] = copy_if_exists(
        post_dir / f"{step_prefix}_sync_report.json",
        post_dir / f"{step_prefix}_sync_report.json",
        "Sync report JSON",
    )

    report = {
        "test_name": test_name,
        "step_index": step_index,
        "step_prefix": step_prefix,
        "time": now_text(),
        "post_dir": str(post_dir),
        "copied": copied,
    }

    save_json(post_dir / f"{step_prefix}_post_copy_report.json", report)

    print("========== STEP POST OUTPUT COPY DONE ==========\n")
    return report


# ============================================================
# CLI USAGE
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["vision", "post", "both"],
        help="Which outputs to copy.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step index. If omitted, uses STEP_INDEX env/capture_config.",
    )

    args = parser.parse_args()

    if args.mode in ["vision", "both"]:
        copy_vision_outputs_for_step(args.step)

    if args.mode in ["post", "both"]:
        copy_post_outputs_for_step(args.step)
