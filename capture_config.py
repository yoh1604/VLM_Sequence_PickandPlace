import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================
# File ini berada di root project:
# /home/b401/Documents/pick_place_occlusion_noetic/capture_config.py
#
# Jadi PROJECT_DIR otomatis menjadi:
# /home/b401/Documents/pick_place_occlusion_noetic

PROJECT_DIR = Path(__file__).resolve().parent


# .env berada di root project
ENV_PATH = PROJECT_DIR / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# TEST CONFIG
# ============================================================
# Ganti ini saja setiap test.

TEST_NAME = "soda_valid_09"
USER_QUERY = "I want to drink coca cola"
# USER_QUERY = "I want to eat cereal with milk. Please find the milk."

# STEP yang sedang diproses untuk post-check
STEP_INDEX = 2


# ============================================================
# BASE DATA PATH
# ============================================================
# Input utama dari kamera berada di:
# /home/b401/Documents/pick_place_occlusion_noetic/data/d455_capture

BASE_DIR = PROJECT_DIR / "data" / "d455_capture"

# Input utama dari kamera
IMAGE_PATH = BASE_DIR / "current_scene_rgb.jpg"
DEPTH_PATH = BASE_DIR / "depth_raw.npy"
INTRINSICS_PATH = BASE_DIR / "camera_intrinsics.json"
DEPTH_VIS_PATH = BASE_DIR / "current_scene_depth.png"

# Post-check image
POST_IMAGE_PATH = BASE_DIR / "post_scene_rgb.jpg"

# ============================================================
# MODEL PATHS
# ============================================================

MODELS_DIR = Path(PROJECT_DIR) / "models"

YOLO_WORLD_MODEL_PATH = MODELS_DIR / "yolov8l-worldv2.pt"
FASTSAM_MODEL_PATH = MODELS_DIR / "FastSAM-s.pt"


# ============================================================
# PER-TEST OUTPUT FOLDER
# ============================================================
# Output hasil pipeline disimpan di:
# /home/b401/Documents/pick_place_occlusion_noetic/outputs/data/d455_capture/tests/<TEST_NAME>

OUTPUT_BASE_DIR = PROJECT_DIR / "outputs"
TEST_OUTPUT_DIR = OUTPUT_BASE_DIR / TEST_NAME

VLM_OUTPUT_DIR = TEST_OUTPUT_DIR / "vlm_output"
VISION_OUTPUT_DIR = TEST_OUTPUT_DIR / "vision_output"
POST_OUTPUT_DIR = TEST_OUTPUT_DIR / "post_check_output"

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(VLM_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISION_OUTPUT_DIR, exist_ok=True)
os.makedirs(POST_OUTPUT_DIR, exist_ok=True)


# ============================================================
# MAIN PIPELINE OUTPUTS
# ============================================================

ACTION_PLAN_JSON = VLM_OUTPUT_DIR / "action_plan_real.json"
VLM_DETECTIONS_JSON = VLM_OUTPUT_DIR / "detections_from_vlm.json"
VALIDATION_JSON = VLM_OUTPUT_DIR / "validation_result_real.json"

YOLO_DETECTIONS_JSON = VISION_OUTPUT_DIR / "detections_yolo.json"
YOLO_RESULT_IMAGE = VISION_OUTPUT_DIR / "yolo_world_result.jpg"

FASTSAM_MASK_PATH = VISION_OUTPUT_DIR / "fastsam_mask.png"
FASTSAM_RESULT_IMAGE = VISION_OUTPUT_DIR / "fastsam_result.jpg"

OBJECT_POSITION_JSON = VISION_OUTPUT_DIR / "object_position_camera.json"


# ============================================================
# POST-CHECK OUTPUTS
# ============================================================

STEP_PREFIX = f"STEP_{STEP_INDEX}"

POST_CHECK_JSON = POST_OUTPUT_DIR / f"{STEP_PREFIX}_post_check_result.json"
POST_YOLO_JSON = POST_OUTPUT_DIR / f"{STEP_PREFIX}_post_check_detections_yolo.json"
POST_YOLO_IMAGE = POST_OUTPUT_DIR / f"{STEP_PREFIX}_post_check_yolo_result.jpg"
POST_FASTSAM_MASK = POST_OUTPUT_DIR / f"{STEP_PREFIX}_post_check_fastsam_mask.png"
POST_FASTSAM_IMAGE = POST_OUTPUT_DIR / f"{STEP_PREFIX}_post_check_fastsam_result.jpg"
REMAINING_PLAN_JSON = POST_OUTPUT_DIR / f"{STEP_PREFIX}_remaining_plan.json"


# ============================================================
# OPTIONAL SNAPSHOT OUTPUTS
# ============================================================

SNAPSHOT_DIR = TEST_OUTPUT_DIR / "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

SNAPSHOT_CURRENT_RGB = SNAPSHOT_DIR / "current_scene_rgb.jpg"
SNAPSHOT_CURRENT_DEPTH = SNAPSHOT_DIR / "current_scene_depth.png"
SNAPSHOT_POST_RGB = SNAPSHOT_DIR / "post_scene_rgb.jpg"


# ============================================================
# CONVERT Path OBJECTS TO STRINGS
# ============================================================
# Supaya tetap kompatibel dengan kode lama yang memakai string path.

PROJECT_DIR = str(PROJECT_DIR)
ENV_PATH = str(ENV_PATH)

BASE_DIR = str(BASE_DIR)
OUTPUT_BASE_DIR = str(OUTPUT_BASE_DIR)
TEST_OUTPUT_DIR = str(TEST_OUTPUT_DIR)

VLM_OUTPUT_DIR = str(VLM_OUTPUT_DIR)
VISION_OUTPUT_DIR = str(VISION_OUTPUT_DIR)
POST_OUTPUT_DIR = str(POST_OUTPUT_DIR)

IMAGE_PATH = str(IMAGE_PATH)
DEPTH_PATH = str(DEPTH_PATH)
INTRINSICS_PATH = str(INTRINSICS_PATH)
DEPTH_VIS_PATH = str(DEPTH_VIS_PATH)
POST_IMAGE_PATH = str(POST_IMAGE_PATH)

MODELS_DIR = str(MODELS_DIR)
YOLO_WORLD_MODEL_PATH = str(YOLO_WORLD_MODEL_PATH)
FASTSAM_MODEL_PATH = str(FASTSAM_MODEL_PATH)

ACTION_PLAN_JSON = str(ACTION_PLAN_JSON)
VLM_DETECTIONS_JSON = str(VLM_DETECTIONS_JSON)
VALIDATION_JSON = str(VALIDATION_JSON)

YOLO_DETECTIONS_JSON = str(YOLO_DETECTIONS_JSON)
YOLO_RESULT_IMAGE = str(YOLO_RESULT_IMAGE)

FASTSAM_MASK_PATH = str(FASTSAM_MASK_PATH)
FASTSAM_RESULT_IMAGE = str(FASTSAM_RESULT_IMAGE)

OBJECT_POSITION_JSON = str(OBJECT_POSITION_JSON)

POST_CHECK_JSON = str(POST_CHECK_JSON)
POST_YOLO_JSON = str(POST_YOLO_JSON)
POST_YOLO_IMAGE = str(POST_YOLO_IMAGE)
POST_FASTSAM_MASK = str(POST_FASTSAM_MASK)
POST_FASTSAM_IMAGE = str(POST_FASTSAM_IMAGE)
REMAINING_PLAN_JSON = str(REMAINING_PLAN_JSON)

SNAPSHOT_DIR = str(SNAPSHOT_DIR)
SNAPSHOT_CURRENT_RGB = str(SNAPSHOT_CURRENT_RGB)
SNAPSHOT_CURRENT_DEPTH = str(SNAPSHOT_CURRENT_DEPTH)
SNAPSHOT_POST_RGB = str(SNAPSHOT_POST_RGB)

BEST_GRASP_CAMERA_JSON = f"{VISION_OUTPUT_DIR}/best_grasp_camera.json"
GRASP_CANDIDATES_CAMERA_JSON = f"{VISION_OUTPUT_DIR}/grasp_candidates_camera.json"

BEST_GRASP_BASE_JSON = f"{VISION_OUTPUT_DIR}/best_grasp_base.json"
TOOL0_PREGRASP_TARGET_JSON = f"{VISION_OUTPUT_DIR}/tool0_pregrasp_target.json"

# ============================================================
# DEBUG PRINT
# ============================================================

def print_config():
    print("\n========== D455 TEST CONFIG ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("ENV_PATH:", ENV_PATH)
    print("TEST_NAME:", TEST_NAME)
    print("USER_QUERY:", USER_QUERY)
    print("STEP_INDEX:", STEP_INDEX)

    print("\n========== INPUT PATHS ==========")
    print("BASE_DIR:", BASE_DIR)
    print("IMAGE_PATH:", IMAGE_PATH)
    print("DEPTH_PATH:", DEPTH_PATH)
    print("INTRINSICS_PATH:", INTRINSICS_PATH)
    print("DEPTH_VIS_PATH:", DEPTH_VIS_PATH)
    print("POST_IMAGE_PATH:", POST_IMAGE_PATH)

    print("\n========== OUTPUT PATHS ==========")
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("TEST_OUTPUT_DIR:", TEST_OUTPUT_DIR)
    print("VLM_OUTPUT_DIR:", VLM_OUTPUT_DIR)
    print("VISION_OUTPUT_DIR:", VISION_OUTPUT_DIR)
    print("POST_OUTPUT_DIR:", POST_OUTPUT_DIR)

    print("\n========== MAIN OUTPUT FILES ==========")
    print("ACTION_PLAN_JSON:", ACTION_PLAN_JSON)
    print("VLM_DETECTIONS_JSON:", VLM_DETECTIONS_JSON)
    print("VALIDATION_JSON:", VALIDATION_JSON)
    print("YOLO_DETECTIONS_JSON:", YOLO_DETECTIONS_JSON)
    print("YOLO_RESULT_IMAGE:", YOLO_RESULT_IMAGE)
    print("FASTSAM_MASK_PATH:", FASTSAM_MASK_PATH)
    print("FASTSAM_RESULT_IMAGE:", FASTSAM_RESULT_IMAGE)
    print("OBJECT_POSITION_JSON:", OBJECT_POSITION_JSON)

    print("\n========== POST-CHECK OUTPUT FILES ==========")
    print("POST_CHECK_JSON:", POST_CHECK_JSON)
    print("POST_YOLO_JSON:", POST_YOLO_JSON)
    print("POST_YOLO_IMAGE:", POST_YOLO_IMAGE)
    print("POST_FASTSAM_MASK:", POST_FASTSAM_MASK)
    print("POST_FASTSAM_IMAGE:", POST_FASTSAM_IMAGE)
    print("REMAINING_PLAN_JSON:", REMAINING_PLAN_JSON)

    print("\n========== SNAPSHOT OUTPUTS ==========")
    print("SNAPSHOT_DIR:", SNAPSHOT_DIR)
    print("SNAPSHOT_CURRENT_RGB:", SNAPSHOT_CURRENT_RGB)
    print("SNAPSHOT_CURRENT_DEPTH:", SNAPSHOT_CURRENT_DEPTH)
    print("SNAPSHOT_POST_RGB:", SNAPSHOT_POST_RGB)

    print("=====================================\n")