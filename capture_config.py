import os
from dotenv import load_dotenv


PROJECT_DIR = "/home/b401/Documents/pick_place_occlusion"
ENV_PATH = f"{PROJECT_DIR}/.env"

load_dotenv(ENV_PATH)

# ============================================================
# TEST CONFIG
# ============================================================
# Ganti ini saja setiap test.

TEST_NAME = "test_07_try_milk"
USER_QUERY = "I want to eat cereal with milk. Please find the milk."

# STEP yang sedang diproses untuk post-check
STEP_INDEX = 2


# ============================================================
# BASE DATA PATH
# ============================================================

BASE_DIR = f"{PROJECT_DIR}/data/d455_capture"

# Input utama dari kamera
IMAGE_PATH = f"{BASE_DIR}/current_scene_rgb.jpg"
DEPTH_PATH = f"{BASE_DIR}/depth_raw.npy"
INTRINSICS_PATH = f"{BASE_DIR}/camera_intrinsics.json"
DEPTH_VIS_PATH = f"{BASE_DIR}/current_scene_depth.png"

# Post-check image
POST_IMAGE_PATH = f"{BASE_DIR}/post_scene_rgb.jpg"


# ============================================================
# PER-TEST OUTPUT FOLDER
# ============================================================

TEST_OUTPUT_DIR = f"{BASE_DIR}/tests/{TEST_NAME}"

VLM_OUTPUT_DIR = f"{TEST_OUTPUT_DIR}/vlm_output"
VISION_OUTPUT_DIR = f"{TEST_OUTPUT_DIR}/vision_output"
POST_OUTPUT_DIR = f"{TEST_OUTPUT_DIR}/post_check_output"

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(VLM_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISION_OUTPUT_DIR, exist_ok=True)
os.makedirs(POST_OUTPUT_DIR, exist_ok=True)


# ============================================================
# MAIN PIPELINE OUTPUTS
# ============================================================

ACTION_PLAN_JSON = f"{VLM_OUTPUT_DIR}/action_plan_real.json"
VLM_DETECTIONS_JSON = f"{VLM_OUTPUT_DIR}/detections_from_vlm.json"
VALIDATION_JSON = f"{VLM_OUTPUT_DIR}/validation_result_real.json"

YOLO_DETECTIONS_JSON = f"{VISION_OUTPUT_DIR}/detections_yolo.json"
YOLO_RESULT_IMAGE = f"{VISION_OUTPUT_DIR}/yolo_world_result.jpg"

FASTSAM_MASK_PATH = f"{VISION_OUTPUT_DIR}/fastsam_mask.png"
FASTSAM_RESULT_IMAGE = f"{VISION_OUTPUT_DIR}/fastsam_result.jpg"

OBJECT_POSITION_JSON = f"{VISION_OUTPUT_DIR}/object_position_camera.json"


# ============================================================
# POST-CHECK OUTPUTS
# ============================================================

STEP_PREFIX = f"STEP_{STEP_INDEX}"

POST_CHECK_JSON = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_post_check_result.json"
POST_YOLO_JSON = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_post_check_detections_yolo.json"
POST_YOLO_IMAGE = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_post_check_yolo_result.jpg"
POST_FASTSAM_MASK = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_post_check_fastsam_mask.png"
POST_FASTSAM_IMAGE = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_post_check_fastsam_result.jpg"
REMAINING_PLAN_JSON = f"{POST_OUTPUT_DIR}/{STEP_PREFIX}_remaining_plan.json"


# ============================================================
# OPTIONAL SNAPSHOT OUTPUTS
# ============================================================

SNAPSHOT_DIR = f"{TEST_OUTPUT_DIR}/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

SNAPSHOT_CURRENT_RGB = f"{SNAPSHOT_DIR}/current_scene_rgb.jpg"
SNAPSHOT_CURRENT_DEPTH = f"{SNAPSHOT_DIR}/current_scene_depth.png"
SNAPSHOT_POST_RGB = f"{SNAPSHOT_DIR}/post_scene_rgb.jpg"


# ============================================================
# DEBUG PRINT
# ============================================================

def print_config():
    print("\n========== D455 TEST CONFIG ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("TEST_NAME:", TEST_NAME)
    print("USER_QUERY:", USER_QUERY)
    print("STEP_INDEX:", STEP_INDEX)
    print("BASE_DIR:", BASE_DIR)
    print("TEST_OUTPUT_DIR:", TEST_OUTPUT_DIR)
    print("IMAGE_PATH:", IMAGE_PATH)
    print("POST_IMAGE_PATH:", POST_IMAGE_PATH)
    print("ACTION_PLAN_JSON:", ACTION_PLAN_JSON)
    print("VALIDATION_JSON:", VALIDATION_JSON)
    print("YOLO_DETECTIONS_JSON:", YOLO_DETECTIONS_JSON)
    print("FASTSAM_RESULT_IMAGE:", FASTSAM_RESULT_IMAGE)
    print("OBJECT_POSITION_JSON:", OBJECT_POSITION_JSON)
    print("POST_CHECK_JSON:", POST_CHECK_JSON)
    print("=====================================\n")