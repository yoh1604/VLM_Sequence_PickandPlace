#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FULL MULTI STEP PICK-PLACE PIPELINE
#
# STEP 1:
#   1. Capture D455 current scene
#   2. run_d455_pipeline.py
#   3. convert gripper_tip -> tool0 target
#   4. apply nudge
#   5. grasp.py
#   6. discard
#
# STEP 2 dst:
#   1. Capture D455 current scene after discard
#   2. copy current_scene_rgb.jpg -> post_scene_rgb.jpg
#   3. run_post_complete.py
#   4. GraspNet for next target
#   5. transform grasp camera -> base
#   6. convert gripper_tip -> tool0 target
#   7. apply nudge
#   8. grasp.py
#   9. discard
#
# Catatan:
# - Post script yang digunakan: run_post_complete.py
# - Semua RGB/depth utama tetap dari data/d455_capture/
# - post_scene_rgb.jpg hanya copy dari current_scene_rgb.jpg terbaru
# - Nudge diterapkan setelah convert tool0 target
# ============================================================


# ============================================================
# BASIC CONFIG
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/pick_place_occlusion_noetic}"

PLANNER_ENV="${PLANNER_ENV:-ur5_pickplace}"
ANYGRASP_ENV="${ANYGRASP_ENV:-anygrasp_py310}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CATKIN_SETUP="${CATKIN_SETUP:-$HOME/ur5_noetic_ws/devel/setup.bash}"

ROBOT_IP="${ROBOT_IP:-192.168.200.1}"

# Usage:
#   ./scripts/run_full_multi_pipeline.sh plan
#   ./scripts/run_full_multi_pipeline.sh execute
MODE="${1:-plan}"

case "$MODE" in
  plan)
    EXECUTE_FLAG=""
    ;;
  execute)
    EXECUTE_FLAG="--execute"
    ;;
  *)
    echo "[ERROR] MODE harus plan atau execute"
    echo "Contoh:"
    echo "  ./scripts/run_full_multi_pipeline.sh plan"
    echo "  ./scripts/run_full_multi_pipeline.sh execute"
    exit 1
    ;;
esac


# ============================================================
# SCRIPT PATHS
# ============================================================

CAPTURE_SCRIPT="${CAPTURE_SCRIPT:-perception/capture_d455_once.py}"
VISION_SCRIPT="${VISION_SCRIPT:-run_d455_pipeline.py}"

# INI YANG BENAR UNTUK POST
POST_SCRIPT="${POST_SCRIPT:-run_post_complete.py}"

GRASPNET_SCRIPT="${GRASPNET_SCRIPT:-models/graspnet-baseline/demo_d455.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-models/graspnet-baseline/logs/log_rs/checkpoint.tar}"

TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT:-perception/transform_grasp_to_base.py}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-perception/convert_gripper_tip_to_tool0_target.py}"
NUDGE_SCRIPT="${NUDGE_SCRIPT:-robot_executor/nudge_tool0_target.py}"

GRASP_SCRIPT="${GRASP_SCRIPT:-robot_executor/grasp.py}"
DISCARD_SCRIPT="${DISCARD_SCRIPT:-robot_executor/execute_from_current_pregrasp_to_discard.py}"
WAYPOINTS_JSON="${WAYPOINTS_JSON:-configs/waypoints_ur5.json}"


# ============================================================
# TUNING PARAMS
# ============================================================

# Offset tool0 ke ujung gripper.
TOOL0_TO_GRIPPER_TIP="${TOOL0_TO_GRIPPER_TIP:-0 0 0.17}"

# Tinggi pre-grasp di atas grasp point.
PREGRASP_Z="${PREGRASP_Z:-0.10}"

# Jarak turun dari pre-grasp ke grasp.
DESCEND_Z="${DESCEND_Z:-0.10}"

# Wait di pre-grasp setelah gripper open sebelum descend.
PREGRASP_WAIT="${PREGRASP_WAIT:-5.0}"

# Nudge hasil tuning kamu.
# Satuan meter.
NUDGE_DX="${NUDGE_DX:--0.05}"
NUDGE_DY="${NUDGE_DY:-0.037}"
NUDGE_DZ="${NUDGE_DZ:-0.0}"

# Lift setelah objek tergenggam sebelum discard.
LIFT_UP="${LIFT_UP:-0.03}"

# Robot movement scale.
VELOCITY="${VELOCITY:-0.05}"
ACCELERATION="${ACCELERATION:-0.05}"

# Jumlah maksimum step pick-place.
MAX_STEPS="${MAX_STEPS:-2}"

# Default skip observation agar tidak menambah titik gagal.
# Kalau ingin singgah OBSERVATION:
#   SKIP_OBSERVATION=0 ./scripts/run_full_multi_pipeline.sh execute
SKIP_OBSERVATION="${SKIP_OBSERVATION:-0}"

# Kalau 1, robot tidak dijalankan. Hanya perception + target generation.
DISABLE_GRIPPER="${DISABLE_GRIPPER:-0}"
ORIENTATION_MODE="${ORIENTATION_MODE:-current}"
USE_SHOULDER_CONSTRAINT="${USE_SHOULDER_CONSTRAINT:-1}"
SHOULDER_TOLERANCE="${SHOULDER_TOLERANCE:-0.6}"

DRY_RUN_ROBOT="${DRY_RUN_ROBOT:-0}"


# ============================================================
# HELPERS
# ============================================================

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

need_file() {
  local path="$1"
  local label="$2"

  if [ ! -f "$path" ]; then
    echo "[ERROR] $label tidak ditemukan:"
    echo "  $path"
    exit 1
  fi

  echo "[OK] $label:"
  echo "  $path"
}

activate_conda() {
  local env_name="$1"

  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  else
    echo "[ERROR] conda.sh tidak ditemukan."
    echo "Cek:"
    echo "  conda info --base"
    exit 1
  fi

  set +u
  conda activate "$env_name"
  set -u

  echo "[INFO] Active env: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "[INFO] Python    : $(which python)"
  python --version
}

deactivate_all_conda() {
  set +u
  conda deactivate >/dev/null 2>&1 || true
  conda deactivate >/dev/null 2>&1 || true
  conda deactivate >/dev/null 2>&1 || true
  set -u

  unset CONDA_EXE || true
  unset CONDA_PREFIX || true
  unset CONDA_PREFIX_1 || true
  unset CONDA_PREFIX_2 || true
  unset CONDA_DEFAULT_ENV || true
  unset CONDA_PROMPT_MODIFIER || true
  unset CONDA_PYTHON_EXE || true
  unset _CE_CONDA || true
  unset _CE_M || true

  unset PYTHONPATH || true
  unset PYTHONHOME || true
  unset LD_LIBRARY_PATH || true
  unset PKG_CONFIG_PATH || true
  unset CMAKE_PREFIX_PATH || true

  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  hash -r
}

get_test_name_from_capture_config() {
  cd "$PROJECT_DIR"

  local output
  local status

  set +e
  output="$(
    conda run -n "$PLANNER_ENV" python - <<'PY'
import sys
from pathlib import Path

project = Path.cwd()
sys.path.insert(0, str(project))

try:
    import capture_config as cfg
    name = str(cfg.TEST_NAME).strip()
    if not name:
        raise RuntimeError("capture_config.TEST_NAME kosong")
    print(name)
except Exception as e:
    print(f"ERROR_IMPORT_CAPTURE_CONFIG:{repr(e)}")
    sys.exit(1)
PY
  )"
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    echo "$output"
    return 1
  fi

  echo "$output" | awk 'NF {line=$0} END {print line}'
}

get_remaining_count() {
  local remaining_json="$1"

  if [ ! -f "$remaining_json" ]; then
    echo "999"
    return
  fi

  python3 - <<PY
import json

p = "$remaining_json"

try:
    with open(p) as f:
        data = json.load(f)

    if isinstance(data, list):
        print(len(data))
    elif isinstance(data, dict) and "remaining_plan" in data:
        rp = data.get("remaining_plan")
        print(len(rp) if isinstance(rp, list) else 0)
    else:
        print(0)

except Exception:
    print(999)
PY
}

check_teach_pendant() {
  if [ "$DRY_RUN_ROBOT" = "1" ]; then
    return
  fi

  log_section "TEACH PENDANT CHECK"

  echo "Pastikan sebelum robot bergerak:"
  echo "1. External Control program aktif."
  echo "2. Tombol Play sudah ditekan."
  echo "3. Robot tidak protective stop."
  echo "4. Speed slider > 0."
  echo ""

#   if [ "$MODE" = "execute" ]; then
#     read -r -p "Sudah Play di teach pendant? ketik y lalu ENTER: " ans
#     if [ "$ans" != "y" ]; then
#       echo "[STOP] Tekan Play dulu di teach pendant."
#       exit 1
#     fi
#   fi
}

print_input_files() {
  log_section "INPUT FILE CHECK"

  ls -lh "$RGB_PATH" || true
  ls -lh "$POST_RGB_PATH" || true
  ls -lh "$DEPTH_PATH" || true
  ls -lh "$INTRINSICS_PATH" || true
  ls -lh "$MASK_PATH" || true
  ls -lh "$TOOL0_TARGET_JSON" || true
}

safe_target_check() {
  local target_json="$1"

  python3 - <<PY
import json
import sys

p = "$target_json"

with open(p) as f:
    d = json.load(f)

pos = d.get("translation_tool0_pregrasp")
quat = d.get("quaternion_tool0_xyzw")
nudge = d.get("manual_nudge_base_m")

print("[TARGET CHECK] translation_tool0_pregrasp:", pos)
print("[TARGET CHECK] quaternion_tool0_xyzw:", quat)
print("[TARGET CHECK] manual_nudge_base_m:", nudge)

if not pos or len(pos) != 3:
    print("[ERROR] translation_tool0_pregrasp tidak valid.")
    sys.exit(1)

x, y, z = [float(v) for v in pos]

warnings = []

if z < 0.18:
    warnings.append("Z pregrasp terlalu rendah (<0.18 m)")
if z > 0.70:
    warnings.append("Z pregrasp terlalu tinggi (>0.70 m)")
if x < -1.10 or x > 0.30:
    warnings.append("X target mencurigakan")
if abs(y) > 0.80:
    warnings.append("Y target mencurigakan")

if warnings:
    print("[WARN] Target mencurigakan:")
    for w in warnings:
        print(" -", w)
    print("[WARN] Cek RViz/fisik sebelum execute.")
else:
    print("[OK] Target kasar masuk akal.")
PY
}


# ============================================================
# TEST NAME + PATHS
# ============================================================

cd "$PROJECT_DIR"

if [ -n "${TEST_NAME:-}" ]; then
  echo "[INFO] TEST_NAME dari environment: $TEST_NAME"
else
  echo "[INFO] Membaca TEST_NAME dari capture_config.py ..."
  if ! TEST_NAME="$(get_test_name_from_capture_config)"; then
    echo "[ERROR] Gagal membaca TEST_NAME dari capture_config.py."
    echo "$TEST_NAME"
    echo ""
    echo "Solusi:"
    echo "  export TEST_NAME=nama_test_kamu"
    echo "  ./scripts/run_full_multi_pipeline.sh execute"
    exit 1
  fi
fi

if [ -z "$TEST_NAME" ]; then
  echo "[ERROR] TEST_NAME kosong."
  exit 1
fi

export TEST_NAME="$TEST_NAME"

VISION_OUTPUT_DIR="$PROJECT_DIR/outputs/$TEST_NAME/vision_output"
POST_OUTPUT_DIR="$PROJECT_DIR/outputs/$TEST_NAME/post_check_output"
REMAINING_PLAN_JSON="$POST_OUTPUT_DIR/remaining_plan.json"

RGB_PATH="$PROJECT_DIR/data/d455_capture/current_scene_rgb.jpg"
POST_RGB_PATH="$PROJECT_DIR/data/d455_capture/post_scene_rgb.jpg"
DEPTH_PATH="$PROJECT_DIR/data/d455_capture/depth_raw.npy"
INTRINSICS_PATH="$PROJECT_DIR/data/d455_capture/camera_intrinsics.json"

MASK_PATH="$VISION_OUTPUT_DIR/fastsam_mask.png"
BEST_GRASP_CAMERA_JSON="$VISION_OUTPUT_DIR/best_grasp_camera.json"
BEST_GRASP_BASE_JSON="$VISION_OUTPUT_DIR/best_grasp_base.json"
TOOL0_TARGET_JSON="$VISION_OUTPUT_DIR/tool0_pregrasp_target.json"

mkdir -p "$VISION_OUTPUT_DIR"
mkdir -p "$POST_OUTPUT_DIR"


# ============================================================
# CONFIG PRINT
# ============================================================

log_section "FULL MULTI PIPELINE CONFIG"

echo "PROJECT_DIR              : $PROJECT_DIR"
echo "TEST_NAME                : $TEST_NAME"
echo "MODE                     : $MODE"
echo "MAX_STEPS                : $MAX_STEPS"
echo "ROBOT_IP                 : $ROBOT_IP"
echo "VISION_OUTPUT_DIR        : $VISION_OUTPUT_DIR"
echo "POST_OUTPUT_DIR          : $POST_OUTPUT_DIR"
echo "RGB_PATH                 : $RGB_PATH"
echo "POST_RGB_PATH            : $POST_RGB_PATH"
echo "DEPTH_PATH               : $DEPTH_PATH"
echo "MASK_PATH                : $MASK_PATH"
echo "TOOL0_TARGET_JSON        : $TOOL0_TARGET_JSON"
echo "POST_SCRIPT              : $POST_SCRIPT"
echo "TOOL0_TO_GRIPPER_TIP     : $TOOL0_TO_GRIPPER_TIP"
echo "PREGRASP_Z               : $PREGRASP_Z"
echo "DESCEND_Z                : $DESCEND_Z"
echo "PREGRASP_WAIT            : $PREGRASP_WAIT"
echo "NUDGE_DX                 : $NUDGE_DX"
echo "NUDGE_DY                 : $NUDGE_DY"
echo "NUDGE_DZ                 : $NUDGE_DZ"
echo "LIFT_UP                  : $LIFT_UP"
echo "SKIP_OBSERVATION         : $SKIP_OBSERVATION"
echo "DISABLE_GRIPPER          : $DISABLE_GRIPPER"
echo "ORIENTATION_MODE         : $ORIENTATION_MODE"
echo "USE_SHOULDER_CONSTRAINT  : $USE_SHOULDER_CONSTRAINT"
echo "SHOULDER_TOLERANCE       : $SHOULDER_TOLERANCE"
echo "DRY_RUN_ROBOT            : $DRY_RUN_ROBOT"


# ============================================================
# STAGE FUNCTIONS
# ============================================================

capture_current_scene() {
  log_section "CAPTURE CURRENT SCENE"

  activate_conda "$PLANNER_ENV"

  need_file "$CAPTURE_SCRIPT" "capture script"

  python "$CAPTURE_SCRIPT"

  need_file "$RGB_PATH" "current_scene_rgb.jpg"
  need_file "$DEPTH_PATH" "depth_raw.npy"
  need_file "$INTRINSICS_PATH" "camera_intrinsics.json"

  conda deactivate || true
}

run_first_pipeline() {
  log_section "STEP 1: RUN D455 PIPELINE"

  activate_conda "$PLANNER_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$VISION_SCRIPT" "run_d455_pipeline.py"

  python "$VISION_SCRIPT"

  need_file "$MASK_PATH" "fastsam_mask.png"
  need_file "$BEST_GRASP_CAMERA_JSON" "best_grasp_camera.json"

  # run_d455_pipeline.py kamu sudah menghasilkan best_grasp_base.json.
  need_file "$BEST_GRASP_BASE_JSON" "best_grasp_base.json"

  conda deactivate || true
}

run_post_complete_stage() {
  local completed_step_index="$1"
  log_section "POST CHECK + PREPARE NEXT TARGET FOR COMPLETED STEP ${completed_step_index}"

  activate_conda "$PLANNER_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$RGB_PATH" "current_scene_rgb.jpg"
  need_file "$DEPTH_PATH" "depth_raw.npy"
  need_file "$POST_SCRIPT" "run_post_complete.py"

  # Semua post memakai capture terbaru dari data/d455_capture.
  # post_scene_rgb.jpg hanya salinan current_scene_rgb.jpg terbaru.
  cp "$RGB_PATH" "$POST_RGB_PATH"

  need_file "$POST_RGB_PATH" "post_scene_rgb.jpg"

  python "$POST_SCRIPT" --step_index "$completed_step_index"

  need_file "$POST_OUTPUT_DIR/STEP_${completed_step_index}_sync_report.json" "sync report"
  need_file "$MASK_PATH" "fastsam_mask.png next target"
  need_file "$VISION_OUTPUT_DIR/object_position_camera.json" "object_position_camera.json next target"

  conda deactivate || true
}

run_graspnet_for_post_target() {
  log_section "GRASPNET FOR POST TARGET"

  activate_conda "$ANYGRASP_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$CHECKPOINT_PATH" "GraspNet checkpoint"
  need_file "$RGB_PATH" "current_scene_rgb.jpg"
  need_file "$DEPTH_PATH" "depth_raw.npy"
  need_file "$INTRINSICS_PATH" "camera_intrinsics.json"
  need_file "$MASK_PATH" "fastsam_mask.png"

  echo "[INFO] Running GraspNet for post target."
  echo "[INFO] Trying settings aligned with run_d455_pipeline.py."

  set +e
  python "$GRASPNET_SCRIPT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --test_name "$TEST_NAME" \
    --num_point 10000 \
    --num_view 300 \
    --collision_thresh -1 \
    --voxel_size 0.01 \
    --max_center_dist 0.10 \
    --mask_dilate_iter 1 \
    --no_vis
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    echo "[WARN] GraspNet extended args gagal. Coba fallback tanpa max_center_dist/mask_dilate_iter."

    python "$GRASPNET_SCRIPT" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --test_name "$TEST_NAME" \
      --num_point 10000 \
      --num_view 300 \
      --collision_thresh -1 \
      --voxel_size 0.01 \
      --no_vis
  fi

  need_file "$BEST_GRASP_CAMERA_JSON" "best_grasp_camera.json"

  conda deactivate || true
}

run_transform_for_post_target() {
  log_section "TRANSFORM GRASP CAMERA TO BASE"

  activate_conda "$PLANNER_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$TRANSFORM_SCRIPT" "transform script"
  need_file "$BEST_GRASP_CAMERA_JSON" "best_grasp_camera.json"

  python "$TRANSFORM_SCRIPT"

  need_file "$BEST_GRASP_BASE_JSON" "best_grasp_base.json"

  conda deactivate || true
}

run_convert_to_tool0_and_nudge() {
  log_section "CONVERT GRIPPER TIP TO TOOL0 TARGET + NUDGE"

  activate_conda "$PLANNER_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$CONVERT_SCRIPT" "convert script"
  need_file "$NUDGE_SCRIPT" "nudge script"
  need_file "$BEST_GRASP_BASE_JSON" "best_grasp_base.json"

  python "$CONVERT_SCRIPT" \
    --best_grasp_base "$BEST_GRASP_BASE_JSON" \
    --output "$TOOL0_TARGET_JSON" \
    --tool0_to_gripper_tip "$TOOL0_TO_GRIPPER_TIP" \
    --pregrasp_z "$PREGRASP_Z"

  need_file "$TOOL0_TARGET_JSON" "tool0_pregrasp_target.json sebelum nudge"

  python3 "$NUDGE_SCRIPT" \
    --target_json "$TOOL0_TARGET_JSON" \
    --dx "$NUDGE_DX" \
    --dy "$NUDGE_DY" \
    --dz "$NUDGE_DZ"

  need_file "$TOOL0_TARGET_JSON" "tool0_pregrasp_target.json setelah nudge"

  safe_target_check "$TOOL0_TARGET_JSON"

  conda deactivate || true
}

run_robot_pick_and_discard() {
  log_section "ROBOT PICK + DISCARD"

  if [ "$DRY_RUN_ROBOT" = "1" ]; then
    echo "[DRY_RUN_ROBOT=1] Skip robot execution."
    return
  fi

  check_teach_pendant

  deactivate_all_conda

  source "$ROS_SETUP"

  if [ -f "$CATKIN_SETUP" ]; then
    source "$CATKIN_SETUP"
  else
    echo "[WARN] CATKIN_SETUP tidak ditemukan: $CATKIN_SETUP"
  fi

  echo "[INFO] Python ROS: $(which python3)"
  python3 --version

  echo "[CHECK] moveit_commander import..."
  python3 - <<'PY'
import moveit_commander
print("moveit_commander OK")
PY

  need_file "$GRASP_SCRIPT" "grasp.py"
  need_file "$DISCARD_SCRIPT" "discard script"
  need_file "$WAYPOINTS_JSON" "waypoints json"
  need_file "$TOOL0_TARGET_JSON" "tool0 target"

  local observation_flag=""
  if [ "$SKIP_OBSERVATION" = "1" ]; then
    observation_flag="--skip_observation"
  fi

  local gripper_flag=""
  if [ "$DISABLE_GRIPPER" = "1" ]; then
    gripper_flag="--disable_gripper"
  fi

  local shoulder_flag=""
  if [ "$USE_SHOULDER_CONSTRAINT" = "1" ]; then
    shoulder_flag="--use_shoulder_constraint"
  fi

  log_section "PICK / GRASP"

  python3 "$GRASP_SCRIPT" \
    --target_json "$TOOL0_TARGET_JSON" \
    --robot_ip "$ROBOT_IP" \
    --waypoints_json "$WAYPOINTS_JSON" \
    --orientation_mode "$ORIENTATION_MODE" \
    --shoulder_tolerance "$SHOULDER_TOLERANCE" \
    --descend_z "$DESCEND_Z" \
    --pregrasp_wait "$PREGRASP_WAIT" \
    --velocity "$VELOCITY" \
    --acceleration "$ACCELERATION" \
    $shoulder_flag \
    $gripper_flag \
    $observation_flag \
    $EXECUTE_FLAG

  log_section "DISCARD"

  python3 "$DISCARD_SCRIPT" \
    --robot_ip "$ROBOT_IP" \
    --waypoints_json "$WAYPOINTS_JSON" \
    --lift_up "$LIFT_UP" \
    $EXECUTE_FLAG
}


# ============================================================
# MAIN FLOW
# ============================================================

print_input_files

# ------------------------------
# STEP 1
# ------------------------------

capture_current_scene
print_input_files

run_first_pipeline

# Step pertama:
# run_d455_pipeline.py sudah menghasilkan best_grasp_base.json.
# Maka cukup convert -> nudge -> robot.
run_convert_to_tool0_and_nudge

run_robot_pick_and_discard


# ------------------------------
# STEP 2 dst
# ------------------------------

for step_loop in $(seq 2 "$MAX_STEPS"); do
  log_section "NEXT STEP LOOP $step_loop"

  # Setelah discard, capture scene terbaru.
  capture_current_scene
  print_input_files

  # run_post_complete.py:
  # - verifikasi target sebelumnya
  # - siapkan target berikutnya
  # - post_scene_rgb.jpg = copy current_scene_rgb.jpg terbaru
  completed_step_index=$((step_loop - 1))
  run_post_complete_stage "$completed_step_index"
  print_input_files

  remaining_count="$(get_remaining_count "$REMAINING_PLAN_JSON")"
  echo "[INFO] Remaining plan count: $remaining_count"

  if [ "$remaining_count" = "0" ]; then
    log_section "ALL STEPS DONE"
    echo "[DONE] Semua step action plan sudah selesai."
    exit 0
  fi

  run_graspnet_for_post_target
  run_transform_for_post_target
  run_convert_to_tool0_and_nudge
  run_robot_pick_and_discard
done

log_section "STOPPED BY MAX_STEPS"
echo "[WARN] Berhenti karena MAX_STEPS=$MAX_STEPS"