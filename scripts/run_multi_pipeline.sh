#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MULTI-STEP PIPELINE WITH SAME-STEP RETRY
#
# Flow:
#   action step N
#   -> discard
#   -> capture post scene
#   -> post-check step N
#   -> if REMOVED_SUCCESS: continue to N+1
#   -> if STILL_FOUND: retry same step N, do NOT increment STEP_INDEX
#
# Important:
# - VLM/action_plan is not regenerated on retry.
# - Retry uses post-check mask/detection of the same target.
# - Next target still uses remaining_plan from run_post_complete.
# ============================================================


# ============================================================
# USER CONFIG
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/pick_place_occlusion_noetic}"
PLANNER_ENV="${PLANNER_ENV:-ur5_pickplace}"

TEST_NAME="${1:-water}"
MODE="${2:-plan}"

MAX_STEPS="${MAX_STEPS:-5}"
START_STEP="${START_STEP:-1}"
MAX_RETRIES_PER_STEP="${MAX_RETRIES_PER_STEP:-2}"

# Kalau 1: langsung post-check START_STEP, cocok kalau action sudah dilakukan manual.
RESUME_AFTER_ACTION="${RESUME_AFTER_ACTION:-0}"

# Planner / validator biasanya sudah ada. Set 1 kalau mau generate ulang.
RUN_PLANNER="${RUN_PLANNER:-0}"
RUN_VALIDATOR="${RUN_VALIDATOR:-0}"

# Capture setelah discard sebelum post-check.
CAPTURE_BEFORE_POST="${CAPTURE_BEFORE_POST:-1}"
POST_CAPTURE_DELAY_SEC="${POST_CAPTURE_DELAY_SEC:-2}"

# Core scripts
RUN_FIRST_SCRIPT="${RUN_FIRST_SCRIPT:-scripts/run_first_pipeline.sh}"
CAPTURE_SCRIPT="${CAPTURE_SCRIPT:-perception/capture_d455_once.py}"
POST_SCRIPT_FIXED="${POST_SCRIPT_FIXED:-run_post_complete_fixed.py}"
POST_SCRIPT_FALLBACK="${POST_SCRIPT_FALLBACK:-run_post_complete.py}"
CAPTURE_CONFIG="${CAPTURE_CONFIG:-capture_config.py}"

# Discard dipanggil dari controller, bukan dari run_first_pipeline.
DISCARD_SCRIPT="${DISCARD_SCRIPT:-robot_executor/execute_from_current_pregrasp_to_discard.py}"
WAYPOINTS_JSON="${WAYPOINTS_JSON:-configs/waypoints_ur5.json}"
ROBOT_IP="${ROBOT_IP:-192.168.200.1}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CATKIN_SETUP="${CATKIN_SETUP:-$HOME/ur5_noetic_ws/devel/setup.bash}"

# Stable grasp config. Kita paksa lewat env, jadi LOCK_STABLE_CONFIG=0.
LOCK_STABLE_CONFIG="${LOCK_STABLE_CONFIG:-0}"

PREGRASP_Z="${PREGRASP_Z:-0.13}"
DESCEND_Z="${DESCEND_Z:-0.12}"

NUDGE_DX="${NUDGE_DX:--0.05}"
NUDGE_DY="${NUDGE_DY:-0.038}"
NUDGE_DZ="${NUDGE_DZ:-0.0}"

SAFE_LIFT_Z="${SAFE_LIFT_Z:-0.0}"
SAFE_LIFT_EEF_STEP="${SAFE_LIFT_EEF_STEP:-0.005}"
CARTESIAN_MIN_FRACTION="${CARTESIAN_MIN_FRACTION:-0.80}"

MAX_XYZ_DISTANCE="${MAX_XYZ_DISTANCE:-0.80}"
MAX_BASE_DELTA="${MAX_BASE_DELTA:-1.80}"
MAX_WRIST_DELTA="${MAX_WRIST_DELTA:-1.50}"

VELOCITY="${VELOCITY:-0.05}"
ACCELERATION="${ACCELERATION:-0.05}"

# Discard stable config.
DISCARD_LIFT_UP="${DISCARD_LIFT_UP:-0.04}"
DISCARD_VELOCITY="${DISCARD_VELOCITY:-0.03}"
DISCARD_ACCELERATION="${DISCARD_ACCELERATION:-0.03}"

# 1 = skip Cartesian lift di discard. Ini menghindari error fraction lift rendah.
DISCARD_SKIP_LIFT="${DISCARD_SKIP_LIFT:-1}"

DISABLE_GRIPPER="${DISABLE_GRIPPER:-0}"


# ============================================================
# HELPERS
# ============================================================

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

need_file() {
  local path="$1"
  local label="$2"

  if [ ! -f "$path" ]; then
    fail "$label tidak ditemukan: $path"
  fi

  echo "[OK] $label: $path"
}

conda_run() {
  local env_name="$1"
  shift

  if command -v conda >/dev/null 2>&1; then
    conda run -n "$env_name" "$@"
  elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    "$HOME/miniconda3/bin/conda" run -n "$env_name" "$@"
  elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    "$HOME/anaconda3/bin/conda" run -n "$env_name" "$@"
  elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    "$HOME/miniforge3/bin/conda" run -n "$env_name" "$@"
  else
    fail "conda tidak ditemukan. Cek: conda info --base"
  fi
}

patch_post_status_bug() {
  for f in "$POST_SCRIPT_FIXED" "$POST_SCRIPT_FALLBACK"; do
    if [ -f "$f" ]; then
      sed -i 's/"REMOVED_SUCCESS" "STILL_FOUND"/"REMOVED_SUCCESS"/g' "$f"
    fi
  done
}

update_capture_config() {
  local test_name="$1"
  local step_index="$2"

  need_file "$CAPTURE_CONFIG" "capture_config.py"

  python3 - <<PY
from pathlib import Path
import re

p = Path(r"$CAPTURE_CONFIG")
s = p.read_text()

if re.search(r'^TEST_NAME\s*=', s, flags=re.M):
    s = re.sub(r'^TEST_NAME\s*=.*$', 'TEST_NAME = "$test_name"', s, flags=re.M)
else:
    raise RuntimeError("TEST_NAME tidak ditemukan di capture_config.py")

if re.search(r'^STEP_INDEX\s*=', s, flags=re.M):
    s = re.sub(r'^STEP_INDEX\s*=\s*\d+', 'STEP_INDEX = $step_index', s, flags=re.M)
else:
    raise RuntimeError("STEP_INDEX tidak ditemukan di capture_config.py")

p.write_text(s)
print("[OK] capture_config.py updated: TEST_NAME=$test_name, STEP_INDEX=$step_index")
PY

  grep -n "TEST_NAME\|STEP_INDEX" "$CAPTURE_CONFIG" || true
}

get_post_result_path() {
  local step_index="$1"
  echo "outputs/$TEST_NAME/post_check_output/STEP_${step_index}_post_check_result.json"
}

get_remaining_path() {
  local step_index="$1"
  echo "outputs/$TEST_NAME/post_check_output/STEP_${step_index}_remaining_plan.json"
}

get_post_status() {
  local step_index="$1"
  local p
  p="$(get_post_result_path "$step_index")"

  python3 - <<PY
import json
from pathlib import Path

p = Path(r"$p")
if not p.exists():
    raise SystemExit(f"[ERROR] Post-check result tidak ditemukan: {p}")

d = json.load(open(p))
print(d.get("post_check_status", "UNKNOWN"))
PY
}

get_remaining_count() {
  local step_index="$1"
  local p
  p="$(get_remaining_path "$step_index")"

  python3 - <<PY
import json
from pathlib import Path

p = Path(r"$p")
if not p.exists():
    raise SystemExit(f"[ERROR] Remaining plan tidak ditemukan: {p}")

d = json.load(open(p))
if not isinstance(d, list):
    raise SystemExit(f"[ERROR] Remaining plan bukan list: {p}")

print(len(d))
PY
}

show_next_ready() {
  local p="outputs/$TEST_NAME/vision_output/next_target_ready.json"

  if [ ! -f "$p" ]; then
    echo "[WARN] next_target_ready.json tidak ditemukan: $p"
    return 0
  fi

  python3 - <<PY
import json

p = r"$p"
d = json.load(open(p))
print("[NEXT_READY] ready:", d.get("ready"))
print("[NEXT_READY] next_step_number:", d.get("next_step_number"))
print("[NEXT_READY] next_target:", d.get("next_target"))
print("[NEXT_READY] target_query_used:", d.get("target_query_used"))
print("[NEXT_READY] conf_used:", d.get("confidence_threshold_used"))
PY
}

run_ros_python() {
  local cmd
  printf -v cmd '%q ' "$@"

  env \
    -u PYTHONPATH \
    -u PYTHONHOME \
    -u LD_LIBRARY_PATH \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    bash -lc "
      set -e
      source '$ROS_SETUP'
      if [ -f '$CATKIN_SETUP' ]; then
        source '$CATKIN_SETUP'
      fi
      cd '$PROJECT_DIR'
      $cmd
    "
}


# ============================================================
# ACTION RUNNERS
# ============================================================

run_first_target_grasp_only() {
  log_section "ACTION STEP 1: FULL PIPELINE UNTIL GRASP ONLY"

  local skip_planner="1"
  local skip_validator="1"

  if [ "$RUN_PLANNER" = "1" ]; then
    skip_planner="0"
  fi

  if [ "$RUN_VALIDATOR" = "1" ]; then
    skip_validator="0"
  fi

  env \
    LOCK_STABLE_CONFIG="$LOCK_STABLE_CONFIG" \
    PREGRASP_Z="$PREGRASP_Z" \
    DESCEND_Z="$DESCEND_Z" \
    NUDGE_DX="$NUDGE_DX" \
    NUDGE_DY="$NUDGE_DY" \
    NUDGE_DZ="$NUDGE_DZ" \
    SAFE_LIFT_Z="$SAFE_LIFT_Z" \
    SAFE_LIFT_EEF_STEP="$SAFE_LIFT_EEF_STEP" \
    CARTESIAN_MIN_FRACTION="$CARTESIAN_MIN_FRACTION" \
    MAX_XYZ_DISTANCE="$MAX_XYZ_DISTANCE" \
    MAX_BASE_DELTA="$MAX_BASE_DELTA" \
    MAX_WRIST_DELTA="$MAX_WRIST_DELTA" \
    VELOCITY="$VELOCITY" \
    ACCELERATION="$ACCELERATION" \
    SKIP_CAPTURE=0 \
    SKIP_PLANNER="$skip_planner" \
    SKIP_VALIDATOR="$skip_validator" \
    SKIP_VISION=0 \
    SKIP_GRASPNET=0 \
    SKIP_TRANSFORM=0 \
    SKIP_CONVERT=0 \
    SKIP_NUDGE=0 \
    SKIP_BASE_LINK_CONVERT=0 \
    SKIP_GRASP=0 \
    SKIP_DISCARD=1 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
}

run_next_target_grasp_only() {
  local step_index="$1"

  log_section "ACTION STEP $step_index: NEXT TARGET UNTIL GRASP ONLY"

  show_next_ready

  need_file "outputs/$TEST_NAME/vision_output/fastsam_mask.png" "FastSAM mask"
  need_file "outputs/$TEST_NAME/vision_output/object_position_camera.json" "Object position"

  env \
    LOCK_STABLE_CONFIG="$LOCK_STABLE_CONFIG" \
    PREGRASP_Z="$PREGRASP_Z" \
    DESCEND_Z="$DESCEND_Z" \
    NUDGE_DX="$NUDGE_DX" \
    NUDGE_DY="$NUDGE_DY" \
    NUDGE_DZ="$NUDGE_DZ" \
    SAFE_LIFT_Z="$SAFE_LIFT_Z" \
    SAFE_LIFT_EEF_STEP="$SAFE_LIFT_EEF_STEP" \
    CARTESIAN_MIN_FRACTION="$CARTESIAN_MIN_FRACTION" \
    MAX_XYZ_DISTANCE="$MAX_XYZ_DISTANCE" \
    MAX_BASE_DELTA="$MAX_BASE_DELTA" \
    MAX_WRIST_DELTA="$MAX_WRIST_DELTA" \
    VELOCITY="$VELOCITY" \
    ACCELERATION="$ACCELERATION" \
    SKIP_CAPTURE=1 \
    SKIP_PLANNER=1 \
    SKIP_VALIDATOR=1 \
    SKIP_VISION=1 \
    SKIP_GRASPNET=0 \
    SKIP_TRANSFORM=0 \
    SKIP_CONVERT=0 \
    SKIP_NUDGE=0 \
    SKIP_BASE_LINK_CONVERT=0 \
    SKIP_GRASP=0 \
    SKIP_DISCARD=1 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
}

run_retry_same_step_grasp_only() {
  local step_index="$1"

  log_section "RETRY ACTION STEP $step_index: SAME TARGET UNTIL GRASP ONLY"

  need_file "outputs/$TEST_NAME/vision_output/fastsam_mask.png" "Retry FastSAM mask"
  need_file "data/d455_capture/current_scene_rgb.jpg" "Current RGB from failed post-check"
  need_file "data/d455_capture/depth_raw.npy" "Current depth from failed post-check"

  env \
    LOCK_STABLE_CONFIG="$LOCK_STABLE_CONFIG" \
    PREGRASP_Z="$PREGRASP_Z" \
    DESCEND_Z="$DESCEND_Z" \
    NUDGE_DX="$NUDGE_DX" \
    NUDGE_DY="$NUDGE_DY" \
    NUDGE_DZ="$NUDGE_DZ" \
    SAFE_LIFT_Z="$SAFE_LIFT_Z" \
    SAFE_LIFT_EEF_STEP="$SAFE_LIFT_EEF_STEP" \
    CARTESIAN_MIN_FRACTION="$CARTESIAN_MIN_FRACTION" \
    MAX_XYZ_DISTANCE="$MAX_XYZ_DISTANCE" \
    MAX_BASE_DELTA="$MAX_BASE_DELTA" \
    MAX_WRIST_DELTA="$MAX_WRIST_DELTA" \
    VELOCITY="$VELOCITY" \
    ACCELERATION="$ACCELERATION" \
    SKIP_CAPTURE=1 \
    SKIP_PLANNER=1 \
    SKIP_VALIDATOR=1 \
    SKIP_VISION=1 \
    SKIP_GRASPNET=0 \
    SKIP_TRANSFORM=0 \
    SKIP_CONVERT=0 \
    SKIP_NUDGE=0 \
    SKIP_BASE_LINK_CONVERT=0 \
    SKIP_GRASP=0 \
    SKIP_DISCARD=1 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
}

run_discard_action() {
  log_section "DISCARD AFTER GRASP"

  need_file "$DISCARD_SCRIPT" "Discard script"
  need_file "$WAYPOINTS_JSON" "Waypoints JSON"

  local discard_cmd=(
    python3 "$DISCARD_SCRIPT"
    --robot_ip "$ROBOT_IP"
    --waypoints_json "$WAYPOINTS_JSON"
    --lift_up "$DISCARD_LIFT_UP"
    --velocity "$DISCARD_VELOCITY"
    --acceleration "$DISCARD_ACCELERATION"
  )

  if [ "$DISCARD_SKIP_LIFT" = "1" ]; then
    discard_cmd+=(--skip_lift)
  fi

  if [ "$DISABLE_GRIPPER" = "1" ]; then
    discard_cmd+=(--disable_gripper)
  fi

  if [ "$MODE" = "execute" ]; then
    discard_cmd+=(--execute)
  fi

  echo "[CMD] ${discard_cmd[*]}"
  run_ros_python "${discard_cmd[@]}"
}


# ============================================================
# POST-CHECK + RETRY PREPARATION
# ============================================================

capture_for_post_check() {
  log_section "CAPTURE POST-SCENE"

  if [ "$CAPTURE_BEFORE_POST" != "1" ]; then
    echo "[SKIP] CAPTURE_BEFORE_POST=0"
    return 0
  fi

  echo "[INFO] Waiting $POST_CAPTURE_DELAY_SEC second(s) after discard before capture..."
  sleep "$POST_CAPTURE_DELAY_SEC"

  need_file "$CAPTURE_SCRIPT" "Capture script"

  echo "[RUN] conda run -n $PLANNER_ENV python $CAPTURE_SCRIPT"
  conda_run "$PLANNER_ENV" python "$CAPTURE_SCRIPT"

  need_file "data/d455_capture/current_scene_rgb.jpg" "current_scene_rgb.jpg"
  need_file "data/d455_capture/depth_raw.npy" "depth_raw.npy"
  need_file "data/d455_capture/camera_intrinsics.json" "camera_intrinsics.json"

  # Untuk fallback post script. Fixed script akan sync ulang dan buat report.
  cp data/d455_capture/current_scene_rgb.jpg data/d455_capture/post_scene_rgb.jpg
}

run_post_check_for_step() {
  local step_index="$1"

  log_section "POST-CHECK STEP $step_index"

  update_capture_config "$TEST_NAME" "$step_index"
  capture_for_post_check

  local post_script="$POST_SCRIPT_FIXED"
  if [ ! -f "$post_script" ]; then
    echo "[WARN] $POST_SCRIPT_FIXED tidak ditemukan. Fallback ke $POST_SCRIPT_FALLBACK"
    post_script="$POST_SCRIPT_FALLBACK"
  fi

  need_file "$post_script" "Post-check script"

  echo "[RUN] conda run -n $PLANNER_ENV python $post_script"
  conda_run "$PLANNER_ENV" python "$post_script"

  POST_STATUS="$(get_post_status "$step_index")"
  echo "[CHECK] STEP $step_index post_check_status: $POST_STATUS"

  if [ "$POST_STATUS" = "REMOVED_SUCCESS" ]; then
    REMAINING_COUNT="$(get_remaining_count "$step_index")"
    echo "[CHECK] STEP $step_index remaining_count: $REMAINING_COUNT"

    if [ "$REMAINING_COUNT" -gt 0 ]; then
      show_next_ready
    fi
  else
    REMAINING_COUNT="-1"
    echo "[WARN] STEP $step_index belum sukses. Status: $POST_STATUS"
  fi
}

prepare_retry_same_step_from_post_check() {
  local step_index="$1"

  log_section "PREPARE RETRY SAME STEP $step_index FROM POST-CHECK OUTPUT"

  python3 - <<PY
import json
import shutil
from pathlib import Path

test_name = "$TEST_NAME"
step_index = int("$step_index")

base = Path("outputs") / test_name
post_dir = base / "post_check_output"
vision_dir = base / "vision_output"
vision_dir.mkdir(parents=True, exist_ok=True)

result_path = post_dir / f"STEP_{step_index}_post_check_result.json"
if not result_path.exists():
    raise SystemExit(f"[ERROR] Missing post-check result: {result_path}")

result = json.load(open(result_path))
status = result.get("post_check_status")

if status != "STILL_FOUND":
    raise SystemExit(f"[ERROR] Retry hanya boleh dari STILL_FOUND, status={status}")

post_yolo_json = post_dir / f"STEP_{step_index}_post_check_detections_yolo.json"
post_yolo_image = Path(result.get("post_yolo_image", post_dir / f"STEP_{step_index}_post_check_yolo_result.jpg"))
post_mask = Path(result.get("post_fastsam_mask", post_dir / f"STEP_{step_index}_post_check_fastsam_mask.png"))
post_fastsam_image = Path(result.get("post_fastsam_image", post_dir / f"STEP_{step_index}_post_check_fastsam_result.jpg"))

mappings = [
    (post_yolo_json, vision_dir / "detections_yolo.json"),
    (post_yolo_image, vision_dir / "yolo_world_result.jpg"),
    (post_mask, vision_dir / "fastsam_mask.png"),
    (post_fastsam_image, vision_dir / "fastsam_result.jpg"),
]

for src, dst in mappings:
    if not src.exists():
        raise SystemExit(f"[ERROR] File retry tidak ditemukan: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[COPY RETRY] {src} -> {dst}")

retry_ready = {
    "ready": True,
    "retry_same_step": True,
    "test_name": test_name,
    "step_index": step_index,
    "target": result.get("target"),
    "source_post_check_result": str(result_path),
    "standard_outputs": {
        "detections_yolo": str(vision_dir / "detections_yolo.json"),
        "yolo_world_result": str(vision_dir / "yolo_world_result.jpg"),
        "fastsam_mask": str(vision_dir / "fastsam_mask.png"),
        "fastsam_result": str(vision_dir / "fastsam_result.jpg"),
    },
    "note": "Target masih terlihat setelah post-check. Output post-check dicopy ke vision_output untuk retry GraspNet pada step yang sama."
}

with open(vision_dir / "retry_same_step_ready.json", "w") as f:
    json.dump(retry_ready, f, indent=2)

print("[OK] retry_same_step_ready.json saved:", vision_dir / "retry_same_step_ready.json")
PY
}


# ============================================================
# MAIN
# ============================================================

cd "$PROJECT_DIR"

case "$MODE" in
  plan|execute) ;;
  *) fail "MODE tidak valid: $MODE. Gunakan plan atau execute." ;;
esac

need_file "$RUN_FIRST_SCRIPT" "run_first_pipeline.sh"
need_file "$CAPTURE_CONFIG" "capture_config.py"
need_file "$DISCARD_SCRIPT" "Discard script"

patch_post_status_bug

log_section "MULTI PIPELINE RETRY CONFIG"
echo "PROJECT_DIR              : $PROJECT_DIR"
echo "TEST_NAME                : $TEST_NAME"
echo "MODE                     : $MODE"
echo "PLANNER_ENV              : $PLANNER_ENV"
echo "MAX_STEPS                : $MAX_STEPS"
echo "START_STEP               : $START_STEP"
echo "MAX_RETRIES_PER_STEP     : $MAX_RETRIES_PER_STEP"
echo "RESUME_AFTER_ACTION      : $RESUME_AFTER_ACTION"
echo "RUN_PLANNER              : $RUN_PLANNER"
echo "RUN_VALIDATOR            : $RUN_VALIDATOR"
echo "LOCK_STABLE_CONFIG       : $LOCK_STABLE_CONFIG"
echo "PREGRASP_Z               : $PREGRASP_Z"
echo "DESCEND_Z                : $DESCEND_Z"
echo "NUDGE_DX/DY/DZ           : $NUDGE_DX / $NUDGE_DY / $NUDGE_DZ"
echo "MAX_WRIST_DELTA          : $MAX_WRIST_DELTA"
echo "CAPTURE_BEFORE_POST      : $CAPTURE_BEFORE_POST"
echo "DISCARD_SKIP_LIFT        : $DISCARD_SKIP_LIFT"
echo "POST_SCRIPT_FIXED        : $POST_SCRIPT_FIXED"
echo "POST_SCRIPT_FALLBACK     : $POST_SCRIPT_FALLBACK"

STEP_INDEX="$START_STEP"

while true; do
  if [ "$STEP_INDEX" -gt "$MAX_STEPS" ]; then
    fail "STEP_INDEX=$STEP_INDEX melebihi MAX_STEPS=$MAX_STEPS"
  fi

  ATTEMPT=1
  POST_STATUS="UNKNOWN"
  REMAINING_COUNT="-1"

  while [ "$ATTEMPT" -le "$MAX_RETRIES_PER_STEP" ]; do
    log_section "STEP $STEP_INDEX / ATTEMPT $ATTEMPT"

    if [ "$RESUME_AFTER_ACTION" = "1" ]; then
      echo "[INFO] RESUME_AFTER_ACTION=1: skip action untuk STEP $STEP_INDEX."
      echo "[INFO] Diasumsikan robot sudah grasp + discard untuk step ini."
      RESUME_AFTER_ACTION="0"
    else
      if [ "$ATTEMPT" -eq 1 ]; then
        if [ "$STEP_INDEX" -eq 1 ]; then
          run_first_target_grasp_only
        else
          run_next_target_grasp_only "$STEP_INDEX"
        fi
      else
        run_retry_same_step_grasp_only "$STEP_INDEX"
      fi

      run_discard_action
    fi

    run_post_check_for_step "$STEP_INDEX"

    if [ "$POST_STATUS" = "REMOVED_SUCCESS" ]; then
      echo "[OK] STEP $STEP_INDEX berhasil pada attempt $ATTEMPT."
      break
    fi

    echo "[WARN] STEP $STEP_INDEX masih gagal pada attempt $ATTEMPT. Status: $POST_STATUS"

    if [ "$ATTEMPT" -ge "$MAX_RETRIES_PER_STEP" ]; then
      echo ""
      echo "[FAIL] STEP $STEP_INDEX gagal setelah $MAX_RETRIES_PER_STEP attempt."
      echo "       Pipeline dihentikan untuk safety."
      echo "       Cek post image, gripper, discard, atau YOLO false positive."
      exit 2
    fi

    prepare_retry_same_step_from_post_check "$STEP_INDEX"

    ATTEMPT=$((ATTEMPT + 1))
    echo "[INFO] Retry STEP $STEP_INDEX dengan target yang sama."
  done

  if [ "$REMAINING_COUNT" -eq 0 ]; then
    log_section "DONE"
    echo "✅ Semua step dalam action_plan sudah selesai."
    echo "TEST_NAME: $TEST_NAME"
    echo "Last completed STEP_INDEX: $STEP_INDEX"
    break
  fi

  echo ""
  echo "[INFO] Masih ada $REMAINING_COUNT step berikutnya."
  echo "[INFO] Lanjut ke STEP $((STEP_INDEX + 1))."

  STEP_INDEX=$((STEP_INDEX + 1))
done