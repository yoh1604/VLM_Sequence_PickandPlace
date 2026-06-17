#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MULTI-STEP PICK PIPELINE CONTROLLER
#
# Purpose:
#   Controller untuk menjalankan pipeline multi-object:
#   step action -> discard -> post-check -> next target -> repeat.
#
# Dependencies:
#   - scripts/run_first_pipeline.sh
#   - capture_config.py
#   - run_post_complete_fixed.py  (preferred)
#   - perception/capture_d455_once.py
#
# Usage:
#   ./scripts/run_multi_pipeline.sh water plan
#   ./scripts/run_multi_pipeline.sh water execute
#
# Recommended first test:
#   ./scripts/run_multi_pipeline.sh water plan
#
# Notes:
#   MODE=plan mengikuti behavior grasp.py kamu sekarang:
#   ada RViz preview dan ENTER manual sebelum execute.
# ============================================================

# ============================================================
# USER CONFIG
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/pick_place_occlusion_noetic}"
PLANNER_ENV="${PLANNER_ENV:-ur5_pickplace}"

TEST_NAME="${1:-water_new_01}"
MODE="${2:-plan}"

MAX_STEPS="${MAX_STEPS:-5}"
START_STEP="${START_STEP:-1}"

MAX_RETRIES_PER_STEP="${MAX_RETRIES_PER_STEP:-2}"

# Kalau 1, script tidak menjalankan aksi untuk START_STEP,
# tetapi langsung post-check START_STEP.
# Berguna kalau robot sudah kamu gerakkan manual sebelumnya.
RESUME_AFTER_ACTION="${RESUME_AFTER_ACTION:-0}"

# Untuk first target:
# Default mengikuti kebiasaanmu: action_plan/validation biasanya sudah ada.
# Kalau ingin planner/validator jalan ulang:
#   RUN_PLANNER=1 RUN_VALIDATOR=1 ./scripts/run_multi_pipeline.sh water plan
RUN_PLANNER="${RUN_PLANNER:-0}"
RUN_VALIDATOR="${RUN_VALIDATOR:-0}"

# Capture scene setelah discard sebelum post-check.
CAPTURE_BEFORE_POST="${CAPTURE_BEFORE_POST:-1}"
POST_CAPTURE_DELAY_SEC="${POST_CAPTURE_DELAY_SEC:-2}"

# Script paths
RUN_FIRST_SCRIPT="${RUN_FIRST_SCRIPT:-scripts/run_first_pipeline.sh}"
CAPTURE_SCRIPT="${CAPTURE_SCRIPT:-perception/capture_d455_once.py}"
POST_SCRIPT_FIXED="${POST_SCRIPT_FIXED:-run_post_complete_fixed.py}"
POST_SCRIPT_FALLBACK="${POST_SCRIPT_FALLBACK:-run_post_complete.py}"

CAPTURE_CONFIG="${CAPTURE_CONFIG:-capture_config.py}"

# Stable parameters. run_first_pipeline.sh juga sudah punya LOCK_STABLE_CONFIG.
LOCK_STABLE_CONFIG="${LOCK_STABLE_CONFIG:-1}"

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

need_executable_or_file() {
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
    raise RuntimeError('TEST_NAME tidak ditemukan di capture_config.py')

if re.search(r'^STEP_INDEX\s*=', s, flags=re.M):
    s = re.sub(r'^STEP_INDEX\s*=\s*\d+', 'STEP_INDEX = $step_index', s, flags=re.M)
else:
    raise RuntimeError('STEP_INDEX tidak ditemukan di capture_config.py')

p.write_text(s)
print('[OK] capture_config.py updated: TEST_NAME=$test_name, STEP_INDEX=$step_index')
PY

  grep -n "TEST_NAME\|STEP_INDEX" "$CAPTURE_CONFIG" || true
}

get_json_value() {
  local json_path="$1"
  local expr="$2"

  python3 - <<PY
import json
from pathlib import Path
p = Path(r"$json_path")
if not p.exists():
    raise SystemExit(f"MISSING_JSON:{p}")
d = json.load(open(p))
print($expr)
PY
}

get_remaining_count() {
  local step_index="$1"
  local p="outputs/$TEST_NAME/post_check_output/STEP_${step_index}_remaining_plan.json"

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

get_post_status() {
  local step_index="$1"
  local p="outputs/$TEST_NAME/post_check_output/STEP_${step_index}_post_check_result.json"

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

# ============================================================
# ACTION RUNNERS
# ============================================================

run_first_target_action() {
  log_section "ACTION STEP 1: FULL FIRST PIPELINE"

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
    SKIP_DISCARD=0 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
}

run_next_target_action() {
  local step_index="$1"

  log_section "ACTION STEP $step_index: NEXT TARGET FROM POST-CHECK OUTPUT"

  show_next_ready

  need_file "outputs/$TEST_NAME/vision_output/fastsam_mask.png" "FastSAM mask next target"
  need_file "outputs/$TEST_NAME/vision_output/object_position_camera.json" "Object position next target"

  env \
    LOCK_STABLE_CONFIG="$LOCK_STABLE_CONFIG" \
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
    SKIP_DISCARD=0 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
}

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
}

run_retry_same_step_action() {
  local step_index="$1"

  log_section "RETRY ACTION STEP $step_index: SAME TARGET"

  env \
    LOCK_STABLE_CONFIG="$LOCK_STABLE_CONFIG" \
    SKIP_CAPTURE=1 \
    SKIP_PLANNER=1 \
    SKIP_VALIDATOR=1 \
    SKIP_VISION=0 \
    SKIP_GRASPNET=0 \
    SKIP_TRANSFORM=0 \
    SKIP_CONVERT=0 \
    SKIP_NUDGE=0 \
    SKIP_BASE_LINK_CONVERT=0 \
    SKIP_GRASP=0 \
    SKIP_DISCARD=0 \
    "$RUN_FIRST_SCRIPT" "$TEST_NAME" "$MODE"
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

  local status
  status="$(get_post_status "$step_index")"
  echo "[CHECK] STEP $step_index post_check_status: $status"

  # if [ "$status" != "REMOVED_SUCCESS" ]; then
  #   echo ""
  #   echo "[STOP] STEP $step_index belum sukses menurut post-check."
  #   echo "       Jangan lanjut ke target berikutnya."
  #   echo "       Cek post image dan ulangi aksi/discard bila perlu."
  #   exit 2
  # fi

  local remaining_count
  remaining_count="$(get_remaining_count "$step_index")"
  echo "[CHECK] STEP $step_index remaining_count: $remaining_count"

  if [ "$remaining_count" -gt 0 ]; then
    show_next_ready
  fi

  echo "$remaining_count"
}

# ============================================================
# START
# ============================================================

cd "$PROJECT_DIR"

case "$MODE" in
  plan|execute) ;;
  *) fail "MODE tidak valid: $MODE. Gunakan plan atau execute." ;;
esac

need_executable_or_file "$RUN_FIRST_SCRIPT" "run_first_pipeline.sh"
need_file "$CAPTURE_CONFIG" "capture_config.py"

log_section "MULTI PIPELINE CONFIG"
echo "PROJECT_DIR              : $PROJECT_DIR"
echo "TEST_NAME                : $TEST_NAME"
echo "MODE                     : $MODE"
echo "PLANNER_ENV              : $PLANNER_ENV"
echo "MAX_STEPS                : $MAX_STEPS"
echo "START_STEP               : $START_STEP"
echo "RESUME_AFTER_ACTION      : $RESUME_AFTER_ACTION"
echo "RUN_PLANNER              : $RUN_PLANNER"
echo "RUN_VALIDATOR            : $RUN_VALIDATOR"
echo "LOCK_STABLE_CONFIG       : $LOCK_STABLE_CONFIG"
echo "CAPTURE_BEFORE_POST      : $CAPTURE_BEFORE_POST"
echo "POST_SCRIPT_FIXED        : $POST_SCRIPT_FIXED"
echo "POST_SCRIPT_FALLBACK     : $POST_SCRIPT_FALLBACK"

STEP_INDEX="$START_STEP"

attempt=1

while [ "$attempt" -le "$MAX_RETRIES_PER_STEP" ]; do
  echo "[INFO] STEP $STEP_INDEX attempt $attempt / $MAX_RETRIES_PER_STEP"

  if [ "$RESUME_AFTER_ACTION" = "1" ]; then
    echo "[INFO] Resume mode: skip action."
    RESUME_AFTER_ACTION="0"
  else
    if [ "$STEP_INDEX" -eq 1 ] && [ "$attempt" -eq 1 ]; then
      run_first_target_action
    else
      run_retry_same_step_action "$STEP_INDEX"
    fi
  fi

  capture_for_post_check
  update_capture_config "$TEST_NAME" "$STEP_INDEX"

  conda_run "$PLANNER_ENV" python "$post_script"

  status="$(get_post_status "$STEP_INDEX")"

  if [ "$status" = "REMOVED_SUCCESS" ]; then
    echo "[OK] STEP $STEP_INDEX success."
    break
  fi

  echo "[WARN] STEP $STEP_INDEX masih gagal: $status"
  attempt=$((attempt + 1))
done

if [ "$status" != "REMOVED_SUCCESS" ]; then
  echo "[FAIL] STEP $STEP_INDEX gagal setelah $MAX_RETRIES_PER_STEP attempt."
  exit 2
fi

# while true; do
#   if [ "$STEP_INDEX" -gt "$MAX_STEPS" ]; then
#     fail "STEP_INDEX=$STEP_INDEX melebihi MAX_STEPS=$MAX_STEPS"
#   fi

#   if [ "$RESUME_AFTER_ACTION" = "1" ]; then
#     log_section "RESUME MODE: SKIP ACTION STEP $STEP_INDEX"
#     echo "[INFO] Diasumsikan STEP $STEP_INDEX sudah grasp + discard."
#     RESUME_AFTER_ACTION="0"
#   else
#     if [ "$STEP_INDEX" -eq 1 ]; then
#       run_first_target_action
#     else
#       run_next_target_action "$STEP_INDEX"
#     fi
#   fi

#   remaining_count="$(run_post_check_for_step "$STEP_INDEX" | tail -n 1)"

#   if [ "$remaining_count" -eq 0 ]; then
#     log_section "DONE"
#     echo "✅ Semua step dalam action_plan sudah selesai."
#     echo "TEST_NAME: $TEST_NAME"
#     echo "Last completed STEP_INDEX: $STEP_INDEX"
#     break
#   fi

#   echo ""
#   echo "[INFO] Masih ada $remaining_count step berikutnya. Lanjut ke action berikutnya."
#   STEP_INDEX=$((STEP_INDEX + 1))
# done
