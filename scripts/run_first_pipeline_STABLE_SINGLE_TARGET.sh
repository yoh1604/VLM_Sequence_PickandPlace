#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FIRST PICK PIPELINE WRAPPER - STABLE grasp config
#
# Alur:
# 1. Resolve TEST_NAME dari argumen atau capture_config.py.
# 2. Capture D455.
# 3. Run vision pipeline: YOLO-World / FastSAM / depth.
# 4. Run GraspNet / AnyGrasp.
# 5. Transform grasp camera -> base.
# 6. Convert gripper_tip -> tool0 pregrasp target dalam frame base.
# 7. Apply nudge dalam frame base.
# 8. Convert target base -> base_link.
# 9. ROS / MoveIt sanity check.
# 10. Run grasp.py:
#     current/IDLE -> optional safe lift -> pregrasp Cartesian -> descend -> close.
# 11. Optional discard.
#
# Catatan penting:
# - Nudge dilakukan SETELAH convert gripper_tip -> tool0 dan SEBELUM base -> base_link.
# - Jangan set SKIP_CONVERT=1 dan SKIP_NUDGE=0, karena nudge akan menumpuk ke file lama.
# - MODE=plan tetap bisa execute setelah ENTER, sesuai behavior grasp.py sekarang.
# ============================================================


# ============================================================
# USER CONFIG
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/pick_place_occlusion_noetic}"

PLANNER_ENV="${PLANNER_ENV:-ur5_pickplace}"
ANYGRASP_ENV="${ANYGRASP_ENV:-anygrasp_py310}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CATKIN_SETUP="${CATKIN_SETUP:-$HOME/ur5_noetic_ws/devel/setup.bash}"

ROBOT_IP="${ROBOT_IP:-192.168.200.1}"

# Argumen:
#   $1 = TEST_NAME atau auto
#   $2 = plan / execute
MODE="${2:-plan}"


# ============================================================
# SCRIPT NAMES
# ============================================================

CAPTURE_SCRIPT="${CAPTURE_SCRIPT:-perception/capture_d455_once.py}"

PLANNER_SCRIPT="${PLANNER_SCRIPT:-run_planner_real.py}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-run_validator_real.py}"
VISION_SCRIPT="${VISION_SCRIPT:-run_d455_pipeline.py}"

ANYGRASP_SCRIPT="${ANYGRASP_SCRIPT:-models/graspnet-baseline/demo_d455.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-models/graspnet-baseline/logs/log_rs/checkpoint.tar}"

TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT:-perception/transform_grasp_to_base.py}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-perception/convert_gripper_tip_to_tool0_target.py}"
NUDGE_SCRIPT="${NUDGE_SCRIPT:-robot_executor/nudge_tool0_target.py}"
BASE_TO_BASE_LINK_SCRIPT="${BASE_TO_BASE_LINK_SCRIPT:-perception/convert_tool0_target_base_to_base_link.py}"

GRIPPER_SCRIPT="${GRIPPER_SCRIPT:-robot_executor/robotiq_socket_control.py}"
GRASP_SCRIPT="${GRASP_SCRIPT:-robot_executor/grasp.py}"
FALLBACK_GRASP_SCRIPT="${FALLBACK_GRASP_SCRIPT:-robot_executor/move_to_tool0_pregrasp.py}"

DISCARD_SCRIPT="${DISCARD_SCRIPT:-robot_executor/execute_from_current_pregrasp_to_discard.py}"
WAYPOINTS_JSON="${WAYPOINTS_JSON:-configs/waypoints_ur5.json}"


# ============================================================
# GRASP PARAMS - STABLE CONFIG
# ============================================================

# Kalau 1, script MEMAKSA pakai config grasp yang sudah stabil,
# sehingga export lama seperti DESCEND_Z=0.10 tidak ikut terbawa.
# Kalau ingin tuning manual, jalankan: export LOCK_STABLE_CONFIG=0
LOCK_STABLE_CONFIG="${LOCK_STABLE_CONFIG:-1}"

# Harus string satu argumen, jangan ditulis sebagai 3 argumen.
TOOL0_TO_GRIPPER_TIP="${TOOL0_TO_GRIPPER_TIP:-0 0 0.17}"

if [ "$LOCK_STABLE_CONFIG" = "1" ]; then
  # Stable grasp config terakhir yang sudah benar saat direct grasp.py.
  PREGRASP_Z="0.13"
  DESCEND_Z="0.12"

  NUDGE_DX="-0.05"
  NUDGE_DY="0.038"
  NUDGE_DZ="0.0"

  SAFE_LIFT_Z="0.0"
  SAFE_LIFT_EEF_STEP="0.005"
  CARTESIAN_MIN_FRACTION="0.80"

  MAX_XYZ_DISTANCE="0.80"
  MAX_BASE_DELTA="1.80"
  MAX_WRIST_DELTA="1.40"
else
  # Mode tuning: boleh override dari export environment.
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
  MAX_WRIST_DELTA="${MAX_WRIST_DELTA:-1.40}"
fi

PREGRASP_WAIT="${PREGRASP_WAIT:-5.0}"
RVIZ_PREVIEW_WAIT="${RVIZ_PREVIEW_WAIT:-3.0}"

LIFT_UP="${LIFT_UP:-0.04}"

VELOCITY="${VELOCITY:-0.05}"
ACCELERATION="${ACCELERATION:-0.05}"

# Gripper:
# - grasp.py akan open di pregrasp dan close setelah descend.
# - PREOPEN_GRIPPER=1 hanya kalau ingin gripper dibuka sebelum robot bergerak.
DISABLE_GRIPPER="${DISABLE_GRIPPER:-0}"
PREOPEN_GRIPPER="${PREOPEN_GRIPPER:-0}"


# ============================================================
# SKIP OPTIONS
# ============================================================

SKIP_CAPTURE="${SKIP_CAPTURE:-0}"
SKIP_PLANNER="${SKIP_PLANNER:-1}"
SKIP_VALIDATOR="${SKIP_VALIDATOR:-1}"
SKIP_VISION="${SKIP_VISION:-0}"
SKIP_GRASPNET="${SKIP_GRASPNET:-0}"
SKIP_TRANSFORM="${SKIP_TRANSFORM:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_NUDGE="${SKIP_NUDGE:-0}"
SKIP_BASE_LINK_CONVERT="${SKIP_BASE_LINK_CONVERT:-0}"
SKIP_GRASP="${SKIP_GRASP:-0}"

# Default skip discard untuk safety. Aktifkan dengan:
#   export SKIP_DISCARD=0
SKIP_DISCARD="${SKIP_DISCARD:-1}"


# ============================================================
# MODE
# ============================================================

case "$MODE" in
  plan)
    # Pada grasp.py sekarang, tanpa --execute tetap ada prompt ENTER lalu execute.
    # Jadi "plan" = manual-confirm execution, bukan dry-run murni.
    EXECUTE_FLAG=""
    ;;
  execute)
    EXECUTE_FLAG="--execute"
    ;;
  *)
    echo "[ERROR] MODE tidak valid: $MODE"
    echo "Gunakan: plan atau execute"
    exit 1
    ;;
esac

if [ "$DISABLE_GRIPPER" = "1" ]; then
  GRIPPER_FLAG="--disable_gripper"
else
  GRIPPER_FLAG=""
fi


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
  local name="$2"

  if [ ! -f "$path" ]; then
    echo "[ERROR] $name tidak ditemukan:"
    echo "  $path"
    exit 1
  fi

  echo "[OK] $name:"
  echo "  $path"
}

run_if_exists() {
  local script_path="$1"
  shift

  if [ -f "$script_path" ]; then
    echo "[RUN] python $script_path $*"
    python "$script_path" "$@"
  else
    echo "[SKIP] Script tidak ditemukan: $script_path"
  fi
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
    echo "Cek dengan:"
    echo "  conda info --base"
    exit 1
  fi

  set +u
  conda activate "$env_name"
  set -u

  echo "[INFO] Active conda env : ${CONDA_DEFAULT_ENV:-unknown}"
  echo "[INFO] Python           : $(which python)"
  python --version

  if [ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]; then
    echo "[ERROR] Gagal activate conda env: $env_name"
    exit 1
  fi
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
  unset CONDA_PREFIX_3 || true
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


# ============================================================
# START
# ============================================================

cd "$PROJECT_DIR"

# Guard supaya nudge tidak menumpuk pada file lama.
if [ "$SKIP_CONVERT" = "1" ] && [ "$SKIP_NUDGE" = "0" ]; then
  echo "[ERROR] Kombinasi berbahaya:"
  echo "  SKIP_CONVERT=1 dan SKIP_NUDGE=0"
  echo ""
  echo "Ini akan nudge file tool0_pregrasp_target.json lama secara in-place,"
  echo "sehingga nudge bisa menumpuk berkali-kali."
  echo ""
  echo "Pilih salah satu:"
  echo "  A) Regenerate bersih: export SKIP_CONVERT=0; export SKIP_NUDGE=0"
  echo "  B) Pakai target lama: export SKIP_CONVERT=1; export SKIP_NUDGE=1"
  exit 1
fi


# ============================================================
# TEST_NAME RESOLUTION
# ============================================================

TEST_ARG="${1:-auto}"

if [ "$TEST_ARG" = "auto" ]; then
  echo "[INFO] TEST_NAME tidak diberi. Membaca dari capture_config.py ..."

  if ! TEST_NAME="$(get_test_name_from_capture_config)"; then
    echo "[ERROR] Gagal membaca TEST_NAME dari capture_config.py."
    echo "$TEST_NAME"
    echo ""
    echo "Solusi cepat:"
    echo "  ./scripts/run_first_pipeline.sh water_bottle $MODE"
    exit 1
  fi
else
  TEST_NAME="$TEST_ARG"
fi

if [ -z "$TEST_NAME" ]; then
  echo "[ERROR] TEST_NAME kosong."
  exit 1
fi

if [[ "$TEST_NAME" == ERROR_IMPORT_CAPTURE_CONFIG:* ]]; then
  echo "[ERROR] Gagal membaca TEST_NAME dari capture_config.py"
  echo "$TEST_NAME"
  echo "Solusi: jalankan dengan argumen manual, contoh:"
  echo "  ./scripts/run_first_pipeline.sh water_bottle plan"
  exit 1
fi

export TEST_NAME="$TEST_NAME"

VISION_OUTPUT_DIR="$PROJECT_DIR/outputs/$TEST_NAME/vision_output"

BEST_GRASP_CAMERA_JSON="$VISION_OUTPUT_DIR/best_grasp_camera.json"
BEST_GRASP_BASE_JSON="$VISION_OUTPUT_DIR/best_grasp_base.json"
TOOL0_PREGRASP_TARGET_JSON="$VISION_OUTPUT_DIR/tool0_pregrasp_target.json"
TOOL0_PREGRASP_TARGET_BASE_LINK_JSON="$VISION_OUTPUT_DIR/tool0_pregrasp_target_base_link.json"

RGB_PATH="$PROJECT_DIR/data/d455_capture/current_scene_rgb.jpg"
DEPTH_PATH="$PROJECT_DIR/data/d455_capture/depth_raw.npy"
INTRINSICS_PATH="$PROJECT_DIR/data/d455_capture/camera_intrinsics.json"
MASK_PATH="$VISION_OUTPUT_DIR/fastsam_mask.png"

mkdir -p "$VISION_OUTPUT_DIR"

log_section "FULL PIPELINE CONFIG"

echo "PROJECT_DIR              : $PROJECT_DIR"
echo "TEST_NAME                : $TEST_NAME"
echo "MODE                     : $MODE"
echo "PLANNER_ENV              : $PLANNER_ENV"
echo "ANYGRASP_ENV             : $ANYGRASP_ENV"
echo "ROBOT_IP                 : $ROBOT_IP"
echo "VISION_OUTPUT_DIR        : $VISION_OUTPUT_DIR"
echo "BEST_GRASP_CAMERA_JSON   : $BEST_GRASP_CAMERA_JSON"
echo "BEST_GRASP_BASE_JSON     : $BEST_GRASP_BASE_JSON"
echo "TOOL0_PREGRASP_TARGET    : $TOOL0_PREGRASP_TARGET_JSON"
echo "TOOL0_PREGRASP_TARGET_BL : $TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"
echo "TOOL0_TO_GRIPPER_TIP     : $TOOL0_TO_GRIPPER_TIP"
echo "PREGRASP_Z               : $PREGRASP_Z"
echo "SAFE_LIFT_Z              : $SAFE_LIFT_Z"
echo "SAFE_LIFT_EEF_STEP       : $SAFE_LIFT_EEF_STEP"
echo "CARTESIAN_MIN_FRACTION   : $CARTESIAN_MIN_FRACTION"
echo "MAX_XYZ_DISTANCE         : $MAX_XYZ_DISTANCE"
echo "MAX_BASE_DELTA           : $MAX_BASE_DELTA"
echo "MAX_WRIST_DELTA          : $MAX_WRIST_DELTA"
echo "DESCEND_Z                : $DESCEND_Z"
echo "NUDGE_DX/DY/DZ           : $NUDGE_DX / $NUDGE_DY / $NUDGE_DZ"
echo "DISABLE_GRIPPER          : $DISABLE_GRIPPER"
echo "PREOPEN_GRIPPER          : $PREOPEN_GRIPPER"
echo "LOCK_STABLE_CONFIG       : $LOCK_STABLE_CONFIG"
echo "SKIP_DISCARD             : $SKIP_DISCARD"


# ============================================================
# STEP 1: CAPTURE D455
# ============================================================

if [ "$SKIP_CAPTURE" = "0" ]; then
  log_section "STEP 1: CAPTURE D455"

  activate_conda "$PLANNER_ENV"

  need_file "$CAPTURE_SCRIPT" "Capture script"

  echo "[RUN] python $CAPTURE_SCRIPT"
  python "$CAPTURE_SCRIPT"

  need_file "$RGB_PATH" "RGB image"
  need_file "$DEPTH_PATH" "Depth npy"
  need_file "$INTRINSICS_PATH" "Camera intrinsics"

  conda deactivate || true
else
  log_section "STEP 1: SKIP CAPTURE"
  need_file "$RGB_PATH" "RGB image"
  need_file "$DEPTH_PATH" "Depth npy"
  need_file "$INTRINSICS_PATH" "Camera intrinsics"
fi


# ============================================================
# STEP 2: ur5_pickplace - PLANNER / VALIDATOR / VISION
# ============================================================

if [ "$SKIP_VISION" = "0" ]; then
  log_section "STEP 2: ur5_pickplace - PLANNER / VALIDATOR / VISION"

  activate_conda "$PLANNER_ENV"

  export TEST_NAME="$TEST_NAME"

  if [ "$SKIP_PLANNER" = "0" ]; then
    run_if_exists "$PLANNER_SCRIPT" --test_name "$TEST_NAME"
  else
    echo "[SKIP] Planner skipped."
  fi

  if [ "$SKIP_VALIDATOR" = "0" ]; then
    run_if_exists "$VALIDATOR_SCRIPT" --test_name "$TEST_NAME"
  else
    echo "[SKIP] Validator skipped."
  fi

  need_file "$VISION_SCRIPT" "Vision script"

  echo "[RUN] python $VISION_SCRIPT --test_name $TEST_NAME"
  if python "$VISION_SCRIPT" --test_name "$TEST_NAME"; then
    echo "[OK] Vision script selesai dengan --test_name."
  else
    echo "[WARN] Vision script gagal dengan --test_name. Mencoba tanpa argumen."
    python "$VISION_SCRIPT"
  fi

  need_file "$MASK_PATH" "FastSAM mask"

  conda deactivate || true
else
  log_section "STEP 2: SKIP VISION"
  need_file "$MASK_PATH" "FastSAM mask"
fi


# ============================================================
# STEP 3: anygrasp_py310 - GRASPNET
# ============================================================

if [ "$SKIP_GRASPNET" = "0" ]; then
  log_section "STEP 3: anygrasp_py310 - GRASPNET"

  activate_conda "$ANYGRASP_ENV"

  export TEST_NAME="$TEST_NAME"

  need_file "$CHECKPOINT_PATH" "GraspNet checkpoint"
  need_file "$MASK_PATH" "FastSAM mask"

  echo "[RUN] python $ANYGRASP_SCRIPT --checkpoint_path $CHECKPOINT_PATH --test_name $TEST_NAME ..."
  python "$ANYGRASP_SCRIPT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --test_name "$TEST_NAME" \
    --num_point 20000 \
    --num_view 300 \
    --collision_thresh 0.01 \
    --voxel_size 0.01 \
    --no_vis

  need_file "$BEST_GRASP_CAMERA_JSON" "best_grasp_camera.json"

  conda deactivate || true
else
  log_section "STEP 3: SKIP GRASPNET"
  need_file "$BEST_GRASP_CAMERA_JSON" "best_grasp_camera.json"
fi


# ============================================================
# STEP 4: ur5_pickplace - TRANSFORM + CONVERT + NUDGE + BASE_LINK
# ============================================================

log_section "STEP 4: ur5_pickplace - TRANSFORM + CONVERT + NUDGE + BASE_LINK"

activate_conda "$PLANNER_ENV"

export TEST_NAME="$TEST_NAME"

if [ "$SKIP_TRANSFORM" = "0" ]; then
  need_file "$TRANSFORM_SCRIPT" "Transform script"
  echo "[RUN] python $TRANSFORM_SCRIPT"
  python "$TRANSFORM_SCRIPT"
  need_file "$BEST_GRASP_BASE_JSON" "best_grasp_base.json"
else
  echo "[SKIP] Transform skipped."
  need_file "$BEST_GRASP_BASE_JSON" "best_grasp_base.json"
fi

if [ "$SKIP_CONVERT" = "0" ]; then
  need_file "$CONVERT_SCRIPT" "Convert gripper_tip -> tool0 script"

  echo "[RUN] python $CONVERT_SCRIPT ..."
  python "$CONVERT_SCRIPT" \
    --best_grasp_base "$BEST_GRASP_BASE_JSON" \
    --output "$TOOL0_PREGRASP_TARGET_JSON" \
    --tool0_to_gripper_tip "$TOOL0_TO_GRIPPER_TIP" \
    --pregrasp_z "$PREGRASP_Z"

  need_file "$TOOL0_PREGRASP_TARGET_JSON" "tool0_pregrasp_target.json sebelum nudge"
else
  echo "[SKIP] Convert skipped."
  need_file "$TOOL0_PREGRASP_TARGET_JSON" "tool0_pregrasp_target.json"
fi

if [ "$SKIP_NUDGE" = "0" ]; then
  need_file "$NUDGE_SCRIPT" "Nudge script"

  echo "[RUN] python3 $NUDGE_SCRIPT --dx $NUDGE_DX --dy $NUDGE_DY --dz $NUDGE_DZ"
  python3 "$NUDGE_SCRIPT" \
    --target_json "$TOOL0_PREGRASP_TARGET_JSON" \
    --dx "$NUDGE_DX" \
    --dy "$NUDGE_DY" \
    --dz "$NUDGE_DZ"

  need_file "$TOOL0_PREGRASP_TARGET_JSON" "tool0_pregrasp_target.json setelah nudge"
else
  echo "[SKIP] Nudge skipped."
fi

python - <<PY
import json
p = r"$TOOL0_PREGRASP_TARGET_JSON"
with open(p, "r") as f:
    data = json.load(f)
print("[CHECK BASE] success:", data.get("success"))
print("[CHECK BASE] frame:", data.get("frame"))
print("[CHECK BASE] translation_tool0_pregrasp:", data.get("translation_tool0_pregrasp"))
print("[CHECK BASE] quaternion_tool0_xyzw:", data.get("quaternion_tool0_xyzw"))
print("[CHECK BASE] manual_nudge_base_m:", data.get("manual_nudge_base_m"))
PY

if [ "$SKIP_BASE_LINK_CONVERT" = "0" ]; then
  need_file "$BASE_TO_BASE_LINK_SCRIPT" "Base to base_link conversion script"

  # Hapus output lama supaya tidak pernah pakai target base_link stale.
  rm -f "$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"

  echo "[RUN] python $BASE_TO_BASE_LINK_SCRIPT --input $TOOL0_PREGRASP_TARGET_JSON --output $TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"
  python "$BASE_TO_BASE_LINK_SCRIPT" \
    --input "$TOOL0_PREGRASP_TARGET_JSON" \
    --output "$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"

  need_file "$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON" "tool0_pregrasp_target_base_link.json"
else
  echo "[SKIP] Base -> base_link conversion skipped."
  need_file "$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON" "tool0_pregrasp_target_base_link.json"
fi

python - <<PY
import json
p = r"$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"
with open(p, "r") as f:
    data = json.load(f)
print("[CHECK BASE_LINK] success:", data.get("success"))
print("[CHECK BASE_LINK] frame:", data.get("frame"))
print("[CHECK BASE_LINK] translation_tool0_pregrasp:", data.get("translation_tool0_pregrasp"))
print("[CHECK BASE_LINK] original base:", data.get("translation_tool0_pregrasp_original_base"))
print("[CHECK BASE_LINK] manual_nudge_base_m:", data.get("manual_nudge_base_m"))
PY

conda deactivate || true


# ============================================================
# STEP 5: ROS NOETIC SYSTEM PYTHON
# ============================================================

log_section "STEP 5: ROS NOETIC SYSTEM PYTHON"

deactivate_all_conda

if [ ! -f "$ROS_SETUP" ]; then
  echo "[ERROR] ROS setup tidak ditemukan:"
  echo "  $ROS_SETUP"
  exit 1
fi

source "$ROS_SETUP"

if [ -f "$CATKIN_SETUP" ]; then
  source "$CATKIN_SETUP"
else
  echo "[WARN] Catkin setup tidak ditemukan:"
  echo "  $CATKIN_SETUP"
fi

echo "[CHECK] Clean ROS Python import..."

python3 - <<'PY'
import os
print("[DEBUG] python executable check via system python")
print("[DEBUG] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))

import ctypes
print("[OK] ctypes import OK")

import moveit_commander
print("[OK] moveit_commander import OK")
PY

echo "[INFO] Python ROS: $(which python3)"
python3 --version

echo "[CHECK] rosnode /move_group..."
if ! rosnode list 2>/tmp/rosnode_check_err.txt | grep -q "/move_group"; then
  echo "[ERROR] /move_group tidak aktif."
  echo "Jangan lanjut ke grasp."
  echo ""
  echo "Jalankan dulu di terminal lain:"
  echo "  source /opt/ros/noetic/setup.bash"
  echo "  source ~/ur5_noetic_ws/devel/setup.bash"
  echo "  roslaunch ur5_moveit_config moveit_planning_execution.launch"
  echo ""
  echo "[DEBUG] rosnode error:"
  cat /tmp/rosnode_check_err.txt || true
  exit 1
else
  echo "[OK] /move_group aktif."
fi

echo "[CHECK] TF base_link -> tool0..."
if ! timeout 3 rosrun tf tf_echo base_link tool0 >/tmp/tf_base_link_tool0_check.txt 2>&1; then
  echo "[WARN] TF base_link -> tool0 belum terbaca dari tf_echo."
  echo "Kalau MoveIt tetap bisa membaca current pose, ini tidak selalu fatal."
else
  echo "[OK] TF base_link -> tool0 terbaca."
fi


# ============================================================
# STEP 6: OPTIONAL PRE-OPEN GRIPPER
# ============================================================

log_section "STEP 6: OPTIONAL PRE-OPEN GRIPPER"

if [ "$DISABLE_GRIPPER" = "1" ]; then
  echo "[SKIP] DISABLE_GRIPPER=1"
elif [ "$PREOPEN_GRIPPER" = "1" ]; then
  need_file "$GRIPPER_SCRIPT" "Gripper script"

  python3 "$GRIPPER_SCRIPT" \
    --robot_ip "$ROBOT_IP" \
    open
else
  echo "[SKIP] PREOPEN_GRIPPER=0"
  echo "[INFO] grasp.py akan open gripper di pregrasp."
fi


# ============================================================
# STEP 7: MOVE TO PREGRASP + DESCEND + CLOSE
# ============================================================

if [ "$SKIP_GRASP" = "0" ]; then
  log_section "STEP 7: MOVE TO PREGRASP + DESCEND + CLOSE"

  if [ -f "$GRASP_SCRIPT" ]; then
    ACTIVE_GRASP_SCRIPT="$GRASP_SCRIPT"
  elif [ -f "$FALLBACK_GRASP_SCRIPT" ]; then
    ACTIVE_GRASP_SCRIPT="$FALLBACK_GRASP_SCRIPT"
  else
    echo "[ERROR] Grasp script tidak ditemukan:"
    echo "  $GRASP_SCRIPT"
    echo "  $FALLBACK_GRASP_SCRIPT"
    exit 1
  fi

  echo "[RUN] python3 $ACTIVE_GRASP_SCRIPT --target_json $TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"

  grasp_cmd=(
    python3 "$ACTIVE_GRASP_SCRIPT"
    --target_json "$TOOL0_PREGRASP_TARGET_BASE_LINK_JSON"
    --robot_ip "$ROBOT_IP"
    --reference_frame base_link
    --orientation_mode current
    --skip_observation
    --safe_lift_z "$SAFE_LIFT_Z"
    --safe_lift_eef_step "$SAFE_LIFT_EEF_STEP"
    --cartesian_min_fraction "$CARTESIAN_MIN_FRACTION"
    --max_base_delta "$MAX_BASE_DELTA"
    --max_wrist_delta "$MAX_WRIST_DELTA"
    --max_xyz_distance "$MAX_XYZ_DISTANCE"
    --descend_z "$DESCEND_Z"
    --pregrasp_wait "$PREGRASP_WAIT"
    --rviz_preview_wait "$RVIZ_PREVIEW_WAIT"
    --velocity "$VELOCITY"
    --acceleration "$ACCELERATION"
  )

  if [ "$DISABLE_GRIPPER" = "1" ]; then
    grasp_cmd+=(--disable_gripper)
  fi

  if [ -n "$EXECUTE_FLAG" ]; then
    grasp_cmd+=("$EXECUTE_FLAG")
  fi

  echo "[CMD] ${grasp_cmd[*]}"
  "${grasp_cmd[@]}"

  echo "[OK] Grasp script selesai."
else
  log_section "STEP 7: SKIP GRASP"
  echo "[INFO] Diasumsikan objek sudah tergenggam."
fi


# ============================================================
# STEP 8: DISCARD
# ============================================================

if [ "$SKIP_DISCARD" = "0" ]; then
  log_section "STEP 8: LIFT + DISCARD WAYPOINTS + OPEN + IDLE"

  need_file "$DISCARD_SCRIPT" "Discard script"
  need_file "$WAYPOINTS_JSON" "Waypoints JSON"

  discard_cmd=(
    python3 "$DISCARD_SCRIPT"
    --robot_ip "$ROBOT_IP"
    --waypoints_json "$WAYPOINTS_JSON"
    --lift_up "$LIFT_UP"
    --velocity "$VELOCITY"
    --acceleration "$ACCELERATION"
  )

  if [ -n "$GRIPPER_FLAG" ]; then
    discard_cmd+=("$GRIPPER_FLAG")
  fi

  if [ -n "$EXECUTE_FLAG" ]; then
    discard_cmd+=("$EXECUTE_FLAG")
  fi

  echo "[CMD] ${discard_cmd[*]}"
  "${discard_cmd[@]}"
else
  log_section "STEP 8: SKIP DISCARD"
fi


log_section "DONE"
echo "[DONE] First pipeline selesai."
