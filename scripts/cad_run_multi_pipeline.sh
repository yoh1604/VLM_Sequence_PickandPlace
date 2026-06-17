#!/usr/bin/env bash
set -euo pipefail

TEST_NAME="${1:-water}"
MAX_STEPS="${MAX_STEPS:-5}"

export PREGRASP_Z=0.13
export DESCEND_Z=0.12
export NUDGE_DX=-0.05
export NUDGE_DY=0.038
export NUDGE_DZ=0.0
export MAX_WRIST_DELTA=1.40
export MAX_BASE_DELTA=1.80
export MAX_XYZ_DISTANCE=0.80
export SAFE_LIFT_Z=0.0
export CARTESIAN_MIN_FRACTION=0.80

# STEP 1: run first pipeline normal
export SKIP_DISCARD=0
./scripts/run_first_pipeline.sh "$TEST_NAME" plan

STEP_INDEX=1

while true; do
  echo "========== POST CHECK STEP $STEP_INDEX =========="

  # 1. Pastikan robot sudah IDLE setelah discard
  # 2. Capture scene baru
  conda run -n ur5_pickplace python perception/capture_d455_once.py

  # 3. Copy ke post scene
  cp data/d455_capture/current_scene_rgb.jpg data/d455_capture/post_scene_rgb.jpg

  # 4. Update STEP_INDEX di capture_config.py
  python3 - <<PY
from pathlib import Path
import re

p = Path("capture_config.py")
s = p.read_text()
s = re.sub(r'^TEST_NAME\s*=.*$', 'TEST_NAME = "$TEST_NAME"', s, flags=re.M)
s = re.sub(r'^STEP_INDEX\s*=\s*\d+', 'STEP_INDEX = $STEP_INDEX', s, flags=re.M)
p.write_text(s)
print("[OK] STEP_INDEX updated:", $STEP_INDEX)
PY

  # 5. Run post-check
  conda run -n ur5_pickplace python run_post_complete.py

  REMAINING="outputs/$TEST_NAME/post_check_output/STEP_${STEP_INDEX}_remaining_plan.json"

  # 6. Kalau remaining kosong, selesai
  python3 - <<PY
import json, sys
p = "$REMAINING"
data = json.load(open(p))
print("[CHECK] remaining_plan:", data)
sys.exit(0 if len(data) == 0 else 1)
PY

  if [ $? -eq 0 ]; then
    echo "[DONE] Semua step selesai."
    break
  fi

  # 7. Kalau masih ada target berikutnya,
  # run robot target generation dari hasil post-check
  echo "[INFO] Ada target berikutnya. Generate robot target."

  export SKIP_CAPTURE=1
  export SKIP_PLANNER=1
  export SKIP_VALIDATOR=1
  export SKIP_VISION=1

  export SKIP_GRASPNET=0
  export SKIP_TRANSFORM=0
  export SKIP_CONVERT=0
  export SKIP_NUDGE=0
  export SKIP_BASE_LINK_CONVERT=0

  export SKIP_GRASP=1
  export SKIP_DISCARD=1

  ./scripts/run_first_pipeline.sh "$TEST_NAME" plan

  # 8. Pastikan base_link target fresh
  rm -f outputs/$TEST_NAME/vision_output/tool0_pregrasp_target_base_link.json
  conda run -n ur5_pickplace python perception/convert_tool0_target_base_to_base_link.py \
    --input outputs/$TEST_NAME/vision_output/tool0_pregrasp_target.json \
    --output outputs/$TEST_NAME/vision_output/tool0_pregrasp_target_base_link.json

  # 9. Grasp target berikutnya
  python3 robot_executor/grasp.py \
    --target_json outputs/$TEST_NAME/vision_output/tool0_pregrasp_target_base_link.json \
    --robot_ip 192.168.200.1 \
    --reference_frame base_link \
    --orientation_mode current \
    --skip_observation \
    --safe_lift_z 0.0 \
    --safe_lift_eef_step 0.005 \
    --cartesian_min_fraction 0.80 \
    --max_base_delta 1.80 \
    --max_wrist_delta 1.40 \
    --max_xyz_distance 0.80 \
    --descend_z 0.12 \
    --pregrasp_wait 5.0 \
    --rviz_preview_wait 3.0 \
    --execute

  # 10. Discard
  python3 robot_executor/execute_from_current_pregrasp_to_discard.py \
    --robot_ip 192.168.200.1 \
    --waypoints_json configs/waypoints_ur5.json \
    --lift_up 0.04 \
    --velocity 0.05 \
    --acceleration 0.05 \
    --execute

  STEP_INDEX=$((STEP_INDEX + 1))

  if [ "$STEP_INDEX" -gt "$MAX_STEPS" ]; then
    echo "[STOP] STEP_INDEX melebihi MAX_STEPS."
    break
  fi
done