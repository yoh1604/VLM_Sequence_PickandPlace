import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT CONFIG
# ============================================================

PROJECT_DIR = Path.home() / "Documents" / "pick_place_occlusion_noetic"


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path):
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {
            "error": str(e),
            "path": str(path),
        }


def file_exists(path):
    return Path(path).exists()


def file_status(path):
    path = Path(path)
    if path.exists():
        return "✅ exists"
    return "❌ missing"


def file_mtime(path):
    path = Path(path)
    if not path.exists():
        return "-"
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def show_json(title, path):
    path = Path(path)
    st.subheader(title)
    st.caption(str(path))

    data = load_json(path)
    if data is None:
        st.warning("File belum ada.")
        return None

    st.json(data)
    return data


def show_image(title, path):
    path = Path(path)
    st.subheader(title)
    st.caption(str(path))

    if not path.exists():
        st.warning("Image belum ada.")
        return

    st.image(str(path), use_container_width=True)


def show_file_info(path):
    path = Path(path)
    if not path.exists():
        st.warning(f"Missing: {path}")
        return

    st.caption(f"Modified: {file_mtime(path)} | Size: {path.stat().st_size} bytes")


# ============================================================
# STEP HELPERS
# ============================================================

def get_post_steps(post_dir):
    post_dir = Path(post_dir)
    if not post_dir.exists():
        return []

    steps = []
    for p in sorted(post_dir.glob("STEP_*_post_check_result.json")):
        try:
            # STEP_1_post_check_result.json
            step_num = int(p.name.split("_")[1])
            steps.append(step_num)
        except Exception:
            pass

    return sorted(set(steps))


def get_next_target_steps(post_dir):
    post_dir = Path(post_dir)
    if not post_dir.exists():
        return []

    steps = []
    for p in sorted(post_dir.glob("STEP_*_next_target")):
        try:
            # STEP_2_next_target
            step_num = int(p.name.split("_")[1])
            steps.append(step_num)
        except Exception:
            pass

    return sorted(set(steps))


def summarize_action_plan(vlm_dir):
    validation_path = Path(vlm_dir) / "validation_result_real.json"
    action_path = Path(vlm_dir) / "action_plan_real.json"

    validation = load_json(validation_path)
    action_plan_json = load_json(action_path)

    plan = []

    if isinstance(validation, dict):
        plan = validation.get("final_action_plan", [])

    if not plan and isinstance(action_plan_json, dict):
        plan = action_plan_json.get("action_plan", [])

    if not isinstance(plan, list):
        return pd.DataFrame()

    rows = []
    for item in plan:
        if not isinstance(item, dict):
            continue

        rows.append({
            "step": item.get("step"),
            "action": item.get("action"),
            "target": item.get("target"),
            "target_role": item.get("target_role"),
            "step_intent": item.get("step_intent"),
            "explanation": item.get("explanation"),
        })

    return pd.DataFrame(rows)


def summarize_post_checks(post_dir):
    post_dir = Path(post_dir)
    rows = []

    for step in get_post_steps(post_dir):
        p = post_dir / f"STEP_{step}_post_check_result.json"
        data = load_json(p) or {}

        best = data.get("best_detection") or {}

        rows.append({
            "step": step,
            "target": data.get("target"),
            "status": data.get("post_check_status"),
            "found_after_action": data.get("target_found_after_action"),
            "confidence": best.get("confidence"),
            "modified": file_mtime(p),
        })

    return pd.DataFrame(rows)


def summarize_next_targets(post_dir):
    post_dir = Path(post_dir)
    rows = []

    for step in get_next_target_steps(post_dir):
        next_dir = post_dir / f"STEP_{step}_next_target"
        result_path = next_dir / f"STEP_{step}_next_target_result.json"
        data = load_json(result_path) or {}

        best = data.get("best_detection") or {}

        rows.append({
            "prepared_for_step": step,
            "next_target": data.get("next_target"),
            "query_used": data.get("target_query_used"),
            "conf_used": data.get("confidence_threshold_used"),
            "det_confidence": best.get("confidence"),
            "modified": file_mtime(result_path),
        })

    return pd.DataFrame(rows)


def summarize_file_table(vlm_dir, vision_dir, post_dir):
    files = [
        ("VLM action_plan_real.json", Path(vlm_dir) / "action_plan_real.json"),
        ("VLM validation_result_real.json", Path(vlm_dir) / "validation_result_real.json"),
        ("Vision detections_yolo.json", Path(vision_dir) / "detections_yolo.json"),
        ("Vision yolo_world_result.jpg", Path(vision_dir) / "yolo_world_result.jpg"),
        ("Vision fastsam_mask.png", Path(vision_dir) / "fastsam_mask.png"),
        ("Vision fastsam_result.jpg", Path(vision_dir) / "fastsam_result.jpg"),
        ("Vision object_position_camera.json", Path(vision_dir) / "object_position_camera.json"),
        ("Grasp best_grasp_camera.json", Path(vision_dir) / "best_grasp_camera.json"),
        ("Grasp best_grasp_base.json", Path(vision_dir) / "best_grasp_base.json"),
        ("Grasp tool0_pregrasp_target.json", Path(vision_dir) / "tool0_pregrasp_target.json"),
        ("Grasp tool0_pregrasp_target_base_link.json", Path(vision_dir) / "tool0_pregrasp_target_base_link.json"),
        ("Next next_target_ready.json", Path(vision_dir) / "next_target_ready.json"),
        ("Retry retry_same_step_ready.json", Path(vision_dir) / "retry_same_step_ready.json"),
    ]

    rows = []
    for label, path in files:
        rows.append({
            "name": label,
            "status": file_status(path),
            "modified": file_mtime(path),
            "path": str(path),
        })

    return pd.DataFrame(rows)


# ============================================================
# UI SETUP
# ============================================================

st.set_page_config(
    page_title="UR5 Occluded Grasping Dashboard",
    layout="wide",
)

st.title("UR5 Occluded Grasping Dashboard")
st.caption("Dashboard read-only untuk melihat output VLM, vision, grasp, post-check, retry, dan remaining plan.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuration")

default_test_name = "soda_valid_09"

test_name = st.sidebar.text_input(
    "TEST_NAME",
    value=default_test_name,
)

test_dir = PROJECT_DIR / "outputs" / test_name
vlm_dir = test_dir / "vlm_output"
vision_dir = test_dir / "vision_output"
post_dir = test_dir / "post_check_output"
snapshot_dir = test_dir / "snapshots"
log_path = PROJECT_DIR / "logs" / f"{test_name}.log"

st.sidebar.caption(f"PROJECT_DIR: {PROJECT_DIR}")
st.sidebar.caption(f"TEST_DIR: {test_dir}")

if st.sidebar.button("Refresh"):
    st.rerun()


# ============================================================
# MAIN PATH STATUS
# ============================================================

st.header("1. Test Folder Status")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Test folder", "exists" if test_dir.exists() else "missing")
with col_b:
    post_steps = get_post_steps(post_dir)
    st.metric("Post-check steps", len(post_steps))
with col_c:
    next_steps = get_next_target_steps(post_dir)
    st.metric("Next target folders", len(next_steps))

st.dataframe(
    summarize_file_table(vlm_dir, vision_dir, post_dir),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ACTION PLAN
# ============================================================

st.header("2. Action Plan")

plan_df = summarize_action_plan(vlm_dir)

if len(plan_df) == 0:
    st.warning("Belum ada action plan / validation result.")
else:
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

with st.expander("Show raw VLM JSON"):
    tab_vlm_1, tab_vlm_2, tab_vlm_3 = st.tabs([
        "action_plan_real.json",
        "validation_result_real.json",
        "detections_from_vlm.json",
    ])

    with tab_vlm_1:
        show_json("Action Plan Real", vlm_dir / "action_plan_real.json")

    with tab_vlm_2:
        show_json("Validation Result Real", vlm_dir / "validation_result_real.json")

    with tab_vlm_3:
        show_json("Detections From VLM", vlm_dir / "detections_from_vlm.json")


# ============================================================
# POST-CHECK SUMMARY
# ============================================================

st.header("3. Post-check Summary")

post_df = summarize_post_checks(post_dir)

if len(post_df) == 0:
    st.warning("Belum ada post-check result.")
else:
    st.dataframe(post_df, use_container_width=True, hide_index=True)


# ============================================================
# NEXT TARGET SUMMARY
# ============================================================

st.header("4. Next Target Summary")

next_df = summarize_next_targets(post_dir)

if len(next_df) == 0:
    st.info("Belum ada folder STEP_X_next_target.")
else:
    st.dataframe(next_df, use_container_width=True, hide_index=True)


# ============================================================
# STEP VIEWER
# ============================================================

# ============================================================
# STEP VIEWER: PRE-CHECK + POST-CHECK
# ============================================================

st.header("5. Step Viewer: PRE-check and POST-check")

pre_dir = test_dir / "pre_check_output"

steps = sorted(set(get_post_steps(post_dir) + [
    int(p.name.split("_")[1])
    for p in pre_dir.glob("STEP_*_latest")
    if pre_dir.exists() and p.is_dir()
]))

if not steps:
    st.warning("Belum ada step output.")
    selected_step = None
else:
    selected_step = st.selectbox(
        "Pilih STEP_INDEX",
        steps,
        index=0,
    )


def show_precheck(selected_step):
    st.subheader(f"STEP {selected_step} PRE-check")

    archived_pre_dir = pre_dir / f"STEP_{selected_step}_latest"

    # Fallback lama:
    # STEP 1 awal biasanya dari vision_output.
    # STEP 2 dst biasanya dari post_check_output/STEP_X_next_target.
    next_target_dir = post_dir / f"STEP_{selected_step}_next_target"

    if archived_pre_dir.exists():
        st.success(f"PRE-check archive ditemukan: {archived_pre_dir}")

        col1, col2 = st.columns(2)

        with col1:
            show_image(
                f"STEP {selected_step} PRE Scene RGB",
                archived_pre_dir / "pre_scene_rgb.jpg",
            )

            show_image(
                f"STEP {selected_step} PRE YOLO Result",
                archived_pre_dir / "pre_yolo_result.jpg",
            )

            show_image(
                f"STEP {selected_step} PRE FastSAM Result",
                archived_pre_dir / "pre_fastsam_result.jpg",
            )

        with col2:
            show_json(
                f"STEP {selected_step} PRE Detections YOLO",
                archived_pre_dir / "pre_detections_yolo.json",
            )

            show_json(
                f"STEP {selected_step} PRE Object Position",
                archived_pre_dir / "pre_object_position_camera.json",
            )

            show_json(
                f"STEP {selected_step} PRE Info",
                archived_pre_dir / "precheck_info.json",
            )

        with st.expander(f"STEP {selected_step} PRE Grasp Target"):
            show_json(
                f"STEP {selected_step} PRE Best Grasp Camera",
                archived_pre_dir / "pre_best_grasp_camera.json",
            )

            show_json(
                f"STEP {selected_step} PRE Best Grasp Base",
                archived_pre_dir / "pre_best_grasp_base.json",
            )

            show_json(
                f"STEP {selected_step} PRE Tool0 Target",
                archived_pre_dir / "pre_tool0_pregrasp_target.json",
            )

            show_json(
                f"STEP {selected_step} PRE Tool0 Target Base Link",
                archived_pre_dir / "pre_tool0_pregrasp_target_base_link.json",
            )

    elif selected_step == 1:
        st.warning(
            "PRE-check archive STEP 1 belum ada. "
            "Fallback ke vision_output. Perhatian: vision_output bisa tertimpa oleh step berikutnya."
        )

        col1, col2 = st.columns(2)

        with col1:
            show_image(
                "STEP 1 PRE / Latest vision_output YOLO",
                vision_dir / "yolo_world_result.jpg",
            )

            show_image(
                "STEP 1 PRE / Latest vision_output FastSAM",
                vision_dir / "fastsam_result.jpg",
            )

        with col2:
            show_json(
                "STEP 1 PRE / Latest YOLO Detections",
                vision_dir / "detections_yolo.json",
            )

            show_json(
                "STEP 1 PRE / Latest Object Position",
                vision_dir / "object_position_camera.json",
            )

    elif next_target_dir.exists():
        st.info(
            f"PRE-check archive STEP {selected_step} belum ada. "
            f"Fallback ke folder next target: {next_target_dir}"
        )

        col1, col2 = st.columns(2)

        with col1:
            show_image(
                f"STEP {selected_step} PRE YOLO Result",
                next_target_dir / f"STEP_{selected_step}_next_yolo_result.jpg",
            )

            show_image(
                f"STEP {selected_step} PRE FastSAM Result",
                next_target_dir / f"STEP_{selected_step}_next_fastsam_result.jpg",
            )

            show_image(
                f"STEP {selected_step} PRE FastSAM Mask",
                next_target_dir / f"STEP_{selected_step}_next_fastsam_mask.png",
            )

        with col2:
            show_json(
                f"STEP {selected_step} PRE Next Target Result",
                next_target_dir / f"STEP_{selected_step}_next_target_result.json",
            )

            show_json(
                f"STEP {selected_step} PRE Detections YOLO",
                next_target_dir / f"STEP_{selected_step}_next_detections_yolo.json",
            )

            show_json(
                f"STEP {selected_step} PRE Object Position",
                next_target_dir / f"STEP_{selected_step}_next_object_position_camera.json",
            )

    else:
        st.warning(f"PRE-check untuk STEP {selected_step} belum ditemukan.")


def show_postcheck(selected_step):
    st.subheader(f"STEP {selected_step} POST-check")

    step_post_result = post_dir / f"STEP_{selected_step}_post_check_result.json"
    step_post_yolo_json = post_dir / f"STEP_{selected_step}_post_check_detections_yolo.json"
    step_post_yolo_image = post_dir / f"STEP_{selected_step}_post_check_yolo_result.jpg"
    step_post_fastsam_mask = post_dir / f"STEP_{selected_step}_post_check_fastsam_mask.png"
    step_post_fastsam_image = post_dir / f"STEP_{selected_step}_post_check_fastsam_result.jpg"
    step_remaining_plan = post_dir / f"STEP_{selected_step}_remaining_plan.json"
    step_sync_report = post_dir / f"STEP_{selected_step}_sync_report.json"

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            f"STEP {selected_step} POST YOLO Result",
            step_post_yolo_image,
        )

        if step_post_fastsam_image.exists():
            show_image(
                f"STEP {selected_step} POST FastSAM Result",
                step_post_fastsam_image,
            )
        else:
            st.info(
                "POST FastSAM tidak ada. Ini normal jika target sudah tidak ditemukan "
                "atau status post-check adalah REMOVED_SUCCESS."
            )

        if step_post_fastsam_mask.exists():
            show_image(
                f"STEP {selected_step} POST FastSAM Mask",
                step_post_fastsam_mask,
            )

    with col2:
        show_json(
            f"STEP {selected_step} POST Result",
            step_post_result,
        )

        show_json(
            f"STEP {selected_step} Remaining Plan",
            step_remaining_plan,
        )

        show_json(
            f"STEP {selected_step} Sync Report",
            step_sync_report,
        )

    with st.expander(f"STEP {selected_step} POST Raw YOLO JSON"):
        show_json(
            f"STEP {selected_step} POST YOLO Detections",
            step_post_yolo_json,
        )


if selected_step is not None:
    tab_pre, tab_post, tab_compare = st.tabs([
        "PRE-check",
        "POST-check",
        "PRE vs POST",
    ])

    with tab_pre:
        show_precheck(selected_step)

    with tab_post:
        show_postcheck(selected_step)

    with tab_compare:
        st.subheader(f"STEP {selected_step} PRE vs POST")

        archived_pre_dir = pre_dir / f"STEP_{selected_step}_latest"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### PRE")
            if archived_pre_dir.exists():
                show_image(
                    f"STEP {selected_step} PRE YOLO",
                    archived_pre_dir / "pre_yolo_result.jpg",
                )
                show_image(
                    f"STEP {selected_step} PRE FastSAM",
                    archived_pre_dir / "pre_fastsam_result.jpg",
                )
            elif selected_step == 1:
                show_image(
                    "STEP 1 PRE YOLO fallback",
                    vision_dir / "yolo_world_result.jpg",
                )
                show_image(
                    "STEP 1 PRE FastSAM fallback",
                    vision_dir / "fastsam_result.jpg",
                )
            else:
                next_target_dir = post_dir / f"STEP_{selected_step}_next_target"
                show_image(
                    f"STEP {selected_step} PRE YOLO fallback",
                    next_target_dir / f"STEP_{selected_step}_next_yolo_result.jpg",
                )
                show_image(
                    f"STEP {selected_step} PRE FastSAM fallback",
                    next_target_dir / f"STEP_{selected_step}_next_fastsam_result.jpg",
                )

        with col2:
            st.markdown("### POST")
            show_image(
                f"STEP {selected_step} POST YOLO",
                post_dir / f"STEP_{selected_step}_post_check_yolo_result.jpg",
            )

            post_fastsam = post_dir / f"STEP_{selected_step}_post_check_fastsam_result.jpg"
            if post_fastsam.exists():
                show_image(
                    f"STEP {selected_step} POST FastSAM",
                    post_fastsam,
                )
            else:
                st.info("POST FastSAM tidak ada jika target sudah hilang.")

steps = get_post_steps(post_dir)

if not steps:
    st.warning("Belum ada STEP_X_post_check_result.json.")
    selected_step = None
else:
    selected_step = st.selectbox(
        "Pilih STEP_INDEX untuk post-check",
        steps,
        index=0,  # default tampilkan STEP 1, bukan step terakhir
    )

if selected_step is not None:
    st.subheader(f"STEP {selected_step} Pre-check vs Post-check")

    step_post_result = post_dir / f"STEP_{selected_step}_post_check_result.json"
    step_post_yolo_json = post_dir / f"STEP_{selected_step}_post_check_detections_yolo.json"
    step_post_yolo_image = post_dir / f"STEP_{selected_step}_post_check_yolo_result.jpg"
    step_post_fastsam_mask = post_dir / f"STEP_{selected_step}_post_check_fastsam_mask.png"
    step_pre_fastsam_image = post_dir / f"STEP_{selected_step}_post_check_fastsam_result.jpg"
    step_remaining_plan = post_dir / f"STEP_{selected_step}_remaining_plan.json"
    step_sync_report = post_dir / f"STEP_{selected_step}_sync_report.json"

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            f"STEP {selected_step} Pre-check Result",
            step_pre_fastsam_image,
        )

        show_image(
            f"STEP {selected_step} Post-check YOLO Result",
            step_post_yolo_image,
        )

        

    with col2:
        show_json(
            f"STEP {selected_step} Post-check Result",
            step_post_result,
        )

        show_json(
            f"STEP {selected_step} Remaining Plan",
            step_remaining_plan,
        )

        show_json(
            f"STEP {selected_step} Sync Report",
            step_sync_report,
        )

    with st.expander(f"STEP {selected_step} Raw Post-check Files"):
        show_json(
            f"STEP {selected_step} YOLO Detections",
            step_post_yolo_json,
        )

        if step_post_fastsam_mask.exists():
            st.caption(str(step_post_fastsam_mask))
            st.image(str(step_post_fastsam_mask), use_container_width=True)
        else:
            st.warning("FastSAM mask belum ada untuk step ini.")


# ============================================================
# NEXT TARGET VIEWER
# ============================================================

st.header("6. Next Target Viewer")

if selected_step is None:
    st.info("Pilih STEP_INDEX dulu.")
else:
    prepared_step = selected_step + 1
    next_dir = post_dir / f"STEP_{prepared_step}_next_target"

    st.subheader(f"Next target prepared for STEP {prepared_step}")
    st.caption(str(next_dir))

    if not next_dir.exists():
        st.info(f"Tidak ada folder next target: {next_dir}")
    else:
        next_yolo_result = next_dir / f"STEP_{prepared_step}_next_yolo_result.jpg"
        next_fastsam_result = next_dir / f"STEP_{prepared_step}_next_fastsam_result.jpg"
        next_fastsam_mask = next_dir / f"STEP_{prepared_step}_next_fastsam_mask.png"
        next_target_result = next_dir / f"STEP_{prepared_step}_next_target_result.json"
        next_object_position = next_dir / f"STEP_{prepared_step}_next_object_position_camera.json"
        next_alias_debug = next_dir / f"STEP_{prepared_step}_next_yolo_alias_debug.json"

        col1, col2 = st.columns(2)

        with col1:
            show_image(
                f"STEP {prepared_step} Next YOLO Result",
                next_yolo_result,
            )

            show_image(
                f"STEP {prepared_step} Next FastSAM Result",
                next_fastsam_result,
            )

            if next_fastsam_mask.exists():
                show_image(
                    f"STEP {prepared_step} Next FastSAM Mask",
                    next_fastsam_mask,
                )

        with col2:
            show_json(
                f"STEP {prepared_step} Next Target Result",
                next_target_result,
            )

            show_json(
                f"STEP {prepared_step} Next Object Position",
                next_object_position,
            )

            show_json(
                f"STEP {prepared_step} YOLO Alias Debug",
                next_alias_debug,
            )


# ============================================================
# LATEST ACTIVE VISION OUTPUT
# ============================================================

st.header("7. Latest Active Vision Output")

st.info(
    "Folder vision_output adalah output target aktif terakhir. "
    "File di sini bisa tertimpa oleh step berikutnya, jadi jangan dipakai sebagai arsip step 1."
)

col1, col2 = st.columns(2)

with col1:
    show_image("Latest YOLO Result", vision_dir / "yolo_world_result.jpg")
    show_image("Latest FastSAM Result", vision_dir / "fastsam_result.jpg")
    show_image("Latest Depth Visualization", PROJECT_DIR / "data/d455_capture/current_scene_depth.png")

with col2:
    show_json("Latest YOLO Detections", vision_dir / "detections_yolo.json")
    show_json("Latest Object Position Camera", vision_dir / "object_position_camera.json")
    show_json("Next Target Ready", vision_dir / "next_target_ready.json")
    show_json("Retry Same Step Ready", vision_dir / "retry_same_step_ready.json")


# ============================================================
# GRASP OUTPUT
# ============================================================

st.header("8. Latest Grasp Output")

tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
    "best_grasp_camera",
    "best_grasp_base",
    "tool0_target_base",
    "tool0_target_base_link",
])

with tab_g1:
    show_json("Best Grasp Camera", vision_dir / "best_grasp_camera.json")

with tab_g2:
    show_json("Best Grasp Base", vision_dir / "best_grasp_base.json")

with tab_g3:
    show_json("Tool0 Pregrasp Target Base", vision_dir / "tool0_pregrasp_target.json")

with tab_g4:
    show_json("Tool0 Pregrasp Target Base Link", vision_dir / "tool0_pregrasp_target_base_link.json")


# ============================================================
# SNAPSHOTS AND RAW CAMERA FILES
# ============================================================

st.header("9. Snapshots and Camera Files")

col1, col2 = st.columns(2)

with col1:
    show_image("Current Scene RGB", PROJECT_DIR / "data/d455_capture/current_scene_rgb.jpg")
    show_image("Current Scene Depth", PROJECT_DIR / "data/d455_capture/current_scene_depth.png")

with col2:
    show_image("Post Scene RGB", PROJECT_DIR / "data/d455_capture/post_scene_rgb.jpg")
    show_image("Snapshot Post RGB", snapshot_dir / "post_scene_rgb.jpg")


# ============================================================
# FILE TREE HINT
# ============================================================

st.header("10. Output Folder Notes")

st.markdown(
    f"""
**Important folder logic:**

`vision_output/` = latest active target.  
This is usually the newest target and may look like STEP 2 or later.

`post_check_output/STEP_X_*` = archived per-step post-check output.  
Use the Step Viewer above to inspect STEP 1, STEP 2, etc.

`post_check_output/STEP_Y_next_target/` = prepared next target for STEP Y.  
For example, after STEP 1 succeeds, the next target folder is often `STEP_2_next_target`.

Current test folder:

`{test_dir}`
"""
)


# ============================================================
# LOG VIEWER
# ============================================================

st.header("11. Pipeline Log")

st.caption(str(log_path))

if not log_path.exists():
    st.warning("Log file belum ada. Jalankan pipeline dengan tee agar log tersimpan.")
    st.code(
        f"""mkdir -p logs
./scripts/run_multi_pipeline_retry.sh {test_name} execute 2>&1 | tee logs/{test_name}.log""",
        language="bash",
    )
else:
    log_text = log_path.read_text(errors="ignore")
    lines = log_text.splitlines()

    tail_lines = st.slider(
        "Show last N log lines",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
    )

    st.code("\n".join(lines[-tail_lines:]), language="bash")