import json
import os
import re
import select
import subprocess
import time
from datetime import datetime
from pathlib import Path

import gradio as gr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DATA_DIR = PROJECT_DIR / "data" / "d455_capture"
TESTS_DIR = DATA_DIR / "tests"


# ============================================================
# BASIC HELPERS
# ============================================================

def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_int(value, default=1):
    try:
        return int(value)
    except Exception:
        return default


def load_json(path: Path):
    path = Path(path)

    if not path.exists():
        return {
            "_missing": True,
            "_path": str(path),
        }

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {
            "_error": True,
            "_path": str(path),
            "_message": str(e),
        }


def image_or_none(path: Path):
    path = Path(path)
    if path.exists():
        return str(path)
    return None


def find_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return str(p)
    return None


def short_text(value, max_len=140):
    if value is None:
        return "-"

    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
    except Exception:
        text = str(value)

    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def clean_label(value):
    if value in [None, "", [], {}]:
        return "-"
    return str(value).replace("_", " ")


def format_confidence(value):
    if value in [None, "", "-", [], {}]:
        return "-"

    try:
        v = float(value)

        if 0 <= v <= 1:
            return f"{v * 100:.1f}%"

        if 1 < v <= 100:
            return f"{v:.1f}%"

        return str(value)
    except Exception:
        return str(value)


def format_bool_status(value, true_text="Ya", false_text="Tidak"):
    if isinstance(value, bool):
        return true_text if value else false_text

    if isinstance(value, str):
        v = value.lower().strip()

        if v in ["true", "yes", "ya", "pass", "passed", "success", "succeeded", "ok", "done"]:
            return true_text

        if v in ["false", "no", "tidak", "fail", "failed", "error", "missing"]:
            return false_text

    return short_text(value)


def list_previous_tests():
    names = []

    if OUTPUTS_DIR.exists():
        for p in OUTPUTS_DIR.iterdir():
            if p.is_dir():
                names.append(p.name)

    if TESTS_DIR.exists():
        for p in TESTS_DIR.iterdir():
            if p.is_dir():
                names.append(p.name)

    return sorted(list(set(names)), reverse=True)


def resolve_test_dir(test_name: str):
    test_name = str(test_name).strip()

    output_candidate = OUTPUTS_DIR / test_name
    d455_candidate = TESTS_DIR / test_name

    if output_candidate.exists():
        return output_candidate

    if d455_candidate.exists():
        return d455_candidate

    return output_candidate


def get_dirs(test_name: str):
    test_dir = resolve_test_dir(test_name)

    return {
        "test_dir": test_dir,
        "vlm_dir": test_dir / "vlm_output",
        "vision_dir": test_dir / "vision_output",
        "post_dir": test_dir / "post_check_output",
        "snapshot_dir": test_dir / "snapshots",
    }


# ============================================================
# JSON FIELD HELPERS
# ============================================================

def get_field(item, keys, default=None):
    if not isinstance(item, dict):
        return default

    for key in keys:
        if key in item and item[key] not in [None, "", [], {}]:
            return item[key]

    return default


def extract_action_items(action_plan_data):
    if isinstance(action_plan_data, list):
        return action_plan_data

    if not isinstance(action_plan_data, dict):
        return []

    possible_keys = [
        "action_plan",
        "plan",
        "steps",
        "actions",
        "sequence",
        "action_sequence",
    ]

    for key in possible_keys:
        value = action_plan_data.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            for nested_key in possible_keys:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    return nested_value

    return []


def extract_detection_items(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    possible_keys = [
        "detections",
        "all_detections",
        "results",
        "predictions",
        "objects",
        "boxes",
    ]

    for key in possible_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    return []


def choose_best_detection(detections):
    if not detections:
        return None

    # Prefer explicitly selected/best detection
    for det in detections:
        if not isinstance(det, dict):
            continue

        selected = get_field(det, ["selected", "is_best", "best", "matched"], None)
        if selected is True or str(selected).lower() in ["true", "yes", "1"]:
            return det

    # Otherwise choose highest confidence
    best_det = None
    best_score = -1

    for det in detections:
        if not isinstance(det, dict):
            continue

        score = get_field(det, ["confidence", "conf", "score", "probability"], 0)

        try:
            score = float(score)
        except Exception:
            score = 0

        if score > best_score:
            best_score = score
            best_det = det

    return best_det or detections[0]


# ============================================================
# ACTION PLAN VISUALIZATION
# ============================================================

def make_action_plan_cards(action_plan_data):
    items = extract_action_items(action_plan_data)

    if not items:
        return """
<div class="empty-card">
  <b>Action plan belum tersedia.</b><br>
  Jalankan VLM planner terlebih dahulu.
</div>
"""

    cards = ""

    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            step = get_field(item, ["step", "step_index", "id", "order"], idx)
            action = get_field(item, ["action", "operation", "task", "command"], "pick")
            target = get_field(item, ["target", "object", "object_name", "target_object"], "objek")
            reason = get_field(item, ["reason", "rationale", "why", "description"], "")
        else:
            step = idx
            action = str(item)
            target = "objek"
            reason = ""

        action_text = clean_label(action).title()
        target_text = clean_label(target)

        if reason:
            reason_text = clean_label(reason)
        else:
            reason_text = "Langkah ini dipilih berdasarkan urutan rencana dari VLM."

        cards += f"""
<div class="step-card">
  <div class="step-badge">STEP {step}</div>
  <div class="step-title">{action_text}: {target_text}</div>
  <div class="step-desc">{reason_text}</div>
</div>
"""

    return f"""
<div class="card-grid">
{cards}
</div>
"""


# ============================================================
# PRE-CHECK VISUALIZATION
# ============================================================

def make_precheck_summary(detections_json, object_position_json, selected_step):
    detections = extract_detection_items(detections_json)
    best_det = choose_best_detection(detections)

    if best_det and isinstance(best_det, dict):
        object_name = get_field(
            best_det,
            ["label", "class", "name", "target", "object", "class_name"],
            "objek terdeteksi",
        )

        confidence = get_field(
            best_det,
            ["confidence", "conf", "score", "probability"],
            None,
        )

        bbox = get_field(
            best_det,
            ["bbox", "box", "xyxy", "bbox_xyxy", "bounding_box"],
            None,
        )

        found_text = "Objek ditemukan"
        confidence_text = format_confidence(confidence)
        bbox_text = short_text(bbox)
    else:
        object_name = "-"
        found_text = "Objek belum ditemukan"
        confidence_text = "-"
        bbox_text = "-"

    pixel_center = "-"
    depth_text = "-"
    point_camera = "-"

    if isinstance(object_position_json, dict):
        pixel_center = get_field(
            object_position_json,
            ["pixel_center_uv", "center_uv", "center", "uv"],
            "-",
        )

        depth_value = get_field(
            object_position_json,
            ["median_depth_m", "depth_m", "z_m", "depth"],
            None,
        )

        if depth_value not in [None, "", "-", [], {}]:
            try:
                depth_text = f"{float(depth_value):.3f} meter"
            except Exception:
                depth_text = short_text(depth_value)

        point_camera = get_field(
            object_position_json,
            ["point_camera_m", "object_position_camera", "target_position_camera", "camera_point"],
            "-",
        )

    return f"""
<div class="summary-grid">
  <div class="metric-card">
    <div class="metric-label">Status</div>
    <div class="metric-value">{found_text}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Objek utama</div>
    <div class="metric-value">{clean_label(object_name)}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Keyakinan deteksi</div>
    <div class="metric-value">{confidence_text}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Jarak kamera ke objek</div>
    <div class="metric-value">{depth_text}</div>
  </div>
</div>

<div class="explain-card">
  <b>Arti hasil pre-check {selected_step}:</b><br>
  Sistem melihat objek <b>{clean_label(object_name)}</b> sebelum robot bergerak.
  Posisi objek di gambar adalah <b>{short_text(pixel_center)}</b>, dan estimasi posisi 3D kamera adalah
  <b>{short_text(point_camera)}</b>.
</div>
"""


def make_precheck_simple_table(detections_json, object_position_json):
    detections = extract_detection_items(detections_json)
    rows = []

    if detections:
        for idx, det in enumerate(detections, start=1):
            if not isinstance(det, dict):
                continue

            name = get_field(
                det,
                ["label", "class", "name", "target", "object", "class_name"],
                f"objek {idx}",
            )

            confidence = get_field(
                det,
                ["confidence", "conf", "score", "probability"],
                "-",
            )

            bbox = get_field(
                det,
                ["bbox", "box", "xyxy", "bbox_xyxy", "bounding_box"],
                "-",
            )

            rows.append([
                idx,
                clean_label(name),
                format_confidence(confidence),
                short_text(bbox),
            ])

    if not rows:
        rows.append([
            "-",
            "Belum ada objek terdeteksi",
            "-",
            "-",
        ])

    return rows


# ============================================================
# POST-CHECK VISUALIZATION
# ============================================================

def infer_post_status(post_json):
    if not isinstance(post_json, dict):
        return {
            "main_status": "Belum ada hasil post-check",
            "removed": "-",
            "still_visible": "-",
            "next_action": "-",
            "message": "Jalankan post-check setelah robot menyelesaikan aksi.",
        }

    if post_json.get("_missing"):
        return {
            "main_status": "Belum ada hasil post-check",
            "removed": "-",
            "still_visible": "-",
            "next_action": "-",
            "message": "File post-check belum ditemukan.",
        }

    status = get_field(
        post_json,
        ["status", "result", "validation", "post_check"],
        None,
    )

    removed = get_field(
        post_json,
        ["removed", "target_removed", "object_removed", "success"],
        None,
    )

    still_visible = get_field(
        post_json,
        ["still_visible", "target_still_visible", "is_target_still_visible", "object_still_visible"],
        None,
    )

    next_target = get_field(
        post_json,
        ["next_target", "next_object", "next_step", "remaining_target"],
        None,
    )

    message = get_field(
        post_json,
        ["message", "reason", "description", "note"],
        None,
    )

    # Make simple public-facing conclusion
    if removed is True:
        main_status = "Aksi berhasil"
    elif removed is False:
        main_status = "Aksi belum berhasil"
    elif status:
        status_text = str(status).lower()
        if status_text in ["pass", "passed", "success", "succeeded", "ok", "true"]:
            main_status = "Aksi berhasil"
        elif status_text in ["fail", "failed", "false", "error"]:
            main_status = "Aksi belum berhasil"
        else:
            main_status = clean_label(status)
    else:
        main_status = "Hasil sudah tersedia"

    removed_text = format_bool_status(removed) if removed is not None else "-"
    visible_text = format_bool_status(still_visible, true_text="Masih terlihat", false_text="Sudah tidak terlihat") if still_visible is not None else "-"

    if next_target:
        next_action = f"Lanjut ke {clean_label(next_target)}"
    else:
        next_action = "Ikuti remaining plan / langkah berikutnya"

    if not message:
        message = "Post-check digunakan untuk memastikan apakah objek sudah berubah setelah robot bergerak."

    return {
        "main_status": main_status,
        "removed": removed_text,
        "still_visible": visible_text,
        "next_action": next_action,
        "message": clean_label(message),
    }


def make_postcheck_summary(post_json, selected_step):
    info = infer_post_status(post_json)

    return f"""
<div class="summary-grid">
  <div class="metric-card">
    <div class="metric-label">Kesimpulan</div>
    <div class="metric-value">{info["main_status"]}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Objek berhasil dipindah?</div>
    <div class="metric-value">{info["removed"]}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Objek masih terlihat?</div>
    <div class="metric-value">{info["still_visible"]}</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Langkah berikutnya</div>
    <div class="metric-value">{info["next_action"]}</div>
  </div>
</div>

<div class="explain-card">
  <b>Arti hasil post-check {selected_step}:</b><br>
  {info["message"]}
</div>
"""


def make_postcheck_simple_table(post_json):
    info = infer_post_status(post_json)

    return [
        ["Kesimpulan", info["main_status"]],
        ["Objek berhasil dipindah?", info["removed"]],
        ["Objek masih terlihat?", info["still_visible"]],
        ["Langkah berikutnya", info["next_action"]],
        ["Penjelasan", info["message"]],
    ]


# ============================================================
# STEP FILE FINDERS
# ============================================================

def step_number(selected_step):
    if selected_step is None:
        return 1

    match = re.search(r"(\d+)", str(selected_step))
    if match:
        return int(match.group(1))

    return 1


def available_steps(test_name):
    dirs = get_dirs(test_name)
    steps = set()

    action_plan_json = load_json(dirs["vlm_dir"] / "action_plan_real.json")
    action_items = extract_action_items(action_plan_json)

    for idx, item in enumerate(action_items, start=1):
        if isinstance(item, dict):
            step_value = get_field(item, ["step", "step_index", "id", "order"], idx)
            try:
                steps.add(int(step_value))
            except Exception:
                steps.add(idx)
        else:
            steps.add(idx)

    for folder in [
        dirs["vision_dir"],
        dirs["post_dir"],
        dirs["snapshot_dir"],
        dirs["vlm_dir"],
    ]:
        if folder.exists():
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue

                match = re.search(r"(?:STEP|step)[_-]?(\d+)", p.name)
                if match:
                    steps.add(int(match.group(1)))

    if not steps:
        steps.add(1)

    return [f"STEP_{n}" for n in sorted(steps)]


def find_precheck_image(test_name, step_num):
    dirs = get_dirs(test_name)
    vision_dir = dirs["vision_dir"]
    snapshot_dir = dirs["snapshot_dir"]

    prefixes = [
        f"STEP_{step_num}",
        f"step_{step_num}",
        f"Step_{step_num}",
    ]

    candidates = []

    for prefix in prefixes:
        candidates += [
            vision_dir / f"{prefix}_yolo_world_result.jpg",
            vision_dir / f"{prefix}_yolo_result.jpg",
            snapshot_dir / f"{prefix}_yolo_world_result.jpg",
            snapshot_dir / f"{prefix}_yolo_result.jpg",
        ]

    candidates += [
        vision_dir / "yolo_world_result.jpg",
        vision_dir / "yolo_precheck.jpg",
        DATA_DIR / "current_scene_rgb.jpg",
    ]

    return find_first_existing(candidates)


def find_postcheck_image(test_name, step_num):
    dirs = get_dirs(test_name)
    post_dir = dirs["post_dir"]
    vision_dir = dirs["vision_dir"]
    snapshot_dir = dirs["snapshot_dir"]

    prefixes = [
        f"STEP_{step_num}",
        f"step_{step_num}",
        f"Step_{step_num}",
    ]

    candidates = []

    for prefix in prefixes:
        candidates += [
            post_dir / f"{prefix}_post_check_yolo_result.jpg",
            post_dir / f"{prefix}_postcheck_yolo_result.jpg",
            post_dir / f"{prefix}_post_check.jpg",
            snapshot_dir / f"{prefix}_post_check_yolo_result.jpg",
        ]

    candidates += [
        vision_dir / "yolo_postcheck_scene.jpg",
        DATA_DIR / "post_scene_rgb.jpg",
    ]

    return find_first_existing(candidates)


def find_precheck_jsons(test_name, step_num):
    dirs = get_dirs(test_name)
    vision_dir = dirs["vision_dir"]
    snapshot_dir = dirs["snapshot_dir"]

    prefixes = [
        f"STEP_{step_num}",
        f"step_{step_num}",
        f"Step_{step_num}",
    ]

    detection_candidates = []
    position_candidates = []

    for prefix in prefixes:
        detection_candidates += [
            vision_dir / f"{prefix}_detections_yolo.json",
            snapshot_dir / f"{prefix}_detections_yolo.json",
        ]

        position_candidates += [
            vision_dir / f"{prefix}_object_position_camera.json",
            snapshot_dir / f"{prefix}_object_position_camera.json",
        ]

    detection_candidates += [
        vision_dir / "detections_yolo.json",
    ]

    position_candidates += [
        vision_dir / "object_position_camera.json",
    ]

    detection_path = None
    position_path = None

    for p in detection_candidates:
        if p.exists():
            detection_path = p
            break

    for p in position_candidates:
        if p.exists():
            position_path = p
            break

    detections_json = load_json(detection_path) if detection_path else {
        "_missing": True,
        "_message": "No detections_yolo JSON found",
    }

    object_position_json = load_json(position_path) if position_path else {
        "_missing": True,
        "_message": "No object_position_camera JSON found",
    }

    return detections_json, object_position_json


def find_postcheck_json(test_name, step_num):
    dirs = get_dirs(test_name)
    post_dir = dirs["post_dir"]

    prefixes = [
        f"STEP_{step_num}",
        f"step_{step_num}",
        f"Step_{step_num}",
    ]

    candidates = []

    for prefix in prefixes:
        candidates += [
            post_dir / f"{prefix}_post_check_result.json",
            post_dir / f"{prefix}_postcheck_result.json",
        ]

    for p in candidates:
        if p.exists():
            return load_json(p)

    if post_dir.exists():
        files = sorted(post_dir.glob("STEP_*_post_check_result.json"))
        if files:
            return load_json(files[-1])

    return {
        "_missing": True,
        "_message": "No post-check JSON found",
    }


# ============================================================
# PAGE COLLECTOR
# ============================================================

def collect_page(test_name, selected_step, logs=None, running=False):
    if logs is None:
        logs = []

    if not test_name or not str(test_name).strip():
        test_name = "ui_test_01"

    test_name = str(test_name).strip()
    dirs = get_dirs(test_name)

    step_choices = available_steps(test_name)

    if selected_step is None or selected_step not in step_choices:
        selected_step = step_choices[0] if step_choices else "STEP_1"

    step_num = step_number(selected_step)

    action_plan_json = load_json(dirs["vlm_dir"] / "action_plan_real.json")
    action_plan_cards = make_action_plan_cards(action_plan_json)

    precheck_image = find_precheck_image(test_name, step_num)
    postcheck_image = find_postcheck_image(test_name, step_num)

    detections_json, object_position_json = find_precheck_jsons(test_name, step_num)
    post_json = find_postcheck_json(test_name, step_num)

    precheck_summary = make_precheck_summary(
        detections_json,
        object_position_json,
        selected_step,
    )

    precheck_table = make_precheck_simple_table(
        detections_json,
        object_position_json,
    )

    postcheck_summary = make_postcheck_summary(
        post_json,
        selected_step,
    )

    postcheck_table = make_postcheck_simple_table(post_json)

    status_html = f"""
<div class="status-card">
  <b>Vision Validation</b><br>
  Status pipeline: <b>{"Sedang berjalan" if running else "Tidak berjalan"}</b><br>
  Waktu update: <b>{now_text()}</b><br>
  TEST_NAME: <b>{test_name}</b><br>
  Step yang ditampilkan: <b>{selected_step}</b>
</div>
"""

    log_text = "\n".join(logs[-300:])

    return (
        gr.update(choices=step_choices, value=selected_step),
        status_html,
        action_plan_cards,
        precheck_image,
        precheck_summary,
        precheck_table,
        postcheck_image,
        postcheck_summary,
        postcheck_table,
        log_text,
    )


# ============================================================
# COMMAND BUILDER
# ============================================================

def build_command(run_mode: str):
    if run_mode == "Vision pipeline only":
        wrapper = PROJECT_DIR / "run_d455_pipeline_ui.py"
        if wrapper.exists():
            return ["python3", "-u", "run_d455_pipeline_ui.py"]
        return ["python3", "-u", "run_d455_pipeline.py"]

    if run_mode == "Post-check only":
        wrapper = PROJECT_DIR / "run_post_complete_ui.py"
        if wrapper.exists():
            return ["python3", "-u", "run_post_complete_ui.py"]

        post_complete = PROJECT_DIR / "run_post_complete.py"
        post_pipeline = PROJECT_DIR / "run_post_pipeline.py"

        if post_complete.exists():
            return ["python3", "-u", "run_post_complete.py"]

        if post_pipeline.exists():
            return ["python3", "-u", "run_post_pipeline.py"]

        return ["python3", "-u", "run_post_complete.py"]

    if run_mode == "Full multi pipeline shell":
        fixed_script = PROJECT_DIR / "scripts" / "run_full_multi_pipeline_fixed.sh"
        normal_script = PROJECT_DIR / "scripts" / "run_full_multi_pipeline.sh"

        if fixed_script.exists():
            return ["stdbuf", "-oL", "-eL", "bash", "scripts/run_full_multi_pipeline_fixed.sh"]

        if normal_script.exists():
            return ["stdbuf", "-oL", "-eL", "bash", "scripts/run_full_multi_pipeline.sh"]

        return ["stdbuf", "-oL", "-eL", "bash", "scripts/run_full_multi_pipeline_fixed.sh"]

    raise ValueError(f"Unknown run mode: {run_mode}")


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline_live(user_query, test_name, step_index, run_mode, selected_step):
    logs = []

    user_query = str(user_query).strip()
    test_name = str(test_name).strip()
    step_index = safe_int(step_index, default=1)

    if not user_query:
        user_query = "I want to pick the target object."

    if not test_name:
        test_name = time.strftime("ui_test_%Y%m%d_%H%M%S")

    selected_step = f"STEP_{step_index}"

    env = os.environ.copy()
    env["USER_QUERY"] = user_query
    env["TEST_NAME"] = test_name
    env["STEP_INDEX"] = str(step_index)

    cmd = build_command(run_mode)

    logs.append("========== GRADIO VISION VALIDATION START ==========")
    logs.append(f"TIME        : {now_text()}")
    logs.append(f"PROJECT_DIR : {PROJECT_DIR}")
    logs.append(f"USER_QUERY  : {user_query}")
    logs.append(f"TEST_NAME   : {test_name}")
    logs.append(f"STEP_INDEX  : {step_index}")
    logs.append(f"RUN_MODE    : {run_mode}")
    logs.append(f"COMMAND     : {' '.join(cmd)}")
    logs.append("====================================================")

    yield collect_page(
        test_name=test_name,
        selected_step=selected_step,
        logs=logs,
        running=True,
    )

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        last_update = time.time()

        while True:
            ready, _, _ = select.select([process.stdout], [], [], 0.5)

            if ready:
                line = process.stdout.readline()
                if line:
                    logs.append(line.rstrip())

            if time.time() - last_update >= 0.7:
                yield collect_page(
                    test_name=test_name,
                    selected_step=selected_step,
                    logs=logs,
                    running=True,
                )
                last_update = time.time()

            return_code = process.poll()
            if return_code is not None:
                for line in process.stdout:
                    logs.append(line.rstrip())

                logs.append("====================================================")
                logs.append(f"PROCESS FINISHED WITH CODE: {return_code}")
                logs.append(f"TIME: {now_text()}")
                logs.append("====================================================")
                break

    except Exception as e:
        logs.append("====================================================")
        logs.append(f"ERROR: {repr(e)}")
        logs.append(f"TIME: {now_text()}")
        logs.append("====================================================")

    yield collect_page(
        test_name=test_name,
        selected_step=selected_step,
        logs=logs,
        running=False,
    )


# ============================================================
# UI CALLBACKS
# ============================================================

def refresh_page(test_name, selected_step):
    logs = [
        "Manual refresh.",
        f"TIME: {now_text()}",
    ]

    return collect_page(
        test_name=test_name,
        selected_step=selected_step,
        logs=logs,
        running=False,
    )


def load_previous_output(selected_test):
    if not selected_test:
        selected_test = "ui_test_01"

    logs = [
        "Loaded previous output.",
        f"TIME: {now_text()}",
        f"SELECTED TEST_NAME: {selected_test}",
    ]

    return (selected_test,) + collect_page(
        test_name=selected_test,
        selected_step=None,
        logs=logs,
        running=False,
    )


def refresh_previous_test_list():
    names = list_previous_tests()

    if names:
        return gr.update(choices=names, value=names[0])

    return gr.update(choices=[], value=None)


# ============================================================
# GRADIO ONE-PAGE UI
# ============================================================

previous_tests = list_previous_tests()
default_test = previous_tests[0] if previous_tests else "ui_test_01"

CUSTOM_CSS = """
#main-title {
    text-align: center;
}

.status-card {
    background: #f7f7f9;
    border: 1px solid #dedee3;
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 10px;
    margin-bottom: 16px;
    font-size: 15px;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-top: 10px;
    margin-bottom: 18px;
}

.step-card {
    background: white;
    border: 1px solid #dddddd;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.step-badge {
    display: inline-block;
    background: #eef2ff;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}

.step-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 6px;
}

.step-desc {
    font-size: 14px;
    color: #555;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(160px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.metric-card {
    background: white;
    border: 1px solid #dddddd;
    border-radius: 14px;
    padding: 13px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.metric-label {
    font-size: 13px;
    color: #666;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 17px;
    font-weight: 800;
}

.explain-card {
    background: #f8fafc;
    border-left: 5px solid #94a3b8;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    font-size: 15px;
}

.empty-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    margin-top: 18px;
}
"""

with gr.Blocks(
    title="Vision Validation UI",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
# 🤖 Vision Validation UI

Tampilan ini dibuat untuk orang awam:

**Rencana Robot → Pre-check → Post-check**

Tidak menampilkan raw JSON.  
Output JSON diterjemahkan menjadi kartu, ringkasan, dan tabel sederhana.
""",
        elem_id="main-title",
    )

    # --------------------------------------------------------
    # INPUT CONTROLS
    # --------------------------------------------------------

    with gr.Row():
        user_query = gr.Textbox(
            label="User Query",
            value="I want to eat cereal with milk. Please find the milk.",
            lines=2,
            scale=2,
        )

    with gr.Row():
        test_name = gr.Textbox(
            label="Current / New TEST_NAME",
            value=default_test,
            scale=1,
        )

        previous_test_dropdown = gr.Dropdown(
            label="View Previous Output",
            choices=previous_tests,
            value=default_test if previous_tests else None,
            scale=1,
        )

    with gr.Row():
        step_index = gr.Number(
            label="STEP_INDEX to Run",
            value=1,
            precision=0,
            scale=1,
        )

        step_view = gr.Dropdown(
            label="Step yang ditampilkan",
            choices=["STEP_1"],
            value="STEP_1",
            scale=1,
        )

        run_mode = gr.Dropdown(
            label="Run Mode",
            choices=[
                "Vision pipeline only",
                "Post-check only",
                "Full multi pipeline shell",
            ],
            value="Vision pipeline only",
            scale=1,
        )

    with gr.Row():
        run_button = gr.Button("Run Pipeline", variant="primary")
        show_step_button = gr.Button("Show Selected Step")
        refresh_button = gr.Button("Refresh Current Output")
        refresh_list_button = gr.Button("Refresh Previous Test List")

    status_html = gr.HTML()

    # --------------------------------------------------------
    # ACTION PLAN
    # --------------------------------------------------------

    gr.Markdown("## 1. Rencana Robot")

    action_plan_cards = gr.HTML()

    # --------------------------------------------------------
    # PRE-CHECK
    # --------------------------------------------------------

    gr.Markdown("## 2. Pre-check")

    with gr.Row():
        with gr.Column(scale=1):
            precheck_image = gr.Image(
                label="Gambar sebelum robot bergerak",
                type="filepath",
                height=420,
            )

        with gr.Column(scale=1):
            precheck_summary = gr.HTML()

            precheck_table = gr.Dataframe(
                headers=[
                    "No",
                    "Objek yang Terlihat",
                    "Keyakinan",
                    "Lokasi di Gambar",
                ],
                label="Objek yang Ditemukan Kamera",
                interactive=False,
            )

    # --------------------------------------------------------
    # POST-CHECK
    # --------------------------------------------------------

    gr.Markdown("## 3. Post-check")

    with gr.Row():
        with gr.Column(scale=1):
            postcheck_image = gr.Image(
                label="Gambar setelah robot bergerak",
                type="filepath",
                height=420,
            )

        with gr.Column(scale=1):
            postcheck_summary = gr.HTML()

            postcheck_table = gr.Dataframe(
                headers=[
                    "Poin Validasi",
                    "Hasil yang Mudah Dipahami",
                ],
                label="Kesimpulan Validasi",
                interactive=False,
            )

    # --------------------------------------------------------
    # LOGS
    # --------------------------------------------------------

    gr.Markdown("## 4. Live Logs")

    logs_box = gr.Textbox(
        label="Terminal Logs",
        lines=15,
        max_lines=28,
        interactive=False,
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    page_outputs = [
        step_view,
        status_html,
        action_plan_cards,
        precheck_image,
        precheck_summary,
        precheck_table,
        postcheck_image,
        postcheck_summary,
        postcheck_table,
        logs_box,
    ]

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    run_button.click(
        fn=run_pipeline_live,
        inputs=[
            user_query,
            test_name,
            step_index,
            run_mode,
            step_view,
        ],
        outputs=page_outputs,
    )

    show_step_button.click(
        fn=refresh_page,
        inputs=[
            test_name,
            step_view,
        ],
        outputs=page_outputs,
    )

    refresh_button.click(
        fn=refresh_page,
        inputs=[
            test_name,
            step_view,
        ],
        outputs=page_outputs,
    )

    previous_test_dropdown.change(
        fn=load_previous_output,
        inputs=[
            previous_test_dropdown,
        ],
        outputs=[test_name] + page_outputs,
    )

    refresh_list_button.click(
        fn=refresh_previous_test_list,
        inputs=[],
        outputs=[
            previous_test_dropdown,
        ],
    )

    step_view.change(
        fn=refresh_page,
        inputs=[
            test_name,
            step_view,
        ],
        outputs=page_outputs,
    )

    demo.load(
        fn=refresh_page,
        inputs=[
            test_name,
            step_view,
        ],
        outputs=page_outputs,
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        # auth=("yohana", "change_this_password"),
        show_error=True,
    )