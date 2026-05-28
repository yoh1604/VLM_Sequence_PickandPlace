import os
import json
import shutil
from dotenv import load_dotenv

from vlm_engine import VLMEngine
from validator_engine import LogicValidator
from yolo_world_engine import YoloWorldEngine
from fastsam_engine import FastSAMEngine
from depth_engine import DepthEngine
from capture_config import (
    PROJECT_DIR,
    ENV_PATH,
    BASE_DIR,
    TEST_NAME,
    USER_QUERY,
    IMAGE_PATH,
    DEPTH_PATH,
    INTRINSICS_PATH,
    DEPTH_VIS_PATH,
    VLM_OUTPUT_DIR,
    VISION_OUTPUT_DIR,
    ACTION_PLAN_JSON,
    VLM_DETECTIONS_JSON,
    VALIDATION_JSON,
    YOLO_DETECTIONS_JSON,
    YOLO_RESULT_IMAGE,
    FASTSAM_MASK_PATH,
    FASTSAM_RESULT_IMAGE,
    OBJECT_POSITION_JSON,
    SNAPSHOT_CURRENT_RGB,
    SNAPSHOT_CURRENT_DEPTH,
    print_config,
)

def get_provider_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    print("ENV path:", ENV_PATH)
    print("OPENAI_API_KEY detected:", bool(openai_key))
    print("GEMINI_API_KEY detected:", bool(gemini_key))

    return openai_key, gemini_key

def save_input_snapshots():
    if os.path.exists(IMAGE_PATH):
        shutil.copyfile(IMAGE_PATH, SNAPSHOT_CURRENT_RGB)
        print("Saved RGB snapshot:", SNAPSHOT_CURRENT_RGB)

    if os.path.exists(DEPTH_VIS_PATH):
        shutil.copyfile(DEPTH_VIS_PATH, SNAPSHOT_CURRENT_DEPTH)
        print("Saved depth snapshot:", SNAPSHOT_CURRENT_DEPTH)

def create_planner(openai_key, gemini_key):
    # if openai_key:
    #     print("Menggunakan OpenAI Cloud VLM Planner...")
    #     return VLMEngine(
    #         api_key=openai_key,
    #         provider="openai"
    #     )

    if gemini_key:
        print("Menggunakan Gemini Cloud VLM Planner...")
        return VLMEngine(
            api_key=gemini_key,
            provider="gemini"
        )

    raise RuntimeError("Tidak ada API key untuk planner.")


def create_validator(openai_key, gemini_key):
    if openai_key:
        print("Menggunakan OpenAI Cloud LogicValidator...")
        return LogicValidator(
            api_key=openai_key,
            provider="openai"
        )

    if gemini_key:
        print("Menggunakan Gemini Cloud LogicValidator...")
        return LogicValidator(
            api_key=gemini_key,
            provider="gemini"
        )

    raise RuntimeError("Tidak ada API key untuk validator.")


def save_vlm_detections(planner_result, output_path):
    visual_analysis = planner_result.get("visual_analysis", [])

    detections = []

    for item in visual_analysis:
        if not isinstance(item, dict):
            continue

        obj = item.get("object", "")
        bbox_yxyx = item.get("bbox", [0, 0, 0, 0])

        if not obj:
            continue

        if not isinstance(bbox_yxyx, list) or len(bbox_yxyx) != 4:
            bbox_yxyx = [0, 0, 0, 0]

        ymin, xmin, ymax, xmax = bbox_yxyx

        detections.append({
            "label": obj,
            "confidence": None,
            "bbox_format": "normalized_0_1000_xyxy",
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_original_format": "normalized_0_1000_yxyx",
            "bbox_original_yxyx": [ymin, xmin, ymax, xmax],
            "source": "vlm_visual_analysis",
            "description": item.get("description", ""),
            "status": item.get("status", ""),
            "target_role": item.get("target_role", "")
        })

    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)

    return detections

def normalize_repeated_words(text):
    words = str(text).lower().strip().split()
    clean = []

    for w in words:
        if len(clean) == 0 or clean[-1] != w:
            clean.append(w)

    return " ".join(clean)


def get_first_target_from_validation(validation_result):
    status = str(validation_result.get("validation_status", "FAIL")).upper()

    if status != "PASS":
        raise RuntimeError("Plan belum PASS. Tidak boleh lanjut ke YOLO/FastSAM.")

    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list) or len(final_plan) == 0:
        raise RuntimeError("final_action_plan kosong.")

    first_step = final_plan[0]
    target = first_step.get("target")
    target = normalize_repeated_words(target)

    if not target:
        raise RuntimeError("Target pada step pertama kosong.")

    print("\nTarget pertama untuk YOLO-World:", target)

    return target, first_step, final_plan


def run_vlm_and_validator(user_query):
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    openai_key, gemini_key = get_provider_keys()

    planner = create_planner(openai_key, gemini_key)

    planner_result = planner.get_strategy(
        img_path=IMAGE_PATH,
        user_query=user_query
    )

    if planner_result is None:
        raise RuntimeError("VLM Planner tidak menghasilkan action plan.")

    if isinstance(planner_result, dict) and "error" in planner_result:
        raise RuntimeError(f"VLM Planner error: {planner_result['error']}")

    with open(ACTION_PLAN_JSON, "w") as f:
        json.dump(planner_result, f, indent=2, ensure_ascii=False)

    vlm_detections = save_vlm_detections(
        planner_result,
        VLM_DETECTIONS_JSON
    )

    print("\nAction plan berhasil dibuat.")
    print(f"Saved to: {ACTION_PLAN_JSON}")
    print(json.dumps(planner_result, indent=2, ensure_ascii=False))

    print("\nVLM detections berhasil dibuat.")
    print(f"Saved to: {VLM_DETECTIONS_JSON}")
    print(json.dumps(vlm_detections, indent=2, ensure_ascii=False))

    validator = create_validator(openai_key, gemini_key)

    validation_result = validator.validate_strategy(
        image_path=IMAGE_PATH,
        user_query=user_query,
        planner_json=planner_result
    )

    with open(VALIDATION_JSON, "w") as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)

    print("\nValidation berhasil dibuat.")
    print(f"Saved to: {VALIDATION_JSON}")
    print(json.dumps(validation_result, indent=2, ensure_ascii=False))

    return planner_result, validation_result


def run_spatial_pipeline(validation_result):
    target, first_step, final_plan = get_first_target_from_validation(
        validation_result
    )

    yolo = YoloWorldEngine(
        model_name="yolov8l-worldv2.pt",
        conf=0.50,
        output_dir=VISION_OUTPUT_DIR
    )

    best_detection, all_detections = yolo.detect_target(
        image_path=IMAGE_PATH,
        target=target,
        output_json=YOLO_DETECTIONS_JSON,
        output_image=YOLO_RESULT_IMAGE,
        conf=0.50,
        use_generic_fallback=True
    )

    bbox = best_detection["bbox"]

    fastsam = FastSAMEngine(
        model_name="FastSAM-s.pt",
        device="cpu",
        imgsz=640,
        conf=0.4,
        iou=0.9,
        output_dir=VISION_OUTPUT_DIR
    )

    mask_path, fastsam_result = fastsam.segment_bbox(
        image_path=IMAGE_PATH,
        bbox=bbox,
        mask_path=FASTSAM_MASK_PATH,
        result_image_path=FASTSAM_RESULT_IMAGE
    )

    depth = DepthEngine(
        depth_path=DEPTH_PATH,
        intrinsics_path=INTRINSICS_PATH,
        min_depth=0.1,
        max_depth=2.0
    )

    object_position = depth.extract_from_mask(
        target=target,
        mask_path=mask_path,
        output_path=OBJECT_POSITION_JSON
    )

    return {
        "target": target,
        "first_step": first_step,
        "best_detection": best_detection,
        "object_position": object_position
    }


def main():
    print_config()
    save_input_snapshots()

    planner_result, validation_result = run_vlm_and_validator(USER_QUERY)

    result = run_spatial_pipeline(validation_result)

    print("\nFULL D455 PIPELINE SELESAI.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()