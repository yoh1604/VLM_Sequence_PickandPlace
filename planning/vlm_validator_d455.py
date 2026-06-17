import os
import json
from dotenv import load_dotenv

from VLM_Sequence_PickandPlace.planning.vlm_engine import VLMEngine
from VLM_Sequence_PickandPlace.planning.validator_engine import LogicValidator


# =========================
# PROJECT CONFIG
# =========================

PROJECT_DIR = "/home/b401/Documents/pick_place_occlusion"
ENV_PATH = f"{PROJECT_DIR}/.env"

load_dotenv(ENV_PATH)

BASE_DIR = f"{PROJECT_DIR}/data/d455_capture"
IMAGE_PATH = f"{BASE_DIR}/current_scene_rgb.jpg"

OUTPUT_DIR = f"{BASE_DIR}/vlm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_PLAN_JSON = f"{OUTPUT_DIR}/action_plan_real.json"
VLM_DETECTIONS_JSON = f"{OUTPUT_DIR}/detections_from_vlm.json"
VALIDATION_JSON = f"{OUTPUT_DIR}/validation_result_real.json"


# =========================
# USER QUERY
# =========================

USER_QUERY = "aku mau minum soda"


# =========================
# HELPER FUNCTIONS
# =========================

def save_vlm_detections(planner_result, output_path):
    """
    Simpan visual_analysis dari VLM menjadi detections_from_vlm.json.

    Catatan:
    - VLM bbox mengikuti prompt kamu: [ymin, xmin, ymax, xmax] normalized 0-1000.
    - Output bbox dibuat [xmin, ymin, xmax, ymax] agar lebih mudah dipakai downstream.
    - Ini bukan hasil YOLO-World, tapi hasil observasi VLM.
    """
    visual_analysis = planner_result.get("visual_analysis", [])

    detections = []

    for item in visual_analysis:
        if not isinstance(item, dict):
            continue

        obj = item.get("object", "")
        bbox_yxyx = item.get("bbox", [0, 0, 0, 0])

        if not obj:
            continue

        if not (
            isinstance(bbox_yxyx, list)
            and len(bbox_yxyx) == 4
        ):
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

    print(f"\nVLM detections saved to: {output_path}")
    print(json.dumps(detections, indent=2, ensure_ascii=False))

    return detections


def get_provider_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    print("ENV path:", ENV_PATH)
    print("OPENAI_API_KEY detected:", bool(openai_key))
    print("GEMINI_API_KEY detected:", bool(gemini_key))

    return openai_key, gemini_key


def create_planner(openai_key, gemini_key):
    """
    Prioritaskan OpenAI.
    Kalau OpenAI tidak ada, fallback ke Gemini.
    """
    # if openai_key:
    #     print("Menggunakan OpenAI Cloud VLM Planner...")
    #     return VLMEngine(
    #         api_key=openai_key,
    #         provider="openai"
    #     ), "openai"

    if gemini_key:
        print("Menggunakan Gemini Cloud VLM Planner...")
        return VLMEngine(
            api_key=gemini_key,
            provider="gemini"
        ), "gemini"

    raise RuntimeError("Tidak ada API key terdeteksi. Cek file .env.")


def create_validator(openai_key, gemini_key):
    """
    Pakai provider yang sama: OpenAI prioritas.
    """

    # if gemini_key:
    #     print("Menggunakan Gemini Cloud LogicValidator...")
    #     return LogicValidator(
    #         api_key=gemini_key,
    #         provider="gemini"
    #     )
    
    if openai_key:
        print("Menggunakan OpenAI Cloud LogicValidator...")
        return LogicValidator(
            api_key=openai_key,
            provider="openai"
        )

    raise RuntimeError("Tidak ada API key untuk validator.")


# =========================
# MAIN PIPELINE
# =========================

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    openai_key, gemini_key = get_provider_keys()

    # 1. Planner
    planner, provider = create_planner(openai_key, gemini_key)

    planner_result = planner.get_strategy(
        img_path=IMAGE_PATH,
        user_query=USER_QUERY
    )

    if planner_result is None:
        raise RuntimeError("VLM Planner tidak menghasilkan action plan.")

    if isinstance(planner_result, dict) and "error" in planner_result:
        raise RuntimeError(f"VLM Planner error: {planner_result['error']}")

    with open(ACTION_PLAN_JSON, "w") as f:
        json.dump(planner_result, f, indent=2, ensure_ascii=False)

    print("\nAction plan berhasil dibuat.")
    print(f"Saved to: {ACTION_PLAN_JSON}")
    print("\n===== ACTION PLAN =====")
    print(json.dumps(planner_result, indent=2, ensure_ascii=False))

    # 2. Simpan visual_analysis sebagai detections_from_vlm.json
    save_vlm_detections(planner_result, VLM_DETECTIONS_JSON)

    # 3. Validator
    validator = create_validator(openai_key, gemini_key)

    validation_result = validator.validate_strategy(
        image_path=IMAGE_PATH,
        user_query=USER_QUERY,
        planner_json=planner_result
    )

    with open(VALIDATION_JSON, "w") as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)

    print("\nValidation berhasil dibuat.")
    print(f"Saved to: {VALIDATION_JSON}")
    print("\n===== VALIDATION RESULT =====")
    print(json.dumps(validation_result, indent=2, ensure_ascii=False))

    # 4. Gate untuk langkah berikutnya
    status = str(validation_result.get("validation_status", "FAIL")).upper()

    if status == "PASS":
        print("\nVALIDATION PASS.")
        print("Lanjut ke YOLO-World/FastSAM/depth extraction untuk target dari action_plan.")
    else:
        print("\nVALIDATION FAIL.")
        print("Jangan lanjut eksekusi robot. Periksa feedback validator.")
        print("Feedback:", validation_result.get("feedback", ""))


if __name__ == "__main__":
    main()