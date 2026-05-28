import os
import json
import shutil
from dotenv import load_dotenv

from yolo_world_engine import YoloWorldEngine
from fastsam_engine import FastSAMEngine
from depth_engine import DepthEngine

from capture_config import (
    PROJECT_DIR,
    ENV_PATH,
    BASE_DIR,
    TEST_NAME,
    STEP_INDEX,

    IMAGE_PATH,
    DEPTH_PATH,
    INTRINSICS_PATH,

    POST_IMAGE_PATH,
    VALIDATION_JSON,
    POST_OUTPUT_DIR,
    POST_CHECK_JSON,
    POST_YOLO_JSON,
    POST_YOLO_IMAGE,
    POST_FASTSAM_MASK,
    POST_FASTSAM_IMAGE,
    REMAINING_PLAN_JSON,

    VISION_OUTPUT_DIR,
    SNAPSHOT_POST_RGB,
    print_config,
)

def save_post_snapshot():
    if os.path.exists(POST_IMAGE_PATH):
        shutil.copyfile(POST_IMAGE_PATH, SNAPSHOT_POST_RGB)
        print("Saved post RGB snapshot:", SNAPSHOT_POST_RGB)

def normalize_repeated_words(text):
    """
    Membersihkan target dengan kata berulang.
    Contoh:
    - "orange orange" -> "orange"
    - "red red soda can" -> "red soda can"
    """

    if text is None:
        return ""

    words = str(text).lower().strip().split()
    clean_words = []

    for word in words:
        if not clean_words or clean_words[-1] != word:
            clean_words.append(word)

    return " ".join(clean_words)


def load_validation_result(validation_path):
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation JSON tidak ditemukan: {validation_path}")

    with open(validation_path, "r") as f:
        validation_result = json.load(f)

    return validation_result


def get_target_step_from_validation(validation_result, step_index):
    """
    Mengambil target berdasarkan STEP_INDEX dari config.

    STEP_INDEX bersifat 1-based:
    STEP_INDEX = 1 -> final_plan[0]
    STEP_INDEX = 2 -> final_plan[1]
    STEP_INDEX = 3 -> final_plan[2]
    """

    status = str(validation_result.get("validation_status", "FAIL")).upper()

    if status != "PASS":
        raise RuntimeError("Plan belum PASS. Tidak boleh lanjut ke post-check.")

    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list) or len(final_plan) == 0:
        raise RuntimeError("final_action_plan kosong.")

    if not isinstance(step_index, int):
        raise RuntimeError(f"STEP_INDEX harus integer, dapat: {step_index}")

    if step_index < 1:
        raise RuntimeError(f"STEP_INDEX harus mulai dari 1, dapat: {step_index}")

    list_index = step_index - 1

    if list_index >= len(final_plan):
        raise RuntimeError(
            f"STEP_INDEX={step_index} melebihi jumlah action_plan. "
            f"Jumlah step tersedia: {len(final_plan)}"
        )

    selected_step = final_plan[list_index]

    target = selected_step.get("target")
    target = normalize_repeated_words(target)

    if not target:
        raise RuntimeError(f"Target pada STEP_INDEX={step_index} kosong.")

    print(f"\nSTEP_INDEX dari config: {step_index}")
    print("Target untuk post-check:", target)

    return target, selected_step, final_plan


def run_post_check(target):
    """
    Jalankan YOLO-World pada gambar setelah aksi manual.
    Jika target masih ditemukan, status STILL_FOUND.
    Jika target tidak ditemukan, status REMOVED_SUCCESS.
    """

    if not os.path.exists(POST_IMAGE_PATH):
        raise FileNotFoundError(
            f"Gambar post-check tidak ditemukan: {POST_IMAGE_PATH}\n"
            f"Capture ulang scene setelah objek dipindahkan, lalu simpan sebagai post_scene_rgb.jpg"
        )

    yolo = YoloWorldEngine(
        model_name="yolov8l-worldv2.pt",
        conf=0.5,
        output_dir=POST_OUTPUT_DIR
    )

    try:
      best_detection, all_detections = yolo.detect_target(
          image_path=POST_IMAGE_PATH,
          target=target,
          output_json=POST_YOLO_JSON,
          output_image=POST_YOLO_IMAGE,
          conf=0.5,
          use_generic_fallback=False
      )

      target_found = True
      post_status = "STILL_FOUND"

      # ============================================================
      # FASTSAM POST-CHECK
      # ============================================================
      bbox = best_detection["bbox"]

      fastsam = FastSAMEngine(
          model_name="FastSAM-s.pt",
          device="cpu",
          imgsz=640,
          conf=0.4,
          iou=0.9,
          output_dir=POST_OUTPUT_DIR
      )

      post_fastsam_mask, post_fastsam_image = fastsam.segment_bbox(
          image_path=POST_IMAGE_PATH,
          bbox=bbox,
          mask_path=POST_FASTSAM_MASK,
          result_image_path=POST_FASTSAM_IMAGE
      )

      result = {
          "target": target,
          "post_check_status": post_status,
          "target_found_after_action": target_found,
          "best_detection": best_detection,
          "all_detections": all_detections,
          "post_image_path": POST_IMAGE_PATH,
          "post_yolo_image": POST_YOLO_IMAGE,
          "post_fastsam_mask": post_fastsam_mask,
          "post_fastsam_image": post_fastsam_image,
          "note": "Target masih terdeteksi setelah aksi manual. FastSAM dijalankan untuk memvisualisasikan target yang masih tersisa."
      }
    except RuntimeError as e:
      error_msg = str(e)

      if "tidak menemukan target" in error_msg.lower():
          target_found = False
          post_status = "REMOVED_SUCCESS"

          result = {
              "target": target,
              "post_check_status": post_status,
              "target_found_after_action": target_found,
              "best_detection": None,
              "all_detections": [],
              "post_image_path": POST_IMAGE_PATH,
              "post_yolo_image": POST_YOLO_IMAGE,
              "post_fastsam_mask": None,
              "post_fastsam_image": None,
              "note": "Target tidak ditemukan setelah aksi manual. FastSAM tidak dijalankan karena tidak ada bbox target. Step dianggap berhasil."
          }
      else:
          raise

    with open(POST_CHECK_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nPost-check selesai.")
    print(f"Saved to: {POST_CHECK_JSON}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result

def process_next_target_after_success(remaining_plan):
    """
    Setelah post-check sukses, fungsi ini langsung memproses target berikutnya:
    YOLO-World -> FastSAM -> Depth extraction.

    remaining_plan[0] dianggap sebagai step berikutnya.
    """

    if not isinstance(remaining_plan, list) or len(remaining_plan) == 0:
        print("\n✅ Tidak ada target berikutnya untuk diproses.")
        return None

    next_step = remaining_plan[0]
    next_target = normalize_repeated_words(next_step.get("target", ""))

    if not next_target:
        print("\n⚠️ Target berikutnya kosong. Tidak bisa lanjut YOLO/FastSAM.")
        return None

    next_step_number = next_step.get("step", STEP_INDEX + 1)

    print("\n===== PROCESS NEXT TARGET AFTER POST-CHECK SUCCESS =====")
    print("Next step:")
    print(json.dumps(next_step, indent=2, ensure_ascii=False))
    print("Next target untuk YOLO/FastSAM:", next_target)

    next_output_dir = f"{POST_OUTPUT_DIR}/STEP_{next_step_number}_next_target"
    os.makedirs(next_output_dir, exist_ok=True)

    next_yolo_json = f"{next_output_dir}/STEP_{next_step_number}_next_detections_yolo.json"
    next_yolo_image = f"{next_output_dir}/STEP_{next_step_number}_next_yolo_result.jpg"

    next_fastsam_mask = f"{next_output_dir}/STEP_{next_step_number}_next_fastsam_mask.png"
    next_fastsam_image = f"{next_output_dir}/STEP_{next_step_number}_next_fastsam_result.jpg"

    next_object_position_json = f"{next_output_dir}/STEP_{next_step_number}_next_object_position_camera.json"

    # ============================================================
    # YOLO-WORLD NEXT TARGET
    # ============================================================
    yolo = YoloWorldEngine(
        model_name="yolov8l-worldv2.pt",
        conf=0.6,
        output_dir=next_output_dir
    )

    best_detection, all_detections = yolo.detect_target(
        image_path=POST_IMAGE_PATH,
        target=next_target,
        output_json=next_yolo_json,
        output_image=next_yolo_image,
        conf=0.6,
        use_generic_fallback=True
    )

    bbox = best_detection["bbox"]

    # ============================================================
    # FASTSAM NEXT TARGET
    # ============================================================
    fastsam = FastSAMEngine(
        model_name="FastSAM-s.pt",
        device="cpu",
        imgsz=640,
        conf=0.4,
        iou=0.9,
        output_dir=next_output_dir
    )

    mask_path, fastsam_result_path = fastsam.segment_bbox(
        image_path=POST_IMAGE_PATH,
        bbox=bbox,
        mask_path=next_fastsam_mask,
        result_image_path=next_fastsam_image
    )

    # ============================================================
    # DEPTH NEXT TARGET
    # ============================================================
    depth = DepthEngine(
        depth_path=DEPTH_PATH,
        intrinsics_path=INTRINSICS_PATH,
        min_depth=0.1,
        max_depth=2.0
    )

    object_position = depth.extract_from_mask(
        target=next_target,
        mask_path=mask_path,
        output_path=next_object_position_json
    )

    next_result = {
        "next_step": next_step,
        "next_target": next_target,
        "best_detection": best_detection,
        "all_detections": all_detections,
        "next_yolo_json": next_yolo_json,
        "next_yolo_image": next_yolo_image,
        "next_fastsam_mask": next_fastsam_mask,
        "next_fastsam_image": next_fastsam_image,
        "next_object_position_json": next_object_position_json,
        "object_position": object_position,
        "note": "Next target has been detected, segmented, and localized after previous step was verified as successful."
    }

    next_result_json = f"{next_output_dir}/STEP_{next_step_number}_next_target_result.json"

    with open(next_result_json, "w") as f:
        json.dump(next_result, f, indent=2, ensure_ascii=False)

    print("\n✅ Next target pipeline selesai.")
    print("Saved to:", next_result_json)
    print(json.dumps(next_result, indent=2, ensure_ascii=False))

    return next_result

def create_remaining_plan_after_success(validation_result, post_check_result, step_index):
    """
    Jika post-check sukses, hapus semua step sampai STEP_INDEX
    lalu simpan remaining_plan.json.

    Contoh:
    final_plan = [step1, step2, step3]

    STEP_INDEX = 1 sukses -> remaining = [step2, step3]
    STEP_INDEX = 2 sukses -> remaining = [step3]
    STEP_INDEX = 3 sukses -> remaining = []
    """

    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list):
        final_plan = []

    if post_check_result.get("post_check_status") != "REMOVED_SUCCESS":
        print("\nStep belum sukses. remaining_plan tidak diubah.")
        return final_plan

    if step_index < 1:
        raise RuntimeError(f"STEP_INDEX tidak valid: {step_index}")

    remaining_plan = final_plan[step_index:]

    with open(REMAINING_PLAN_JSON, "w") as f:
        json.dump(remaining_plan, f, indent=2, ensure_ascii=False)

    print("\nRemaining plan saved to:", REMAINING_PLAN_JSON)
    print(json.dumps(remaining_plan, indent=2, ensure_ascii=False))

    return remaining_plan


def main():
    print_config()
    save_post_snapshot()
    print("\n===== POST-CHECK PIPELINE START =====")

    validation_result = load_validation_result(VALIDATION_JSON)

    target, selected_step, final_plan = get_target_step_from_validation(
        validation_result,
        STEP_INDEX
    )

    print(f"\nStep yang diverifikasi: STEP_INDEX={STEP_INDEX}")
    print(json.dumps(selected_step, indent=2, ensure_ascii=False))

    post_check_result = run_post_check(target)

    remaining_plan = create_remaining_plan_after_success(
        validation_result,
        post_check_result,
        STEP_INDEX
    )

    if post_check_result.get("post_check_status") == "REMOVED_SUCCESS":
      print("\n✅ Step yang diverifikasi berhasil menurut post-check.")

      if len(remaining_plan) > 0:
          print("➡️ Masih ada step berikutnya.")
          print("Target berikutnya:", remaining_plan[0].get("target"))

          print("\n🔍 Memproses YOLO/FastSAM/depth untuk target berikutnya...")
          next_result = process_next_target_after_success(remaining_plan)

          if next_result is not None:
              print("\n✅ Target berikutnya sudah siap untuk aksi manual berikutnya.")
              print("Next target:", next_result["next_target"])
              print("Next FastSAM:", next_result["next_fastsam_image"])
              print("Next 3D camera point:", next_result["object_position"]["point_camera_m"])

      else:
          print("✅ Semua step dalam action_plan sudah selesai.")
    else:
      print("\n⚠️ Step yang diverifikasi belum berhasil.")
      print("Target masih terlihat. Ulangi aksi manual atau capture ulang.")


if __name__ == "__main__":
    main()