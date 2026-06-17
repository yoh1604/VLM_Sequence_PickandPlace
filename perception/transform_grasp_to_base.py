import json
import argparse
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# DEFAULT PATHS FROM PIPELINE CONFIG
# ============================================================

def load_default_paths():
    """
    Ambil path default dari capture_config.py supaya otomatis mengikuti
    TEST_NAME dan VISION_OUTPUT_DIR yang sedang dipakai pipeline utama.
    """

    try:
        import capture_config as cfg

        vision_output_dir = Path(cfg.VISION_OUTPUT_DIR).expanduser()
        if not vision_output_dir.is_absolute():
            vision_output_dir = PROJECT_DIR / vision_output_dir

        best_grasp_camera = vision_output_dir / "best_grasp_camera.json"
        best_grasp_base = vision_output_dir / "best_grasp_base.json"

        test_name = getattr(cfg, "TEST_NAME", "unknown_test")

        print("[INFO] Menggunakan capture_config.py")
        print("[INFO] TEST_NAME:", test_name)
        print("[INFO] VISION_OUTPUT_DIR:", vision_output_dir)

        return {
            "test_name": test_name,
            "tf_json": PROJECT_DIR / "configs" / "T_base_camera.json",
            "input": best_grasp_camera,
            "output": best_grasp_base,
        }

    except Exception as e:
        print("[WARN] Gagal import capture_config.py:", e)
        print("[WARN] Fallback ke outputs/test_grasp/vision_output")

        vision_output_dir = PROJECT_DIR / "outputs" / "test_grasp" / "vision_output"

        return {
            "test_name": "test_grasp",
            "tf_json": PROJECT_DIR / "configs" / "T_base_camera.json",
            "input": vision_output_dir / "best_grasp_camera.json",
            "output": vision_output_dir / "best_grasp_base.json",
        }


def resolve_path(path):
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


def load_json(path, name):
    path = resolve_path(path)

    print(f"[CHECK] {name}: {path}")

    if not path.exists():
        raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    with open(path, "r") as f:
        return json.load(f), path


def save_json(path, data):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def validate_rotation_matrix(R):
    """
    Validasi ringan untuk rotation matrix.
    Tidak menghentikan program, hanya memberi warning.
    """

    R = np.asarray(R, dtype=np.float64)

    det = np.linalg.det(R)
    orth_error = np.linalg.norm(R.T @ R - np.eye(3))

    if abs(det - 1.0) > 0.05:
        print(f"[WARN] Determinant rotation matrix tidak dekat 1: det={det}")

    if orth_error > 0.05:
        print(f"[WARN] Rotation matrix kurang orthonormal: error={orth_error}")

    return float(det), float(orth_error)


def transform_grasp(tf_data, grasp_data):
    if not tf_data.get("success", True):
        raise RuntimeError("T_base_camera.json success=False")

    if not grasp_data.get("success", False):
        raise RuntimeError("best_grasp_camera.json success=False")

    if "T_base_camera" not in tf_data:
        raise KeyError("T_base_camera tidak ditemukan di file transform.")

    if "best_grasp" not in grasp_data:
        raise KeyError("best_grasp tidak ditemukan di best_grasp_camera.json.")

    T_base_camera = np.array(tf_data["T_base_camera"], dtype=np.float64)

    if T_base_camera.shape != (4, 4):
        raise ValueError(f"T_base_camera harus 4x4, sekarang: {T_base_camera.shape}")

    R_base_camera = T_base_camera[:3, :3]
    t_base_camera = T_base_camera[:3, 3]

    best_grasp = grasp_data["best_grasp"]

    p_camera = np.array(best_grasp["translation_camera"], dtype=np.float64)
    R_camera_grasp = np.array(best_grasp["rotation_matrix_camera"], dtype=np.float64)

    if p_camera.shape != (3,):
        raise ValueError(f"translation_camera harus 3D, sekarang: {p_camera.shape}")

    if R_camera_grasp.shape != (3, 3):
        raise ValueError(f"rotation_matrix_camera harus 3x3, sekarang: {R_camera_grasp.shape}")

    p_base = R_base_camera @ p_camera + t_base_camera
    R_base_grasp = R_base_camera @ R_camera_grasp

    det, orth_error = validate_rotation_matrix(R_base_grasp)

    result = {
        "success": True,
        "method": "camera_to_base_transform",
        "translation_base": p_base.tolist(),
        "rotation_matrix_base": R_base_grasp.tolist(),
        "score": best_grasp.get("score"),
        "width": best_grasp.get("width"),
        "height": best_grasp.get("height"),
        "depth": best_grasp.get("depth"),
        "object_id": best_grasp.get("object_id"),
        "frame": "UR5 base frame",
        "source_frame": "RealSense D455 aligned color camera frame",
        "rotation_validation": {
            "determinant": det,
            "orthonormal_error": orth_error,
        },
        "camera_grasp_used": {
            "translation_camera": best_grasp.get("translation_camera"),
            "rotation_matrix_camera": best_grasp.get("rotation_matrix_camera"),
        },
        "note": (
            "Pose grasp sudah ditransformasikan dari camera frame ke UR5 base frame. "
            "Untuk test pertama, gunakan pre-grasp dengan z ditambah 0.10 m dan jangan langsung descend."
        ),
    }

    return result


def main():
    defaults = load_default_paths()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tf_json",
        default=str(defaults["tf_json"]),
        help="Path ke configs/T_base_camera.json",
    )

    parser.add_argument(
        "--input",
        default=str(defaults["input"]),
        help="Path ke best_grasp_camera.json",
    )

    parser.add_argument(
        "--output",
        default=str(defaults["output"]),
        help="Path output best_grasp_base.json",
    )

    parser.add_argument(
        "--pregrasp_offset_z",
        type=float,
        default=0.10,
        help="Offset z untuk preview pre-grasp dalam meter.",
    )

    args = parser.parse_args()

    tf_data, tf_path = load_json(args.tf_json, "T_BASE_CAMERA_JSON")
    grasp_data, grasp_camera_path = load_json(args.input, "BEST_GRASP_CAMERA_JSON")

    result = transform_grasp(tf_data, grasp_data)

    p_base = np.array(result["translation_base"], dtype=np.float64)
    p_pregrasp = p_base.copy()
    p_pregrasp[2] += args.pregrasp_offset_z

    result["pre_grasp_preview"] = {
        "translation_base": p_pregrasp.tolist(),
        "offset_z_m": float(args.pregrasp_offset_z),
        "note": "Gunakan pose ini untuk test pertama agar robot hanya bergerak ke atas objek.",
    }

    result["source_file"] = str(grasp_camera_path)
    result["source_transform"] = str(tf_path)

    output_path = save_json(args.output, result)

    print("\n========== GRASP CAMERA TO BASE ==========")
    print("Input grasp camera:", grasp_camera_path)
    print("Transform:", tf_path)
    print("Output grasp base:", output_path)
    print("-----------------------------------------")
    print("translation_base:", result["translation_base"])
    print("pre_grasp_translation_base:", result["pre_grasp_preview"]["translation_base"])
    print("score:", result["score"])
    print("width:", result["width"])
    print("=========================================\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()