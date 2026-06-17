#!/usr/bin/env python3

import json
import argparse
import sys
import ast
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT + CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CAPTURE_CONFIG_PATH = PROJECT_DIR / "capture_config.py"

sys.path.append(str(PROJECT_DIR))


def parse_capture_config_literals(config_path):
    """
    Membaca variabel literal dari capture_config.py TANPA import.

    Ini penting karena pada beda conda/env, import capture_config.py bisa gagal
    karena dependency seperti python-dotenv tidak ada.

    Yang dibaca dengan aman:
    - TEST_NAME = "..."
    - PROJECT_DIR = "..."
    - BASE_DIR = "..."
    dll jika literal biasa.

    Untuk f-string seperti:
        VISION_OUTPUT_DIR = f"{OUTPUT_DIR}/vision_output"
    tidak dievaluasi di sini. Kita bentuk ulang path berdasarkan TEST_NAME.
    """
    result = {}

    config_path = Path(config_path)

    if not config_path.exists():
        return result

    try:
        tree = ast.parse(config_path.read_text())
    except Exception as e:
        print("[WARN] Gagal parse capture_config.py:", e)
        return result

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id

                    if isinstance(node.value, ast.Constant):
                        result[name] = node.value.value

    return result


def load_capture_config():
    """
    Coba import capture_config.py normal.
    Kalau gagal, script tetap bisa jalan memakai parser literal.
    """
    try:
        import capture_config as cfg
        print("[INFO] capture_config.py imported successfully.")
        return cfg
    except Exception as e:
        print("[WARN] Gagal import capture_config.py:", e)
        return None


CFG = load_capture_config()
CFG_LITERALS = parse_capture_config_literals(CAPTURE_CONFIG_PATH)


def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


def get_cfg_value(attr_name, default=None):
    """
    Prioritas:
    1. Variabel dari capture_config.py jika import berhasil
    2. Literal dari hasil parse capture_config.py
    3. default
    """
    if CFG is not None and hasattr(CFG, attr_name):
        return getattr(CFG, attr_name)

    if attr_name in CFG_LITERALS:
        return CFG_LITERALS[attr_name]

    return default


def get_test_name(cli_test_name=None):
    """
    Prioritas TEST_NAME:
    1. --test_name dari CLI
    2. capture_config.TEST_NAME jika import berhasil
    3. TEST_NAME hasil parse literal dari capture_config.py
    4. fallback test_default
    """
    if cli_test_name:
        return str(cli_test_name)

    test_name = get_cfg_value("TEST_NAME", None)

    if test_name:
        return str(test_name)

    return "test_default"


def get_vision_output_dir(test_name):
    """
    Ambil VISION_OUTPUT_DIR.

    Kalau import capture_config.py berhasil dan punya VISION_OUTPUT_DIR,
    pakai langsung.

    Kalau import gagal, bentuk manual:
        outputs/<TEST_NAME>/vision_output
    """
    if CFG is not None and hasattr(CFG, "VISION_OUTPUT_DIR"):
        return resolve_path(CFG.VISION_OUTPUT_DIR)

    return resolve_path(PROJECT_DIR / "outputs" / test_name / "vision_output")


def get_default_best_grasp_base(test_name):
    """
    Prioritas:
    1. capture_config.BEST_GRASP_BASE_JSON jika import berhasil
    2. outputs/<TEST_NAME>/vision_output/best_grasp_base.json
    """
    if CFG is not None and hasattr(CFG, "BEST_GRASP_BASE_JSON"):
        return resolve_path(CFG.BEST_GRASP_BASE_JSON)

    vision_dir = get_vision_output_dir(test_name)
    return vision_dir / "best_grasp_base.json"


def get_default_tool0_pregrasp_output(test_name):
    """
    Prioritas:
    1. capture_config.TOOL0_PREGRASP_TARGET_JSON jika import berhasil
    2. outputs/<TEST_NAME>/vision_output/tool0_pregrasp_target.json
    """
    if CFG is not None and hasattr(CFG, "TOOL0_PREGRASP_TARGET_JSON"):
        return resolve_path(CFG.TOOL0_PREGRASP_TARGET_JSON)

    vision_dir = get_vision_output_dir(test_name)
    return vision_dir / "tool0_pregrasp_target.json"


# ============================================================
# JSON UTILS
# ============================================================

def load_json(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("[OK] Saved:", path)


# ============================================================
# MATH UTILS
# ============================================================

def parse_vec3(text):
    vals = [float(v) for v in str(text).replace(",", " ").split()]

    if len(vals) != 3:
        raise ValueError("Harus 3 angka. Contoh: '0 0 0.17'")

    return np.array(vals, dtype=np.float64)


def parse_quat_xyzw(text):
    vals = [float(v) for v in str(text).replace(",", " ").split()]

    if len(vals) != 4:
        raise ValueError("Quaternion harus 4 angka. Contoh: '-0.283 0.666 0.641 -0.255'")

    q = np.array(vals, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm terlalu kecil.")

    return (q / norm).tolist()


def rpy_to_rot(roll, pitch, yaw):
    """
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr],
        ],
        dtype=np.float64,
    )

    Ry = np.array(
        [
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ],
        dtype=np.float64,
    )

    Rz = np.array(
        [
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


def quat_xyzw_to_rot(q):
    """
    Quaternion format: [x, y, z, w]
    """
    q = np.asarray(q, dtype=np.float64)

    if q.shape != (4,):
        raise ValueError("Quaternion harus berbentuk [x, y, z, w].")

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm terlalu kecil.")

    x, y, z, w = q / norm

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    return R


def rot_to_quat_xyzw(R):
    """
    Convert rotation matrix ke quaternion [x, y, z, w].
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q = q / np.linalg.norm(q)

    return q.tolist()


# ============================================================
# GRASP JSON PARSING
# ============================================================

def get_gripper_tip_grasp_translation_base(data):
    """
    Mendukung beberapa format best_grasp_base.json.

    Format 1:
    {
      "translation_base": [...]
    }

    Format 2:
    {
      "best_grasp": {
        "translation_base": [...]
      }
    }
    """
    if "translation_base" in data:
        return np.array(data["translation_base"], dtype=np.float64)

    if "best_grasp" in data and "translation_base" in data["best_grasp"]:
        return np.array(data["best_grasp"]["translation_base"], dtype=np.float64)

    if "best_grasp" in data and "translation_camera" in data["best_grasp"]:
        raise KeyError(
            "File ini masih camera frame, bukan base frame. "
            "Jalankan perception/transform_grasp_to_base.py terlebih dahulu."
        )

    raise KeyError(
        "Tidak menemukan translation_base di best_grasp_base.json. "
        "Pastikan file best_grasp_base.json benar."
    )


def get_optional_rotation_base(data):
    """
    Ambil rotation_matrix_base jika tersedia.

    Saat ini script default tetap memakai fixed_rpy/fixed_quat karena kamu
    memakai orientasi fixed tool0 untuk eksekusi.
    """
    if "rotation_matrix_base" in data:
        return data["rotation_matrix_base"]

    if "best_grasp" in data and "rotation_matrix_base" in data["best_grasp"]:
        return data["best_grasp"]["rotation_matrix_base"]

    return None


def get_optional_score_width(data):
    """
    Ambil score/width/depth/height jika ada, supaya ikut tersimpan.
    """
    score = None
    width = None
    depth = None
    height = None

    if "score" in data:
        score = data.get("score")
    if "width" in data:
        width = data.get("width")
    if "depth" in data:
        depth = data.get("depth")
    if "height" in data:
        height = data.get("height")

    if "best_grasp" in data:
        bg = data["best_grasp"]
        score = bg.get("score", score)
        width = bg.get("width", width)
        depth = bg.get("depth", depth)
        height = bg.get("height", height)

    return score, width, depth, height


# ============================================================
# MAIN
# ============================================================

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)

    pre_parser.add_argument(
        "--test_name",
        default=None,
        help="Override TEST_NAME dari capture_config.py. Contoh: --test_name test_grasp_2",
    )

    pre_args, _ = pre_parser.parse_known_args()
    active_test_name = get_test_name(pre_args.test_name)

    default_best_grasp_base = get_default_best_grasp_base(active_test_name)
    default_output = get_default_tool0_pregrasp_output(active_test_name)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description=(
            "Convert target gripper_tip dalam base frame menjadi target tool0 "
            "untuk MoveIt. Default path mengikuti capture_config.py atau TEST_NAME "
            "yang diparse langsung dari capture_config.py."
        ),
    )

    parser.add_argument(
        "--best_grasp_base",
        default=str(default_best_grasp_base),
        help="Path best_grasp_base.json. Default mengikuti capture_config.py / --test_name.",
    )

    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output tool0_pregrasp_target.json. Default mengikuti capture_config.py / --test_name.",
    )

    parser.add_argument(
        "--tool0_to_gripper_tip",
        required=True,
        help='Offset tool0 ke gripper_tip dalam frame tool0. Contoh: "0 0 0.17"',
    )

    parser.add_argument(
        "--pregrasp_z",
        type=float,
        default=0.10,
        help="Offset naik dalam frame base untuk pre-grasp. Default 0.10 m.",
    )

    parser.add_argument(
        "--fixed_rpy",
        default="1.619 0.024 -2.360",
        help=(
            "Orientasi tool0 fixed dalam RPY radian. "
            "Default dari tf_echo base tool0 yang pernah dipakai."
        ),
    )

    parser.add_argument(
        "--fixed_quat",
        default=None,
        help=(
            'Opsional. Quaternion tool0 fixed [x y z w]. '
            'Kalau diisi, akan mengabaikan --fixed_rpy. '
            'Contoh: "-0.283 0.666 0.641 -0.255"'
        ),
    )

    args = parser.parse_args()

    best_grasp_base_path = resolve_path(args.best_grasp_base)
    output_path = resolve_path(args.output)

    print("\n========== CONVERT GRIPPER TIP TO TOOL0 ==========")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("capture_config imported:", CFG is not None)

    if CFG is not None:
        print("capture_config TEST_NAME:", getattr(CFG, "TEST_NAME", "N/A"))
        print("capture_config VISION_OUTPUT_DIR:", getattr(CFG, "VISION_OUTPUT_DIR", "N/A"))
    else:
        print("capture_config parsed TEST_NAME:", active_test_name)

    print("ACTIVE_TEST_NAME:", active_test_name)
    print("Input best_grasp_base:", best_grasp_base_path)
    print("Output tool0 target:", output_path)
    print("tool0_to_gripper_tip:", args.tool0_to_gripper_tip)
    print("pregrasp_z:", args.pregrasp_z)
    print("=================================================\n")

    data = load_json(best_grasp_base_path)

    # Target hasil transform grasp adalah target gripper_tip dalam base frame.
    p_tip_grasp_base = get_gripper_tip_grasp_translation_base(data)

    # Pre-grasp: gripper_tip dinaikkan di arah Z base.
    p_tip_pregrasp_base = p_tip_grasp_base.copy()
    p_tip_pregrasp_base[2] += float(args.pregrasp_z)

    offset_tool_tip = parse_vec3(args.tool0_to_gripper_tip)

    # Orientasi fixed tool0.
    if args.fixed_quat is not None:
        q_tool0_xyzw = parse_quat_xyzw(args.fixed_quat)
        R_base_tool0 = quat_xyzw_to_rot(q_tool0_xyzw)
        rpy_used = None
        orientation_source = "fixed_quat"
    else:
        roll, pitch, yaw = parse_vec3(args.fixed_rpy)
        R_base_tool0 = rpy_to_rot(roll, pitch, yaw)
        q_tool0_xyzw = rot_to_quat_xyzw(R_base_tool0)
        rpy_used = [float(roll), float(pitch), float(yaw)]
        orientation_source = "fixed_rpy"

    # Inti konversi:
    # p_tip = p_tool0 + R_base_tool0 @ offset_tool_tip
    # maka:
    # p_tool0 = p_tip - R_base_tool0 @ offset_tool_tip
    p_tool0_pregrasp_base = p_tip_pregrasp_base - (R_base_tool0 @ offset_tool_tip)

    score, width, depth, height = get_optional_score_width(data)

    result = {
        "success": True,
        "frame": "base",
        "target_link": "tool0",
        "source_target": "gripper_tip",
        "active_test_name": active_test_name,
        "translation_tool0_pregrasp": p_tool0_pregrasp_base.tolist(),
        "quaternion_tool0_xyzw": q_tool0_xyzw,
        "rpy_tool0_rad": rpy_used,
        "orientation_source": orientation_source,
        "gripper_tip_grasp_base": p_tip_grasp_base.tolist(),
        "gripper_tip_pregrasp_base": p_tip_pregrasp_base.tolist(),
        "tool0_to_gripper_tip_offset_tool_frame": offset_tool_tip.tolist(),
        "pregrasp_z_m": float(args.pregrasp_z),
        "score": score,
        "width": width,
        "depth": depth,
        "height": height,
        "source_file": str(best_grasp_base_path),
        "note": (
            "Kirim translation_tool0_pregrasp + quaternion_tool0_xyzw ke MoveIt "
            "sebagai target tool0. Jangan kirim gripper_tip_grasp_base langsung ke MoveIt."
        ),
    }

    save_json(output_path, result)

    print("\n========== RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("============================\n")


if __name__ == "__main__":
    main()
    
# #!/usr/bin/env python3

# import json
# import argparse
# import sys
# import ast
# from pathlib import Path

# import numpy as np


# # ============================================================
# # PROJECT + CONFIG
# # ============================================================

# PROJECT_DIR = Path(__file__).resolve().parent.parent
# CAPTURE_CONFIG_PATH = PROJECT_DIR / "capture_config.py"

# sys.path.append(str(PROJECT_DIR))


# def parse_capture_config_literals(config_path):
#     """
#     Membaca variabel literal dari capture_config.py TANPA import.

#     Ini penting karena pada beda conda/env, import capture_config.py bisa gagal
#     karena dependency seperti python-dotenv tidak ada.

#     Yang dibaca dengan aman:
#     - TEST_NAME = "..."
#     - PROJECT_DIR = "..."
#     - BASE_DIR = "..."
#     dll jika literal biasa.

#     Untuk f-string seperti:
#         VISION_OUTPUT_DIR = f"{OUTPUT_DIR}/vision_output"
#     tidak dievaluasi di sini. Kita bentuk ulang path berdasarkan TEST_NAME.
#     """
#     result = {}

#     config_path = Path(config_path)

#     if not config_path.exists():
#         return result

#     try:
#         tree = ast.parse(config_path.read_text())
#     except Exception as e:
#         print("[WARN] Gagal parse capture_config.py:", e)
#         return result

#     for node in tree.body:
#         if isinstance(node, ast.Assign):
#             for target in node.targets:
#                 if isinstance(target, ast.Name):
#                     name = target.id

#                     if isinstance(node.value, ast.Constant):
#                         result[name] = node.value.value

#     return result


# def load_capture_config():
#     """
#     Coba import capture_config.py normal.
#     Kalau gagal, script tetap bisa jalan memakai parser literal.
#     """
#     try:
#         import capture_config as cfg
#         print("[INFO] capture_config.py imported successfully.")
#         return cfg
#     except Exception as e:
#         print("[WARN] Gagal import capture_config.py:", e)
#         return None


# CFG = load_capture_config()
# CFG_LITERALS = parse_capture_config_literals(CAPTURE_CONFIG_PATH)


# def resolve_path(path_like):
#     path = Path(str(path_like)).expanduser()

#     if not path.is_absolute():
#         path = PROJECT_DIR / path

#     return path.resolve()


# def get_cfg_value(attr_name, default=None):
#     """
#     Prioritas:
#     1. Variabel dari capture_config.py jika import berhasil
#     2. Literal dari hasil parse capture_config.py
#     3. default
#     """
#     if CFG is not None and hasattr(CFG, attr_name):
#         return getattr(CFG, attr_name)

#     if attr_name in CFG_LITERALS:
#         return CFG_LITERALS[attr_name]

#     return default


# def get_test_name(cli_test_name=None):
#     """
#     Prioritas TEST_NAME:
#     1. --test_name dari CLI
#     2. capture_config.TEST_NAME jika import berhasil
#     3. TEST_NAME hasil parse literal dari capture_config.py
#     4. fallback test_default
#     """
#     if cli_test_name:
#         return str(cli_test_name)

#     test_name = get_cfg_value("TEST_NAME", None)

#     if test_name:
#         return str(test_name)

#     return "test_default"


# def get_vision_output_dir(test_name):
#     """
#     Ambil VISION_OUTPUT_DIR.

#     Kalau import capture_config.py berhasil dan punya VISION_OUTPUT_DIR,
#     pakai langsung.

#     Kalau import gagal, bentuk manual:
#         outputs/<TEST_NAME>/vision_output
#     """
#     if CFG is not None and hasattr(CFG, "VISION_OUTPUT_DIR"):
#         return resolve_path(CFG.VISION_OUTPUT_DIR)

#     return resolve_path(PROJECT_DIR / "outputs" / test_name / "vision_output")


# def get_default_best_grasp_base(test_name):
#     """
#     Prioritas:
#     1. capture_config.BEST_GRASP_BASE_JSON jika import berhasil
#     2. outputs/<TEST_NAME>/vision_output/best_grasp_base.json
#     """
#     if CFG is not None and hasattr(CFG, "BEST_GRASP_BASE_JSON"):
#         return resolve_path(CFG.BEST_GRASP_BASE_JSON)

#     vision_dir = get_vision_output_dir(test_name)
#     return vision_dir / "best_grasp_base.json"


# def get_default_tool0_pregrasp_output(test_name):
#     """
#     Prioritas:
#     1. capture_config.TOOL0_PREGRASP_TARGET_JSON jika import berhasil
#     2. outputs/<TEST_NAME>/vision_output/tool0_pregrasp_target.json
#     """
#     if CFG is not None and hasattr(CFG, "TOOL0_PREGRASP_TARGET_JSON"):
#         return resolve_path(CFG.TOOL0_PREGRASP_TARGET_JSON)

#     vision_dir = get_vision_output_dir(test_name)
#     return vision_dir / "tool0_pregrasp_target.json"


# # ============================================================
# # JSON UTILS
# # ============================================================

# def load_json(path):
#     path = resolve_path(path)

#     if not path.exists():
#         raise FileNotFoundError(f"File tidak ditemukan: {path}")

#     with open(path, "r") as f:
#         return json.load(f)


# def save_json(path, data):
#     path = resolve_path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)

#     with open(path, "w") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

#     print("[OK] Saved:", path)


# # ============================================================
# # MATH UTILS
# # ============================================================

# def parse_vec3(text):
#     vals = [float(v) for v in str(text).replace(",", " ").split()]

#     if len(vals) != 3:
#         raise ValueError("Harus 3 angka. Contoh: '0 0 0.17'")

#     return np.array(vals, dtype=np.float64)


# def parse_quat_xyzw(text):
#     vals = [float(v) for v in str(text).replace(",", " ").split()]

#     if len(vals) != 4:
#         raise ValueError("Quaternion harus 4 angka. Contoh: '-0.283 0.666 0.641 -0.255'")

#     q = np.array(vals, dtype=np.float64)
#     norm = np.linalg.norm(q)

#     if norm < 1e-12:
#         raise ValueError("Quaternion norm terlalu kecil.")

#     return (q / norm).tolist()


# def rpy_to_rot(roll, pitch, yaw):
#     """
#     R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#     """
#     cr, sr = np.cos(roll), np.sin(roll)
#     cp, sp = np.cos(pitch), np.sin(pitch)
#     cy, sy = np.cos(yaw), np.sin(yaw)

#     Rx = np.array(
#         [
#             [1, 0, 0],
#             [0, cr, -sr],
#             [0, sr, cr],
#         ],
#         dtype=np.float64,
#     )

#     Ry = np.array(
#         [
#             [cp, 0, sp],
#             [0, 1, 0],
#             [-sp, 0, cp],
#         ],
#         dtype=np.float64,
#     )

#     Rz = np.array(
#         [
#             [cy, -sy, 0],
#             [sy, cy, 0],
#             [0, 0, 1],
#         ],
#         dtype=np.float64,
#     )

#     return Rz @ Ry @ Rx


# def quat_xyzw_to_rot(q):
#     """
#     Quaternion format: [x, y, z, w]
#     """
#     q = np.asarray(q, dtype=np.float64)

#     if q.shape != (4,):
#         raise ValueError("Quaternion harus berbentuk [x, y, z, w].")

#     norm = np.linalg.norm(q)

#     if norm < 1e-12:
#         raise ValueError("Quaternion norm terlalu kecil.")

#     x, y, z, w = q / norm

#     R = np.array(
#         [
#             [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
#             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
#             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
#         ],
#         dtype=np.float64,
#     )

#     return R


# def rot_to_quat_xyzw(R):
#     """
#     Convert rotation matrix ke quaternion [x, y, z, w].
#     """
#     R = np.asarray(R, dtype=np.float64)
#     tr = np.trace(R)

#     if tr > 0:
#         s = np.sqrt(tr + 1.0) * 2.0
#         w = 0.25 * s
#         x = (R[2, 1] - R[1, 2]) / s
#         y = (R[0, 2] - R[2, 0]) / s
#         z = (R[1, 0] - R[0, 1]) / s

#     elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
#         s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
#         w = (R[2, 1] - R[1, 2]) / s
#         x = 0.25 * s
#         y = (R[0, 1] + R[1, 0]) / s
#         z = (R[0, 2] + R[2, 0]) / s

#     elif R[1, 1] > R[2, 2]:
#         s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
#         w = (R[0, 2] - R[2, 0]) / s
#         x = (R[0, 1] + R[1, 0]) / s
#         y = 0.25 * s
#         z = (R[1, 2] + R[2, 1]) / s

#     else:
#         s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
#         w = (R[1, 0] - R[0, 1]) / s
#         x = (R[0, 2] + R[2, 0]) / s
#         y = (R[1, 2] + R[2, 1]) / s
#         z = 0.25 * s

#     q = np.array([x, y, z, w], dtype=np.float64)
#     q = q / np.linalg.norm(q)

#     return q.tolist()


# # ============================================================
# # GRASP JSON PARSING
# # ============================================================

# def get_gripper_tip_grasp_translation_base(data):
#     """
#     Mendukung beberapa format best_grasp_base.json.

#     Format 1:
#     {
#       "translation_base": [...]
#     }

#     Format 2:
#     {
#       "best_grasp": {
#         "translation_base": [...]
#       }
#     }
#     """
#     if "translation_base" in data:
#         return np.array(data["translation_base"], dtype=np.float64)

#     if "best_grasp" in data and "translation_base" in data["best_grasp"]:
#         return np.array(data["best_grasp"]["translation_base"], dtype=np.float64)

#     if "best_grasp" in data and "translation_camera" in data["best_grasp"]:
#         raise KeyError(
#             "File ini masih camera frame, bukan base frame. "
#             "Jalankan perception/transform_grasp_to_base.py terlebih dahulu."
#         )

#     raise KeyError(
#         "Tidak menemukan translation_base di best_grasp_base.json. "
#         "Pastikan file best_grasp_base.json benar."
#     )


# def get_optional_rotation_base(data):
#     """
#     Ambil rotation_matrix_base jika tersedia.

#     Saat ini script default tetap memakai fixed_rpy/fixed_quat karena kamu
#     memakai orientasi fixed tool0 untuk eksekusi.
#     """
#     if "rotation_matrix_base" in data:
#         return data["rotation_matrix_base"]

#     if "best_grasp" in data and "rotation_matrix_base" in data["best_grasp"]:
#         return data["best_grasp"]["rotation_matrix_base"]

#     return None


# def get_optional_score_width(data):
#     """
#     Ambil score/width/depth/height jika ada, supaya ikut tersimpan.
#     """
#     score = None
#     width = None
#     depth = None
#     height = None

#     if "score" in data:
#         score = data.get("score")
#     if "width" in data:
#         width = data.get("width")
#     if "depth" in data:
#         depth = data.get("depth")
#     if "height" in data:
#         height = data.get("height")

#     if "best_grasp" in data:
#         bg = data["best_grasp"]
#         score = bg.get("score", score)
#         width = bg.get("width", width)
#         depth = bg.get("depth", depth)
#         height = bg.get("height", height)

#     return score, width, depth, height


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     pre_parser = argparse.ArgumentParser(add_help=False)

#     pre_parser.add_argument(
#         "--test_name",
#         default=None,
#         help="Override TEST_NAME dari capture_config.py. Contoh: --test_name test_grasp_2",
#     )

#     pre_args, _ = pre_parser.parse_known_args()
#     active_test_name = get_test_name(pre_args.test_name)

#     default_best_grasp_base = get_default_best_grasp_base(active_test_name)
#     default_output = get_default_tool0_pregrasp_output(active_test_name)

#     parser = argparse.ArgumentParser(
#         parents=[pre_parser],
#         description=(
#             "Convert target gripper_tip dalam base frame menjadi target tool0 "
#             "untuk MoveIt. Default path mengikuti capture_config.py atau TEST_NAME "
#             "yang diparse langsung dari capture_config.py."
#         ),
#     )

#     parser.add_argument(
#         "--best_grasp_base",
#         default=str(default_best_grasp_base),
#         help="Path best_grasp_base.json. Default mengikuti capture_config.py / --test_name.",
#     )

#     parser.add_argument(
#         "--output",
#         default=str(default_output),
#         help="Output tool0_pregrasp_target.json. Default mengikuti capture_config.py / --test_name.",
#     )

#     parser.add_argument(
#         "--tool0_to_gripper_tip",
#         required=True,
#         help='Offset tool0 ke gripper_tip dalam frame tool0. Contoh: "0 0 0.17"',
#     )

#     parser.add_argument(
#         "--pregrasp_z",
#         type=float,
#         default=0.10,
#         help="Offset naik dalam frame base untuk pre-grasp. Default 0.10 m.",
#     )

#     parser.add_argument(
#         "--fixed_rpy",
#         default="1.619 0.024 -2.360",
#         help=(
#             "Orientasi tool0 fixed dalam RPY radian. "
#             "Default dari tf_echo base tool0 yang pernah dipakai."
#         ),
#     )

#     parser.add_argument(
#         "--fixed_quat",
#         default=None,
#         help=(
#             'Opsional. Quaternion tool0 fixed [x y z w]. '
#             'Kalau diisi, akan mengabaikan --fixed_rpy. '
#             'Contoh: "-0.283 0.666 0.641 -0.255"'
#         ),
#     )

#     args = parser.parse_args()

#     best_grasp_base_path = resolve_path(args.best_grasp_base)
#     output_path = resolve_path(args.output)

#     print("\n========== CONVERT GRIPPER TIP TO TOOL0 ==========")
#     print("PROJECT_DIR:", PROJECT_DIR)
#     print("capture_config imported:", CFG is not None)

#     if CFG is not None:
#         print("capture_config TEST_NAME:", getattr(CFG, "TEST_NAME", "N/A"))
#         print("capture_config VISION_OUTPUT_DIR:", getattr(CFG, "VISION_OUTPUT_DIR", "N/A"))
#     else:
#         print("capture_config parsed TEST_NAME:", active_test_name)

#     print("ACTIVE_TEST_NAME:", active_test_name)
#     print("Input best_grasp_base:", best_grasp_base_path)
#     print("Output tool0 target:", output_path)
#     print("tool0_to_gripper_tip:", args.tool0_to_gripper_tip)
#     print("pregrasp_z:", args.pregrasp_z)
#     print("=================================================\n")

#     data = load_json(best_grasp_base_path)

#     # Target hasil transform grasp adalah target gripper_tip dalam base frame.
#     p_tip_grasp_base = get_gripper_tip_grasp_translation_base(data)

#     # Pre-grasp: gripper_tip dinaikkan di arah Z base.
#     p_tip_pregrasp_base = p_tip_grasp_base.copy()
#     p_tip_pregrasp_base[2] += float(args.pregrasp_z)

#     offset_tool_tip = parse_vec3(args.tool0_to_gripper_tip)

#     # Orientasi fixed tool0.
#     if args.fixed_quat is not None:
#         q_tool0_xyzw = parse_quat_xyzw(args.fixed_quat)
#         R_base_tool0 = quat_xyzw_to_rot(q_tool0_xyzw)
#         rpy_used = None
#         orientation_source = "fixed_quat"
#     else:
#         roll, pitch, yaw = parse_vec3(args.fixed_rpy)
#         R_base_tool0 = rpy_to_rot(roll, pitch, yaw)
#         q_tool0_xyzw = rot_to_quat_xyzw(R_base_tool0)
#         rpy_used = [float(roll), float(pitch), float(yaw)]
#         orientation_source = "fixed_rpy"

#     # Inti konversi:
#     # p_tip = p_tool0 + R_base_tool0 @ offset_tool_tip
#     # maka:
#     # p_tool0 = p_tip - R_base_tool0 @ offset_tool_tip
#     p_tool0_pregrasp_base = p_tip_pregrasp_base - (R_base_tool0 @ offset_tool_tip)

#     score, width, depth, height = get_optional_score_width(data)

#     result = {
#         "success": True,
#         "frame": "base",
#         "target_link": "tool0",
#         "source_target": "gripper_tip",
#         "active_test_name": active_test_name,
#         "translation_tool0_pregrasp": p_tool0_pregrasp_base.tolist(),
#         "quaternion_tool0_xyzw": q_tool0_xyzw,
#         "rpy_tool0_rad": rpy_used,
#         "orientation_source": orientation_source,
#         "gripper_tip_grasp_base": p_tip_grasp_base.tolist(),
#         "gripper_tip_pregrasp_base": p_tip_pregrasp_base.tolist(),
#         "tool0_to_gripper_tip_offset_tool_frame": offset_tool_tip.tolist(),
#         "pregrasp_z_m": float(args.pregrasp_z),
#         "score": score,
#         "width": width,
#         "depth": depth,
#         "height": height,
#         "source_file": str(best_grasp_base_path),
#         "note": (
#             "Kirim translation_tool0_pregrasp + quaternion_tool0_xyzw ke MoveIt "
#             "sebagai target tool0. Jangan kirim gripper_tip_grasp_base langsung ke MoveIt."
#         ),
#     }

#     save_json(output_path, result)

#     print("\n========== RESULT ==========")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
#     print("============================\n")


# if __name__ == "__main__":
#     main()