#!/usr/bin/env python3
"""
Convert tool0 pregrasp target from ROS frame `base` to `base_link`.

For this UR5 setup, tf_echo showed:
  base and base_link have the same origin, but differ by yaw 180 deg around Z.
Therefore point conversion is:
  x_base_link = -x_base
  y_base_link = -y_base
  z_base_link =  z_base

Run AFTER nudge, because the current nudge values were tuned in frame `base`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def resolve_path(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def convert_point_base_to_base_link(p_base: List[float]) -> List[float]:
    if not isinstance(p_base, list) or len(p_base) != 3:
        raise ValueError(f"Expected 3D list point, got: {p_base!r}")
    return [-float(p_base[0]), -float(p_base[1]), float(p_base[2])]


def convert_vec3_key(data: Dict[str, Any], key: str) -> None:
    """Convert a top-level 3D point and preserve its original value."""
    if key not in data:
        return
    val = data[key]
    if not isinstance(val, list) or len(val) != 3:
        return
    data[f"{key}_original_base"] = val
    data[key] = convert_point_base_to_base_link(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tool0_pregrasp_target.json from base frame to base_link frame."
    )
    parser.add_argument("--input", required=True, help="Input target JSON in frame base.")
    parser.add_argument("--output", required=True, help="Output target JSON in frame base_link.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow conversion even if input already says frame=base_link.",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    with open(input_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    if not data.get("success", False):
        raise RuntimeError("Input target JSON success=False. Jangan lanjut execute.")

    input_frame = data.get("frame", "base")
    if input_frame == "base_link" and not args.force:
        raise RuntimeError(
            "Input sudah frame=base_link. Stop agar tidak double-convert. "
            "Gunakan --force hanya kalau benar-benar yakin."
        )

    if "translation_tool0_pregrasp" not in data:
        raise KeyError("translation_tool0_pregrasp tidak ada di input JSON.")

    # Preserve metadata.
    data["frame_original"] = input_frame
    data["frame"] = "base_link"
    data["converted_from_frame"] = "base"
    data["converted_to_frame"] = "base_link"
    data["base_to_base_link_conversion"] = {
        "type": "yaw_180_deg_same_origin",
        "formula": "[x_bl, y_bl, z_bl] = [-x_base, -y_base, z_base]",
        "reason": "tf_echo showed base and base_link differ by 180 deg yaw around Z.",
    }

    # Main point used by grasp.py.
    convert_vec3_key(data, "translation_tool0_pregrasp")

    # Extra audit/debug points if present.
    convert_vec3_key(data, "gripper_tip_grasp_base")
    convert_vec3_key(data, "gripper_tip_pregrasp_base")
    convert_vec3_key(data, "translation_base")

    # Nested preview if present.
    preview = data.get("pre_grasp_preview")
    if isinstance(preview, dict):
        if "translation_base" in preview and isinstance(preview["translation_base"], list):
            old = preview["translation_base"]
            preview["translation_base_original_base"] = old
            preview["translation_base"] = convert_point_base_to_base_link(old)
            preview["frame"] = "base_link"

    # Quaternion deliberately preserved. In the current execution script, orientation is taken
    # from current robot pose with --orientation_mode current, so JSON quaternion is not used.
    data["quaternion_preserved_not_rotated"] = True
    data["quaternion_note"] = "Preserved because grasp.py should run with --orientation_mode current."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n========== BASE -> BASE_LINK TARGET CONVERSION ==========")
    print("Input :", input_path)
    print("Output:", output_path)
    print("Input frame :", input_frame)
    print("Output frame:", data.get("frame"))
    print("translation_tool0_pregrasp original base:")
    print(" ", data.get("translation_tool0_pregrasp_original_base"))
    print("translation_tool0_pregrasp new base_link:")
    print(" ", data.get("translation_tool0_pregrasp"))
    print("========================================================\n")


if __name__ == "__main__":
    main()
