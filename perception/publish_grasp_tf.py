#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import numpy as np
import rospy
import tf
from tf.transformations import quaternion_from_matrix


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path):
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


def load_json(path):
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "r") as f:
        return json.load(f), path


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix 3x3 ke quaternion xyzw untuk ROS TF.
    """
    T = np.eye(4)
    T[:3, :3] = R

    q = quaternion_from_matrix(T)
    return q


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grasp_base_json",
        default=None,
        help="Path best_grasp_base.json. Jika kosong, pakai capture_config.VISION_OUTPUT_DIR.",
    )

    parser.add_argument(
        "--base_frame",
        default="base",
        help="Frame base robot. Biasanya base atau base_link.",
    )

    parser.add_argument(
        "--grasp_frame",
        default="best_grasp",
        help="Nama TF frame untuk pose grasp.",
    )

    parser.add_argument(
        "--pre_grasp_frame",
        default="pre_grasp",
        help="Nama TF frame untuk pose pre-grasp.",
    )

    parser.add_argument(
        "--pregrasp_offset_z",
        type=float,
        default=0.10,
        help="Offset pre-grasp ke atas dalam frame base, meter.",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Publish rate TF.",
    )

    args = parser.parse_args()

    if args.grasp_base_json is None:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_DIR))
            import capture_config as cfg

            grasp_base_json = Path(cfg.VISION_OUTPUT_DIR) / "best_grasp_base.json"
            print("[INFO] Menggunakan path dari capture_config.py")
            print("[INFO] TEST_NAME:", getattr(cfg, "TEST_NAME", "unknown"))
            print("[INFO] grasp_base_json:", grasp_base_json)

        except Exception as e:
            print("[WARN] Gagal import capture_config.py:", e)
            grasp_base_json = (
                PROJECT_DIR
                / "outputs"
                / "test_grasp"
                / "vision_output"
                / "best_grasp_base.json"
            )
            print("[WARN] Fallback:", grasp_base_json)
    else:
        grasp_base_json = args.grasp_base_json

    data, json_path = load_json(grasp_base_json)

    if not data.get("success", False):
        raise RuntimeError("best_grasp_base.json success=False")

    if "translation_base" not in data:
        raise KeyError("translation_base tidak ditemukan di best_grasp_base.json")

    if "rotation_matrix_base" not in data:
        raise KeyError("rotation_matrix_base tidak ditemukan di best_grasp_base.json")

    p_grasp = np.array(data["translation_base"], dtype=float)
    R_grasp = np.array(data["rotation_matrix_base"], dtype=float)

    if p_grasp.shape != (3,):
        raise ValueError(f"translation_base harus 3D, sekarang: {p_grasp.shape}")

    if R_grasp.shape != (3, 3):
        raise ValueError(f"rotation_matrix_base harus 3x3, sekarang: {R_grasp.shape}")

    q_grasp = rotation_matrix_to_quaternion(R_grasp)

    if "pre_grasp_preview" in data and "translation_base" in data["pre_grasp_preview"]:
        p_pre = np.array(data["pre_grasp_preview"]["translation_base"], dtype=float)
        print("[INFO] Menggunakan pre_grasp_preview dari JSON.")
    else:
        p_pre = p_grasp.copy()
        p_pre[2] += args.pregrasp_offset_z
        print("[INFO] pre_grasp_preview tidak ada. Membuat dari z + offset.")

    q_pre = q_grasp

    rospy.init_node("publish_grasp_tf", anonymous=True)

    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(args.rate)

    print("\n========== PUBLISH GRASP TF ==========")
    print("JSON:", json_path)
    print("base_frame:", args.base_frame)
    print("best_grasp frame:", args.grasp_frame)
    print("pre_grasp frame:", args.pre_grasp_frame)
    print("best_grasp translation:", p_grasp.tolist())
    print("pre_grasp translation:", p_pre.tolist())
    print("quaternion xyzw:", q_grasp.tolist())
    print("=====================================\n")

    print("Buka RViz → Add → TF → cari frame best_grasp dan pre_grasp.")
    print("Tekan Ctrl+C untuk stop publisher.\n")

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        broadcaster.sendTransform(
            tuple(p_grasp.tolist()),
            tuple(q_grasp.tolist()),
            now,
            args.grasp_frame,
            args.base_frame,
        )

        broadcaster.sendTransform(
            tuple(p_pre.tolist()),
            tuple(q_pre.tolist()),
            now,
            args.pre_grasp_frame,
            args.base_frame,
        )

        rate.sleep()


if __name__ == "__main__":
    main()
