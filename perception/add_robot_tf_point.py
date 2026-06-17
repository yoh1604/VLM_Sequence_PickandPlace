#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

import rospy
import tf


PROJECT_DIR = Path(__file__).resolve().parent.parent
PENDING_DIR = PROJECT_DIR / "configs" / "calibration_pending"
OUTPUT_JSON = PROJECT_DIR / "configs" / "camera_base_points.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_existing_output():
    if not OUTPUT_JSON.exists():
        return {
            "description": "Camera-base calibration point pairs.",
            "frame_camera": "RealSense D455 aligned color camera frame",
            "frame_base": "UR5 base frame",
            "samples": [],
            "camera_points": [],
            "base_points": []
        }

    with open(OUTPUT_JSON, "r") as f:
        return json.load(f)


def remove_existing_sample(data, sample_id):
    samples = data.get("samples", [])
    samples = [s for s in samples if s.get("id") != sample_id]

    data["samples"] = samples
    data["camera_points"] = [s["point_camera_m"] for s in samples]
    data["base_points"] = [s["point_base_m"] for s in samples]

    return data


def get_tf_point(base_frame, tip_frame, timeout_sec):
    listener = tf.TransformListener()

    print(f"[INFO] Menunggu TF {base_frame} -> {tip_frame} ...")

    listener.waitForTransform(
        base_frame,
        tip_frame,
        rospy.Time(0),
        rospy.Duration(timeout_sec)
    )

    trans, rot = listener.lookupTransform(
        base_frame,
        tip_frame,
        rospy.Time(0)
    )

    return trans, rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Contoh: P01")
    parser.add_argument("--base_frame", default="base")
    parser.add_argument("--tip_frame", default="gripper_tip")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    rospy.init_node("add_robot_tf_point", anonymous=True)

    camera_file = PENDING_DIR / f"{args.id}_camera.json"

    if not camera_file.exists():
        raise FileNotFoundError(
            f"Camera sample belum ada: {camera_file}\n"
            f"Jalankan record_camera_marker_point.py dulu."
        )

    camera_sample = load_json(camera_file)

    trans, rot = get_tf_point(
        base_frame=args.base_frame,
        tip_frame=args.tip_frame,
        timeout_sec=args.timeout
    )

    sample = {
        **camera_sample,
        "point_base_m": [float(trans[0]), float(trans[1]), float(trans[2])],
        "base_quaternion_xyzw": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
        "frame_base": args.base_frame,
        "tip_frame": args.tip_frame,
        "base_added_at": datetime.now().isoformat(),
        "note_robot": f"Robot point recorded from TF {args.base_frame} -> {args.tip_frame}."
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    data = load_existing_output()

    existing_ids = [s.get("id") for s in data.get("samples", [])]

    if args.id in existing_ids:
        if not args.replace:
            raise RuntimeError(
                f"Sample {args.id} sudah ada. Pakai --replace untuk menimpa."
            )
        data = remove_existing_sample(data, args.id)

    data["samples"].append(sample)
    data["camera_points"].append(sample["point_camera_m"])
    data["base_points"].append(sample["point_base_m"])
    data["updated_at"] = datetime.now().isoformat()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print("\n========== ROBOT TF POINT SAVED ==========")
    print("Sample ID:", args.id)
    print("Camera point:", sample["point_camera_m"])
    print("Robot base point:", sample["point_base_m"])
    print("Base quaternion:", sample["base_quaternion_xyzw"])
    print("Saved:", OUTPUT_JSON)
    print("Total samples:", len(data["samples"]))
    print("=========================================\n")


if __name__ == "__main__":
    main()
