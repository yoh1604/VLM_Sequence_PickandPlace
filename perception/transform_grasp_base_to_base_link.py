#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import rospy
import tf
from geometry_msgs.msg import PointStamped


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


def save_json(path, data):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def transform_point(listener, point, from_frame, to_frame):
    p = PointStamped()
    p.header.frame_id = from_frame
    p.header.stamp = rospy.Time(0)
    p.point.x = float(point[0])
    p.point.y = float(point[1])
    p.point.z = float(point[2])

    listener.waitForTransform(
        to_frame,
        from_frame,
        rospy.Time(0),
        rospy.Duration(5.0)
    )

    p_out = listener.transformPoint(to_frame, p)

    return [
        float(p_out.point.x),
        float(p_out.point.y),
        float(p_out.point.z),
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="outputs/test_grasp/vision_output/best_grasp_base.json",
        help="Input grasp dalam frame base."
    )

    parser.add_argument(
        "--output",
        default="outputs/test_grasp/vision_output/best_grasp_base_link.json",
        help="Output grasp dalam frame base_link."
    )

    parser.add_argument("--from_frame", default="base")
    parser.add_argument("--to_frame", default="base_link")

    args = parser.parse_args()

    rospy.init_node("transform_grasp_base_to_base_link", anonymous=True)
    listener = tf.TransformListener()

    data, input_path = load_json(args.input)

    if not data.get("success", False):
        raise RuntimeError("Input grasp success=False")

    if "translation_base" not in data:
        raise KeyError("translation_base tidak ada di input JSON")

    translation_base_link = transform_point(
        listener=listener,
        point=data["translation_base"],
        from_frame=args.from_frame,
        to_frame=args.to_frame
    )

    result = dict(data)
    result["translation_base_original"] = data["translation_base"]
    result["translation_base"] = translation_base_link
    result["frame"] = args.to_frame
    result["source_frame_before_tf"] = args.from_frame
    result["source_file"] = str(input_path)

    if "pre_grasp_preview" in result and "translation_base" in result["pre_grasp_preview"]:
        pre_base_link = transform_point(
            listener=listener,
            point=result["pre_grasp_preview"]["translation_base"],
            from_frame=args.from_frame,
            to_frame=args.to_frame
        )

        result["pre_grasp_preview"]["translation_base_original"] = result["pre_grasp_preview"]["translation_base"]
        result["pre_grasp_preview"]["translation_base"] = pre_base_link
        result["pre_grasp_preview"]["frame"] = args.to_frame

    output_path = save_json(args.output, result)

    print("\n========== TRANSFORM GRASP FRAME ==========")
    print("Input:", input_path)
    print("Output:", output_path)
    print("From frame:", args.from_frame)
    print("To frame:", args.to_frame)
    print("grasp original:", data["translation_base"])
    print("grasp transformed:", result["translation_base"])
    if "pre_grasp_preview" in result:
        print("pre_grasp transformed:", result["pre_grasp_preview"]["translation_base"])
    print("==========================================\n")


if __name__ == "__main__":
    main()

