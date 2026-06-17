import json
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

OUT_DIR = Path("/home/b401/Documents/pick_place_occlusion_noetic/data/d455_capture")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align = rs.align(rs.stream.color)

profile = pipeline.start(config)

try:
    # Warm-up supaya auto exposure stabil
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError("RGB/depth frame tidak lengkap")

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intr = color_frame.profile.as_video_stream_profile().intrinsics

    cv2.imwrite(str(OUT_DIR / "current_scene_rgb.jpg"), color)
    np.save(str(OUT_DIR / "depth_raw.npy"), depth)

    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
    cv2.imwrite(str(OUT_DIR / "current_scene_depth.png"), depth_vis)

    intrinsics = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "depth_scale": depth_scale
    }

    with open(OUT_DIR / "camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    print("[OK] Saved:")
    print(OUT_DIR / "current_scene_rgb.jpg")
    print(OUT_DIR / "depth_raw.npy")
    print(OUT_DIR / "current_scene_depth.png")
    print(OUT_DIR / "camera_intrinsics.json")

finally:
    pipeline.stop()
