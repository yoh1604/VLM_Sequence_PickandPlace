import cv2
import numpy as np
import json
import os

base_dir = "/home/b401/Documents/pick_place_occlusion/data/d455_capture"

depth_path = f"{base_dir}/depth_raw.npy"
intr_path = f"{base_dir}/camera_intrinsics.json"
mask_path = f"{base_dir}/vision_output/fastsam_mask.png"
output_path = f"{base_dir}/vision_output/object_position_camera.json"

depth_raw = np.load(depth_path)

with open(intr_path, "r") as f:
    intr = json.load(f)

depth_scale = intr["depth_scale"]

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise RuntimeError(f"Mask tidak ditemukan: {mask_path}")

if mask.shape != depth_raw.shape:
    mask = cv2.resize(mask, (depth_raw.shape[1], depth_raw.shape[0]))

mask_bool = mask > 0

depth_m = depth_raw.astype(np.float32) * depth_scale

valid_depth = depth_m[mask_bool]
valid_depth = valid_depth[(valid_depth > 0.1) & (valid_depth < 2.0)]

print("Jumlah pixel mask:", int(mask_bool.sum()))
print("Jumlah depth valid pada mask:", len(valid_depth))

if len(valid_depth) == 0:
    raise RuntimeError("Tidak ada depth valid pada area mask.")

median_depth = float(np.median(valid_depth))

ys, xs = np.where(mask_bool)
u = int(np.median(xs))
v = int(np.median(ys))

fx = intr["fx"]
fy = intr["fy"]
ppx = intr["ppx"]
ppy = intr["ppy"]

x = (u - ppx) * median_depth / fx
y = (v - ppy) * median_depth / fy
z = median_depth

result = {
    "object": "orange",
    "pixel_center_uv": [u, v],
    "median_depth_m": median_depth,
    "point_camera_m": [float(x), float(y), float(z)],
    "frame": "RealSense D455 aligned color camera frame",
    "note": "Belum ditransformasikan ke base frame UR5."
}

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print("Depth extraction selesai.")
print(json.dumps(result, indent=2))
print("Saved to:", output_path)
