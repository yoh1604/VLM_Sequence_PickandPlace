import numpy as np
import json

depth = np.load("d455_capture/depth_raw.npy")

with open("d455_capture/camera_intrinsics.json", "r") as f:
    intr = json.load(f)

scale = intr["depth_scale"]
depth_m = depth.astype(np.float32) * scale

valid = depth_m[(depth_m > 0.1) & (depth_m < 3.0)]

print("Depth shape:", depth.shape)
print("Raw min:", depth.min())
print("Raw max:", depth.max())
print("Jumlah depth valid:", len(valid))

if len(valid) > 0:
    print(f"Median valid depth: {np.median(valid):.3f} m")
    print(f"Min valid depth: {np.min(valid):.3f} m")
    print(f"Max valid depth: {np.max(valid):.3f} m")
else:
    print("Tidak ada depth valid.")

