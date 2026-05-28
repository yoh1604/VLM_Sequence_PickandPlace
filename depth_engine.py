import os
import json
import cv2
import numpy as np


class DepthEngine:
    """
    Mengambil median depth dari area mask FastSAM.

    Input:
    - depth_raw.npy
    - camera_intrinsics.json
    - fastsam_mask.png

    Output:
    - object_position_camera.json
    """

    def __init__(
        self,
        depth_path,
        intrinsics_path,
        min_depth=0.1,
        max_depth=2.0
    ):
        self.depth_path = depth_path
        self.intrinsics_path = intrinsics_path
        self.min_depth = min_depth
        self.max_depth = max_depth

        if not os.path.exists(self.depth_path):
            raise FileNotFoundError(f"Depth file tidak ditemukan: {self.depth_path}")

        if not os.path.exists(self.intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file tidak ditemukan: {self.intrinsics_path}")

        self.depth_raw = np.load(self.depth_path)

        with open(self.intrinsics_path, "r") as f:
            self.intrinsics = json.load(f)

        self.depth_scale = self.intrinsics["depth_scale"]

    def extract_from_mask(
        self,
        target,
        mask_path,
        output_path
    ):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Gagal membaca mask: {mask_path}")

        if mask.shape != self.depth_raw.shape:
            mask = cv2.resize(
                mask,
                (self.depth_raw.shape[1], self.depth_raw.shape[0])
            )

        mask_bool = mask > 0

        depth_m = self.depth_raw.astype(np.float32) * self.depth_scale

        valid_depth = depth_m[mask_bool]
        valid_depth = valid_depth[
            (valid_depth > self.min_depth)
            & (valid_depth < self.max_depth)
        ]

        print("\n[DepthEngine] Jumlah pixel mask:", int(mask_bool.sum()))
        print("[DepthEngine] Jumlah depth valid pada mask:", len(valid_depth))

        if len(valid_depth) == 0:
            raise RuntimeError("Tidak ada depth valid pada area mask.")

        median_depth = float(np.median(valid_depth))

        ys, xs = np.where(mask_bool)

        u = int(np.median(xs))
        v = int(np.median(ys))

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        x = (u - ppx) * median_depth / fx
        y = (v - ppy) * median_depth / fy
        z = median_depth

        result = {
            "object": target,
            "pixel_center_uv": [u, v],
            "median_depth_m": median_depth,
            "point_camera_m": [float(x), float(y), float(z)],
            "frame": "RealSense D455 aligned color camera frame",
            "note": "Belum ditransformasikan ke base frame UR5."
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("[DepthEngine] Object position saved to:", output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result