import os
import json
import cv2
import numpy as np


class DepthEngine:
    """
    Mengambil posisi 3D objek dari area mask FastSAM menggunakan depth RealSense.

    Input:
    - depth_raw.npy
    - camera_intrinsics.json
    - fastsam_mask.png

    Output:
    - object_position_camera.json

    Catatan:
    - Mask FastSAM menentukan area objek secara 2D.
    - Depth valid menentukan jarak objek.
    - Pixel mask yang depth-nya invalid akan diisi menggunakan median depth valid.
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
            raise FileNotFoundError(
                f"Depth file tidak ditemukan: {self.depth_path}"
            )

        if not os.path.exists(self.intrinsics_path):
            raise FileNotFoundError(
                f"Intrinsics file tidak ditemukan: {self.intrinsics_path}"
            )

        self.depth_raw = np.load(self.depth_path)

        with open(self.intrinsics_path, "r") as f:
            self.intrinsics = json.load(f)

        self.depth_scale = float(self.intrinsics.get("depth_scale", 0.001))

        self.fx = float(self.intrinsics["fx"])
        self.fy = float(self.intrinsics["fy"])

        # RealSense memakai ppx/ppy, bukan cx/cy
        self.ppx = float(self.intrinsics.get("ppx", self.intrinsics.get("cx")))
        self.ppy = float(self.intrinsics.get("ppy", self.intrinsics.get("cy")))

        if self.ppx is None or self.ppy is None:
            raise KeyError(
                "Intrinsics harus memiliki ppx/ppy atau cx/cy."
            )

    def _load_and_prepare_mask(self, mask_path):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Gagal membaca mask: {mask_path}")

        print("[DepthEngine] Mask asli shape:", mask.shape)
        print("[DepthEngine] Depth shape:", self.depth_raw.shape)
        print("[DepthEngine] Mask asli unique values:", np.unique(mask)[:20])

        # Resize mask ke resolusi depth jika berbeda
        if mask.shape != self.depth_raw.shape:
            print(
                "[DepthEngine] Resize mask:",
                mask.shape,
                "->",
                self.depth_raw.shape
            )
            mask = cv2.resize(
                mask,
                (self.depth_raw.shape[1], self.depth_raw.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Binary saja, tanpa morphological cleaning
        mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)

        print("[DepthEngine] Pixel mask setelah binary:", int((mask_bin > 0).sum()))

        return mask_bin

    def extract_from_mask(
        self,
        target,
        mask_path,
        output_path
    ):
        mask = self._load_and_prepare_mask(mask_path)

        mask_bool = mask > 0
        mask_pixel_count = int(mask_bool.sum())

        if mask_pixel_count == 0:
            raise RuntimeError(
                "Mask kosong. Tidak bisa menghitung posisi objek."
            )

        # Convert depth raw ke meter
        depth_m = self.depth_raw.astype(np.float32) * self.depth_scale

        ys_all, xs_all = np.where(mask_bool)
        zs_all = depth_m[ys_all, xs_all]

        valid_depth_mask = (
            np.isfinite(zs_all)
            & (zs_all > self.min_depth)
            & (zs_all < self.max_depth)
        )

        valid_depth_values = zs_all[valid_depth_mask]
        valid_depth_count = int(valid_depth_values.size)

        print("\n[DepthEngine] Jumlah pixel mask:", mask_pixel_count)
        print("[DepthEngine] Jumlah depth valid pada mask:", valid_depth_count)

        if valid_depth_count == 0:
            raise RuntimeError(
                "Tidak ada depth valid pada area mask. "
                "Tidak bisa mendapatkan posisi 3D objek."
            )

        # Median depth dari area mask yang valid
        median_depth = float(np.median(valid_depth_values))
        mean_depth = float(np.mean(valid_depth_values))

        print("[DepthEngine] Median depth valid:", median_depth)
        print("[DepthEngine] Mean depth valid:", mean_depth)

        # ============================================================
        # Gunakan SEMUA pixel mask
        # Pixel yang depth-nya invalid diisi median_depth
        # ============================================================

        zs_filled = zs_all.copy()
        zs_filled[~valid_depth_mask] = median_depth

        final_valid = (
            np.isfinite(zs_filled)
            & (zs_filled > self.min_depth)
            & (zs_filled < self.max_depth)
        )

        xs = xs_all[final_valid].astype(np.float32)
        ys = ys_all[final_valid].astype(np.float32)
        zs = zs_filled[final_valid].astype(np.float32)

        if len(zs) == 0:
            raise RuntimeError(
                "Tidak ada titik valid setelah depth filling."
            )

        # ============================================================
        # Pixel 2D + depth -> koordinat 3D camera frame
        # ============================================================

        X = (xs - self.ppx) * zs / self.fx
        Y = (ys - self.ppy) * zs / self.fy
        Z = zs

        points = np.stack([X, Y, Z], axis=1)

        # Median center lebih stabil untuk robot dibanding mean
        center_median = np.median(points, axis=0)
        center_mean = np.mean(points, axis=0)

        # Pixel center dari mask, bukan hanya dari depth valid
        u_median = int(np.median(xs_all))
        v_median = int(np.median(ys_all))

        u_mean = int(np.mean(xs_all))
        v_mean = int(np.mean(ys_all))

        valid_depth_ratio = float(valid_depth_count / mask_pixel_count)

        result = {
            "object": target,

            "pixel_center_uv": [u_median, v_median],
            "pixel_center_uv_mean": [u_mean, v_mean],

            "median_depth_m": median_depth,
            "mean_depth_m": mean_depth,

            "point_camera_m": [
                float(center_median[0]),
                float(center_median[1]),
                float(center_median[2])
            ],

            "point_camera_mean_m": [
                float(center_mean[0]),
                float(center_mean[1]),
                float(center_mean[2])
            ],

            "debug": {
                "mask_pixels": mask_pixel_count,
                "valid_depth_pixels": valid_depth_count,
                "valid_depth_ratio": valid_depth_ratio,
                "used_pixels_after_depth_fill": int(len(points)),
                "min_depth_m": float(self.min_depth),
                "max_depth_m": float(self.max_depth),
                "depth_scale": float(self.depth_scale),
                "fx": float(self.fx),
                "fy": float(self.fy),
                "ppx": float(self.ppx),
                "ppy": float(self.ppy),
                "method": "all_mask_pixels_with_invalid_depth_filled_by_median_depth"
            },

            "frame": "RealSense D455 aligned color camera frame",
            "note": "Belum ditransformasikan ke base frame UR5."
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("[DepthEngine] Object position saved to:", output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result