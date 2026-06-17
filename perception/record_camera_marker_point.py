import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
PENDING_DIR = PROJECT_DIR / "configs" / "calibration_pending"


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def default_paths():
    return (
        PROJECT_DIR / "data" / "d455_capture" / "current_scene_rgb.jpg",
        PROJECT_DIR / "data" / "d455_capture" / "depth_raw.npy",
        PROJECT_DIR / "data" / "d455_capture" / "camera_intrinsics.json",
    )


def load_intrinsics(path):
    with open(path, "r") as f:
        intr = json.load(f)

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr.get("ppx", intr.get("cx")))
    cy = float(intr.get("ppy", intr.get("cy")))
    depth_scale = float(intr.get("depth_scale", 0.001))

    return fx, fy, cx, cy, depth_scale, intr


def select_pixel(rgb_bgr):
    selected = {"u": None, "v": None}
    window_name = "Click marker center, then press ENTER"

    image = rgb_bgr.copy()

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["u"] = int(x)
            selected["v"] = int(y)

            temp = image.copy()
            cv2.circle(temp, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                temp,
                f"u={x}, v={y}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imshow(window_name, temp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    cv2.setMouseCallback(window_name, cb)

    print("\nKlik pusat marker pada gambar.")
    print("Tekan ENTER setelah klik.")
    print("Tekan ESC untuk batal.\n")

    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 13:
            break

        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Dibatalkan user.")

    cv2.destroyAllWindows()

    if selected["u"] is None or selected["v"] is None:
        raise RuntimeError("Belum ada pixel yang diklik.")

    return selected["u"], selected["v"]


def depth_median(depth_m, u, v, radius, min_depth, max_depth):
    h, w = depth_m.shape

    u1 = max(0, u - radius)
    u2 = min(w, u + radius + 1)
    v1 = max(0, v - radius)
    v2 = min(h, v + radius + 1)

    patch = depth_m[v1:v2, u1:u2]

    valid = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)
    values = patch[valid]

    if len(values) == 0:
        raise RuntimeError(
            f"Tidak ada depth valid di sekitar pixel ({u}, {v}). "
            f"Coba klik area marker yang lebih jelas atau pakai --radius lebih besar."
        )

    return float(np.median(values)), int(len(values))


def pixel_to_camera(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return [float(x), float(y), float(z)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Contoh: P01")
    parser.add_argument("--rgb_path", default=None)
    parser.add_argument("--depth_path", default=None)
    parser.add_argument("--intrinsics_path", default=None)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=2.0)
    args = parser.parse_args()

    default_rgb, default_depth, default_intr = default_paths()

    rgb_path = resolve_path(args.rgb_path) if args.rgb_path else default_rgb
    depth_path = resolve_path(args.depth_path) if args.depth_path else default_depth
    intrinsics_path = resolve_path(args.intrinsics_path) if args.intrinsics_path else default_intr

    print("\n========== INPUT CAMERA FILES ==========")
    print("RGB:", rgb_path)
    print("DEPTH:", depth_path)
    print("INTRINSICS:", intrinsics_path)
    print("=======================================\n")

    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    if not depth_path.exists():
        raise FileNotFoundError(depth_path)

    if not intrinsics_path.exists():
        raise FileNotFoundError(intrinsics_path)

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Gagal membaca RGB: {rgb_path}")

    depth_raw = np.load(str(depth_path)).astype(np.float32)

    fx, fy, cx, cy, depth_scale, intr = load_intrinsics(intrinsics_path)

    h, w = depth_raw.shape

    if rgb_bgr.shape[:2] != (h, w):
        print("[INFO] Resize RGB:", rgb_bgr.shape[:2], "->", (h, w))
        rgb_bgr = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    depth_m = depth_raw * depth_scale

    u, v = select_pixel(rgb_bgr)

    z, valid_count = depth_median(
        depth_m=depth_m,
        u=u,
        v=v,
        radius=args.radius,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    point_camera_m = pixel_to_camera(u, v, z, fx, fy, cx, cy)

    sample = {
        "id": args.id,
        "timestamp": datetime.now().isoformat(),
        "pixel_uv": [int(u), int(v)],
        "median_depth_m": float(z),
        "valid_depth_count": int(valid_count),
        "point_camera_m": point_camera_m,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "intrinsics_path": str(intrinsics_path),
        "intrinsics": intr,
        "frame_camera": "RealSense D455 aligned color camera frame",
        "note": "Camera point recorded from RGB-D files already transferred from laptop."
    }

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PENDING_DIR / f"{args.id}_camera.json"

    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)

    print("\n========== CAMERA POINT SAVED ==========")
    print("Saved:", out_path)
    print(json.dumps(sample, indent=2))
    print("=======================================\n")


if __name__ == "__main__":
    main()
