import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
POINTS_JSON = PROJECT_DIR / "configs" / "camera_base_points.json"
OUTPUT_JSON = PROJECT_DIR / "configs" / "T_base_camera.json"


def estimate_rigid_transform(camera_points, base_points):
    A = np.asarray(camera_points, dtype=np.float64)
    B = np.asarray(base_points, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"Shape berbeda: camera={A.shape}, base={B.shape}")

    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("Points harus Nx3.")

    if A.shape[0] < 3:
        raise ValueError("Minimal 3 titik. Disarankan 8-10 titik.")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("[WARN] Reflection detected. Fixing.")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def main():
    if not POINTS_JSON.exists():
        raise FileNotFoundError(f"Tidak ada file: {POINTS_JSON}")

    with open(POINTS_JSON, "r") as f:
        data = json.load(f)

    camera_points = np.array(data["camera_points"], dtype=np.float64)
    base_points = np.array(data["base_points"], dtype=np.float64)

    R, t = estimate_rigid_transform(camera_points, base_points)

    pred_base = (R @ camera_points.T).T + t
    errors = np.linalg.norm(pred_base - base_points, axis=1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    result = {
        "success": True,
        "T_base_camera": T.tolist(),
        "R_base_camera": R.tolist(),
        "t_base_camera": t.tolist(),
        "num_points": int(len(camera_points)),
        "errors_m": errors.tolist(),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "frame_from": "RealSense D455 aligned color camera frame",
        "frame_to": "UR5 base frame",
        "formula": "p_base = R_base_camera @ p_camera + t_base_camera"
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n========== CAMERA TO BASE TRANSFORM ==========")
    print("Saved:", OUTPUT_JSON)
    print("num_points:", result["num_points"])
    print("mean_error_m:", result["mean_error_m"])
    print("median_error_m:", result["median_error_m"])
    print("max_error_m:", result["max_error_m"])
    print("=============================================\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
