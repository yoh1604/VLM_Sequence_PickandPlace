import json
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

POINTS_JSON = PROJECT_DIR / "configs" / "camera_base_points.json"
OUTPUT_JSON = PROJECT_DIR / "configs" / "T_base_camera.json"


def estimate_rigid_transform(A, B):
    """
    Mencari transform rigid dari A ke B.

    A: Nx3 points dalam camera frame
    B: Nx3 points dalam base frame

    Output:
    R, t sehingga:
    B ≈ R @ A + t
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"Shape A dan B harus sama. A={A.shape}, B={B.shape}")

    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A dan B harus berbentuk Nx3.")

    if A.shape[0] < 3:
        raise ValueError("Minimal butuh 3 titik korespondensi.")

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Koreksi reflection jika determinant negatif
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def main():
    if not POINTS_JSON.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {POINTS_JSON}")

    with open(POINTS_JSON, "r") as f:
        data = json.load(f)

    camera_points = np.array(data["camera_points"], dtype=np.float64)
    base_points = np.array(data["base_points"], dtype=np.float64)

    R, t = estimate_rigid_transform(camera_points, base_points)

    transformed = (R @ camera_points.T).T + t
    errors = np.linalg.norm(transformed - base_points, axis=1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    result = {
        "T_base_camera": T.tolist(),
        "R_base_camera": R.tolist(),
        "t_base_camera": t.tolist(),
        "num_points": int(len(camera_points)),
        "errors_m": errors.tolist(),
        "mean_error_m": float(np.mean(errors)),
        "max_error_m": float(np.max(errors)),
        "note": "Transform dari RealSense D455 camera frame ke UR5 base frame."
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved transform to:", OUTPUT_JSON)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()