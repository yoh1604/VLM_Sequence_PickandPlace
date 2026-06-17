#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()
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

    print("[OK] Saved:", path)


def main():
    parser = argparse.ArgumentParser(
        description="Geser target tool0_pregrasp_target.json secara manual dalam frame base."
    )

    parser.add_argument(
        "--target_json",
        required=True,
        help="Path ke tool0_pregrasp_target.json",
    )

    parser.add_argument("--dx", type=float, default=-0.0, help="Geser X base dalam meter")
    parser.add_argument("--dy", type=float, default=0.0, help="Geser Y base dalam meter")
    parser.add_argument("--dz", type=float, default=0.0, help="Geser Z base dalam meter")

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON. Kalau kosong, file target_json akan ditimpa.",
    )

    args = parser.parse_args()

    data, target_path = load_json(args.target_json)

    if "translation_tool0_pregrasp" not in data:
        raise KeyError("JSON tidak punya key translation_tool0_pregrasp")

    old_p = data["translation_tool0_pregrasp"]

    if len(old_p) != 3:
        raise ValueError("translation_tool0_pregrasp harus berisi 3 angka")

    new_p = [
        float(old_p[0]) + float(args.dx),
        float(old_p[1]) + float(args.dy),
        float(old_p[2]) + float(args.dz),
    ]

    # Simpan history supaya tidak bingung kalau file sudah beberapa kali digeser
    history = data.get("manual_nudge_history", [])
    history.append(
        {
            "from": old_p,
            "to": new_p,
            "dx": float(args.dx),
            "dy": float(args.dy),
            "dz": float(args.dz),
            "frame": "base",
        }
    )

    data["translation_tool0_pregrasp"] = new_p
    data["manual_nudge_base_m"] = {
        "dx": float(args.dx),
        "dy": float(args.dy),
        "dz": float(args.dz),
    }
    data["manual_nudge_history"] = history
    data["note_nudge"] = (
        "translation_tool0_pregrasp digeser manual dalam frame base. "
        "dx/dy/dz satuannya meter."
    )

    output_path = resolve_path(args.output) if args.output else target_path

    print("\n========== NUDGE TOOL0 TARGET ==========")
    print("Input :", target_path)
    print("Output:", output_path)
    print("Old translation_tool0_pregrasp:", old_p)
    print("Nudge dx dy dz:", args.dx, args.dy, args.dz)
    print("New translation_tool0_pregrasp:", new_p)
    print("=======================================\n")

    save_json(output_path, data)


if __name__ == "__main__":
    main()
