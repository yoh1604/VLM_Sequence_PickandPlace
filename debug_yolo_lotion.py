from pathlib import Path
import json
import capture_config as cfg
from perception.yolo_world_engine import YoloWorldEngine

image_path = cfg.IMAGE_PATH

queries = [
    "pink lotion bottle",
    "lotion bottle",
    "body lotion",
    "pink bottle",
    "bottle",
]

out_dir = Path(cfg.VISION_OUTPUT_DIR) / "debug_yolo_lotion"
out_dir.mkdir(parents=True, exist_ok=True)

print("IMAGE_PATH:", image_path)
print("YOLO_WORLD_MODEL_PATH:", cfg.YOLO_WORLD_MODEL_PATH)
print("OUT_DIR:", out_dir)

for conf in [0.60, 0.50, 0.40, 0.30, 0.20]:
    print("\n=================================================")
    print("CONF:", conf)

    yolo = YoloWorldEngine(
        model_name=cfg.YOLO_WORLD_MODEL_PATH,
        conf=conf,
        output_dir=str(out_dir),
    )

    for q in queries:
        print("\nQUERY:", q)

        try:
            best, all_det = yolo.detect_target(
                image_path=image_path,
                target=q,
                output_json=str(out_dir / f"conf_{conf}_{q.replace(' ', '_')}.json"),
                output_image=str(out_dir / f"conf_{conf}_{q.replace(' ', '_')}.jpg"),
                conf=conf,
                use_generic_fallback=True,
            )

            print("[OK] FOUND")
            print(json.dumps(best, indent=2))

        except Exception as e:
            print("[FAIL]", e)

