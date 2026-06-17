from ultralytics import YOLOWorld
import cv2
import json
import os

image_path = "/home/b401/Documents/pick_place_occlusion/data/d455_capture/current_scene_rgb.jpg"
action_plan_path = "/home/b401/Documents/pick_place_occlusion/data/d455_capture/action_plan_real.json"
output_dir = "/home/b401/Documents/Documents/pick_place_occlusion/data/d455_capture/vision_output"
os.makedirs(output_dir, exist_ok=True)

def build_vocab_from_action_plan(action_plan):
    vocab = []

    for step in action_plan.get("steps", []):
        obj = step.get("object", "").strip()
        if obj:
            vocab.append(obj)

            # Tambahkan versi general untuk meningkatkan recall
            words = obj.split()
            if len(words) > 1:
                vocab.append(words[-1])  # contoh: "red can" -> "can"

    # Hapus duplikat, pertahankan urutan
    seen = set()
    clean_vocab = []
    for item in vocab:
        item = item.lower().strip()
        if item and item not in seen:
            clean_vocab.append(item)
            seen.add(item)

    return clean_vocab

with open(action_plan_path, "r") as f:
    action_plan = json.load(f)

classes = build_vocab_from_action_plan(action_plan)

if not classes:
    raise RuntimeError("Tidak ada object vocabulary dari action plan.")

print("Dynamic YOLO-World vocabulary:", classes)

model = YOLOWorld("yolov8s-world.pt")
model.set_classes(classes)

results = model.predict(
    source=image_path,
    conf=0.08,
    save=False
)

img = cv2.imread(image_path)
detections = []

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = model.names[cls_id]

        x1, y1, x2, y2 = xyxy.tolist()

        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

cv2.imwrite(os.path.join(output_dir, "yolo_world_dynamic_result.jpg"), img)

with open(os.path.join(output_dir, "detections_dynamic.json"), "w") as f:
    json.dump(detections, f, indent=2)

print("Deteksi selesai.")
print("Jumlah deteksi:", len(detections))
print(json.dumps(detections, indent=2))
