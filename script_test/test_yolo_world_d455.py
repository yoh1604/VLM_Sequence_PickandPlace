from ultralytics import YOLOWorld
import cv2
import os
import json

image_path = "/home/b401/Documents/pick_place_occlusion/data/d455_capture/current_scene_rgb.jpg"
output_dir = "/home/b401/Documents/pick_place_occlusion/data/d455_capture/vision_output"
os.makedirs(output_dir, exist_ok=True)

model = YOLOWorld("yolov8s-worldv2.pt")

model.set_classes([
    "can",
    "apple",
    "milk",
    "orange",
    "lemon",
    "garlic",
    "potato"
])

results = model.predict(
    source=image_path,
    conf=0.10,
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

cv2.imwrite(os.path.join(output_dir, "yolo_world_result.jpg"), img)

with open(os.path.join(output_dir, "detections.json"), "w") as f:
    json.dump(detections, f, indent=2)

print("YOLO-World selesai.")
print("Jumlah deteksi:", len(detections))
print(json.dumps(detections, indent=2))
