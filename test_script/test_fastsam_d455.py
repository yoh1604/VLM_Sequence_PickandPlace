from ultralytics import FastSAM
import cv2
import os
import json
import numpy as np

base_dir = "/home/b401/Documents/pick_place_occlusion/data/d455_capture"
image_path = f"{base_dir}/current_scene_rgb.jpg"
detections_path = f"{base_dir}/vision_output/detections.json"
output_dir = f"{base_dir}/vision_output"
os.makedirs(output_dir, exist_ok=True)

with open(detections_path, "r") as f:
    detections = json.load(f)

if len(detections) == 0:
    raise RuntimeError("Tidak ada deteksi dari YOLO-World.")

# Untuk awal ambil deteksi pertama
target = detections[0]
bbox = target["bbox"]

print("Target FastSAM:", target)

model = FastSAM("FastSAM-s.pt")

results = model(
    image_path,
    bboxes=[bbox],
    device="cpu",
    retina_masks=True,
    imgsz=640,
    conf=0.4,
    iou=0.9
)

img = cv2.imread(image_path)
mask_saved = False

for r in results:
    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()

    if len(masks) == 0:
        continue

    mask = masks[0]
    mask_uint8 = (mask * 255).astype(np.uint8)

    mask_path = os.path.join(output_dir, "fastsam_mask.png")
    cv2.imwrite(mask_path, mask_uint8)

    overlay = img.copy()
    overlay[mask_uint8 > 0] = (0, 255, 0)

    blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

    result_path = os.path.join(output_dir, "fastsam_result.jpg")
    cv2.imwrite(result_path, blended)

    mask_saved = True
    break

if mask_saved:
    print("FastSAM selesai.")
    print(f"Mask: {output_dir}/fastsam_mask.png")
    print(f"Result: {output_dir}/fastsam_result.jpg")
else:
    print("FastSAM tidak menghasilkan mask.")
