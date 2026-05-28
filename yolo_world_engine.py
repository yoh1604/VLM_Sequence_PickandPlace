import os
import json
import cv2
from ultralytics import YOLOWorld


class YoloWorldEngine:
    """
    Wrapper YOLO-World untuk deteksi target berbasis vocabulary dinamis.

    Input:
    - image_path
    - target dari action_plan VLM

    Output:
    - detections_yolo.json
    - yolo_world_result.jpg
    - best_detection
    """

    def __init__(
        self,
        model_name="yolov8s-world.pt",
        conf=0.08,
        output_dir=None
    ):
        self.model_name = model_name
        self.conf = conf
        self.output_dir = output_dir or "vision_output"

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading YOLO-World model: {self.model_name}")
        self.model = YOLOWorld(self.model_name)

    def _build_dynamic_classes(self, target, use_generic_fallback=False):
      target_clean = str(target).lower().strip()

      if not target_clean:
          raise ValueError("Target kosong untuk YOLO-World.")

      classes = [target_clean]

      if use_generic_fallback:
          words = target_clean.split()
          if len(words) > 1:
              classes.append(words[-1])

      classes = list(dict.fromkeys(classes))
      return classes

    def detect_target(
        self,
        image_path,
        target,
        output_json=None,
        output_image=None,
        conf=None,
        use_generic_fallback=False
    ):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        output_json = output_json or os.path.join(
            self.output_dir,
            "detections_yolo.json"
        )

        output_image = output_image or os.path.join(
            self.output_dir,
            "yolo_world_result.jpg"
        )

        classes = self._build_dynamic_classes(
            target,
            use_generic_fallback=use_generic_fallback
        )

        print(f"\n[YOLO-World] Target dari action plan: {target}")
        print(f"[YOLO-World] Dynamic classes: {classes}")

        self.model.set_classes(classes)

        predict_conf = conf if conf is not None else self.conf

        results = self.model.predict(
            source=image_path,
            conf=predict_conf,
            save=False
        )

        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(f"Gagal membaca image: {image_path}")

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = self.model.names[cls_id]

                x1, y1, x2, y2 = xyxy.tolist()

                det = {
                    "label": label,
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_format": "xyxy_pixel",
                    "source": "yolo_world",
                    "target_query": target
                }

                detections.append(det)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{label} {score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        with open(output_json, "w") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)

        cv2.imwrite(output_image, img)

        print(f"[YOLO-World] Detections saved to: {output_json}")
        print(f"[YOLO-World] Result image saved to: {output_image}")
        print(json.dumps(detections, indent=2, ensure_ascii=False))

        if len(detections) == 0:
            raise RuntimeError(f"YOLO-World tidak menemukan target: {target}")

        best_detection = max(detections, key=lambda d: d["confidence"])

        print("[YOLO-World] Best detection:")
        print(json.dumps(best_detection, indent=2, ensure_ascii=False))

        return best_detection, detections