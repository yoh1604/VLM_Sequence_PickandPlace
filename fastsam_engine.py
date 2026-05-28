import os
import cv2
import numpy as np
from ultralytics import FastSAM


class FastSAMEngine:
    """
    Wrapper FastSAM untuk segmentasi objek berdasarkan bbox YOLO.

    Input:
    - image_path
    - bbox [x1, y1, x2, y2]

    Output:
    - fastsam_mask.png
    - fastsam_result.jpg
    """

    def __init__(
        self,
        model_name="FastSAM-s.pt",
        device="cpu",
        imgsz=640,
        conf=0.4,
        iou=0.9,
        output_dir=None
    ):
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.output_dir = output_dir or "vision_output"

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading FastSAM model: {self.model_name}")
        self.model = FastSAM(self.model_name)

    def segment_bbox(
        self,
        image_path,
        bbox,
        mask_path=None,
        result_image_path=None
    ):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
        ):
            raise ValueError(f"bbox harus [x1, y1, x2, y2], dapat: {bbox}")

        mask_path = mask_path or os.path.join(
            self.output_dir,
            "fastsam_mask.png"
        )

        result_image_path = result_image_path or os.path.join(
            self.output_dir,
            "fastsam_result.jpg"
        )

        print(f"\n[FastSAM] Segmenting bbox: {bbox}")

        results = self.model(
            image_path,
            bboxes=[bbox],
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )

        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(f"Gagal membaca image: {image_path}")

        mask_saved = False

        for r in results:
            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            if len(masks) == 0:
                continue

            mask = masks[0]
            mask_uint8 = (mask * 255).astype(np.uint8)

            cv2.imwrite(mask_path, mask_uint8)

            overlay = img.copy()
            overlay[mask_uint8 > 0] = (0, 255, 0)

            blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

            cv2.imwrite(result_image_path, blended)

            mask_saved = True
            break

        if not mask_saved:
            raise RuntimeError("FastSAM tidak menghasilkan mask.")

        print(f"[FastSAM] Mask saved to: {mask_path}")
        print(f"[FastSAM] Result image saved to: {result_image_path}")

        return mask_path, result_image_path