from typing import List
import cv2
import cv2
import numpy as np
from polars import count
import torch
import time
from ultralytics.utils.nms import non_max_suppression

# 2026-04-17
# Adapted from Matt's Work.

class MockResults:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.as_tensor(xyxy) if not isinstance(xyxy, torch.Tensor) else xyxy
        self.conf = torch.as_tensor(conf) if not isinstance(conf, torch.Tensor) else conf
        self.cls = torch.as_tensor(cls) if not isinstance(cls, torch.Tensor) else cls

    def __len__(self): return len(self.xyxy)
    def __getitem__(self, idx): return MockResults(self.xyxy[idx], self.conf[idx], self.cls[idx])

    @property
    def xywh(self):
        if len(self.xyxy) == 0: return np.empty((0, 4))
        boxes = np.atleast_2d(self.xyxy.cpu().numpy())
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1], axis=1)

    def numpy(self): return self

class SahiTurtleTracker:
    def __init__(self):
        pass

    @staticmethod
    def get_slices(img_h, img_w, slice_size, overlap) -> List[tuple]:
        slices = []
        step = int(slice_size * (1 - overlap))
        for y in range(0, img_h, step):
            if y + slice_size > img_h: y = max(0, img_h - slice_size)
            for x in range(0, img_w, step):
                if x + slice_size > img_w: x = max(0, img_w - slice_size)
                slices.append((x, y, slice_size, slice_size))
                if x + slice_size >= img_w: break
            if y + slice_size >= img_h: break
        return list(set(slices))
    
    @staticmethod
    def custom_sahi_inference(model, frame, slice_size, overlap, conf, aspect_ratio_limit=1.8):
        """Runs parallel batch inference across all slices for high performance."""
        img_h, img_w = frame.shape[:2]
        slices = SahiTurtleTracker.get_slices(img_h, img_w, slice_size, overlap)
        all_boxes = []
        
        # 1. Prepare Tiles
        tiles = []
        for (x_off, y_off, sw, sh) in slices:
            tiles.append(frame[y_off:y_off+sh, x_off:x_off+sw])

        total_start = time.perf_counter()

        # 2. Batch Inference (GPU Parallelism)
        # YOLO handles the batching internally when passed a list of images.
        results_list = model.predict(tiles, conf=conf, verbose=False, imgsz=slice_size)

        # 3. Process Batch Results
        for i, results in enumerate(results_list):
            x_off, y_off, _, _ = slices[i]
            
            if results.boxes:
                # Step 1: Intra-Tile NMS
                indices = non_max_suppression(results.boxes.data.unsqueeze(0), conf_thres=conf, iou_thres=0.3)[0]
                
                for det in indices:
                    x1, y1, x2, y2, c, cls_id = det.cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    
                    # Step 2: Aspect Ratio Filter
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    if aspect_ratio < aspect_ratio_limit:
                        all_boxes.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, c, cls_id])
        
        # 4. Centroid Proximity Suppression
        raw_count = len(all_boxes)
        kept_boxes = []
        if raw_count > 0:
            all_boxes = np.array(all_boxes)
            # Sort by confidence
            all_boxes = all_boxes[all_boxes[:, 4].argsort()[::-1]]
            
            while len(all_boxes) > 0:
                curr = all_boxes[0]
                kept_boxes.append(curr)
                if len(all_boxes) == 1: break
                
                curr_center = np.array([(curr[0]+curr[2])/2, (curr[1]+curr[3])/2])
                other_centers = np.array([(all_boxes[1:,0]+all_boxes[1:,2])/2, 
                                        (all_boxes[1:,1]+all_boxes[1:,3])/2]).T
                
                dists = np.linalg.norm(other_centers - curr_center, axis=1)
                mask = dists > 15 # Reject centers within 15 pixels
                all_boxes = all_boxes[1:][mask]

        total_dt = (time.perf_counter() - total_start) * 1000
        print(f"\n--- Batch Stats ---")
        print(f"Slices: {len(tiles)} | Total Batch Time: {total_dt:.2f}ms")
        print(f"Detections: {raw_count} raw -> {len(kept_boxes)} unique")
        
        return np.array(kept_boxes) if len(kept_boxes) > 0 else np.empty((0, 6))