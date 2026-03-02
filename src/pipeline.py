import os, json
import cv2
import numpy as np
from tqdm import tqdm

from .video_io import iter_frames_mp4
from .rle import binmask_to_rle
from .detector import CocoDetector
from .sam_seg import SamSegmenter

class CocoSamPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        keep = set(cfg["keep_class_ids"])
        self.detector = CocoDetector(
            model_name=cfg["yolo_model"],
            conf_thres=cfg["conf_thres"],
            iou_thres=cfg["iou_thres"],
            keep_class_ids=keep,
        )
        self.sam = SamSegmenter(
            checkpoint_path=cfg["sam_checkpoint"],
            device=cfg.get("device", "cuda"),
        )

    def process_mp4(self, mp4_path: str, out_json_path: str):
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

        clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
        out = {
            "clip_id": clip_id,
            "input_mp4": mp4_path,
            "config": {
                k: self.cfg[k] for k in [
                    "frame_stride","max_frames","yolo_model","conf_thres","iou_thres",
                    "keep_class_ids","max_instances_per_frame","min_area_frac"
                ] if k in self.cfg
            },
            "frames": []
        }

        first_hw = None
        stride = int(self.cfg.get("frame_stride", 1))
        max_frames = self.cfg.get("max_frames", None)
        max_inst = int(self.cfg.get("max_instances_per_frame", 30))
        min_area_frac = float(self.cfg.get("min_area_frac", 0.001))

        for frame_idx, frame_bgr in tqdm(iter_frames_mp4(mp4_path, stride, max_frames),
                                         desc=f"clip {clip_id}"):
            H, W = frame_bgr.shape[:2]
            if first_hw is None:
                first_hw = (H, W)
            img_area = H * W
            min_area = int(min_area_frac * img_area)

            dets = self.detector.detect_xyxy(frame_bgr)

            objects = []
            for d in dets[:max_inst]:
                mask, sam_score = self.sam.segment_box(frame_bgr, d["bbox_xyxy"])
                area = int(mask.sum())
                if area < min_area:
                    continue
                objects.append({
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "det_conf": d["conf"],
                    "bbox_xyxy": d["bbox_xyxy"],
                    "sam_score": sam_score,
                    "mask_rle": binmask_to_rle(mask),
                    "mask_area": area,
                })

            out["frames"].append({
                "frame_idx": int(frame_idx),
                "objects": objects
            })

        tmp = out_json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_json_path)
        