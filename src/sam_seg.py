import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

class SamSegmenter:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def segment_box(self, frame_bgr, bbox_xyxy):
        """
        bbox_xyxy: [x1,y1,x2,y2]
        returns (mask_bool, score)
        """
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img)

        box = np.array(bbox_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])
    