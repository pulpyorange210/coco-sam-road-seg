from ultralytics import YOLO

class CocoDetector:
    def __init__(self, model_name: str, conf_thres: float, iou_thres: float, keep_class_ids: set[int]):
        self.model = YOLO(model_name)
        self.conf = conf_thres
        self.iou = iou_thres
        self.keep = keep_class_ids

    def detect_xyxy(self, frame_bgr):
        """
        Returns a list of detections: dict(label_id, label_name, conf, bbox_xyxy)
        bbox_xyxy is [x1, y1, x2, y2] float.
        """
        res = self.model(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        names = res.names  # id -> string

        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets

        for xyxy, cls_id, conf in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
        ):
            cid = int(cls_id)
            if cid not in self.keep:
                continue
            dets.append({
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "conf": float(conf),
                "bbox_xyxy": [float(x) for x in xyxy.tolist()],
            })
        return dets