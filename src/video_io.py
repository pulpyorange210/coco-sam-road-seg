import cv2

def iter_frames_mp4(mp4_path: str, frame_stride: int = 1, max_frames: int | None = None):
    """
    Yields (frame_idx, frame_bgr) from an mp4 using OpenCV.
    frame_idx is the *decoded* frame index (0-based).
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")

    idx = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            yield idx, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        idx += 1

    cap.release()
    