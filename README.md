# COCO + SAM Road Object Segmentation

This repository provides a simple preprocessing pipeline:

MP4 video → COCO object detection (YOLOv8) → SAM mask refinement → JSON output.

It detects common road objects and generates pixel-level segmentation masks for each frame.

---

WHAT IT DOES

For each input MP4 clip:

1. Runs YOLOv8 (COCO-pretrained) to detect objects.
2. Keeps only these road-relevant classes:
   - person
   - bicycle
   - car
   - motorcycle
   - bus
   - truck
3. Uses Meta Segment Anything (SAM) in box-prompt mode to refine each bounding box into a segmentation mask.
4. Saves results as a JSON file (one per clip).

---

INSTALLATION

Install Python dependencies:

pip install -r requirements.txt

Clone and install Segment Anything:

git clone https://github.com/facebookresearch/segment-anything.git
pip install -e segment-anything

Download the SAM checkpoint:

wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Place the checkpoint file in the root directory of this repository.

---

INPUT FORMAT

Put MP4 clips inside:

data/clips/*.mp4

---

RUN

python run_pipeline.py --config configs/default.yaml

Or:

python run_pipeline.py --input_dir data/clips --output_dir outputs

---

OUTPUT

For each clip, a file is created:

outputs/<clip_id>.json

Each JSON contains:
- clip_id
- input_mp4
- config
- frames
  - frame_idx
  - objects
    - class_id
    - class_name
    - det_conf
    - bbox_xyxy
    - sam_score
    - mask_area
    - mask_rle (COCO RLE format)

---

NOTES

- SAM is used in box-prompt mode (not automatic mask generation).
- Masks are stored in COCO RLE format.
- This repository performs segmentation only.
- Tracking and downstream modeling are not included.

