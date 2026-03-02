import os, glob, argparse, yaml
from src.pipeline import CocoSamPipeline

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.input_dir:  cfg["input_dir"] = args.input_dir
    if args.output_dir: cfg["output_dir"] = args.output_dir

    in_dir = cfg["input_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    mp4s = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    if not mp4s:
        raise SystemExit(f"No mp4 files found in {in_dir}")

    pipe = CocoSamPipeline(cfg)

    for mp4 in mp4s:
        clip_id = os.path.splitext(os.path.basename(mp4))[0]
        out_json = os.path.join(out_dir, f"{clip_id}.json")
        if os.path.exists(out_json):
            print(f"[skip] {clip_id} (already exists)")
            continue
        print(f"[run]  {clip_id}")
        pipe.process_mp4(mp4, out_json)

if __name__ == "__main__":
    main()
    