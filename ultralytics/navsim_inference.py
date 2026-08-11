import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = Path(__file__).resolve().parent / "weights/best.pt"
DEFAULT_CLASSES = list(range(13))


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic light/sign detection on NAVSIM front camera images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the YOLO model weights.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to NAVSIM sensor_blobs directory containing split folders.",
    )
    parser.add_argument("--splits", nargs="+", default=["trainval"], help="NAVSIM splits to process.")
    parser.add_argument("--camera", default="CAM_F0", help="Camera folder name under each scene.")
    parser.add_argument("--project", type=Path, default=Path("traffic_elements/navsim_te"), help="Directory for prediction outputs.")
    parser.add_argument("--device", default=None, help="CUDA device id, for example '0'. Uses Ultralytics default when omitted.")
    parser.add_argument("--imgsz", type=int, default=2048, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="Class ids to keep.")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmented inference.")
    return parser.parse_args()


def predict_scene(model, args, split, scene_dir):
    predict_args = {
        "source": scene_dir / args.camera,
        "save_txt": True,
        "save_conf": True,
        "save": True,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "max_det": args.max_det,
        "augment": not args.no_augment,
        "project": args.project / split,
        "name": scene_dir.name,
        "classes": args.classes,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    model.predict(**predict_args)


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    for split in args.splits:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"NAVSIM split directory not found: {split_dir}")

        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            camera_dir = scene_dir / args.camera
            if camera_dir.is_dir():
                predict_scene(model, args, split, scene_dir)


if __name__ == "__main__":
    main()
