import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = Path(__file__).resolve().parent / "weights/best.pt"
DEFAULT_CLASSES = list(range(13))


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic light/sign detection on nuScenes front camera images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the YOLO model weights.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to nuScenes CAM_FRONT images, for example: /path/to/nuscenes/samples/CAM_FRONT.",
    )
    parser.add_argument("--project", type=Path, default=Path("traffic_elements"), help="Directory for prediction outputs.")
    parser.add_argument("--name", default="nusc_te", help="Prediction run name.")
    parser.add_argument("--device", default=None, help="CUDA device id, for example '0'. Uses Ultralytics default when omitted.")
    parser.add_argument("--imgsz", type=int, default=2048, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="Class ids to keep.")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmented inference.")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    predict_args = {
        "source": args.source,
        "save_txt": True,
        "save_conf": True,
        "save": True,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "max_det": args.max_det,
        "augment": not args.no_augment,
        "project": args.project,
        "name": args.name,
        "classes": args.classes,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    model.predict(**predict_args)


if __name__ == "__main__":
    main()
