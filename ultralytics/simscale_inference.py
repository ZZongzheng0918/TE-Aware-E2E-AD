import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = Path(__file__).resolve().parent / "weights/best.pt"
DEFAULT_CLASSES = list(range(13))
DEFAULT_SPLITS = [
    "navtrain_reaction_pdm_v1.0-1",
    "navtrain_reaction_pdm_v1.0-2",
    "navtrain_reaction_recovery_v1.0-1",
    "navtrain_reaction_recovery_v1.0-2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic light/sign detection on SimScale images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the YOLO model weights.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the NAVSIM sensor_blobs directory containing SimScale split folders.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing one <split>.txt image manifest per SimScale split.",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="SimScale splits to process.")
    parser.add_argument("--camera", default="CAM_F0", help="Camera name recorded in the image manifests.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("traffic_elements/simscale_te"),
        help="Directory for prediction outputs.",
    )
    parser.add_argument("--device", default=None, help="CUDA device id, for example '0'. Uses Ultralytics default when omitted.")
    parser.add_argument("--imgsz", type=int, default=2048, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="Class ids to keep.")
    parser.add_argument("--save-images", action="store_true", help="Save annotated prediction images.")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmented inference.")
    return parser.parse_args()


def load_split_images(args, split):
    manifest_path = args.manifest_dir / f"{split}.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SimScale image manifest not found: {manifest_path}")

    split_dir = args.data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"SimScale split directory not found: {split_dir}")

    images_by_scene = {}
    missing = []
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        relative_path = Path(line.strip())
        if not relative_path.parts:
            continue
        if relative_path.is_absolute() or len(relative_path.parts) != 3:
            raise ValueError(f"Invalid image path at {manifest_path}:{line_number}: {line}")

        scene, camera, _ = relative_path.parts
        if camera != args.camera:
            continue

        image_path = split_dir / relative_path
        if image_path.is_file():
            images_by_scene.setdefault(scene, set()).add(image_path)
        else:
            missing.append(image_path)

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"{len(missing)} SimScale images were not found:\n{preview}{suffix}")

    return {scene: sorted(paths) for scene, paths in sorted(images_by_scene.items())}


def predict_scene(model, args, split, scene, image_paths):
    manifest_dir = args.project / ".manifests" / split
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{scene}.txt"
    manifest_path.write_text("\n".join(str(path.absolute()) for path in image_paths) + "\n")

    predict_args = {
        "source": manifest_path,
        "save_txt": True,
        "save_conf": True,
        "save": args.save_images,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "max_det": args.max_det,
        "augment": not args.no_augment,
        "project": args.project / split,
        "name": scene,
        "classes": args.classes,
        "exist_ok": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    try:
        model.predict(**predict_args)
    finally:
        manifest_path.unlink(missing_ok=True)


def remove_empty_dir(path):
    try:
        path.rmdir()
    except OSError:
        pass


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    for split in args.splits:
        images_by_scene = load_split_images(args, split)
        for scene, image_paths in images_by_scene.items():
            predict_scene(model, args, split, scene, image_paths)
        remove_empty_dir(args.project / ".manifests" / split)

    remove_empty_dir(args.project / ".manifests")


if __name__ == "__main__":
    main()
