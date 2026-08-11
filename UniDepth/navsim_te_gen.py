import argparse
import json
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate NAVSIM front-camera depth and project 2D traffic elements to 3D."
    )
    parser.add_argument("--log-dir", type=Path, required=True, help="Path to NAVSIM log directory.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to NAVSIM sensor_blobs directory.")
    parser.add_argument("--det-dir", type=Path, required=True, help="Path to YOLO prediction output directory.")
    parser.add_argument("--save-dir", type=Path, required=True, help="Directory for generated traffic element JSON files.")
    parser.add_argument("--splits", nargs="+", default=["trainval"], help="NAVSIM splits to process.")
    parser.add_argument("--camera", default="CAM_F0", help="Camera key and folder name to process.")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Minimum detection confidence.")
    parser.add_argument("--model-size", choices=["s", "b", "l"], default="l", help="UniDepth V2 ViT backbone size.")
    parser.add_argument("--device", default=None, help="Torch device, for example 'cuda:0' or 'cpu'.")
    return parser.parse_args()


def load_model(model_size, device_name):
    import torch
    from unidepth.models import UniDepthV2

    model_name = f"unidepth-v2-vit{model_size}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
    model.interpolation_mode = "bilinear"

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(device).eval()


def infer_depth(model, image_path, intrinsic, distortion):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from unidepth.utils.camera import Pinhole

    rgb = np.array(Image.open(image_path))
    rgb = cv2.undistort(rgb, intrinsic, distortion)

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(intrinsic)
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

    predictions = model.infer(rgb_torch, camera)
    return rgb, predictions["depth"].squeeze().cpu().numpy()


def load_detection_labels(label_path, width, height, threshold):
    detections = []
    if not label_path.exists():
        return detections

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:5])
            confidence = float(parts[5])
            if confidence < threshold:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "center": (x_center * width, y_center * height),
                    "size": (box_w * width, box_h * height),
                }
            )

    return detections


def project_detection(detection, depth_pred, intrinsic, distortion, cam2lidar_rotation, cam2lidar_translation):
    import cv2
    import numpy as np
    from scipy.ndimage import map_coordinates

    abs_x_center, abs_y_center = detection["center"]
    abs_box_w, abs_box_h = detection["size"]

    distorted_center = np.array([[abs_x_center, abs_y_center]], dtype=np.float32)
    te_center = cv2.undistortPoints(distorted_center, intrinsic, distortion, P=intrinsic).squeeze()

    x1 = abs_x_center - abs_box_w / 2
    y1 = abs_y_center - abs_box_h / 2
    x2 = abs_x_center + abs_box_w / 2
    y2 = abs_y_center + abs_box_h / 2
    te_points = np.array([[x1, y1], [x2, y2]])

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    height, width = depth_pred.shape
    u = np.clip(te_center[0], 0, width - 1)
    v = np.clip(te_center[1], 0, height - 1)

    depth_value = map_coordinates(depth_pred, np.vstack((v, u)), order=1, mode="nearest")
    point_3d = np.stack(
        ((u - cx) * depth_value / fx, (v - cy) * depth_value / fy, depth_value),
        axis=-1,
    ).reshape(-1, 3)
    point_lidar = ((cam2lidar_rotation @ point_3d.T).T + cam2lidar_translation).squeeze()

    return {
        "attribute": detection["class_id"],
        "confidence": float(detection["confidence"]),
        "2d_points": te_points.tolist(),
        "2d_center": te_center.tolist(),
        "3d_center": point_lidar.tolist(),
    }


def process_frame(model, frame, args, split, log_name, save_dir):
    import numpy as np

    camera_info = frame["cams"][args.camera]
    image_path = args.data_root / split / camera_info["data_path"]
    if not image_path.exists():
        return

    image_token = image_path.stem
    label_path = args.det_dir / split / log_name / "labels" / f"{image_token}.txt"
    save_path = save_dir / f"{image_token}.json"
    if not label_path.exists():
        with save_path.open("w") as f:
            json.dump({"traffic_elements": []}, f, indent=4)
        return

    intrinsic = np.asarray(camera_info["cam_intrinsic"])
    distortion = np.asarray(camera_info["distortion"])
    cam2lidar_rotation = np.asarray(camera_info["sensor2lidar_rotation"])
    cam2lidar_translation = np.asarray(camera_info["sensor2lidar_translation"])
    rgb, depth_pred = infer_depth(model, image_path, intrinsic, distortion)
    height, width = rgb.shape[:2]

    detections = load_detection_labels(label_path, width, height, args.score_threshold)
    traffic_elements = [
        project_detection(
            detection,
            depth_pred,
            intrinsic,
            distortion,
            cam2lidar_rotation,
            cam2lidar_translation,
        )
        for detection in detections
    ]

    with save_path.open("w") as f:
        json.dump({"traffic_elements": traffic_elements}, f, indent=4)


def process_log(model, log_path, args, split):
    log_name = log_path.stem
    save_dir = args.save_dir / split / log_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("rb") as f:
        log = pickle.load(f)

    for frame in log:
        process_frame(model, frame, args, split, log_name, save_dir)


def main():
    from tqdm import tqdm

    args = parse_args()
    model = load_model(args.model_size, args.device)

    for split in args.splits:
        split_log_dir = args.log_dir / split
        if not split_log_dir.is_dir():
            raise FileNotFoundError(f"NAVSIM log split directory not found: {split_log_dir}")

        log_paths = sorted(split_log_dir.glob("*.pkl"))
        for log_path in tqdm(log_paths, desc=f"Processing {split}"):
            process_log(model, log_path, args, split)


if __name__ == "__main__":
    main()
