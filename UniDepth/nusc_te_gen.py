import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3D traffic elements for nuScenes. OpenLaneV2 boxes are used when available; "
            "YOLO boxes are used to fill samples without OpenLaneV2 mappings."
        )
    )
    parser.add_argument("--nusc-root", type=Path, required=True, help="nuScenes dataroot.")
    parser.add_argument("--nusc-version", default="v1.0-trainval", help="nuScenes version.")
    parser.add_argument("--save-dir", type=Path, required=True, help="Output directory for generated JSON files.")
    parser.add_argument("--model-size", choices=["s", "b", "l"], default="l", help="UniDepth V2 ViT backbone size.")
    parser.add_argument("--device", default=None, help="Torch device, for example 'cuda:0' or 'cpu'.")
    parser.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera name.")

    parser.add_argument("--mapping-path", type=Path, required=True, help="OpenLaneV2 to nuScenes mapping JSON.")
    parser.add_argument("--openlane-root", type=Path, required=True, help="OpenLaneV2 dataset root.")
    parser.add_argument("--det-dir", type=Path, help="YOLO label directory for samples without OpenLaneV2 mappings.")
    parser.add_argument("--missing-file", type=Path, help="Path to save samples without OpenLaneV2 mappings.")
    parser.add_argument("--score-threshold", type=float, default=0.6, help="Minimum YOLO confidence.")
    return parser.parse_args()


def load_model(model_size, device_name):
    import torch
    from unidepth.models import UniDepthV2

    model_name = f"unidepth-v2-vit{model_size}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
    model.interpolation_mode = "bilinear"

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(device).eval()


def build_nuscenes(args):
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=args.nusc_version, dataroot=str(args.nusc_root), verbose=True)


def get_calibration(nusc, sample, camera_name):
    import numpy as np
    from pyquaternion import Quaternion

    cam_token = sample["data"][camera_name]
    cam_data = nusc.get("sample_data", cam_token)
    image_path = nusc.get_sample_data_path(cam_token)

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    intrinsic = np.asarray(nusc.get_sample_data(cam_token)[2])

    pose_record = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    lidar_cs_record = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar2ego_t = np.asarray(lidar_cs_record["translation"])
    ego2global_t = np.asarray(pose_record["translation"])
    lidar2ego_r_mat = Quaternion(lidar_cs_record["rotation"]).rotation_matrix
    ego2global_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix

    cam_cs_record = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_pose_record = nusc.get("ego_pose", cam_data["ego_pose_token"])
    cam2ego_t = np.asarray(cam_cs_record["translation"])
    ego2global_t_cam = np.asarray(cam_pose_record["translation"])
    cam2ego_r_mat = Quaternion(cam_cs_record["rotation"]).rotation_matrix
    ego2global_r_mat_cam = Quaternion(cam_pose_record["rotation"]).rotation_matrix

    lidar_to_global_inv = np.linalg.inv(ego2global_r_mat).T @ np.linalg.inv(lidar2ego_r_mat).T
    rotation = (cam2ego_r_mat.T @ ego2global_r_mat_cam.T) @ lidar_to_global_inv
    translation = (cam2ego_t @ ego2global_r_mat_cam.T + ego2global_t_cam) @ lidar_to_global_inv
    translation -= ego2global_t @ lidar_to_global_inv + lidar2ego_t @ np.linalg.inv(lidar2ego_r_mat).T

    return image_path, intrinsic, rotation.T, translation


def infer_depth(model, image_path, intrinsic):
    import numpy as np
    import torch
    from PIL import Image
    from unidepth.utils.camera import Pinhole

    rgb = np.array(Image.open(image_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(intrinsic)
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
    predictions = model.infer(rgb_torch, camera)
    return rgb, predictions["depth"].squeeze().cpu().numpy()


def project_center(center, depth_pred, intrinsic, cam2lidar_rotation, cam2lidar_translation):
    import numpy as np
    from scipy.ndimage import map_coordinates

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    height, width = depth_pred.shape
    u = np.clip(center[0], 0, width - 1)
    v = np.clip(center[1], 0, height - 1)

    depth_value = map_coordinates(depth_pred, np.vstack((v, u)), order=1, mode="nearest")
    point_3d = np.stack(
        ((u - cx) * depth_value / fx, (v - cy) * depth_value / fy, depth_value),
        axis=-1,
    ).reshape(-1, 3)
    return ((cam2lidar_rotation @ point_3d.T).T + cam2lidar_translation).squeeze()


def write_json(path, traffic_elements):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"traffic_elements": traffic_elements}, f, indent=4)


def process_openlane_samples(nusc, model, args):
    import numpy as np
    from openlanev2.centerline.dataset import Collection
    from tqdm import tqdm

    with args.mapping_path.open("r") as f:
        mapping = json.load(f)

    collections = {
        "train": Collection(str(args.openlane_root), str(args.openlane_root), "data_dict_subset_B_train"),
        "val": Collection(str(args.openlane_root), str(args.openlane_root), "data_dict_subset_B_val"),
    }

    missing = []
    output_dir = args.save_dir

    for sample in tqdm(nusc.sample, desc="Processing nuScenes"):
        scene_token = sample["scene_token"]
        mapping_info = mapping.get(scene_token)
        if not mapping_info:
            sample_data = nusc.get("sample_data", sample["data"][args.camera])
            missing.append((scene_token, sample["token"], sample_data.get("filename", "")))
            continue

        image_path, intrinsic, cam2lidar_rotation, cam2lidar_translation = get_calibration(nusc, sample, args.camera)
        _, depth_pred = infer_depth(model, image_path, intrinsic)

        split = mapping_info["split"]
        openlane_index = mapping_info["openlane_index"]
        timestamp = nusc.get("sample_data", sample["data"][args.camera])["timestamp"]
        frame = collections[split].get_frame_via_identifier((split, openlane_index, str(timestamp)))

        traffic_elements = []
        for traffic_element in frame.get_annotations()["traffic_element"]:
            points_2d = traffic_element["points"]
            center_2d = np.array([points_2d[:, 0].mean(), points_2d[:, 1].mean()])
            center_3d = project_center(center_2d, depth_pred, intrinsic, cam2lidar_rotation, cam2lidar_translation)
            traffic_elements.append(
                {
                    "id": traffic_element["id"],
                    "category": traffic_element["category"],
                    "attribute": traffic_element["attribute"],
                    "2d_points": points_2d.tolist(),
                    "2d_center": center_2d.tolist(),
                    "3d_center": center_3d.tolist(),
                }
            )

        write_json(output_dir / f"{sample['token']}.json", traffic_elements)

    missing_file = args.missing_file or args.save_dir / "missing.txt"
    missing_file.parent.mkdir(parents=True, exist_ok=True)
    with missing_file.open("w") as f:
        for scene_token, sample_token, file_path in missing:
            f.write(f"{scene_token}, {sample_token}, {file_path}\n")

    return [sample_token for _, sample_token, _ in missing]


def load_yolo_detections(label_path, width, height, threshold):
    import numpy as np

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

            abs_x_center = x_center * width
            abs_y_center = y_center * height
            abs_box_w = box_w * width
            abs_box_h = box_h * height
            detections.append(
                {
                    "id": "yolo",
                    "category": 1 if class_id < 4 else 2,
                    "attribute": class_id,
                    "confidence": confidence,
                    "2d_center": np.array([abs_x_center, abs_y_center]),
                    "2d_points": np.array(
                        [
                            [abs_x_center - abs_box_w / 2, abs_y_center - abs_box_h / 2],
                            [abs_x_center + abs_box_w / 2, abs_y_center + abs_box_h / 2],
                        ]
                    ),
                }
            )

    return detections


def process_yolo_samples(nusc, model, args, sample_tokens):
    from tqdm import tqdm

    output_dir = args.save_dir / "res"
    for sample_token in tqdm(sample_tokens, desc="Processing YOLO fallback samples"):
        sample = nusc.get("sample", sample_token)
        image_path, intrinsic, cam2lidar_rotation, cam2lidar_translation = get_calibration(nusc, sample, args.camera)
        rgb, depth_pred = infer_depth(model, image_path, intrinsic)
        height, width = rgb.shape[:2]

        label_path = args.det_dir / f"{Path(image_path).stem}.txt"
        traffic_elements = []
        for detection in load_yolo_detections(label_path, width, height, args.score_threshold):
            center_3d = project_center(
                detection["2d_center"],
                depth_pred,
                intrinsic,
                cam2lidar_rotation,
                cam2lidar_translation,
            )
            traffic_elements.append(
                {
                    "id": detection["id"],
                    "category": detection["category"],
                    "attribute": detection["attribute"],
                    "confidence": float(detection["confidence"]),
                    "2d_points": detection["2d_points"].tolist(),
                    "2d_center": detection["2d_center"].tolist(),
                    "3d_center": center_3d.tolist(),
                }
            )

        write_json(output_dir / f"{sample_token}.json", traffic_elements)


def main():
    args = parse_args()
    nusc = build_nuscenes(args)
    model = load_model(args.model_size, args.device)

    missing_tokens = process_openlane_samples(nusc, model, args)
    if args.det_dir is not None and missing_tokens:
        process_yolo_samples(nusc, model, args, missing_tokens)


if __name__ == "__main__":
    main()
