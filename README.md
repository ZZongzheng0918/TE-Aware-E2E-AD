# Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving

> **Accepted by ECCV 2026**

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="ArXiv Coming Soon">
  <a href="https://zzongzheng0918.github.io/TE-Aware-E2E-AD/">
    <img src="https://img.shields.io/badge/Project-Page-2F8F9D?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/Zzz0918/Traffic_Elements">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000" alt="Dataset">
  </a>
</p>


## Traffic Elements Generation

### 2D Traffic Element Detection

Environment reference:

- [Ultralytics](https://github.com/ultralytics/ultralytics)

YOLO weights for 2D Traffic Element detection: **To be released**. Place the downloaded weights at:

```text
TE-Aware-E2E-AD/
└── ultralytics/
    └── weights/
        └── best.pt
```

#### nuScenes

Some nuScenes scenes use OpenLaneV2 2D ground truth, while the remaining scenes use YOLO detections.

```bash
python ultralytics/nusc_inference.py \
  --model ultralytics/weights/best.pt \
  --source /path/to/nuscenes/samples/CAM_FRONT \
  --project traffic_elements \
  --name nusc_te \
  --device 0
```

#### NAVSIM

All NAVSIM scenes use YOLO to generate 2D Traffic Elements.

```bash
python ultralytics/navsim_inference.py \
  --model ultralytics/weights/best.pt \
  --data-root /path/to/navsim/sensor_blobs \
  --splits trainval test \
  --project traffic_elements/navsim_te \
  --device 0
```

### Depth Estimation and 3D Traffic Element Generation

Environment references:

- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)

Combine the 2D Traffic Elements with UniDepth estimates to generate 3D Traffic Element center points.

#### nuScenes

```bash
CUDA_VISIBLE_DEVICES=0 python UniDepth/nusc_te_gen.py \
  --nusc-root /path/to/nuscenes \
  --nusc-version v1.0-trainval \
  --mapping-path UniDepth/mapping_openlane_nusc.json \
  --openlane-root /path/to/OpenLane-V2 \
  --det-dir traffic_elements/nusc_te/labels \
  --save-dir data/nuscenes/traffic_elements \
  --missing-file data/nuscenes/traffic_elements/missing.txt \
  --model-size l \
  --device cuda:0 \
  --camera CAM_FRONT
```

#### NAVSIM

```bash
CUDA_VISIBLE_DEVICES=0 python UniDepth/navsim_te_gen.py \
  --log-dir /path/to/navsim/navsim_logs \
  --data-root /path/to/navsim/sensor_blobs \
  --det-dir traffic_elements/navsim_te \
  --save-dir data/navsim/traffic_elements \
  --splits trainval test \
  --camera CAM_F0 \
  --model-size l \
  --device cuda:0
```

## Data and Model Organization

### Dataset Downloads

| Dataset | Archive | Download |
| --- | --- | --- |
| nuScenes 2D and 3D Traffic Elements | `nuscenes_traffic_elements.tar.gz` | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements) |
| NAVSIM 2D and 3D Traffic Elements | `navsim_traffic_elements.tar.gz` | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements) |

### Directory Structure

The downloaded datasets should be organized under `data/` as follows:

```text
data/
├── nuscenes/
│   ├── traffic_elements/
│   │   └── <sample_token>.json
│   └── traffic_elements_2d/
│       └── labels/
│           └── <image_name>.txt
└── navsim/
    ├── traffic_elements/
    │   ├── trainval/
    │   │   └── <log_name>/
    │   │       └── <image_token>.json
    │   └── test/
    │       └── <log_name>/
    │           └── <image_token>.json
    └── traffic_elements_2d/
        ├── trainval/
        │   └── <log_name>/
        │       └── labels/
        │           └── <image_token>.txt
        └── test/
            └── <log_name>/
                └── labels/
                    └── <image_token>.txt
```
