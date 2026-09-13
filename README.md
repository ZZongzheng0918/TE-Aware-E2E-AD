# Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving

> **Accepted by ECCV 2026**

<p align="center">
  <a href="https://arxiv.org/abs/2608.18035">
    <img src="https://img.shields.io/badge/arXiv-2608.18035-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv:2608.18035">
  </a>
  <a href="https://zzongzheng0918.github.io/TE-Aware-E2E-AD/">
    <img src="https://img.shields.io/badge/Project-Page-2F8F9D?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/Zzz0918/Traffic_Elements">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000" alt="Dataset">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-3DA639?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License">
  </a>
</p>

## Traffic Elements Generation

### 2D Traffic Element Detection

Environment reference:

- [Ultralytics](https://github.com/ultralytics/ultralytics)

The YOLO weights for 2D Traffic Element detection are available on Hugging Face:

- Weights directory: [`weights/yolo/`](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/tree/main/weights/yolo)
- Model file: [`best.pt`](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/blob/main/weights/yolo/best.pt)

Download the model directly to the path expected by the inference scripts:

```bash
mkdir -p ultralytics/weights
curl -L "https://huggingface.co/datasets/Zzz0918/Traffic_Elements/resolve/main/weights/yolo/best.pt?download=true" \
  -o ultralytics/weights/best.pt
```

The resulting directory should be:

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
  --project data/nuscenes \
  --name traffic_elements_2d \
  --device 0
```

#### NAVSIM

All NAVSIM scenes use YOLO to generate 2D Traffic Elements.

```bash
python ultralytics/navsim_inference.py \
  --model ultralytics/weights/best.pt \
  --data-root /path/to/navsim/sensor_blobs \
  --splits trainval test \
  --project data/navsim/traffic_elements_2d \
  --device 0
```

#### SimScale

SimScale uses YOLO to generate 2D Traffic Elements for the images listed in its split manifests.

```bash
python ultralytics/simscale_inference.py \
  --model ultralytics/weights/best.pt \
  --data-root /path/to/navsim/sensor_blobs \
  --manifest-dir ultralytics/simscale_manifests \
  --project data/simscale/traffic_elements_2d \
  --device 0
```

### Depth Estimation and 3D Traffic Element Generation

Environment references:

- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [Simscale](https://github.com/OpenDriveLab/SimScale)

Combine the 2D Traffic Elements with UniDepth estimates to generate 3D Traffic Element center points.

#### nuScenes

```bash
CUDA_VISIBLE_DEVICES=0 python UniDepth/nusc_te_gen.py \
  --nusc-root /path/to/nuscenes \
  --nusc-version v1.0-trainval \
  --mapping-path UniDepth/mapping_openlane_nusc.json \
  --openlane-root /path/to/OpenLane-V2 \
  --det-dir data/nuscenes/traffic_elements_2d/labels \
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
  --det-dir data/navsim/traffic_elements_2d \
  --save-dir data/navsim/traffic_elements \
  --splits trainval test \
  --camera CAM_F0 \
  --model-size l \
  --device cuda:0
```

#### SimScale

```bash
CUDA_VISIBLE_DEVICES=0 python UniDepth/simscale_te_gen.py \
  --log-dir /path/to/simscale/navsim_logs \
  --data-root /path/to/navsim/sensor_blobs \
  --det-dir data/simscale/traffic_elements_2d \
  --save-dir data/simscale/traffic_elements \
  --camera CAM_F0 \
  --model-size l \
  --device cuda:0
```

## Data and Model Organization

### Dataset Downloads

| Dataset                             | Archive                              | Download                                                                |
| ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| nuScenes 2D and 3D Traffic Elements | `nuscenes_traffic_elements.tar.gz` | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements) |
| NAVSIM 2D and 3D Traffic Elements   | `navsim_traffic_elements.tar.gz`   | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements) |
| SimScale 2D and 3D Traffic Elements | `simscale_traffic_elements.tar.gz` | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements) |

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
├── navsim/
│   ├── traffic_elements/
│   │   ├── trainval/
│   │   │   └── <log_name>/
│   │   │       └── <image_token>.json
│   │   └── test/
│   │       └── <log_name>/
│   │           └── <image_token>.json
│   └── traffic_elements_2d/
│       ├── trainval/
│       │   └── <log_name>/
│       │       └── labels/
│       │           └── <image_token>.txt
│       └── test/
│           └── <log_name>/
│               └── labels/
│                   └── <image_token>.txt
└── simscale/
    ├── traffic_elements/
    │   └── <split>/
    │       └── <scene_name>/
    │           └── <image_token>.json
    └── traffic_elements_2d/
        └── <split>/
            └── <scene_name>/
                └── labels/
                    └── <image_token>.txt
```

## End-to-End Driving on NAVSIM

Environment reference:

- [NAVSIM](https://github.com/autonomousvision/navsim)
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive.git)

### Pretrained Weights

The released evaluation checkpoints are available on Hugging Face:

| Agent                            | Hydra config                         | Checkpoint                                                                                                                                                 |
| -------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LTF + Ours                       | `ltf_te_agent`                     | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/blob/main/weights/ltf_te_agent/ltf_te.ckpt)                                         |
| DiffusionDrive + Ours            | `diffusiondrive_te_agent`          | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/blob/main/weights/diffusiondrive_te_agent/diffusiondrive_te.ckpt)                   |
| LTF + SimScale + Ours            | `ltf_te_simscale_agent`            | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/blob/main/weights/ltf_te_simscale_agent/ltf_te_simscale.ckpt)                       |
| DiffusionDrive + SimScale + Ours | `diffusiondrive_te_simscale_agent` | [Hugging Face](https://huggingface.co/datasets/Zzz0918/Traffic_Elements/blob/main/weights/diffusiondrive_te_simscale_agent/diffusiondrive_te_simscale.ckpt) |

Place the downloaded checkpoints and traffic elements at the following paths:

```text
e2e/navsim/exp/
├── ltf_te_agent/
│   └── ltf_te.ckpt
├── diffusiondrive_te_agent/
│   └── diffusiondrive_te.ckpt
├── ltf_te_simscale_agent/
│   └── ltf_te_simscale.ckpt
├── diffusiondrive_te_simscale_agent/
│   └── diffusiondrive_te_simscale.ckpt
└── traffic_elements
```

### NAVHARD Metric Cache

Generate the shared metric cache once before evaluating an agent:

```bash
bash e2e/navsim/scripts/traffic_elements/run_navhard_metric_caching.sh
```

### LTF + Ours

```bash
# Dataset caching
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_agent.sh cache

# Training
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_agent.sh train

# Evaluation on NAVHARD
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_agent.sh eval
```

### DiffusionDrive + Ours

```bash
# Dataset caching
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_agent.sh cache

# Training
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_agent.sh train

# Evaluation on NAVHARD
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_agent.sh eval
```

### LTF + SimScale + Ours

```bash
# Dataset caching
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_simscale_agent.sh cache

# Stage 1: PDM synthetic data
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_simscale_agent.sh pdm

# Stage 2: Recovery synthetic data
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_simscale_agent.sh recovery

# Evaluation on NAVHARD
bash e2e/navsim/scripts/traffic_elements/run_ltf_te_simscale_agent.sh eval
```

### DiffusionDrive + SimScale + Ours

```bash
# Dataset caching
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_simscale_agent.sh cache

# Stage 1: PDM synthetic data
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_simscale_agent.sh pdm

# Stage 2: Recovery synthetic data
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_simscale_agent.sh recovery

# Evaluation on NAVHARD
bash e2e/navsim/scripts/traffic_elements/run_diffusiondrive_te_simscale_agent.sh eval
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026plug,
  title={Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving},
  author={Zhang, Zongzheng and Wang, Jijun and Zhang, Saining and Wang, Shuo and Wang, Yiru and Yang, Hai and Chen, Yang and Heng, Yuwen and Sun, Hao and Jiang, Anqing and others},
  journal={arXiv preprint arXiv:2608.18035},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
