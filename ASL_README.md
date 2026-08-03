# WiLoR Hand Parameter Extraction and Interpolation for ASL Data

This document explains how to process two ASL/sign-language image datasets in this repository:

- How2Sign: `/data/hwu/how2sign`
- YouTube ASL clips: `/data/hwu/youtube_dataset/clip_fps24_img_0`

The complete workflow has two stages:

1. GPU inference: detect hands and save raw per-frame WiLoR/MANO PKLs.
2. CPU postprocessing: organize each frame into fixed `[left, right]` slots and temporally interpolate missing hands.

Main scripts:

- [`extract_how2sign_wilor.py`](./extract_how2sign_wilor.py): batched GPU parameter extraction.
- [`interpolate_how2sign_wilor.py`](./interpolate_how2sign_wilor.py): CPU left/right alignment and temporal interpolation.
- [`run_how2sign_wilor_2gpu.sh`](./run_how2sign_wilor_2gpu.sh): background launcher for How2Sign.
- [`run_youtube_wilor_2gpu.sh`](./run_youtube_wilor_2gpu.sh): background launcher for the YouTube dataset.
- [`demo.py`](./demo.py): visualization for a single clip.

Although the two launcher filenames retain the `_2gpu` suffix, they support any number of GPUs.

## 1. Environment and Required Files

Enter the repository:

```bash
cd /home/student/hwu/Workplace/WiLoR
```

Use the configured Python environment:

```text
/home/student/hwu/miniconda3/envs/wilor/bin/python
```

The following model files must exist:

```text
pretrained_models/wilor_final.ckpt
pretrained_models/model_config.yaml
pretrained_models/detector.pt
mano_data/MANO_RIGHT.pkl
```

Check GPU availability:

```bash
nvidia-smi
```

## 2. Input and Output Conventions

### 2.1 Input clips

Each dataset root contains multiple clip directories. Each clip contains video frames ordered by filename:

```text
<input-root>/
├── <clip-id-0>/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── <clip-id-1>/
└── ...
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`.

### 2.2 Raw WiLoR output

GPU extraction produces:

```text
<output-root>/
└── wilor_params/
    └── <clip-id>/
        ├── frame_000000.pkl
        ├── frame_000001.pkl
        ├── ...
        └── .complete
```

Every input frame has one PKL. The first dimension of raw detections is `N`, where `N` may be 0, 1, 2, or greater.

### 2.3 Interpolated output

CPU interpolation writes to a separate directory and does not modify raw PKLs:

```text
<output-root>/
├── wilor_params/
└── wilor_params_interpolated/
    └── <clip-id>/
        ├── frame_000000.pkl
        ├── frame_000001.pkl
        ├── ...
        └── .complete
```

The interpolated output has fixed hand slots:

```text
index 0 = left hand
index 1 = right hand
```

## 3. Relationship Between GPUs and Shards

The launcher scripts interpret positional arguments as physical GPU IDs:

```bash
bash run_how2sign_wilor_2gpu.sh 2
bash run_how2sign_wilor_2gpu.sh 0 1
bash run_how2sign_wilor_2gpu.sh 0 2 3
```

These commands use one, two, and three GPUs, respectively. The scripts configure:

```text
num_shards = number of GPU arguments
one GPU = one worker = one shard index
```

Example mapping:

```text
physical GPU 0 -> CUDA_VISIBLE_DEVICES=0 -> internal --device 0 -> shard 0
physical GPU 2 -> CUDA_VISIBLE_DEVICES=2 -> internal --device 0 -> shard 1
physical GPU 3 -> CUDA_VISIBLE_DEVICES=3 -> internal --device 0 -> shard 2
```

`--shard-index` is not a GPU ID. Each process sees only one card through `CUDA_VISIBLE_DEVICES`, so its internal device is always `--device 0`.

For backward compatibility, both launchers default to GPUs 0 and 1 when no GPU IDs are provided.

Do not change the GPU count while old workers are still running. After all workers stop, the task may be resumed with a different GPU count; clips with `.complete` are skipped.

## 4. How2Sign Parameter Extraction

### 4.1 Directories

Inputs:

```text
/data/hwu/how2sign/how2sign_images_test
/data/hwu/how2sign/how2sign_images_val
/data/hwu/how2sign/how2sign_images_train
```

Raw outputs:

```text
/data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params
/data/hwu/how2sign/how2sign_images_val_wilor_out/wilor_params
/data/hwu/how2sign/how2sign_images_train_wilor_out/wilor_params
```

### 4.2 Recommended: background launcher

Use only GPU 2:

```bash
cd /home/student/hwu/Workplace/WiLoR
bash run_how2sign_wilor_2gpu.sh 2
```

Use GPUs 0 and 1:

```bash
bash run_how2sign_wilor_2gpu.sh 0 1
```

Use GPUs 0, 2, and 3:

```bash
bash run_how2sign_wilor_2gpu.sh 0 2 3
```

The script processes splits in this order:

```text
test -> val -> train
```

Logs:

```bash
tail -f logs/how2sign_wilor/worker_0.log
tail -f logs/how2sign_wilor/worker_1.log
```

### 4.3 Single-process foreground command

```bash
CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/wilor_matplotlib_single \
YOLO_CONFIG_DIR=/tmp/wilor_ultralytics_single \
PYTHONUNBUFFERED=1 \
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    extract_how2sign_wilor.py \
    --dataset-root /data/hwu/how2sign \
    --splits test val train \
    --device 0 \
    --num-shards 1 \
    --shard-index 0 \
    --detector-batch-size 32 \
    --wilor-batch-size 64 \
    --fast
```

## 5. YouTube Parameter Extraction

### 5.1 Directories

Input:

```text
/data/hwu/youtube_dataset/clip_fps24_img_0
```

This directory contains approximately 44,919 clip directories.

Raw output:

```text
/data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params
```

### 5.2 Recommended: background launcher

Use only GPU 2:

```bash
cd /home/student/hwu/Workplace/WiLoR
bash run_youtube_wilor_2gpu.sh 2
```

Use two GPUs:

```bash
bash run_youtube_wilor_2gpu.sh 0 1
```

Use any three GPUs:

```bash
bash run_youtube_wilor_2gpu.sh 0 2 3
```

Logs:

```bash
tail -f logs/youtube_wilor/worker_0.log
tail -f logs/youtube_wilor/worker_1.log
```

### 5.3 Test only 10 clips

The following command uses physical GPU 2 and processes 10 new clips in this run:

```bash
cd /home/student/hwu/Workplace/WiLoR

CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/youtube_wilor_matplotlib_gpu2 \
YOLO_CONFIG_DIR=/tmp/youtube_wilor_ultralytics_gpu2 \
PYTHONUNBUFFERED=1 \
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    extract_how2sign_wilor.py \
    --input-root /data/hwu/youtube_dataset/clip_fps24_img_0 \
    --output-root /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out \
    --split-name youtube \
    --device 0 \
    --num-shards 1 \
    --shard-index 0 \
    --max-clips 10 \
    --detector-batch-size 32 \
    --wilor-batch-size 64 \
    --fast
```

`--max-clips` is the maximum number of new clips processed by each worker in that run. If three workers each use `--max-clips 10`, they may process up to 30 clips in total. Rerunning the single-worker command skips completed clips and processes the next 10.

### 5.4 Generic direct-directory mode

For another dataset with the same layout:

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    extract_how2sign_wilor.py \
    --input-root /path/to/clip_image_root \
    --output-root /path/to/output_wilor_out \
    --split-name custom_name \
    --device 0 \
    --num-shards 1 \
    --shard-index 0 \
    --detector-batch-size 32 \
    --wilor-batch-size 64 \
    --fast
```

## 6. Raw Per-Frame PKL Fields

For a frame with `N` detected hands:

| Key | Shape | Description |
|---|---|---|
| `num_hands` | scalar | Number of raw detections `N` |
| `hand_pose` | `[N,45]` | Axis-angle for 15 local MANO joints, in radians |
| `global_orient` | `[N,3]` | MANO global axis-angle |
| `mano_pose` | `[N,48]` | Concatenated global orientation and hand pose |
| `hand_pose_rotmat` | `[N,15,3,3]` | Original local rotation matrices from WiLoR |
| `global_orient_rotmat` | `[N,1,3,3]` | Original global rotation matrix from WiLoR |
| `betas` | `[N,10]` | MANO shape coefficients |
| `pred_cam` | `[N,3]` | Crop-camera parameters |
| `cam_t_full` | `[N,3]` | Full-image camera translation |
| `bbox_xyxy` | `[N,4]` | Detection bounding box |
| `detection_confidence` | `[N]` | Detection confidence |
| `is_right` | `[N]` | 0 for left, 1 for right |
| `box_center` | `[N,2]` | Crop center |
| `box_size` | `[N]` | Crop size |
| `image_size` | `[N,2]` | Image width and height |
| `focal_length` | `[N]` | Inference focal length |

A PKL is still written when no hand is detected, with a first dimension of 0. This preserves a one-to-one mapping between input frames and raw outputs.

## 7. Interpolation Logic

[`interpolate_how2sign_wilor.py`](./interpolate_how2sign_wilor.py) supports How2Sign, YouTube, and any dataset with the same raw-output structure.

Each clip is processed independently; interpolation never crosses clip boundaries:

1. Fix the slots to `[left, right]`.
2. If one side has multiple detections, select the highest-confidence candidate.
3. If one side is missing in an interior frame, find the nearest previous and next observations of the same side.
4. Interpolate `global_orient_rotmat` and all 15 `hand_pose_rotmat` joints with quaternion SLERP.
5. Linearly interpolate continuous values such as betas, camera parameters, and bounding boxes.
6. For missing values at the beginning or end of a clip, copy the nearest observation of the same side.
7. If one side never appears anywhere in a clip, use finite neutral values and set `valid_mask=False`.

Interpolation does not create extra video frames or change the source video's 24 FPS. It only fills missing hand parameters in existing frames.

The logged `processing_fps` is the number of PKLs processed per second, not the video playback frame rate.

## 8. How2Sign Interpolation Commands

Interpolation is CPU-only and does not require `CUDA_VISIBLE_DEVICES`.

### 8.1 Process all splits with one worker

```bash
cd /home/student/hwu/Workplace/WiLoR

/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --dataset-root /data/hwu/how2sign \
    --splits test val train \
    --num-shards 1 \
    --shard-index 0
```

Outputs:

```text
/data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params_interpolated
/data/hwu/how2sign/how2sign_images_val_wilor_out/wilor_params_interpolated
/data/hwu/how2sign/how2sign_images_train_wilor_out/wilor_params_interpolated
```

### 8.2 Two CPU workers

Run these commands in separate terminals or tmux panes.

Worker 0:

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --dataset-root /data/hwu/how2sign \
    --splits test val train \
    --num-shards 2 \
    --shard-index 0
```

Worker 1:

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --dataset-root /data/hwu/how2sign \
    --splits test val train \
    --num-shards 2 \
    --shard-index 1
```

### 8.3 Process only the test output root

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --split-output-root \
      /data/hwu/how2sign/how2sign_images_test_wilor_out
```

The How2Sign directory name contains the split, so `test` is inferred automatically.

## 9. YouTube Interpolation Commands

### 9.1 One worker

```bash
cd /home/student/hwu/Workplace/WiLoR

/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --split-output-root \
      /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out \
    --split-name youtube \
    --num-shards 1 \
    --shard-index 0
```

Output:

```text
/data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/
└── wilor_params_interpolated/
```

### 9.2 Two CPU workers

Worker 0:

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --split-output-root \
      /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out \
    --split-name youtube \
    --num-shards 2 \
    --shard-index 0
```

Worker 1:

```bash
/home/student/hwu/miniconda3/envs/wilor/bin/python \
    interpolate_how2sign_wilor.py \
    --split-output-root \
      /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out \
    --split-name youtube \
    --num-shards 2 \
    --shard-index 1
```

## 10. Interpolated Output Fields

All hand-parameter arrays have a fixed first dimension of 2:

```text
hand_pose             [2,45]
global_orient         [2,3]
mano_pose             [2,48]
hand_pose_rotmat      [2,15,3,3]
global_orient_rotmat  [2,1,3,3]
betas                 [2,10]
bbox_xyxy             [2,4]
is_right              [0,1]
```

Status fields:

| Key | Shape | Description |
|---|---|---|
| `source_num_hands` | scalar | Number of detections in the raw frame |
| `num_hands` | scalar | Always 2, representing two fixed slots |
| `observed_mask` | `[2]` bool | This side is a real detection in the current frame |
| `interpolated_mask` | `[2]` bool | This side was interpolated or copied from an edge observation |
| `valid_mask` | `[2]` bool | This side has at least one observation somewhere in the clip |
| `selected_detection_index` | `[2]` | Selected raw detection index |
| `discarded_duplicate_count` | `[2]` | Lower-confidence same-side detections that were discarded |
| `interpolation_prev_frame` | `[2]` | Previous observed frame index for each side |
| `interpolation_next_frame` | `[2]` | Next observed frame index for each side |
| `interpolation_alpha` | `[2]` | Interpolation coefficient |

The `detection_confidence` of an interpolated slot is set to 0 so generated parameters are not mistaken for real detections.

Example:

```python
import pickle

with open("frame_000100.pkl", "rb") as handle:
    data = pickle.load(handle)

left_pose = data["hand_pose"][0]
right_pose = data["hand_pose"][1]

print(data["observed_mask"])
print(data["interpolated_mask"])
print(data["valid_mask"])
```

## 11. Resuming Interrupted Runs

### 11.1 GPU extraction

- Each frame is first written to a temporary PKL and then atomically moved into place.
- A `.complete` marker is created after the entire clip finishes.
- On restart, clips with `.complete` are skipped.
- For an incomplete clip, existing frame PKLs are skipped and only missing frames are processed.
- Do not use `--overwrite` for a normal resume.

The launchers may be rerun directly:

```bash
bash run_how2sign_wilor_2gpu.sh 2
bash run_youtube_wilor_2gpu.sh 2
```

### 11.2 CPU interpolation

- Only raw clips with `.complete` are processed.
- Interpolated outputs have their own `.complete` markers.
- Completed interpolated clips are skipped.
- An incomplete output clip writes only missing output PKLs.
- Do not use `--overwrite` for a normal resume.

## 12. Monitoring Progress

### 12.1 Processes and GPUs

```bash
ps -ef | grep extract_how2sign_wilor.py
nvidia-smi
```

### 12.2 How2Sign logs

```bash
tail -f logs/how2sign_wilor/worker_0.log
```

### 12.3 YouTube logs

```bash
tail -f logs/youtube_wilor/worker_0.log
```

### 12.4 Raw PKL and completed-clip counts

How2Sign test:

```bash
find /data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params \
    -type f -name '*.pkl' | wc -l

find /data/hwu/how2sign/how2sign_images_test_wilor_out/wilor_params \
    -type f -name '.complete' | wc -l
```

YouTube:

```bash
find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params \
    -type f -name '*.pkl' | wc -l

find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params \
    -type f -name '.complete' | wc -l
```

### 12.5 Interpolation progress

```bash
find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params_interpolated \
    -type f -name '*.pkl' | wc -l

find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params_interpolated \
    -type f -name '.complete' | wc -l
```

Running `find` over millions of small files may take a long time.

## 13. Single-Clip Interpolation Visualization

Use `demo.py` to inspect one clip. Do not use it as a replacement for the batch parameter extractor.

How2Sign example:

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/student/hwu/miniconda3/envs/wilor/bin/python demo.py \
    --img_folder \
      /data/hwu/how2sign/how2sign_images_test/-fZc293MpJk_4-1-rgb_front \
    --out_folder demo_out_h2s_params \
    --fast \
    --interpolate_missing_hands \
    --output_video_fps 24
```

YouTube example:

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/student/hwu/miniconda3/envs/wilor/bin/python demo.py \
    --img_folder \
      /data/hwu/youtube_dataset/clip_fps24_img_0/<clip-id> \
    --out_folder /tmp/youtube_wilor_demo/<clip-id> \
    --fast \
    --interpolate_missing_hands \
    --output_video_fps 24
```

Outputs:

```text
<out-folder>/frame_*.jpg
<out-folder>/interpolation_preview.mp4
<out-folder>/interpolation_summary.json
```

Visualization colors:

```text
purple = real detection in the current frame
orange = result interpolated from neighboring frames
```

## 14. MANO Parameter Notes

- `hand_pose [45]` contains the final, complete rotations for 15 local MANO joints. It is not a residual with the MANO mean subtracted.
- Do not add the MANO hand mean again during reconstruction.
- The current output is a full 45D pose and must use `use_pca=False`.
- `num_pca_comps=12` does not apply to this 45D output.
- Set `flat_hand_mean=True` when reconstructing from axis-angle values.
- For the most accurate reconstruction, use `hand_pose_rotmat` and `global_orient_rotmat` directly.

## 15. Frequently Asked Questions

### Why is `processing_fps` much higher than 24?

24 FPS is the video playback rate. `processing_fps` is the number of PKLs the CPU can read, interpolate, and write per second. They are unrelated.

### Why do normal frames also appear in the interpolated directory?

Only missing-hand slots are changed. Normal frames are also written so downstream code receives a complete per-frame sequence with fixed `[left, right]` slots.

### How should `observed=399, interpolated=7` be interpreted?

A 203-frame clip has `203 * 2 = 406` left/right hand slots. `399 + 7 = 406`, so seven slots were filled by interpolation.

### Which shard index should be used with one GPU?

```text
--num-shards 1 --shard-index 0
```

`--num-shards 1 --shard-index 1` is invalid.

### Can only `--shard-index 1` be run?

If `--num-shards 2` is configured but only shard 1 is running, clips assigned to shard 0 will not be processed. Every shard index must have one worker.
