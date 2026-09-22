# WiLoR Hand Parameter Extraction and Interpolation for ASL Data

This document explains how to process two ASL/sign-language image datasets in this repository:

- How2Sign: `/data/hwu/how2sign`
- YouTube ASL clips: `/data/hwu/youtube_dataset/clip_fps24_img_0`

The complete workflow has three stages:

1. GPU inference: detect hands and save raw per-frame WiLoR/MANO PKLs.
2. CPU postprocessing: organize each frame into fixed `[left, right]` slots and temporally interpolate missing hands.
3. GPU refinement: fuse the interpolated hands into whole-body SMPL-X and jointly refine the arms against WiLoR and RTMPose.

Stages 1 and 2 are documented in sections 1 to 10. Stage 3 is documented in section 11 and runs on the OpenASL clip set.

Main scripts:

- [`extract_how2sign_wilor.py`](./extract_how2sign_wilor.py): batched GPU parameter extraction.
- [`interpolate_how2sign_wilor.py`](./interpolate_how2sign_wilor.py): CPU left/right alignment and temporal interpolation.
- [`run_how2sign_wilor_2gpu.sh`](./run_how2sign_wilor_2gpu.sh): background launcher for How2Sign.
- [`run_youtube_wilor_2gpu.sh`](./run_youtube_wilor_2gpu.sh): background launcher for the YouTube dataset.
- [`fuse_shared_aios_wilor.py`](./fuse_shared_aios_wilor.py): merge AIOS person-0 SMPL-X with interpolated WiLoR hands into per-clip NPZ. The output format is described in [`FUSED_AIOS_WILOR_NPZ_FORMAT.md`](./FUSED_AIOS_WILOR_NPZ_FORMAT.md).
- [`refine_aios_arms_with_wilor_rtmpose.py`](./refine_aios_arms_with_wilor_rtmpose.py): joint WiLoR + RTMPose arm refinement.
- [`run_refine_wilor_rtmpose.sh`](./run_refine_wilor_rtmpose.sh): multi-GPU launcher for the refinement.
- [`demo.py`](./demo.py): visualization for a single clip.

Although the two extraction launchers retain the `_2gpu` suffix, they support any number of GPUs.

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

### 5.5 Video-file input mode

When each clip is one video file instead of a directory of frames, pass `--video`:

```bash
bash run_youtube_wilor_2gpu.sh --video 0 1
```

Options must precede the positional GPU IDs.

Each mode has its own default input root, so `--input-root` is only needed for a different dataset. The output root is always the input root with `_wilor_out` appended unless `--output-root` overrides it, which keeps each mode's results beside its own input:

| Mode | Default input root | Resulting output root |
|---|---|---|
| `images`, `auto` | `/data/hwu/youtube_dataset/clip_fps24_without_openasl` | `..._without_openasl_wilor_out` |
| `video` | `/data/hwu/youtube_dataset/clip_fps24` | `clip_fps24_wilor_out` |

Both defaults are plain variables at the top of the launcher, `DEFAULT_IMAGE_INPUT_ROOT` and `DEFAULT_VIDEO_INPUT_ROOT`. Another root is passed as usual, and its output follows along:

```bash
bash run_youtube_wilor_2gpu.sh --video \
    --input-root /data/hwu/youtube_dataset/clip_fps24_other 0 1
```

`--input-mode auto` decides between the two layouts by inspecting the input root, so it uses the image default. It fails deliberately if a root holds both clip directories and video files; name the mode explicitly in that case.

#### Frame naming

In video mode there are no input filenames, so each frame name is generated from its decoded index, and that name's stem becomes the PKL name. The defaults are:

```text
--frame-name-template '{index:06d}.jpg'
--frame-index-start   0
```

This matters when the same clips were, or will be, extracted from images as well. If the image-based naming is ffmpeg's one-based `%06d`, match it explicitly:

```bash
bash run_youtube_wilor_2gpu.sh --video --frame-index-start 1 0 1
```

A mismatch does not fail loudly. It shifts every PKL name by one frame, and the problem only surfaces later during fusion.

### 5.6 Live progress and Weights & Biases

By default the launcher backgrounds every worker with `nohup` and sends its output to a log file. Two options change that.

`--foreground` (short form `--fg`) keeps the workers attached to the terminal, writes no log file, and shows a tqdm progress bar per worker. Ctrl-C stops them all.

`--wandb` streams throughput and detection statistics to Weights & Biases. Each shard opens its own run, and the runs are grouped so one extraction's shards appear together.

The two combine:

```bash
bash run_youtube_wilor_2gpu.sh --fg --wandb --video 0 1
```

Log in once before the first W&B run:

```bash
wandb login
```

The progress bar looks like this, pinned to the bottom of the terminal while the per-clip `DONE` lines scroll above it:

```text
youtube shard0:  12%|████▌      | 2698/22460 [04:11<30:42, 10.7clip/s, done=2698, skip=0, fail=0, fps=241.3]
```

The denominator is the clip count assigned to **that shard**, not the whole dataset. Two shards each reaching 100% means the dataset is done.

Both workers draw their bars on the same terminal and overwrite each other. For one clean bar per view, give each terminal a single GPU. Note that each such command is then `--num-shards 1`, so both terminals walk the same clip list; the `.complete` marker keeps the result correct, but the two workers compete for the same clips instead of dividing them.

#### What is sent to W&B

| Event | Keys |
|---|---|
| Split start | `split/assigned_clips`, `split/total_clips` |
| Each finished clip | `clip/fps`, `clip/hands_per_frame`, `progress/mean_fps`, `progress/clips_done`, `gpu/max_memory_gb` |
| Each failed clip | `failure/clip_id`, `failure/reason` |
| Run end (summary) | `clips_processed`, `frames_processed`, `hands_detected`, `elapsed_seconds`, `mean_fps`, `exit_code` |

Default run group: the split name plus the input root name. Default run name: `shard<index>of<count>`. Override with `--wandb-project` and `--wandb-run-group`.

One point per finished clip is a lot over tens of thousands of clips. To thin it out, call the extractor directly with `--wandb-log-every-n-clips 50`; the launcher forwards only `--wandb`, `--wandb-project`, and `--wandb-run-group`.

Backgrounded runs are unaffected by the tqdm change. The bar is disabled automatically when stdout is not a terminal, so log files keep the same plain timestamped lines as before.

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

## 11. SMPL-X Arm Refinement (WiLoR + RTMPose)

This stage is separate from the two above and runs on the OpenASL clip set, not on How2Sign or YouTube. It takes whole-body SMPL-X clips that already carry WiLoR hands and jointly refines the arms so the SMPL-X wrists line up with the WiLoR hands and the RTMPose arm keypoints.

### 11.1 Inputs and outputs

Produced earlier by [`fuse_shared_aios_wilor.py`](./fuse_shared_aios_wilor.py) and the interpolation stage:

```text
wilor_params_interpolated/                           28G
smplx_params_openasl_tianhao_wilor_fused/            13G
/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192
```

Written by this stage:

```text
smplx_params_openasl_tianhao_wilor_rtmpose_refined/  12G
```

The three repository-root directories are real directories, not symlinks, and all of them are in `.gitignore`. The script defaults point at `shared_samples/`, which holds only the small visual-validation subset, so a full run must pass the roots explicitly. The launcher already defaults to the full roots.

Only `smplx_body_pose` changes. Finger pose, camera, shape, root, face, and every non-arm body joint are copied through unchanged.

### 11.2 Launcher

[`run_refine_wilor_rtmpose.sh`](./run_refine_wilor_rtmpose.sh) starts one worker per shard and spreads the shards over the given GPUs.

```bash
bash run_refine_wilor_rtmpose.sh --help
```

Validate every input first; this loads no SMPL-X or MANO model and is fast:

```bash
bash run_refine_wilor_rtmpose.sh --extra "--dry-run"
```

Then run it:

```bash
bash run_refine_wilor_rtmpose.sh --gpus 1,2
```

### 11.3 Choosing GPUs

Three equivalent forms. These are physical card numbers as shown by `nvidia-smi`; each worker receives its own `CUDA_VISIBLE_DEVICES` and therefore passes `--device cuda:0` internally.

```bash
bash run_refine_wilor_rtmpose.sh 2 3          # positional, after all options
bash run_refine_wilor_rtmpose.sh --gpus 2,3   # order-independent
GPUS=2,3 bash run_refine_wilor_rtmpose.sh     # environment variable
```

The default is GPUs 0 and 1. Card numbers need not be contiguous: `--gpus 1,4,7` is fine.

### 11.4 Shard count independent of GPU count

Unlike the extraction launchers, the number of workers here is set separately from the number of cards. Shard `i` runs on `GPU_IDS[i % number_of_gpus]`.

| Command | Result |
|---|---|
| `--gpus 2,3` | 2 workers, one per card |
| `--gpus 3 --shards 4` | 4 workers all on GPU 3 |
| `--gpus 2,3 --shards 6` | GPU 2 takes shards 0, 2, 4; GPU 3 takes 1, 3, 5 |

The optimizer uses a forward-kinematics-only pass and never computes vertices, so GPU load per clip is low and the bottleneck is often NPZ and pickle I/O. If `nvidia-smi` shows both cards idling, more shards than GPUs is the cheaper win. Limit the CPU threads when packing workers onto one card, otherwise each process tries to use every core:

```bash
bash run_refine_wilor_rtmpose.sh --shards 6 --extra "--torch-threads 2" 2 3
```

Each worker loads its own SMPL-X and MANO model, so check free memory before packing many onto one card.

### 11.5 Live progress instead of logs

The refinement script uses tqdm already. `--fg` keeps the workers on the terminal with no log file and no `nohup`; Ctrl-C stops them.

```bash
bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2
```

Several foreground workers overwrite each other's bars. `--only-shard N` starts just one shard of the split, so each terminal gets one clean bar while the two terminals still cover disjoint clips:

```bash
# terminal 1
bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2 --shards 2 --only-shard 0
# terminal 2
bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2 --shards 2 --only-shard 1
```

This differs from splitting the extraction across terminals, where each command is `--num-shards 1` and the workers duplicate each other's clip list.

### 11.6 Resuming

Rerun the same command. Nothing else is needed, and `--overwrite` must not be used for a normal resume.

- Clips whose output NPZ already exists are skipped and counted in the final `skipped=` total.
- Output is written to a temporary file in the destination directory and then atomically renamed, so an interrupted write never leaves a truncated NPZ that a later run would mistake for finished work.
- The clip that was mid-optimization restarts from iteration 1. There is no checkpoint inside a clip, so at most one clip's work is lost.
- Keep `--shards` the same across restarts. A different count re-splits the clip list and renames the log and PID files. The result stays correct, since the skip test only asks whether the output file exists.

Two costs are paid again on every restart. The validation pass runs over the whole shard before any skipping, reloading every fused NPZ, WiLoR NPZ, and RTMPose pickle, including the clips already finished. And `failed_clips.shard<i>of<n>.txt` is opened in append mode, so a clip that keeps failing validation gains one line per restart; pipe through `sort -u` when counting failures.

A `SIGKILL` (`kill -9`, or the OOM killer) skips the temporary-file cleanup and can leave hidden `.<clip_id>.*.tmp.npz` files in the output directory. They do not affect resuming, because clips are discovered from the fused root rather than the output directory. To clear them:

```bash
find smplx_params_openasl_tianhao_wilor_rtmpose_refined -name '.*.tmp.npz' -delete
```

### 11.7 Per-clip failures

By default a clip that fails validation or refinement is skipped, reported on the console, and appended to `<output-root>/failed_clips.shard<i>of<n>.txt` as one `<clip_id>\t<stage>\t<error>` line. This keeps a whole-dataset run from being lost to one truncated or misaligned clip. Pass `--extra "--strict"` to abort on the first failure instead.

## 12. Resuming Interrupted Runs

### 12.1 GPU extraction

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

### 12.2 CPU interpolation

- Only raw clips with `.complete` are processed.
- Interpolated outputs have their own `.complete` markers.
- Completed interpolated clips are skipped.
- An incomplete output clip writes only missing output PKLs.
- Do not use `--overwrite` for a normal resume.

### 12.3 Arm refinement

Resuming for stage 3 works differently, on whole output NPZ files rather than `.complete` markers. See section 11.6.

## 13. Monitoring Progress

### 13.1 Processes and GPUs

```bash
ps -ef | grep extract_how2sign_wilor.py
nvidia-smi
```

### 13.2 How2Sign logs

```bash
tail -f logs/how2sign_wilor/worker_0.log
```

### 13.3 YouTube logs

```bash
tail -f logs/youtube_wilor/worker_0.log
```

Video-mode runs use a separate directory:

```bash
tail -f logs/youtube_wilor_video/worker_0.log
```

Runs started with `--fg` write no log at all; their output is on the terminal.

### 13.4 Raw PKL and completed-clip counts

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

### 13.5 Interpolation progress

```bash
find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params_interpolated \
    -type f -name '*.pkl' | wc -l

find /data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out/wilor_params_interpolated \
    -type f -name '.complete' | wc -l
```

Running `find` over millions of small files may take a long time.

### 13.6 Refined clip count

Refinement writes one NPZ per clip, so counting is cheap:

```bash
ls smplx_params_openasl_tianhao_wilor_rtmpose_refined/*.npz | wc -l
ls smplx_params_openasl_tianhao_wilor_fused/*.npz | wc -l
```

Failures across all shards:

```bash
sort -u smplx_params_openasl_tianhao_wilor_rtmpose_refined/failed_clips.shard*.txt | wc -l
```

## 14. Single-Clip Interpolation Visualization

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

## 15. MANO Parameter Notes

- `hand_pose [45]` contains the final, complete rotations for 15 local MANO joints. It is not a residual with the MANO mean subtracted.
- Do not add the MANO hand mean again during reconstruction.
- The current output is a full 45D pose and must use `use_pca=False`.
- `num_pca_comps=12` does not apply to this 45D output.
- Set `flat_hand_mean=True` when reconstructing from axis-angle values.
- For the most accurate reconstruction, use `hand_pose_rotmat` and `global_orient_rotmat` directly.

## 16. Frequently Asked Questions

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
