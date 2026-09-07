# Run AIOS + WiLoR Fusion and Arm Refinement

This pipeline always uses AIOS `person_0`. It first generates stabilized fused
files, then jointly uses WiLoR hands and RTMPose arms to refine the SMPL-X
upper body.

## Inputs

```text
shared_samples/aios_smplx_params
shared_samples/wilor_params_interpolated
/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192
```

Run every command from:

```bash
cd /home/student/hwu/Workplace/WiLoR
```

## 1. Run fusion and stabilization

This stage selects AIOS `person_0`, smooths the AIOS motion, preserves the
original WiLoR finger detail, replaces the SMPL-X hands with WiLoR hands, and
fixes the clipwise camera.

```bash
/home/student/hwu/miniconda3/envs/soke/bin/python \
  fuse_shared_aios_wilor.py \
  --aios-root shared_samples/aios_smplx_params \
  --wilor-root shared_samples/wilor_params_interpolated \
  --output-root shared_samples/aios_smoothed_wilor_hands_fused_final
```

Stage-1 output:

```text
shared_samples/aios_smoothed_wilor_hands_fused_final/<clip_id>.npz
```

## 2. Run WiLoR + RTMPose arm refinement

Run this stage only after stage 1 finishes. Change `CUDA_VISIBLE_DEVICES=0` to
the physical GPU you want to use. RTMPose is enabled by default; do not pass
`--no-rtmpose` for the standard pipeline.

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/student/hwu/miniconda3/envs/soke/bin/python \
  refine_aios_arms_with_wilor_rtmpose.py \
  --fused-root shared_samples/aios_smoothed_wilor_hands_fused_final \
  --wilor-root shared_samples/wilor_params_interpolated \
  --rtmpose-root \
    /home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192 \
  --output-root shared_samples/aios_wilor_rtmpose_arm_refined \
  --device cuda:0
```

Final output:

```text
shared_samples/aios_wilor_rtmpose_arm_refined/<clip_id>.npz
```

Use this final directory for downstream processing:

```text
/home/student/hwu/Workplace/WiLoR/shared_samples/aios_wilor_rtmpose_arm_refined
```

The paths shown in the command are already the script defaults, so the
equivalent short command is:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/student/hwu/miniconda3/envs/soke/bin/python \
  refine_aios_arms_with_wilor_rtmpose.py \
  --device cuda:0
```

## Test one clip first

Run the same clip through both stages:

```bash
CLIP_ID='00ADU7t7IWI-00:00:01.106-00:00:08.759'

/home/student/hwu/miniconda3/envs/soke/bin/python \
  fuse_shared_aios_wilor.py \
  --aios-root shared_samples/aios_smplx_params \
  --wilor-root shared_samples/wilor_params_interpolated \
  --output-root shared_samples/aios_smoothed_wilor_hands_fused_final \
  --clip-id "$CLIP_ID"

CUDA_VISIBLE_DEVICES=0 \
/home/student/hwu/miniconda3/envs/soke/bin/python \
  refine_aios_arms_with_wilor_rtmpose.py \
  --clip-id "$CLIP_ID" \
  --device cuda:0
```

## Visualize a final refined clip

```bash
cd /home/student/hwu/Workplace/SOKE

CUDA_VISIBLE_DEVICES=0 \
/home/student/hwu/miniconda3/envs/soke/bin/python \
  visualize_fused_ytb_npz.py \
  --fused-root \
    /home/student/hwu/Workplace/WiLoR/shared_samples/aios_wilor_rtmpose_arm_refined \
  --clip=00ADU7t7IWI-00:00:01.106-00:00:08.759 \
  --output-root \
    /home/student/hwu/Workplace/WiLoR/shared_samples/demo_arm_refined \
  --device cuda:0 \
  --overwrite
```

## Resume and overwrite behavior

- Both processing scripts skip output NPZ files that already exist.
- After an interruption, rerun the same command to continue.
- Add `--overwrite` only when you intentionally want to regenerate existing
  files.
- For arm refinement, use the exact intermediate output directory created by
  the fusion command above.

## Check output counts

```bash
find shared_samples/aios_smoothed_wilor_hands_fused_final \
  -maxdepth 1 -type f -name '*.npz' | wc -l

find shared_samples/aios_wilor_rtmpose_arm_refined \
  -maxdepth 1 -type f -name '*.npz' | wc -l
```

For the current shared sample set, both completed stages should contain 1,000
NPZ files.

## Refinement behavior

`refine_aios_arms_with_wilor_rtmpose.py` is the standard arm-refinement entry
point. WiLoR supervises wrist position and palm orientation, while RTMPose
shoulders and elbows supervise the arm. On frames supervised by WiLoR, the
RTMPose COCO wrist target is disabled by default because it represents a
different anatomical point from the MANO wrist. RTMPose acts as the wrist
fallback when WiLoR supervision is unavailable. A 3-D torso-volume loss also
discourages arm penetration.

Different RTMPose and fused frame counts are aligned through the time interval
encoded in the clip id. A small global offset is accepted only when it improves
the initial robust reprojection score. Low-confidence RTMPose targets are
masked.

Pass `--no-rtmpose` only when WiLoR-only refinement is intentionally required.

## Validate inputs only

Validate all 1,000 shared clips without loading SMPL-X or writing outputs:

```bash
/home/student/hwu/miniconda3/envs/soke/bin/python \
  refine_aios_arms_with_wilor_rtmpose.py \
  --dry-run
```
