#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/student/hwu/Workplace/WiLoR"
PYTHON_BIN="/home/student/hwu/miniconda3/envs/wilor/bin/python"
EXTRACTOR="${REPO_DIR}/extract_how2sign_wilor.py"

# Defaults; every one can be overridden by a flag or by an environment variable.
# One default input root per mode, because the two layouts live in different
# directories. The output root is always derived from whichever one is used,
# so picking a mode also picks where its results land.
DEFAULT_IMAGE_INPUT_ROOT="/data/hwu/youtube_dataset/clip_fps24_without_openasl"
DEFAULT_VIDEO_INPUT_ROOT="/data/hwu/youtube_dataset/clip_fps24"
INPUT_MODE="${INPUT_MODE:-auto}"
INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
SPLIT_NAME="${SPLIT_NAME:-youtube}"
FRAME_NAME_TEMPLATE="${FRAME_NAME_TEMPLATE:-}"
FRAME_INDEX_START="${FRAME_INDEX_START:-}"
DETECTOR_BATCH_SIZE="${DETECTOR_BATCH_SIZE:-32}"
WILOR_BATCH_SIZE="${WILOR_BATCH_SIZE:-64}"
FOREGROUND="${FOREGROUND:-0}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"

usage() {
    cat <<'USAGE'
Usage: bash run_youtube_wilor_2gpu.sh [options] [--] [GPU_ID ...]

Options:
  --input-mode {auto|images|video}
                        How clips are stored: one directory of frames per clip
                        (images), one video file per clip (video), or detect it
                        from the input root (auto, default).
  --video               Shorthand for --input-mode video.
  --input-root PATH     Root holding the clip directories or the clip videos.
                        Defaults per mode, see "Default roots" below.
  --output-root PATH    Output root. Default: <input-root>_wilor_out.
  --frame-name-template STR
                        Video mode only. Frame name built from the decoded
                        frame index, e.g. 'frame_{index:06d}.jpg'. Its stem
                        becomes the pickle name, so match this to an existing
                        image-based extraction. Default: '{index:06d}.jpg'.
  --frame-index-start N Video mode only. Index of the first decoded frame; use
                        1 for ffmpeg-style one-based numbering. Default: 0.
  --split-name NAME     Metadata/sharding name. Default: youtube.
  --detector-batch-size N   Frames per YOLO batch. Default: 32.
  --wilor-batch-size N      Hand crops per WiLoR batch. Default: 64.
  --log-dir PATH        Where worker logs and PID files go.
  --foreground, --fg    Keep the workers attached to this terminal and print
                        straight to it, with no log file and no nohup. Ctrl-C
                        stops them all. Several workers interleave their
                        output; use one GPU per terminal for a clean view.
  --wandb               Track every worker in Weights & Biases. Each shard
                        opens its own run, grouped so one extraction's shards
                        appear together.
  --wandb-project NAME  W&B project. Default: the extractor's own default.
  --wandb-run-group STR W&B group shared by this extraction's shards.
                        Default: split name plus input root name.
  -h, --help            Show this message.

Every option also reads an environment variable: INPUT_MODE, INPUT_ROOT,
OUTPUT_ROOT, FRAME_NAME_TEMPLATE, FRAME_INDEX_START, SPLIT_NAME,
DETECTOR_BATCH_SIZE, WILOR_BATCH_SIZE, LOG_DIR, FOREGROUND, WANDB,
WANDB_PROJECT, WANDB_RUN_GROUP.

Default roots, used when --input-root is not given. The output root follows
from the input root, so each mode writes beside its own input:

  images / auto  in:  /data/hwu/youtube_dataset/clip_fps24_without_openasl
                 out: /data/hwu/youtube_dataset/clip_fps24_without_openasl_wilor_out
  video          in:  /data/hwu/youtube_dataset/clip_fps24
                 out: /data/hwu/youtube_dataset/clip_fps24_wilor_out

Examples:
  # frame directories (unchanged default behaviour), GPUs 0 and 1
  bash run_youtube_wilor_2gpu.sh 0 1

  # one mp4 per clip, default video root, GPUs 0 and 1
  bash run_youtube_wilor_2gpu.sh --video 0 1

  # watch it live in this terminal instead of a log
  bash run_youtube_wilor_2gpu.sh --fg --video 0 1

  # local tqdm bar plus W&B tracking for both shards
  bash run_youtube_wilor_2gpu.sh --fg --wandb --video 0 1

  # a different video root, output derived from it
  bash run_youtube_wilor_2gpu.sh --video \
      --input-root /data/hwu/youtube_dataset/clip_fps24_other 0 1

  # four workers
  bash run_youtube_wilor_2gpu.sh --video 0 1 2 3

  # mp4 clips whose frame names must match ffmpeg's 1-based %06d output
  bash run_youtube_wilor_2gpu.sh --video --frame-index-start 1 2
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --input-mode)
            [[ "$#" -ge 2 ]] || { echo "--input-mode needs a value" >&2; exit 2; }
            INPUT_MODE="$2"
            shift 2
            ;;
        --video)
            INPUT_MODE="video"
            shift
            ;;
        --images)
            INPUT_MODE="images"
            shift
            ;;
        --input-root)
            [[ "$#" -ge 2 ]] || { echo "--input-root needs a value" >&2; exit 2; }
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output-root)
            [[ "$#" -ge 2 ]] || { echo "--output-root needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --frame-name-template)
            [[ "$#" -ge 2 ]] || { echo "--frame-name-template needs a value" >&2; exit 2; }
            FRAME_NAME_TEMPLATE="$2"
            shift 2
            ;;
        --frame-index-start)
            [[ "$#" -ge 2 ]] || { echo "--frame-index-start needs a value" >&2; exit 2; }
            FRAME_INDEX_START="$2"
            shift 2
            ;;
        --split-name)
            [[ "$#" -ge 2 ]] || { echo "--split-name needs a value" >&2; exit 2; }
            SPLIT_NAME="$2"
            shift 2
            ;;
        --detector-batch-size)
            [[ "$#" -ge 2 ]] || { echo "--detector-batch-size needs a value" >&2; exit 2; }
            DETECTOR_BATCH_SIZE="$2"
            shift 2
            ;;
        --wilor-batch-size)
            [[ "$#" -ge 2 ]] || { echo "--wilor-batch-size needs a value" >&2; exit 2; }
            WILOR_BATCH_SIZE="$2"
            shift 2
            ;;
        --log-dir)
            [[ "$#" -ge 2 ]] || { echo "--log-dir needs a value" >&2; exit 2; }
            LOG_DIR="$2"
            shift 2
            ;;
        --foreground|--fg)
            FOREGROUND=1
            shift
            ;;
        --wandb)
            WANDB=1
            shift
            ;;
        --wandb-project)
            [[ "$#" -ge 2 ]] || { echo "--wandb-project needs a value" >&2; exit 2; }
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run-group)
            [[ "$#" -ge 2 ]] || { echo "--wandb-run-group needs a value" >&2; exit 2; }
            WANDB_RUN_GROUP="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

case "${INPUT_MODE}" in
    auto|images|video) ;;
    *)
        echo "invalid --input-mode: ${INPUT_MODE} (auto|images|video)" >&2
        exit 2
        ;;
esac

if [[ -z "${INPUT_ROOT}" ]]; then
    # "auto" resolves the layout by inspecting the root, so it needs a root to
    # inspect; the image default is the one that has always been assumed there.
    if [[ "${INPUT_MODE}" == "video" ]]; then
        INPUT_ROOT="${DEFAULT_VIDEO_INPUT_ROOT}"
    else
        INPUT_ROOT="${DEFAULT_IMAGE_INPUT_ROOT}"
    fi
fi

INPUT_ROOT="${INPUT_ROOT%/}"
if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "input root does not exist: ${INPUT_ROOT}" >&2
    exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${INPUT_ROOT}_wilor_out"
fi
OUTPUT_ROOT="${OUTPUT_ROOT%/}"

if [[ -z "${LOG_DIR}" ]]; then
    if [[ "${INPUT_MODE}" == "video" ]]; then
        LOG_DIR="${REPO_DIR}/logs/youtube_wilor_video"
    else
        LOG_DIR="${REPO_DIR}/logs/youtube_wilor"
    fi
fi

# Foreground workers print to the terminal, so they need no log directory.
[[ "${FOREGROUND}" -eq 1 ]] || mkdir -p "${LOG_DIR}"

if [[ "$#" -eq 0 ]]; then
    GPU_IDS=(0 1)
else
    GPU_IDS=("$@")
fi

declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "invalid GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
        echo "duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPU_IDS["${gpu_id}"]=1
done

NUM_SHARDS="${#GPU_IDS[@]}"

EXTRA_ARGS=()
if [[ -n "${FRAME_NAME_TEMPLATE}" ]]; then
    EXTRA_ARGS+=(--frame-name-template "${FRAME_NAME_TEMPLATE}")
fi
if [[ -n "${FRAME_INDEX_START}" ]]; then
    if [[ ! "${FRAME_INDEX_START}" =~ ^[0-9]+$ ]]; then
        echo "invalid --frame-index-start: ${FRAME_INDEX_START}" >&2
        exit 2
    fi
    EXTRA_ARGS+=(--frame-index-start "${FRAME_INDEX_START}")
fi
if [[ "${WANDB}" -eq 1 ]]; then
    EXTRA_ARGS+=(--wandb)
    if [[ -n "${WANDB_PROJECT}" ]]; then
        EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}")
    fi
    # Without an explicit group each worker derives its own from the split and
    # input-root names, which already matches across shards; an explicit group
    # additionally keeps separate reruns of the same data apart.
    if [[ -n "${WANDB_RUN_GROUP}" ]]; then
        EXTRA_ARGS+=(--wandb-run-group "${WANDB_RUN_GROUP}")
    fi
fi

# Everything except --shard-index is the same for every worker.
WORKER_ARGS=(
    --input-root "${INPUT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --input-mode "${INPUT_MODE}"
    --split-name "${SPLIT_NAME}"
    --device 0
    --num-shards "${NUM_SHARDS}"
    --detector-batch-size "${DETECTOR_BATCH_SIZE}"
    --wilor-batch-size "${WILOR_BATCH_SIZE}"
    --fast
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)

# matplotlib and ultralytics both write into their config directory at import
# time, so each worker gets its own to avoid a race between the shards.
worker_config_dirs() {
    local shard_index="$1"
    mkdir -p \
        "/tmp/youtube_wilor_matplotlib_${shard_index}" \
        "/tmp/youtube_wilor_ultralytics_${shard_index}"
}

start_worker_background() {
    local gpu_id="$1"
    local shard_index="$2"
    local log_path="${LOG_DIR}/worker_${shard_index}.log"
    local pid_path="${LOG_DIR}/worker_${shard_index}.pid"

    if [[ -f "${pid_path}" ]]; then
        local existing_pid
        existing_pid="$(<"${pid_path}")"
        if kill -0 "${existing_pid}" 2>/dev/null; then
            echo "worker ${shard_index} is already running as PID ${existing_pid}"
            return
        fi
    fi

    worker_config_dirs "${shard_index}"

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        MPLCONFIGDIR="/tmp/youtube_wilor_matplotlib_${shard_index}" \
        YOLO_CONFIG_DIR="/tmp/youtube_wilor_ultralytics_${shard_index}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${EXTRACTOR}" \
            "${WORKER_ARGS[@]}" \
            --shard-index "${shard_index}" \
        >"${log_path}" 2>&1 </dev/null &

    local worker_pid="$!"
    echo "${worker_pid}" >"${pid_path}"
    echo "started worker ${shard_index} on physical GPU ${gpu_id}: PID ${worker_pid}"
    echo "log: ${log_path}"
}

start_worker_foreground() {
    local gpu_id="$1"
    local shard_index="$2"
    worker_config_dirs "${shard_index}"
    echo "worker ${shard_index} -> physical GPU ${gpu_id}"
    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        MPLCONFIGDIR="/tmp/youtube_wilor_matplotlib_${shard_index}" \
        YOLO_CONFIG_DIR="/tmp/youtube_wilor_ultralytics_${shard_index}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${EXTRACTOR}" \
            "${WORKER_ARGS[@]}" \
            --shard-index "${shard_index}" &
    FOREGROUND_PIDS+=("$!")
}

cd "${REPO_DIR}"
echo "mode: ${INPUT_MODE}"
echo "input: ${INPUT_ROOT}"
echo "GPUs: ${GPU_IDS[*]} (${NUM_SHARDS} shard(s))"
echo "output: ${OUTPUT_ROOT}/wilor_params"

FOREGROUND_PIDS=()
for shard_index in "${!GPU_IDS[@]}"; do
    if [[ "${FOREGROUND}" -eq 1 ]]; then
        start_worker_foreground "${GPU_IDS[${shard_index}]}" "${shard_index}"
    else
        start_worker_background "${GPU_IDS[${shard_index}]}" "${shard_index}"
    fi
done

if [[ "${FOREGROUND}" -eq 1 ]]; then
    # Ctrl-C reaches the workers directly: a non-interactive shell puts them in
    # this script's process group, so the terminal signals all of them at once.
    # The trap only covers a SIGTERM sent to the script itself.
    trap 'kill "${FOREGROUND_PIDS[@]}" 2>/dev/null || true' INT TERM
    echo
    status=0
    wait || status="$?"
    exit "${status}"
fi
