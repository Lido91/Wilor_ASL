#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/student/hwu/Workplace/WiLoR"
PYTHON_BIN="/home/student/hwu/miniconda3/envs/wilor/bin/python"
EXTRACTOR="${REPO_DIR}/extract_how2sign_wilor.py"
INPUT_ROOT="/data/hwu/youtube_dataset/clip_fps24_img_0"
OUTPUT_ROOT="/data/hwu/youtube_dataset/clip_fps24_img_0_wilor_out"
LOG_DIR="${REPO_DIR}/logs/youtube_wilor"

if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "input root does not exist: ${INPUT_ROOT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash $0 [GPU_ID ...]"
    echo "Examples:"
    echo "  bash $0 2        # one worker on physical GPU 2"
    echo "  bash $0 0 1      # two workers on GPUs 0 and 1"
    echo "  bash $0 0 2 3    # three workers on GPUs 0, 2, and 3"
    exit 0
fi

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

start_worker() {
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

    mkdir -p \
        "/tmp/youtube_wilor_matplotlib_${shard_index}" \
        "/tmp/youtube_wilor_ultralytics_${shard_index}"

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        MPLCONFIGDIR="/tmp/youtube_wilor_matplotlib_${shard_index}" \
        YOLO_CONFIG_DIR="/tmp/youtube_wilor_ultralytics_${shard_index}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${EXTRACTOR}" \
            --input-root "${INPUT_ROOT}" \
            --output-root "${OUTPUT_ROOT}" \
            --split-name youtube \
            --device 0 \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${shard_index}" \
            --detector-batch-size 32 \
            --wilor-batch-size 64 \
            --fast \
        >"${log_path}" 2>&1 </dev/null &

    local worker_pid="$!"
    echo "${worker_pid}" >"${pid_path}"
    echo "started worker ${shard_index} on physical GPU ${gpu_id}: PID ${worker_pid}"
    echo "log: ${log_path}"
}

cd "${REPO_DIR}"
echo "GPUs: ${GPU_IDS[*]} (${NUM_SHARDS} shard(s))"
for shard_index in "${!GPU_IDS[@]}"; do
    start_worker "${GPU_IDS[${shard_index}]}" "${shard_index}"
done

echo "output: ${OUTPUT_ROOT}/wilor_params"
