#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/student/hwu/Workplace/WiLoR"
PYTHON_BIN="${PYTHON_BIN:-/home/student/hwu/miniconda3/envs/wilor/bin/python}"
REFINER="${REPO_DIR}/refine_aios_arms_with_wilor_rtmpose.py"

# Defaults; every one can be overridden by a flag or by an environment variable.
FUSED_ROOT="${FUSED_ROOT:-${REPO_DIR}/smplx_params_openasl_tianhao_wilor_fused}"
WILOR_ROOT="${WILOR_ROOT:-${REPO_DIR}/wilor_params_interpolated}"
RTMPOSE_ROOT="${RTMPOSE_ROOT:-/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/smplx_params_openasl_tianhao_wilor_rtmpose_refined}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/refine_wilor_rtmpose}"
GPUS="${GPUS:-}"
SHARDS="${SHARDS:-}"
ONLY_SHARD="${ONLY_SHARD:-}"
FOREGROUND="${FOREGROUND:-0}"
EXTRA="${EXTRA:-}"

usage() {
    cat <<'USAGE'
Usage: bash run_refine_wilor_rtmpose.sh [options] [GPU_ID ...]

Starts --shards refinement workers and spreads them round-robin over the given
GPUs, so the shard count is independent of how many cards you have. The clip
list is split with the script's own --num-shards/--shard-index, so the workers
process disjoint clips and can share one output root. Re-running the script
resumes: clips whose output NPZ already exists are skipped (pass
--extra --overwrite to force a redo). Keep --shards the same across restarts,
otherwise the clip list is re-split and the log/PID names change.

Which GPUs to use: either trailing positional GPU_IDs (options must come
first) or --gpus, which may appear anywhere. These are physical card numbers
as shown by nvidia-smi; each worker gets CUDA_VISIBLE_DEVICES set to its own.
Default: 0 1.

Options:
  --gpus "2 3"          GPUs to spread the workers over, space- or
                        comma-separated. Same as the positional form, but
                        order-independent. Trailing GPU_IDs win if both given.
  --shards N            Number of worker processes. Default: one per GPU.
                        More shards than GPUs packs several workers onto each
                        card, which pays off when the GPUs sit idle waiting on
                        NPZ/pickle I/O; add --extra "--torch-threads 2" so the
                        packed workers do not oversubscribe the CPU cores.
  --foreground, --fg    Keep the workers attached to this terminal and print
                        straight to it, with no log file and no nohup. Ctrl-C
                        stops them. Several foreground workers interleave
                        their output; pair with --only-shard for one clean
                        progress bar per terminal.
  --only-shard N        Start just this one shard out of --shards. The split
                        is unchanged, so running --shards 2 --only-shard 0 in
                        one terminal and --only-shard 1 in another covers the
                        same clips as launching both at once.
  --fused-root PATH     Fused AIOS+WiLoR NPZ root.
  --wilor-root PATH     Interpolated WiLoR NPZ root.
  --rtmpose-root PATH   RTMPose pickle root. Ignored with --extra --no-rtmpose.
  --output-root PATH    Shared output root for every shard.
  --log-dir PATH        Where worker logs and PID files go.
  --extra "ARGS"        Extra arguments passed verbatim to every worker,
                        e.g. --extra "--iterations 240 --overwrite".
  -h, --help            Show this message.

Every option also reads an environment variable: GPUS, SHARDS, ONLY_SHARD,
FOREGROUND, FUSED_ROOT, WILOR_ROOT, RTMPOSE_ROOT, OUTPUT_ROOT, LOG_DIR, EXTRA,
PYTHON_BIN.

Examples:
  # one worker per GPU on GPUs 0 and 1 (default)
  bash run_refine_wilor_rtmpose.sh

  # watch it live in this terminal instead of a log
  bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2

  # same split, one clean progress bar per terminal
  bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2 --shards 2 --only-shard 0
  bash run_refine_wilor_rtmpose.sh --fg --gpus 1,2 --shards 2 --only-shard 1

  # GPUs 2 and 3 instead
  bash run_refine_wilor_rtmpose.sh 2 3
  bash run_refine_wilor_rtmpose.sh --gpus 2,3

  # six workers over two GPUs: shards 0,2,4 on GPU 0 and 1,3,5 on GPU 1
  bash run_refine_wilor_rtmpose.sh --shards 6 --extra "--torch-threads 2" 0 1

  # four workers all on GPU 3
  bash run_refine_wilor_rtmpose.sh --gpus 3 --shards 4

  # validate the inputs only, no SMPL-X/MANO loaded
  bash run_refine_wilor_rtmpose.sh --extra "--dry-run" 0 1

  # eight GPUs, WiLoR-only refinement
  bash run_refine_wilor_rtmpose.sh --extra "--no-rtmpose" 0 1 2 3 4 5 6 7
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --fused-root)
            [[ "$#" -ge 2 ]] || { echo "--fused-root needs a value" >&2; exit 2; }
            FUSED_ROOT="$2"
            shift 2
            ;;
        --wilor-root)
            [[ "$#" -ge 2 ]] || { echo "--wilor-root needs a value" >&2; exit 2; }
            WILOR_ROOT="$2"
            shift 2
            ;;
        --rtmpose-root)
            [[ "$#" -ge 2 ]] || { echo "--rtmpose-root needs a value" >&2; exit 2; }
            RTMPOSE_ROOT="$2"
            shift 2
            ;;
        --output-root)
            [[ "$#" -ge 2 ]] || { echo "--output-root needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --log-dir)
            [[ "$#" -ge 2 ]] || { echo "--log-dir needs a value" >&2; exit 2; }
            LOG_DIR="$2"
            shift 2
            ;;
        --gpus)
            [[ "$#" -ge 2 ]] || { echo "--gpus needs a value" >&2; exit 2; }
            GPUS="$2"
            shift 2
            ;;
        --shards)
            [[ "$#" -ge 2 ]] || { echo "--shards needs a value" >&2; exit 2; }
            SHARDS="$2"
            shift 2
            ;;
        --only-shard)
            [[ "$#" -ge 2 ]] || { echo "--only-shard needs a value" >&2; exit 2; }
            ONLY_SHARD="$2"
            shift 2
            ;;
        --foreground|--fg)
            FOREGROUND=1
            shift
            ;;
        --extra)
            [[ "$#" -ge 2 ]] || { echo "--extra needs a value" >&2; exit 2; }
            EXTRA="$2"
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

FUSED_ROOT="${FUSED_ROOT%/}"
WILOR_ROOT="${WILOR_ROOT%/}"
RTMPOSE_ROOT="${RTMPOSE_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT%/}"

for entry in "fused root:${FUSED_ROOT}" "WiLoR root:${WILOR_ROOT}"; do
    if [[ ! -d "${entry#*:}" ]]; then
        echo "${entry%%:*} does not exist: ${entry#*:}" >&2
        exit 1
    fi
done
# The refiner itself rejects a missing RTMPose root unless --no-rtmpose is set,
# so only warn here instead of failing an intentionally WiLoR-only run.
if [[ ! -d "${RTMPOSE_ROOT}" && "${EXTRA}" != *--no-rtmpose* ]]; then
    echo "RTMPose root does not exist: ${RTMPOSE_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
# Foreground workers print to the terminal, so they need no log directory.
[[ "${FOREGROUND}" -eq 1 ]] || mkdir -p "${LOG_DIR}"

if [[ "$#" -gt 0 ]]; then
    GPU_IDS=("$@")
elif [[ -n "${GPUS}" ]]; then
    # Accept "2 3" and the CUDA_VISIBLE_DEVICES-style "2,3" alike.
    read -ra GPU_IDS <<<"${GPUS//,/ }"
else
    GPU_IDS=(0 1)
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "no GPU IDs given" >&2
    exit 2
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

NUM_GPUS="${#GPU_IDS[@]}"
# Shard count is independent of the GPU count: shard i runs on GPU_IDS[i % N],
# so one card can host several workers and spare cards simply stay unused.
NUM_SHARDS="${SHARDS:-${NUM_GPUS}}"
if [[ ! "${NUM_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SHARDS}" -lt 1 ]]; then
    echo "--shards must be a positive integer, got: ${NUM_SHARDS}" >&2
    exit 2
fi
if [[ -n "${ONLY_SHARD}" ]]; then
    if [[ ! "${ONLY_SHARD}" =~ ^[0-9]+$ ]] || [[ "${ONLY_SHARD}" -ge "${NUM_SHARDS}" ]]; then
        echo "--only-shard must be in [0, ${NUM_SHARDS}), got: ${ONLY_SHARD}" >&2
        exit 2
    fi
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA}" ]]; then
    read -ra EXTRA_ARGS <<<"${EXTRA}"
fi

# Everything except --shard-index is the same for every worker.
WORKER_ARGS=(
    --fused-root "${FUSED_ROOT}"
    --wilor-root "${WILOR_ROOT}"
    --rtmpose-root "${RTMPOSE_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --device cuda:0
    --num-shards "${NUM_SHARDS}"
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)

# CUDA_VISIBLE_DEVICES renumbers the selected card to 0, so every worker asks
# for cuda:0 and still lands on its own physical GPU.
start_worker_background() {
    local gpu_id="$1"
    local shard_index="$2"
    local log_path="${LOG_DIR}/shard_${shard_index}of${NUM_SHARDS}.log"
    local pid_path="${LOG_DIR}/shard_${shard_index}of${NUM_SHARDS}.pid"

    if [[ -f "${pid_path}" ]]; then
        local existing_pid
        existing_pid="$(<"${pid_path}")"
        if kill -0 "${existing_pid}" 2>/dev/null; then
            echo "shard ${shard_index} is already running as PID ${existing_pid}"
            return
        fi
    fi

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${REFINER}" \
            "${WORKER_ARGS[@]}" \
            --shard-index "${shard_index}" \
        >"${log_path}" 2>&1 </dev/null &

    local worker_pid="$!"
    echo "${worker_pid}" >"${pid_path}"
    echo "started shard ${shard_index} on physical GPU ${gpu_id}: PID ${worker_pid}"
    echo "log: ${log_path}"
}

# Foreground: the worker keeps this terminal, so its tqdm bars and [done]
# lines render live and nothing is written to a log. Output from several
# foreground workers interleaves on one terminal; run one shard per terminal
# (--shards N on each, different --only-shard) if you want clean bars.
start_worker_foreground() {
    local gpu_id="$1"
    local shard_index="$2"
    echo "shard ${shard_index} -> physical GPU ${gpu_id}"
    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${REFINER}" \
            "${WORKER_ARGS[@]}" \
            --shard-index "${shard_index}" &
    FOREGROUND_PIDS+=("$!")
}

cd "${REPO_DIR}"
echo "fused:  ${FUSED_ROOT}"
echo "wilor:  ${WILOR_ROOT}"
echo "output: ${OUTPUT_ROOT}"
echo "GPUs: ${GPU_IDS[*]} (${NUM_GPUS}) -> ${NUM_SHARDS} shard(s)"
[[ -n "${ONLY_SHARD}" ]] && echo "running shard ${ONLY_SHARD} only"
[[ -n "${EXTRA}" ]] && echo "extra: ${EXTRA}"

FOREGROUND_PIDS=()
for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
    if [[ -n "${ONLY_SHARD}" && "${shard_index}" -ne "${ONLY_SHARD}" ]]; then
        continue
    fi
    if [[ "${FOREGROUND}" -eq 1 ]]; then
        start_worker_foreground "${GPU_IDS[shard_index % NUM_GPUS]}" "${shard_index}"
    else
        start_worker_background "${GPU_IDS[shard_index % NUM_GPUS]}" "${shard_index}"
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

echo
echo "follow: tail -f ${LOG_DIR}/shard_*.log"
echo "stop:   kill \$(cat ${LOG_DIR}/shard_*.pid)"
