#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
SPECFORGE="${ROOT_DIR}/.venv/bin/specforge"
MODEL=/hzeng/models/Qwen/Qwen3-8B
INPUT=cache/dataset/nemotron-v2-non-thinking-codealpaca_train.jsonl
OUTPUT=cache/dataset/qwen3-8b-non-thinking-nemotron-v2-codealpaca-regen.jsonl
CONFIG=examples/configs/online/disaggregated/managed-local/qwen3-8b-dflash-non-thinking-target-regen-6epoch-dp3.yaml

server_pids=()
stop_servers() {
    for pid in "${server_pids[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    wait "${server_pids[@]}" 2>/dev/null || true
}
trap stop_servers EXIT INT TERM

for gpu_port in "4 31004" "5 31005" "6 31006" "7 31007"; do
    read -r gpu port <<< "${gpu_port}"
    CUDA_VISIBLE_DEVICES="${gpu}" setsid "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL}" \
        --tp-size 1 \
        --dtype bfloat16 \
        --context-length 40960 \
        --mem-fraction-static 0.85 \
        --cuda-graph-max-bs 128 \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port "${port}" \
        > "outputs/qwen3-8b-target-regen-gpu${gpu}.log" 2>&1 &
    server_pids+=("$!")
done

for port in 31004 31005 31006 31007; do
    for _ in $(seq 1 120); do
        curl -fsS "http://127.0.0.1:${port}/health" >/dev/null && break
        sleep 5
    done
    curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
done

"${PYTHON}" scripts/regenerate_train_data.py \
    --model "${MODEL}" \
    --reasoning disable \
    --temperature 0 \
    --top-p 0.95 \
    --top-k 20 \
    --concurrency 64 \
    --max-tokens 32768 \
    --server-address 127.0.0.1:31004 127.0.0.1:31005 127.0.0.1:31006 127.0.0.1:31007 \
    --input-file-path "${INPUT}" \
    --output-file-path "${OUTPUT}" \
    --resume

stop_servers
server_pids=()
trap - EXIT INT TERM

"${SPECFORGE}" train -c "${CONFIG}"
