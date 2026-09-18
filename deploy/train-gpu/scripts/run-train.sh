#!/usr/bin/env bash

set -euo pipefail

if (($# < 2)); then
  printf 'Usage: %s <storage-account-name> <image-reference> [tlmtc train options...]\n' "$0" >&2
  exit 2
fi

readonly STORAGE_ACCOUNT_NAME="$1"
readonly TRAIN_IMAGE="$2"
shift 2

readonly DATA_MOUNT="/mnt/tlmtc"
readonly INPUTS_DIR="${DATA_MOUNT}/inputs"
readonly WORK_DIR="${DATA_MOUNT}/work"
readonly HF_CACHE_DIR="${DATA_MOUNT}/cache/huggingface"
readonly TRAIN_OUTPUTS_DIR="${WORK_DIR}/train_outputs"
readonly STORAGE_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
readonly TRAINING_INPUTS_URL="${STORAGE_URL}/training-inputs"
readonly HPO_WORKSPACES_URL="${STORAGE_URL}/hpo-workspaces"
readonly TRAINING_OUTPUTS_URL="${STORAGE_URL}/training-outputs"

export AZCOPY_AUTO_LOGIN_TYPE=MSI

log() {
  printf '[run-train] %s\n' "$*"
}

list_run_ids() {
  find "$TRAIN_OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

main() {
  local option read_run_id=false run_id=""

  for option in "$@"; do
    if [[ "$read_run_id" == true ]]; then
      run_id="$option"
      break
    fi

    case "$option" in
      --run-id)
        read_run_id=true
        ;;
      --run-id=*)
        run_id="${option#*=}"
        break
        ;;
    esac
  done

  local gpu_count
  gpu_count="$(nvidia-smi --list-gpus | wc -l)"

  log "Downloading training inputs."
  azcopy copy \
    "${TRAINING_INPUTS_URL}/*" \
    "${INPUTS_DIR}/" \
    --recursive

  install --directory --mode 0755 --owner 10001 --group 10001 "$TRAIN_OUTPUTS_DIR"
  if [[ -n "$run_id" ]]; then
    log "Downloading existing HPO workspace ${run_id}."
    azcopy copy \
      "${HPO_WORKSPACES_URL}/*" \
      "${TRAIN_OUTPUTS_DIR}/" \
      --recursive \
      --include-path "$run_id" \
      --overwrite=ifSourceNewer
    chown -R 10001:10001 "$TRAIN_OUTPUTS_DIR"
  fi

  local existing_runs
  existing_runs="$(list_run_ids)"

  log "Running ${TRAIN_IMAGE} with torchrun on ${gpu_count} GPU(s)."
  docker run \
    --rm \
    --gpus all \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --mount "type=bind,source=${INPUTS_DIR},target=/inputs,readonly" \
    --mount "type=bind,source=${WORK_DIR},target=/workspace" \
    --mount "type=bind,source=${HF_CACHE_DIR},target=/home/tlmtc/.cache/huggingface" \
    --entrypoint torchrun \
    "$TRAIN_IMAGE" \
    --standalone \
    "--nproc-per-node=${gpu_count}" \
    -m tlmtc train \
    "$@" \
    --no-hyperparameter-tuning \
    --transfer-learning \
    --export-onnx \
    --work-dir /workspace \
    --no-use-cpu

  if [[ -z "$run_id" ]]; then
    local -a new_runs=()
    mapfile -t new_runs < <(comm -13 <(printf '%s\n' "$existing_runs") <(list_run_ids))
    run_id="${new_runs[0]}"
  fi

  log "Uploading the training run ${run_id}."
  azcopy copy \
    "${TRAIN_OUTPUTS_DIR}/${run_id}" \
    "$TRAINING_OUTPUTS_URL" \
    --recursive \
    --overwrite=ifSourceNewer
}

main "$@"
