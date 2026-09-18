#!/usr/bin/env bash

set -euo pipefail

if (($# < 2)); then
  printf 'Usage: %s <storage-account-name> <image-reference> [tlmtc train options...]\n' "$0" >&2
  exit 2
fi

readonly STORAGE_ACCOUNT_NAME="$1"
readonly HPO_IMAGE="$2"
shift 2

readonly DATA_MOUNT="/mnt/tlmtc"
readonly INPUTS_DIR="${DATA_MOUNT}/inputs"
readonly WORK_DIR="${DATA_MOUNT}/work"
readonly HF_CACHE_DIR="${DATA_MOUNT}/cache/huggingface"
readonly TRAIN_OUTPUTS_DIR="${WORK_DIR}/train_outputs"
readonly STORAGE_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
readonly TRAINING_INPUTS_URL="${STORAGE_URL}/training-inputs"
readonly HPO_WORKSPACES_URL="${STORAGE_URL}/hpo-workspaces"

export AZCOPY_AUTO_LOGIN_TYPE=MSI

log() {
  printf '[run-hpo] %s\n' "$*"
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

  log "Downloading training inputs."
  azcopy copy \
    "${TRAINING_INPUTS_URL}/*" \
    "${INPUTS_DIR}/" \
    --recursive

  if [[ -n "$run_id" ]]; then
    log "Downloading existing HPO workspace ${run_id}."
    install --directory --mode 0755 --owner 10001 --group 10001 "$TRAIN_OUTPUTS_DIR"
    azcopy copy \
      "${HPO_WORKSPACES_URL}/*" \
      "${TRAIN_OUTPUTS_DIR}/" \
      --recursive \
      --include-path "$run_id" \
      --overwrite=ifSourceNewer
    chown -R 10001:10001 "$TRAIN_OUTPUTS_DIR"
  fi

  log "Running ${HPO_IMAGE}."
  docker run \
    --rm \
    --gpus device=0 \
    --mount "type=bind,source=${INPUTS_DIR},target=/inputs,readonly" \
    --mount "type=bind,source=${WORK_DIR},target=/workspace" \
    --mount "type=bind,source=${HF_CACHE_DIR},target=/home/tlmtc/.cache/huggingface" \
    "$HPO_IMAGE" \
    "$@" \
    --hyperparameter-tuning \
    --no-transfer-learning \
    --work-dir /workspace \
    --no-use-cpu

  log "Uploading the HPO workspace."
  azcopy copy \
    "${TRAIN_OUTPUTS_DIR}/*" \
    "$HPO_WORKSPACES_URL" \
    --recursive \
    --overwrite=ifSourceNewer
}

main "$@"
