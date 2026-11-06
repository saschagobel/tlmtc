#!/usr/bin/env bash

set -euo pipefail

if (($# < 2)); then
  printf 'Usage: %s <storage-account-name> <image-reference> [tlmtc predict options...]\n' "$0" >&2
  exit 2
fi

readonly STORAGE_ACCOUNT_NAME="$1"
readonly PREDICT_IMAGE="$2"
shift 2

readonly DATA_MOUNT="/mnt/tlmtc"
readonly INPUTS_DIR="${DATA_MOUNT}/inputs"
readonly WORK_DIR="${DATA_MOUNT}/work"
readonly TRAIN_OUTPUTS_DIR="${WORK_DIR}/train_outputs"
readonly PREDICTION_OUTPUTS_DIR="${WORK_DIR}/prediction_outputs"
readonly STORAGE_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
readonly TRAINING_OUTPUTS_URL="${STORAGE_URL}/training-outputs"
readonly PREDICTION_INPUTS_URL="${STORAGE_URL}/prediction-inputs"
readonly PREDICTION_OUTPUTS_URL="${STORAGE_URL}/prediction-outputs"

export AZCOPY_AUTO_LOGIN_TYPE=MSI

log() {
  printf '[run-predict] %s\n' "$*"
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

  if [[ -z "$run_id" ]]; then
    printf '[run-predict] ERROR: an explicit --run-id is required.\n' >&2
    exit 2
  fi

  log "Downloading prediction inputs."
  azcopy copy \
    "${PREDICTION_INPUTS_URL}/*" \
    "${INPUTS_DIR}/" \
    --recursive

  log "Downloading training run ${run_id}."
  install --directory --mode 0755 --owner 10001 --group 10001 "$TRAIN_OUTPUTS_DIR"
  azcopy copy \
    "${TRAINING_OUTPUTS_URL}/*" \
    "${TRAIN_OUTPUTS_DIR}/" \
    --recursive \
    --include-path "$run_id" \
    --overwrite=ifSourceNewer
  chown -R 10001:10001 "$TRAIN_OUTPUTS_DIR"

  log "Running ${PREDICT_IMAGE} for run ${run_id}."
  docker run \
    --rm \
    --mount "type=bind,source=${INPUTS_DIR},target=/inputs,readonly" \
    --mount "type=bind,source=${WORK_DIR},target=/workspace" \
    "$PREDICT_IMAGE" \
    "$@" \
    --inference-backend onnx \
    --use-cpu \
    --work-dir /workspace

  log "Uploading prediction run ${run_id}."
  azcopy copy \
    "${PREDICTION_OUTPUTS_DIR}/${run_id}" \
    "$PREDICTION_OUTPUTS_URL" \
    --recursive \
    --overwrite=ifSourceNewer
}

main "$@"
