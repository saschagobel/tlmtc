#!/usr/bin/env bash

set -Eeuo pipefail

readonly DATA_MOUNT="/mnt/tlmtc"
readonly DATA_DISK_LUN="0"
readonly NVIDIA_DRIVER_BRANCH="580"
readonly DOCKER_VERSION="5:29.6.1-1~ubuntu.24.04~noble"
readonly NVIDIA_CONTAINER_TOOLKIT_VERSION="1.19.1-1"
readonly AZCOPY_VERSION="10.32.6"
readonly AZCOPY_SHA256="2ec557656be1976754e97de828c0662f7df0e73ef6fc98fc52a518452a6adfbc"

readonly TEMP_DIR="$(mktemp --directory)"
trap 'rm -rf -- "${TEMP_DIR:?}"' EXIT

log() {
  printf '[bootstrap-gpu-host] %s\n' "$*"
}

fail() {
  printf '[bootstrap-gpu-host] ERROR: %s\n' "$*" >&2
  exit 1
}

apt_install() {
  DEBIAN_FRONTEND=noninteractive apt-get install --yes "$@"
}

find_data_device() {
  local device_link

  udevadm settle
  for device_link in \
    "/dev/disk/azure/data/by-lun/${DATA_DISK_LUN}" \
    "/dev/disk/azure/scsi1/lun${DATA_DISK_LUN}"; do
    if [[ -b "$device_link" ]]; then
      readlink --canonicalize -- "$device_link"
      return
    fi
  done

  fail "Azure data disk at LUN ${DATA_DISK_LUN} was not found."
}

prepare_data_disk() {
  local data_device data_partition data_uuid

  data_device="$(find_data_device)"
  if wipefs --no-act --noheadings "$data_device" | grep --quiet .; then
    fail "Azure data disk ${data_device} is not blank."
  fi

  log "Partitioning Azure data disk ${data_device}."
  parted --script "$data_device" \
    mklabel gpt \
    mkpart primary ext4 0% 100%
  partprobe "$data_device"
  udevadm settle

  data_partition="$(
    lsblk --noheadings --paths --raw --output NAME,TYPE "$data_device" \
      | awk '$2 == "part" { print $1; exit }'
  )"

  mkfs.ext4 -L tlmtc-data "$data_partition"
  data_uuid="$(blkid --match-tag UUID --output value "$data_partition")"
  install --directory --mode 0755 "$DATA_MOUNT"

  printf 'UUID=%s %s ext4 defaults,nofail 0 2\n' "$data_uuid" "$DATA_MOUNT" \
    >>/etc/fstab

  mount "$DATA_MOUNT"

  install --directory --mode 0755 --owner root --group root \
    "$DATA_MOUNT/inputs"
  install --directory --mode 0755 --owner 10001 --group 10001 \
    "$DATA_MOUNT/work" \
    "$DATA_MOUNT/cache/huggingface"
  install --directory --mode 0711 --owner root --group root \
    "$DATA_MOUNT/docker"
}

configure_package_repositories() {
  curl --fail --location \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    --output "$TEMP_DIR/cuda-keyring.deb"
  dpkg --install "$TEMP_DIR/cuda-keyring.deb"

  install --directory --mode 0755 /etc/apt/keyrings
  curl --fail --silent --show-error --location \
    https://download.docker.com/linux/ubuntu/gpg \
    --output /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  cat >/etc/apt/sources.list.d/docker.sources <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF

  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/gpgkey \
    --output "$TEMP_DIR/nvidia-container-toolkit-key"
  gpg --dearmor --yes \
    --output /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    "$TEMP_DIR/nvidia-container-toolkit-key"

  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    --output "$TEMP_DIR/nvidia-container-toolkit.list"
  sed \
    's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    "$TEMP_DIR/nvidia-container-toolkit.list" \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
}

install_host_packages() {
  apt-get update
  apt_install "nvidia-driver-pinning-${NVIDIA_DRIVER_BRANCH}"
  apt_install \
    cuda-drivers \
    "docker-ce=${DOCKER_VERSION}" \
    "docker-ce-cli=${DOCKER_VERSION}" \
    "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
}

configure_docker() {
  systemctl stop docker docker.socket
  containerd config dump \
    | sed "s|^root = .*|root = \"${DATA_MOUNT}/containerd\"|" \
      >"$TEMP_DIR/containerd.toml"
  install --mode 0644 "$TEMP_DIR/containerd.toml" /etc/containerd/config.toml
  systemctl restart containerd

  install --directory --mode 0755 /etc/docker
  cat >/etc/docker/daemon.json <<EOF
{
  "data-root": "${DATA_MOUNT}/docker"
}
EOF

  nvidia-ctk runtime configure --runtime=docker
  systemctl enable docker
  systemctl restart docker
}

install_azcopy() {
  local package="$TEMP_DIR/azcopy.deb"

  curl --fail --location \
    "https://github.com/Azure/azure-storage-azcopy/releases/download/v${AZCOPY_VERSION}/azcopy-${AZCOPY_VERSION}.x86_64.deb" \
    --output "$package"
  printf '%s  %s\n' "$AZCOPY_SHA256" "$package" | sha256sum --check
  apt_install "$package"
}

main() {
  apt-get update
  apt_install ca-certificates curl e2fsprogs gnupg parted

  prepare_data_disk
  configure_package_repositories
  install_host_packages
  configure_docker
  install_azcopy

  log "Rebooting in one minute to activate the NVIDIA driver."
  shutdown --reboot +1 "Activating the NVIDIA driver"
}

main
