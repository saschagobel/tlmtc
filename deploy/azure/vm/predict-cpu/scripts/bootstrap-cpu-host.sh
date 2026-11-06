#!/usr/bin/env bash

set -Eeuo pipefail

readonly DATA_MOUNT="/mnt/tlmtc"
readonly DATA_DISK_LUN="0"
readonly DOCKER_VERSION="5:29.6.1-1~ubuntu.24.04~noble"
readonly AZCOPY_VERSION="10.32.6"
readonly AZCOPY_SHA256="2ec557656be1976754e97de828c0662f7df0e73ef6fc98fc52a518452a6adfbc"

TEMP_DIR="$(mktemp --directory)"
readonly TEMP_DIR
trap 'rm -rf -- "${TEMP_DIR:?}"' EXIT

log() {
  printf '[bootstrap-cpu-host] %s\n' "$*"
}

fail() {
  printf '[bootstrap-cpu-host] ERROR: %s\n' "$*" >&2
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
    "$DATA_MOUNT/work"
  install --directory --mode 0711 --owner root --group root \
    "$DATA_MOUNT/docker"
}

configure_package_repositories() {
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
}

install_host_packages() {
  apt-get update
  apt_install \
    "docker-ce=${DOCKER_VERSION}" \
    "docker-ce-cli=${DOCKER_VERSION}"
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
  apt_install ca-certificates curl e2fsprogs parted

  prepare_data_disk
  configure_package_repositories
  install_host_packages
  configure_docker
  install_azcopy

  log "CPU host bootstrap complete."
}

main
