#!/usr/bin/env bash
set -euo pipefail

VM_DIR="${HOME}/.koton_vm"
DISK_IMG="${VM_DIR}/koton-ubuntu-vm.qcow2"
SEED_ISO="${VM_DIR}/seed.iso"
PID_FILE="${VM_DIR}/qemu.pid"
SSH_PORT="${KOTON_VM_SSH_PORT:-2222}"

if [ ! -f "${DISK_IMG}" ] || [ ! -f "${SEED_ISO}" ]; then
    bash "$(dirname "$0")/setup_ubuntu_vm.sh"
fi

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[KoTon-VM] Ubuntu VM is already running (PID $(cat "${PID_FILE}"), SSH 127.0.0.1:${SSH_PORT})."
    exit 0
fi

KVM_ARGS=""
if [ -e /dev/kvm ] && [ ! -w /dev/kvm ]; then
    sudo -n chmod 666 /dev/kvm 2>/dev/null || true
fi
if [ -w /dev/kvm ]; then
    KVM_ARGS="-enable-kvm -cpu host"
else
    KVM_ARGS="-cpu max"
fi

qemu-system-x86_64 \
    ${KVM_ARGS} \
    -m 1536 \
    -smp 2 \
    -drive file="${DISK_IMG}",if=virtio,format=qcow2,cache=writethrough \
    -drive file="${SEED_ISO}",if=virtio,format=raw \
    -netdev user,id=net0,hostfwd=tcp:127.0.0.1:${SSH_PORT}-:22 \
    -device virtio-net-pci,netdev=net0 \
    -display none \
    -daemonize \
    -pidfile "${PID_FILE}" \
    -serial file:"${VM_DIR}/console.log"

echo "[KoTon-VM] Booted isolated Ubuntu VM (PID $(cat "${PID_FILE}"))."
echo "[KoTon-VM] Connect Window B with: ssh -i ~/.koton_vm/id_ed25519 -p ${SSH_PORT} ubuntu@127.0.0.1"
