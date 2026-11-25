#!/usr/bin/env bash
set -euo pipefail

VM_DIR="${HOME}/.koton_vm"
IMG_URL="https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img"
BASE_IMG="${VM_DIR}/noble-base.img"
DISK_IMG="${VM_DIR}/koton-ubuntu-vm.qcow2"
SEED_ISO="${VM_DIR}/seed.iso"

mkdir -p "${VM_DIR}"

if ! command -v qemu-system-x86_64 >/dev/null 2>&1 || ! command -v cloud-localds >/dev/null 2>&1; then
    echo "[KoTon-VM] Installing QEMU/KVM and cloud-image-utils..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq qemu-system-x86 qemu-utils cloud-image-utils openssh-client netcat-openbsd
fi

if [ ! -f "${BASE_IMG}" ]; then
    echo "[KoTon-VM] Downloading Ubuntu 24.04 Minimal Cloud Image..."
    curl -L --fail -o "${BASE_IMG}" "${IMG_URL}"
fi

if [ ! -f "${DISK_IMG}" ]; then
    qemu-img create -f qcow2 -b "${BASE_IMG}" -F qcow2 "${DISK_IMG}" 12G
fi

if [ ! -f "${VM_DIR}/id_ed25519" ]; then
    ssh-keygen -t ed25519 -N "" -f "${VM_DIR}/id_ed25519" -q
fi
PUB_KEY="$(cat "${VM_DIR}/id_ed25519.pub")"

cat > "${VM_DIR}/user-data" <<EOF
#cloud-config
hostname: ubuntu-vm
manage_etc_hosts: true
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: users, admin, sudo
    home: /home/ubuntu
    shell: /bin/bash
    lock_passwd: false
    ssh_authorized_keys:
      - ${PUB_KEY}
  - name: kotonfwd
    system: true
    no_create_home: true
    shell: /usr/sbin/nologin
chpasswd:
  list: |
    ubuntu:koton
  expire: false
ssh_pwauth: true
package_update: true
packages:
  - auditd
  - audispd-plugins
  - curl
  - netcat-openbsd
write_files:
  - path: /etc/audit/rules.d/99-koton.rules
    permissions: '0640'
    content: |
      -D
      -b 16384
      -f 1
      -a never,exit -F dir=/var/log/audit
      -a always,exit -F arch=b64 -S execve,execveat -k koton_exec
      -a always,exit -F arch=b64 -S clone,fork,vfork -F uid!=kotonfwd -k koton_fork
      -a always,exit -F arch=b64 -S connect -F uid!=kotonfwd -k koton_net
      -a always,exit -F arch=b64 -S openat,read,write -F dir=/etc -F uid!=kotonfwd -k koton_etc
      -a always,exit -F arch=b64 -S openat,read,write -F dir=/tmp -F uid!=kotonfwd -k koton_tmp
      -a always,exit -F arch=b64 -S openat,read,write -F dir=/var/www -F uid!=kotonfwd -k koton_web
  - path: /usr/local/bin/koton_audit_forwarder.py
    permissions: '0755'
    content: |
      #!/usr/bin/env python3
      import os, socket, time

      HOST = os.environ.get("KOTON_HOST", "10.0.2.2")
      PORT = int(os.environ.get("KOTON_PORT", "9999"))
      LOG_PATH = "/var/log/audit/audit.log"

      def stream_forever():
          sock = None
          while not os.path.exists(LOG_PATH):
              time.sleep(0.5)
          with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as fp:
              fp.seek(0, os.SEEK_END)
              while True:
                  line = fp.readline()
                  if not line:
                      time.sleep(0.05)
                      continue
                  payload = f"ubuntu-vm\t{line.strip()}\n".encode("utf-8")
                  while True:
                      try:
                          if sock is None:
                              sock = socket.create_connection((HOST, PORT), timeout=3.0)
                          sock.sendall(payload)
                          break
                      except OSError:
                          if sock is not None:
                              try:
                                  sock.close()
                              except OSError:
                                  pass
                              sock = None
                          time.sleep(1.0)

      if __name__ == "__main__":
          stream_forever()
  - path: /etc/systemd/system/koton-audit-forwarder.service
    permissions: '0644'
    content: |
      [Unit]
      Description=KoTon Real-Time Auditd TCP Forwarde
      After=network-online.target auditd.service
      Wants=network-online.target

      [Service]
      Type=simple
      User=kotonfwd
      SupplementaryGroups=adm
      ExecStartPre=/bin/chmod 0644 /var/log/audit/audit.log
      PermissionsStartOnly=true
      ExecStart=/usr/bin/python3 /usr/local/bin/koton_audit_forwarder.py
      Restart=always
      RestartSec=1

      [Install]
      WantedBy=multi-user.target
  - path: /usr/local/bin/koton_idle_heartbeat.sh
    permissions: '0755'
    content: |
      #!/usr/bin/env bash
      while true; do
          cat /etc/hostname >/dev/null 2>&1 || true
          sleep 3
          cat /etc/os-release >/dev/null 2>&1 || true
          sleep 4
          ls /tmp >/dev/null 2>&1 || true
          sleep 5
      done
  - path: /etc/systemd/system/koton-idle-activity.service
    permissions: '0644'
    content: |
      [Unit]
      Description=KoTon Benign Idle Telemetry Heartbeat
      After=auditd.service koton-audit-forwarder.service

      [Service]
      Type=simple
      User=ubuntu
      ExecStart=/usr/local/bin/koton_idle_heartbeat.sh
      Restart=always
      RestartSec=3

      [Install]
      WantedBy=multi-user.target
runcmd:
  - mkdir -p /var/www/html && chown ubuntu:ubuntu /var/www/html || true
  - sed -i 's/^log_group.*/log_group = adm/' /etc/audit/auditd.conf || true
  - service auditd restart || systemctl restart auditd
  - augenrules --load || auditctl -R /etc/audit/rules.d/99-koton.rules || true
  - chmod 0644 /var/log/audit/audit.log || true
  - systemctl daemon-reload
  - systemctl enable --now koton-audit-forwarder.service
  - systemctl enable --now koton-idle-activity.service
  - sync
EOF

cat > "${VM_DIR}/meta-data" <<EOF
instance-id: koton-ubuntu-vm-01
local-hostname: ubuntu-vm
EOF

cloud-localds "${SEED_ISO}" "${VM_DIR}/user-data" "${VM_DIR}/meta-data"
echo "[KoTon-VM] Ubuntu VM provisioned at ${VM_DIR}"
