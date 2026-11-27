# 실시간 수집·추론 및 3D 웹 관제 상세 정보

---

## 1. 실시간 관제 디렉터리 및 런타임 격리 아키텍처

| 구분 | 경로 | 역할 및 제약 사항 |
| :--- | :--- | :--- |
| **Stream Collector** | `src/collector/audit_stream.py` | TCP(`:9999`), Unix Domain Socket(`/tmp/koton_audit.sock`), 파일 리플레이를 통합 수신하고 `StatefulAuditParser`로 실시간 멀티라인 병합 수행. 원문 로그 라인은 별도 링 버퍼(`maxsize=2,000`)에 적재하여 대시보드 로그 피드로 전달. |
| **Streaming Engine** | `src/serving/engine.py`<br>`checkpoints/koton_all7_hetero_gnn.pt`<br>`configs/threshold_profile{,_vm}.json` | 7-시나리오 통합 GNN 및 Train 전용 정규화 통계량 기반 180초 증분 슬라이딩 윈도우, $O(1)$ Online CPR, 엔티티 타입별 $p$-value + Fisher's Method 통합 점수, 2-Hop 인과 서브그래프 추출, Delta-Patch 산출 및 무재추론($<1\text{ms}$) Conformal Cutoff 판정 담당. |
| **Web Backend & UI** | `web/app.py`<br>`web/templates/index.html`<br>`web/static/{js,css}/` | `Flask` + `Flask-SocketIO`(`threading` 모드) 기반 실시간 증분 푸시 서버, 이중 환경 프로파일 전환(`KOTON_PROFILE_MODE=ait\|vm`, `POST /api/profile`) 및 Custom GLSL `THREE.Points` 3D 보안 관제 대시보드(Window A). 최초 기동 시 빈 클라우드(`0 / 0`)로 시작. |
| **HUD Entity Icons** | `web/images/{process,file,socket}.png` | 사용자 제작 `512×512` RGBA 아이콘(흑색 실루엣). 호버 시 Inspector HUD에 단일 노출되며, 코드에서 재생성·덮어쓰기하지 않음(`scripts/generate_hud_icons.py`는 파일 부재 시에만 대체 아이콘 생성). |
| **Isolated Ubuntu VM** | `scripts/setup_ubuntu_vm.sh`<br>`scripts/start_ubuntu_vm.sh`<br>`scripts/demo_attack.sh` | WSL2 `/dev/kvm` 가속 Ubuntu 24.04 게스트(`~/.koton_vm/`, CoW 디스크 격리) 구동, 순정 커널 `auditd` 6종 syscall 포워딩, 유휴 하트비트 서비스 및 파라미터 선택형 5단계 시연 스크립트(Window B). |
| **Calibration & Verification** | `scripts/calibrate_thresholds.py`<br>`scripts/verify_realtime_soc.py`<br>`scripts/verify_live_vm.py` | AIT-LDS `false_positive_test` 및 Ubuntu VM 5분 유휴 트래픽 기반 이중 환경 Split Conformal 보정, 빈 클라우드·아이콘·로그·단조 전환 검증 및 실제 KVM VM 기반 E2E 검증. |

---

## 2. 격리 Ubuntu VM 및 실시간 커널 `auditd` 수집 파이프라인

### 2.1 QEMU/KVM 격리 게스트 및 자기 감사 루프 차단 (`scripts/setup_ubuntu_vm.sh`, `scripts/start_ubuntu_vm.sh`)
* **커널 `CONFIG_AUDIT` 제약 돌파 및 디스크 무결성 보장**
  * WSL2 호스트 커널의 감사 서브시스템 제약을 우회하기 위해 순정 Linux 커널(`6.8.0-147-generic`)을 탑재한 Ubuntu 24.04 Cloud Image 기반 경량 KVM 가상머신을 분리 구동(`127.0.0.1:2222` SSH 포워딩).
  * 원본 클라우드 이미지를 보존한 채 Copy-on-Write 오버레이(`koton-ubuntu-vm.qcow2`, `cache=writethrough`)로 동작.
* **핵심 6종 시스템콜 로깅 규칙 및 무한 루프 원천 차단 (`/etc/audit/rules.d/99-koton.rules`)**
  * 프로세스 생성·실행(`execve`, `execveat`, `clone`, `fork`, `vfork`), 소켓 연결(`connect`), 민감 디렉터리(`/etc`, `/tmp`, `/var/www`) 파일 입출력(`openat`, `read`, `write`) 수집.
  * 포워더 전용 시스템 계정(`kotonfwd`) 분리 및 `-a never,exit -F dir=/var/log/audit`, `-F uid!=kotonfwd` 필터로 자기 증폭 루프(Self-Audit Loop) 차단.
* **실시간 호스트 태그 포워딩 및 유휴 하트비트**
  * `koton-audit-forwarder.service`: `/var/log/audit/audit.log` 증분 라인을 `<host_id>\t<raw_audit_line>\n` 규격으로 호스트 수집기(`10.0.2.2:9999`)에 자동 재연결 스트리밍.
  * `koton-idle-activity.service`: `ubuntu` 계정으로 3~5초 간격의 정상 파일 조회(`/etc/hostname`, `/etc/os-release`, `/tmp`)를 반복하여, 공격이 없는 유휴 상태에서도 정상(회색) 노드·엣지가 실시간으로 조금씩 추가되도록 보장.

### 2.2 스트리밍 수집기 및 원문 로그 피드 (`src/collector/audit_stream.py`, `src/parser/streaming_parser.py`)
* **파서 파이프라인 100% 재사용 및 저지연 Carry-over 플러시**
  * 오프라인 학습과 동일한 `StatefulAuditParser`(`include_supplementary_logs=False`, `carryover_sec=0.25s`)로 `(host_id, serial_id)` 기준 `SYSCALL`, `EXECVE`, `CWD`, `PATH`, `SOCKADDR`, `PROCTITLE`, `EOE` 레코드 실시간 병합.
* **원문 로그 링 버퍼 (`drain_raw_logs`)**
  * 수신한 모든 라인을 `{"ts", "host", "line"}` 형태로 별도 큐(`maxsize=2,000`, 포화 시 가장 오래된 항목 폐기)에 적재하고, 서버 워커가 0.2초 주기로 최대 60줄씩 꺼내 `audit_log_batch` 이벤트로 푸시.
* **커널 의사 경로(Pseudo-Path) 중복 억제**
  * `nametype=PARENT` 레코드를 실제 대상 경로가 존재할 때 제외하고, 동일 이벤트 내 중복 `(edge_type, dst_key)`를 단일화.

---

## 3. 증분 슬라이딩 윈도우 추론 엔진 및 Significance 제어

### 3.1 슬라이딩 버퍼, Online CPR 및 2-Hop 인과 역추적 (`src/serving/engine.py`)
* **180초 Incremental Sliding Window Buffer:** 신규 스트림 유입 시 또는 5초 마이크로 틱마다 만료 엣지를 `deque` 좌측에서 퇴거하고 현재 윈도우 내부 차수만으로 `HeteroData` 스냅샷 구성.
* **$O(1)$ Online CPR:** $0 \le \Delta t < 1.0\text{s}$ 내 동일 프로세스의 연속 파일 입출력을 `cpr_count`로 병합하고, 타 프로세스 개입 시 해당 파일의 병합 슬롯만 무효화.
* **2-Hop 인과 서브그래프:** 활성 노드·공격 경로 엣지 양 끝단을 시드로 역방향 2-Hop + 정방향 1-Hop BFS를 수행하여 `subgraph_node_ids`와 `dominant_attack_class` 산출.

### 3.2 이중 환경 Split Conformal Calibration 및 활성 상태 판정 (`_evaluate_records_with_cutoff`, `compute_window_fisher_summary`)
* **이중 환경 보정 프로파일 (`configs/threshold_profile.json`, `configs/threshold_profile_vm.json`):**
  * **AIT-LDS 프로파일(`ait`):** 7개 시나리오 미학습 `false_positive_test` 1,576개 순수 정상 윈도우 기반 (`recon_scale=1.1462`, $\alpha=0.01$ Cutoff `0.8493`, Fisher Threshold `30.6756`, Held-out Node FPR **`0.8920%` < 1%**).
  * **Ubuntu VM 프로파일(`vm`):** 실제 Ubuntu VM 5분(300초) 유휴 데몬 트래픽(18,609줄 `auditd` 로그 → 4,803 CPR 엣지, 96개 마이크로틱 윈도우) 기반 (`recon_scale=2.4128`, $\alpha=0.01$ Cutoff `0.6977`, Fisher Threshold `40.1870`, Held-out Node FPR **`1.0140%`**). Ubuntu VM 고유 데몬의 높은 기본 복원 오차를 보정 스케일로 흡수하여 오탐을 억제.
* **엔티티 타입별 $p$-value 및 Fisher's Method 통합 점수:**
  * `process` / `file` / `socket` 타입별 독립 스케일과 정렬 비순응도 분포에서 $O(\log M)$ 경험적 $p$-value($p_t$)를 산출.
  * 윈도우 내 활성 엔티티 타입별 최솟값 $p$-value를 Fisher's Method($T_{\text{Fisher}} = -2 \sum_t \ln p_{W, t}$)로 결합하여 Split Conformal 경험적 $p$-value 및 $\chi^2_{2k}$ 유의확률을 `telemetry.fisher`에 포함.
* **신뢰도 점수 분리 저장 및 동일 기준 판정:** 매 틱 추론 시 노드(`recon_score`, `max_edge_risk`, `cand_cls`)와 엣지(`edge_conf`, `cand_cls`)의 연속 신뢰도를 캐시(`_cached_state`)에 저장하고, 단일 Cutoff와 비교해 활성(공격) / 비활성(정상·회색) 상태를 결정(양 끝단이 모두 활성인 엣지는 `0.72 × Cutoff`부터 점등).
* **Significance(α) → Cutoff 변환 (`configs/threshold_profile.json` 기준, 상한 `0.95`):**

| Significance (α) | AIT-LDS Cutoff | VM Cutoff | 화면 동작 |
| :---: | :---: | :---: | :--- |
| `0.005` | `0.938` | `0.705` | 고신뢰 공격 노드 중심 강조 |
| `0.010` (기본) | `0.849` | `0.698` | **FPR < 1%** 목표 지점, 핵심 공격 단계 활성 |
| `0.050` | `0.632` | `0.632` | 스캔·웹셸·크래킹 포함 전 단계 활성 |
| `0.200` | `0.325` | `0.448` | 복원 오차가 다소 높은 경계 정상 노드까지 민감 표시 |

* **무재추론 갱신:** 슬라이더 조작 또는 `POST /api/profile` 전환 시 GNN을 재실행하지 않고 캐시된 신뢰도 배열만 $O(N)$ 재평가(실측 `0.206 ms`)하여 `graph_delta`(`updated_nodes`, `updated_edges`)와 2-Hop 서브그래프를 즉시 재산출. 클라우드 좌표는 유지되고 **색상·크기·불투명도만** 전환됨.

### 3.3 저지연 추론 가속 (`src/graph/encoder.py`, `src/models/hetero_gnn.py`)
* **FastText 어휘 캐시 선적재 및 온라인 정규화:** 7-시나리오 통합 FastText(`artifacts/all7_cmd_fasttext.model`) 어휘 벡터를 `_tok_vec_cache`에 선적재하고, 체크포인트 내 Train 전용 `feature_stats`를 마이크로틱 스냅샷에 즉시 적용.
* **3-배치 블록 대각 융합 추론 및 ONNX/컴파일 검증:** 원본 그래프와 2-Fold 마스킹 그래프 2개를 단일 블록 대각 그래프로 결합하여 `encode_backbone` 호출을 3회에서 1회로 축소(CUDA Eager 평균 `20.44 ms/window`, Edge MLP ONNX Export `artifacts/koton_edge_classifier.onnx` CPU ORT **1.96x 가속**).
* **Split Conformal $p$-value:** $p_t(s) = \frac{1 + |\{i : s_{t,i} \ge s\}|}{1 + M_t}$ (엔티티 타입별 이진 탐색 $O(\log M_t)$).

---

## 4. Delta-Patch 웹 백엔드 및 통신 인터페이스 명세

### 4.1 REST API 및 Socket.IO 이벤트 규격 (`web/app.py`)

| 프로토콜 / 방향 | 엔드포인트 / 이벤트명 | 역할 및 페이로드 요약 |
| :--- | :--- | :--- |
| **HTTP `GET`** | `/` , `/images/<filename>` | 대시보드 HTML 렌더링 및 사용자 제작 HUD 아이콘(`process.png`, `file.png`, `socket.png`) 서빙. |
| **HTTP `GET`** | `/api/status` , `/api/snapshot` | 수집기 상태 조회 및 현재 윈도우 그래프 스냅샷 반환(최초 기동 직후에는 빈 `nodes`/`edges`). |
| **HTTP `POST`** | `/api/threshold` | `{"alpha": <float>}` 또는 `{"threshold": <float>}` 수신 후 무재추론 갱신 결과 반환 및 브로드캐스트. |
| **HTTP `POST`** | `/api/profile` | `{"mode": "ait" \| "vm"}` 수신 시 `configs/threshold_profile{,_vm}.json`을 무재기동으로 전환하고 `threshold_updated` 브로드캐스트. |
| **HTTP `POST`** | `/api/simulate/live_attack` | VM 없이 동작 확인이 가능하도록 4단계 커널 `auditd` 멀티라인 이벤트 스트림을 수집기에 주입. |
| **WS `Server → Client`** | `graph_init` | 최초 소켓 연결 시 현재 윈도우 상태 1회 전송(서버 최초 기동 시 **빈 클라우드**). |
| **WS `Server → Client`** | `graph_delta` | 신규 스트림 유입 또는 마이크로 틱마다 변경분(`added_*`, `updated_*`, `removed_*`)과 텔레메트리 푸시. |
| **WS `Server → Client`** | `audit_log_batch` | `{"logs": [{"ts", "host", "line"}, ...]}` 형태로 원문 `auditd` 로그를 0.2초 주기 푸시. |
| **WS `Server → Client`** | `attack_alert` | 2-Hop 인과 서브그래프 요약(`dominant_attack_class`, `subgraph_node_ids`) 푸시. |
| **WS `Bidirectional`** | `update_threshold` $\leftrightarrow$ `threshold_updated` | 슬라이더 α 전달 후 `<1ms` 내 갱신된 `graph_delta` 및 경보 요약을 전 클라이언트 동기화. |

### 4.2 Delta-Patch 및 로그 페이로드 스키마 (`src/serving/engine.py`)
```json
{
  "timestamp": 1764090038.19,
  "delta": {
    "added_nodes": [
      {
        "id": "ubuntu-vm_proc_4155_v2",
        "type": "process",
        "label": "/usr/bin/sudo",
        "anomaly_score": 0.925,
        "recon_score": 0.4127,
        "p_value": 0.1043,
        "attack_class": "privilege_escalation",
        "is_malicious": true
      }
    ],
    "updated_nodes": [],
    "removed_node_ids": [],
    "added_edges": [
      {
        "id": "ubuntu-vm_proc_4155_v2->read->ubuntu-vm_file_tmp_mock_shadow#0",
        "source": "ubuntu-vm_proc_4155_v2",
        "target": "ubuntu-vm_file_tmp_mock_shadow",
        "type": "read",
        "count": 1,
        "edge_risk": 0.925,
        "attack_class": "privilege_escalation",
        "is_attack_path": true
      }
    ],
    "updated_edges": [],
    "removed_edge_ids": []
  },
  "telemetry": {
    "latency_ms": 45.35,
    "num_nodes": 365,
    "num_edges": 1386,
    "alpha": 0.01,
    "threshold": 0.6977,
    "estimated_fpr": 0.0089,
    "online_cpr_pct": 14.50,
    "profile_env": "ubuntu_vm_idle",
    "fisher": {
      "fisher_stat": 42.18,
      "fisher_threshold": 40.187,
      "fisher_chi2_pvalue": 0.000001,
      "fisher_conformal_pvalue": 0.0099,
      "is_fisher_anomalous": true
    }
  }
}
```
```json
{
  "logs": [
    { "ts": 1764090037.103, "host": "ubuntu-vm", "line": "type=SYSCALL msg=audit(1764090037.091:4410): arch=c000003e syscall=257 success=yes ... exe=\"/usr/bin/cat\"" }
  ]
}
```

---

## 5. 고성능 Three.js 3D 뷰어 및 Dual-Window 시연 규격

### 5.1 렌더링, 배치 및 상호작용 (`web/static/js/dashboard.js`, `web/static/css/dashboard.css`)

| 시각화 요소 | 구현 방식 | 세부 렌더링 동작 명세 |
| :--- | :--- | :--- |
| **기하 도형 노드 (`THREE.Points`)** | 단일 `BufferGeometry` + Custom GLSL Shader | 프래그먼트 셰이더 SDF로 단일 Draw Call에서 **`process`=원형**, **`file`=정사각형**, **`socket`=마름모** 렌더링(60 FPS 유지). |
| **화면 중앙 정렬** | 원점 시선 카메라 + 질량 중심 보정 | 카메라를 `(0, 0, 390)`에서 원점을 정면 주시하도록 고정하고, 계층 분리 폭을 `±20`으로 대칭 축소한 뒤 매 프레임 전체 노드의 질량 중심을 원점으로 8%씩 보정하여 **상하·좌우 모두 화면 중앙**에 클라우드 유지. |
| **점진 추가 및 자연 응집** | 연속 Force-Directed 물리 + Spawn 페이드 | 빈 상태에서 시작하여 새 노드만 **이미 존재하는 인접 노드 근처**에서 생성되고 크기·불투명도가 0.08 → 1.0으로 서서히 증가. 기존 노드 좌표는 재계산하지 않으며, 중심 인력·반발력·엣지 스프링(감쇠 0.82, 최대 속도 3.2)으로 **매 프레임 조금씩 뭉침**. |
| **정상 vs. 활성 대비** | 불투명도·크기·펄스 분리 + Additive Blending | 정상은 슬레이트 무채색(`#64748b`, `alpha=0.22`), 활성은 **6-Class 네온 팔레트**(`scans` 시안, `webshell` 마젠타, `cracking` 앰버, `privilege_escalation` 레드, `dnsteal` 퍼플, 미분류 `anomaly` 로즈) 및 펄스 적용. |
| **방향성 공격 파티클** | `THREE.Points` 선형 보간 (`MAX_PARTICLES=600`) | 공격 경로 엣지를 따라 노드의 **실시간 위치**를 추적하며 전술 색상 입자가 흐름. |
| **실시간 로그 피드** | 테두리 없는 하단 스택 + TTL 페이드 | `Live Audit Log: ON/OFF` 토글. 최신 로그가 **아래에서 위로 올라오며**(최대 15줄), 9초 후 위로 페이드아웃되며 제거. 배경·테두리 없이 텍스트 그림자만 사용하며, 화면 폭을 넘는 긴 로그는 `pre-wrap` + `overflow-wrap: anywhere`로 **줄바꿈** 처리. |
| **단일 아이콘 Inspector HUD** | 화면 공간 투영 피킹 + DOM 오버레이 | 셰이더 포인트 크기에 맞춘 화면 좌표 반경 피킹으로 커서에 가장 가까운 노드 1개를 선택하고, HUD를 커서 옆에 표시. 사용자 제작 흑색 실루엣 아이콘이 어두운 배경에서 보이도록 CSS `brightness(0) invert(1)` + 시안 글로우를 적용하여 **흰색으로 표시**. 페이지 로드 시 아이콘을 미리 지정해 빈 `src` 상태를 제거. |

### 5.2 파라미터 선택형 시연 스크립트 (`scripts/demo_attack.sh`)
* **사용법**
  ```bash
  /home/ubuntu/demo_attack.sh          # 5개 시나리오를 순서대로 느리게 실행 (단계 간 5.0초, 세부 동작 간 1.2초)
  /home/ubuntu/demo_attack.sh --1      # 지정 시나리오만 실행
  /home/ubuntu/demo_attack.sh --2 --4  # 여러 시나리오를 지정 순서대로 실행
  /home/ubuntu/demo_attack.sh --all    # 전체 시나리오 명시 실행
  /home/ubuntu/demo_attack.sh --help   # 도움말
  ```
  * 속도 조절: `KOTON_ATTACK_DELAY`(단계 간), `KOTON_SUBSTEP_DELAY`(세부 동작 간) 환경 변수.
* **시나리오 ID별 동작 (모두 VM 내부 `127.0.0.1` 및 모의 파일 대상)**

| ID | 전술 (Class) | 커널 syscall 인과 사슬 |
| :---: | :--- | :--- |
| `--1` | `scans` (1) | 로컬 포트 `22, 80, 443, 3306, 8080, 9000`에 순차 `connect` 후 `/etc`, `/var/www`, `/tmp` 열거 |
| `--2` | `webshell/reverse_shell` (2) | `/var/www/html/koton_sim/cmd_webshell.sh` 생성(`write`) → `/bin/sh -c` 자식 체인 실행(`fork` → `execve`) |
| `--3` | `cracking` (3) | 모의 해시 덤프(`/tmp/hash_dump.txt`) 생성 → 사전 대입 비교 → `/tmp/cracked_creds.txt` 기록 |
| `--4` | `privilege_escalation` (4) | `sudo -n -l` 권한 확인, SUID 바이너리 탐색, `/etc/passwd`·`/tmp/mock_shadow` 읽기 |
| `--5` | `service_stop/dnsteal` (5) | 수집 파일을 `/tmp/exfil_customers_secret.tar.gz`로 압축 → 자체 생성 로컬 리스너(`127.0.0.1:4444`)로 전송 |

### 5.3 통합 검증 지표
* **`scripts/verify_realtime_soc.py` (5/5 통과, 7-시나리오 통합 모델 + AIT-LDS 프로파일)**
  * 사용자 아이콘 보존(`512×512` × 3종) 및 `/images/*` 서빙 확인.
  * 최초 기동 시 빈 클라우드(`0 / 0`) 확인.
  * 원문 로그 24줄 → 링 버퍼 24줄 전량 수신, 프로비넌스 엣지 18개 생성.
  * 스트리밍 틱 **`29.01 ms`** (< 250ms SLA).
  * Significance 단조 전환: α=`0.002` → **3개** 활성 / α=`0.050` → **11개** / α=`0.240` → **13개** (무재추론 **`0.206 ms`**).
* **`scripts/verify_live_vm.py` (실제 KVM Ubuntu VM, `KOTON_PROFILE_MODE=vm` 통과)**
  * 유휴 하트비트 스트림이 호스트 수집기에 도달함을 확인(별도 5분 유휴 보정 시 18,609줄 / 4,803 CPR 엣지 수집, CPR **`19.80%`**).
  * `--1` 단독 실행: Stage 1만 실행되고 Stage 2 이후는 실행되지 않음을 확인(원문 로그 80줄, 신규 노드 213개, 틱 `223.21 ms`).
  * 인자 없는 전체 실행: Stage 1~5 순차 완료, Cold Burst **`50.43 ms`**, Steady Tick **`45.35 ms`**, 365 노드 · 1,386 엣지, Online CPR **`14.50%`**, 2-Hop 서브그래프 353 노드(`privilege_escalation`) 검출.

### 5.4 Dual-Window 라이브 데모 구동 절차 (WSL2)
* **Window A (호스트 백엔드 및 3D 관제 대시보드 — `http://127.0.0.1:5000`):**
  ```bash
  cd /mnt/c/Users/dongmin/Desktop/KoTon
  bash scripts/start_ubuntu_vm.sh
  KOTON_PROFILE_MODE=vm .venv/bin/python web/app.py
  ```
  * 환경 변수 `KOTON_PROFILE_MODE`는 `ait`(기본값, `configs/threshold_profile.json`) 또는 `vm`(`configs/threshold_profile_vm.json`)을 지원하며, 기동 중에도 `POST /api/profile`로 무재기동 전환 가능.
* **Window B (격리 Ubuntu VM SSH 접속 및 시나리오 실행):**
  ```bash
  ssh -i ~/.koton_vm/id_ed25519 -p 2222 ubuntu@127.0.0.1
  /home/ubuntu/demo_attack.sh        # 전체 순차 (느린 관찰 모드)
  /home/ubuntu/demo_attack.sh --4    # 특정 시나리오만
  ```
  * 스크립트는 `scripts/verify_live_vm.py` 실행 시 VM의 `/home/ubuntu/demo_attack.sh`로 자동 동기화되며, 수동 동기화 시 `scp -i ~/.koton_vm/id_ed25519 -P 2222 scripts/demo_attack.sh ubuntu@127.0.0.1:/home/ubuntu/` 사용.
