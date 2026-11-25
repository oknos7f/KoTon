# KoTon — Real-Time Provenance Graph Intrusion Detection System

## 1. 프로젝트 개요 및 현재 목표
**KoTon**은 Linux 커널의 `auditd` 로그를 실시간 이종 Provenance Graph로 변환하고, Multi-Task Heterogeneous GNN을 통해 알 수 없는 이상 행위(Anomaly)와 알려진 공격 단계를 동시에 탐지하는 보안 관제 시스템입니다.

---

## 2. 데이터셋 명세, 보존 원칙

### 메인 데이터셋: AIT-LDS v2.0
Linux 네이티브 auditd 로그와 다중 클래스 GT을 제공하는 AIT-LDS v2.0을 메인 데이터셋으로 채택
- 선정 사유: 실제 엔터프라이즈 가상 환경에서 수집된 8개의 시나리오로 구성되며, OS 커널의 순정 auditd 로그를 그대로 포함함.
- 공격 클래스 다양성: 단순 이진 탐지를 넘어 Reconnaissance (Nmap/Dirb 스캔), Initial Access (웹셸/익스플로잇), Privilege Escalation (권한 상승), Credential Access (패스워드 크래킹/브루트포스), Exfiltration (데이터 유출) 등 시연 임팩트가 확실한 다중 공격 단계(Multi-stage) 라벨을 모두 포함함.

### 추후 확장 데이터셋: CAM-LDS (2026)
- 7개 Linux 공격 시나리오, 순정 `auditd` 로그, **13개 MITRE ATT&CK Tactics** 및 **81개 Techniques** 라벨 제공.
- 추후 코드 수정 없이 결합할 수 있도록 분리 설계함.

---

## 3. 이종 Feature 인코딩
**노드 타입별 고유 차원(Type-Specific Input Dimensions)**을 유지한 뒤 모델 입력단의 타입별 선형 투영(`68/36/20 -> 128-dim`) + `LayerNorm` + `GELU`으로 공통 128차원 잠재 공간에 정합시킨다. 절대 타임스탬프, 호스트명, 임시 PID는 Data Leakage 방지를 위해 Feature에서 제외.

| 엔티티 타입 | 차원 | 세부 피처 구성 명세 (`src/graph/encoder.py`, `src/graph/builder.py`) |
| :--- | :---: | :--- |
| **`process` (Node)** | **68-dim** | • **Path HFH (`16-dim`):** 정제된 `exe` 경로의 계층적 해싱(Hierarchical Feature Hashing)<br>• **Cmdline Embedding (`48-dim`):** Train Split `cmdline` 전용 학습 FastText 벡터<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), cmdline_shannon_entropy, uid==0_flag]` |
| **`file` (Node)** | **36-dim** | • **Path HFH (`32-dim`):** `CWD` 결합 및 Sanitizer 처리된 정규 절대경로 HFH<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), path_depth, is_sensitive_path_flag]` |
| **`socket` (Node)** | **20-dim** | • **Subnet HFH (`12-dim`):** IPv4/IPv6 서브넷 프리픽스(A/B/C/D 클래스) 계층 해싱<br>• **Port Encoding (`4-dim`):** 포트 대역 One-hot(Well-known/Registered/Dynamic `3-dim`) + 정규화 포트(`port/65535`, `1-dim`)<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), is_rfc1918_private, log1p(window_conn_freq)]` |
| **`relations` (Edge)** | **2-dim** | • ** 대상 관계(10종):** 정방향 5종(`execute`, `read`, `write`, `fork`, `connect`) + 역방향 5종(`rev_*`)<br>• **Temporal & Compression (`2-dim`):** `[log1p(delta_t), log1p(cpr_count)]` (동일 주체 직전 이벤트 대비 경과 시간 및 CPR 누적 횟수) |

---

## 4. 개발 환경

- **OS / Runtime:** Windows + WSL2
- **가상환경:** `requirements.txt` 참고

---

## 5. 단계별 상세 구현 진행도

### ☑ Step 1: 환경 설정 및 데이터셋 분석
### ☑ Step 2: `auditd` 멀티라인 병합 파서 제작 및 타임스탬프 일치
### ☑ Step 3: Feature 인코딩, 라벨 매핑
### ☑ Step 4: 모델 구현 ([상세 정보](./MODEL.md))
### ☑ Step 5: 프론트엔드 연동을 위한 모델 인터페이스 모듈화
### ☐ Step 6: AIT-LDS v2.0 전체 시나리오 확장 및 모델 최적화
- **전체 시나리오 통합 전처리 (`dataset/`은 읽기 전용 유지):**
  - `wheeler`를 제외하고, `fox` 외 나머지 6개 시나리오(`russellmitchell`, `harrison`, `santos`, `shaw`, `wardbeck`, `wilson`)의 UTC 공격 구간을 `src/dataset/phases.py`의 `RAW_PHASES`에 등록한 뒤 `processed/<scenario_name>/`에 직렬화(`.pt`).
  - 시나리오 간 일반화 성능 검증을 위해 **7-Fold LOOCV**를 적용하며, Transductive Leakage 차단을 위해 FastText(`48-dim`) 및 피처 정규화 통계량은 각 Fold의 Train 시나리오(6개) 내부에서만 독립 재학습.
- **관계 균형 마스킹 및 하이퍼파라미터 최적화 (`src/models/losses.py`, `scripts/train_loocv.py`):**
  - 고빈도 관계(`read`, `write`)의 손실 독점을 방지하고 저빈도·고위험 관계(`execute`, `connect`, `fork`)의 복원 민감도를 높이기 위해 **관계 균형 마스킹(Relation-Balanced Masking)** 도입.
  - `Optuna`(TPE + Hyperband)로 손실 가중치 $\lambda$, Focal Loss $\gamma$, 관계 마스킹 지수, `GATv2Conv` Dropout을 최적화하고, 최종 가중치는 `torch.compile` 또는 ONNX Runtime으로 추론 가속.
  - **목표 성능 지표:**
    - Window-Level Macro-F1 (LOOCV): **$\ge 0.85$**
    - Rare-Class Recall (`PrivEsc`, `Exfiltration`): **$\ge 0.80$**
    - False Positive Rate (FPR): **$< 1.0\%$**
    - 단일 윈도우 추론 지연 (Inference Latency): **$< 50\text{ms}$**

### ☐ Step 7: `false_positive_test` 분석 및 Conformal Prediction 기반 동적 임계치 산정 (`scripts/calibrate_threshold.py`)
- **Split Conformal Calibration 적용:**
  - AIT-LDS v2.0의 순수 정상 구간(`false_positive_test`)을 학습과 완전히 분리된 **Disjoint Calibration Set**으로 구성하여, 엔티티 타입별(`process`, `file`, `socket`) 복원 오차($\mathcal{L}_{recon}$)의 경험적 $p$-value를 산출.
  - 분포 가정 없이 유한 표본에서 **FPR < 1% ($\alpha = 0.01$)**를 수학적으로 보장하는 `default_threshold` 확정.
- **타입 간 증거 융합 및 슬라이더 프로파일 생성 (`configs/threshold_profile.json`):**
  - 노드 타입별 $p$-value를 **Fisher's Method**로 통합하여 단일 이상 점수를 산출하고, 프론트엔드 슬라이더용 목표 오탐 허용률 범위($\alpha \in [0.001, 0.10]$)에 대응하는 분위수 룩업 테이블을 저장.

### ☐ Step 8: Ubuntu VM 기반 실시간 `auditd` 수집 및 스트리밍 추론 엔진 완성 (`src/collector/`, `src/serving/`)
- **격리 Ubuntu VM 기반 `auditd` 실시간 수집 환경 (`src/collector/audit_stream.py`):**
  - WSL2 기본 커널의 `CONFIG_AUDIT` 미지원 제약을 돌파하고 실제 공격 시연의 현실성·격리성을 확보하기 위해 경량 **Ubuntu VM**(Multipass/QEMU 등)을 구성·가동.
  - VM 내부 순정 커널 `auditd`에 핵심 6종 syscall(`execve`, `openat`, `read`, `write`, `connect`, `clone`/`fork`) 로깅 규칙을 적용하고, 실시간 로그 스트림을 TCP/Unix Socket으로 호스트 수집기에 전송(기존 `streaming_parser.py` 파이프라인 100% 재사용).
- **실시간 스트리밍 추론 엔진 고도화 (`src/serving/engine.py`):**
  - `KoTonInferenceEngine`에 180초 윈도우 내 5초 마이크로 틱 기반 **Incremental Sliding Window Buffer** 및 **Online CPR** 통합.
  - 임계치 초과 노드 및 악성 분류 엣지를 시드로 **2-Hop 인과 역추적(Attack Summary Subgraph 추출)**을 수행하며, 이벤트 수신 $\rightarrow$ 인코딩 $\rightarrow$ 추론 $\rightarrow$ 서브그래프 추출 $\rightarrow$ 직렬화까지의 End-to-End 지연 시간을 **500ms 이하**(목표 250ms 이내)로 최적화.

### ☐ Step 9: Flask-SocketIO + Three.js 3D 보안 관제 및 Dual-Window 데모 (`web/`)
- **Delta-Patch 기반 Flask-SocketIO 백엔드 (`web/app.py`):**
  - `Flask` + `Flask-SocketIO` (`threading` 비동기 모드)로 전체 스냅샷 대신 직전 상태 대비 변경분(`graph_delta`)과 경보 요약(`attack_alert`)만 WebSocket으로 푸시하여 브라우저 파싱 병목 제거.
  - **무재추론(Zero-Reinference) 동적 임계치 API:** 슬라이더에서 수신한 신규 임계치(`update_threshold`, $\alpha$)를 GNN 재추론 없이 캐싱된 `anomaly_score` 및 `threshold_profile.json`과 $O(N)$ 대조(1ms 미만)하여 `is_malicious` / `is_attack_path` 플래그 즉시 갱신.
- **Dual-Window 데모 레이아웃 (`web/static/`, `web/templates/`):**
  - **Window A:** 3D 관제 대시보드(3D Provenance Cloud, 실시간 FPR·지연시간 텔레메트리, Conformal 임계치 슬라이더, 공격 타임라인).
  - **Window B:** 공격자 **Ubuntu VM** 터미널 창을 병렬 배치하여 명령 실행 직후 실시간 탐지 및 공격 경로 점등 시연.

### ☐ Step 10: 라이브 공격 시나리오 스크립트 및 고성능 3D 비주얼 고도화 (`scripts/demo_attack.sh`, `web/`)
- **Ubuntu VM 라이브 공격 시나리오 스크립트 (`scripts/demo_attack.sh`):**
  - 격리된 Ubuntu VM 내부에서 실제 커널 `auditd` syscall 체인을 유발하며 AIT-LDS v2.0의 다중 공격 단계를 재현하는 실행 스크립트 구현:
    1. `Scan` (Class 1): 로컬 포트 스캔 및 디렉터리 열거
    2. `Webshell / RCE` (Class 2): 로컬 웹 서비스 기반 `/bin/sh -c` 비정상 자식 프로세스 체인 생성(`fork` $\rightarrow$ `execute`)
    3. `Privilege Escalation` (Class 4): `sudo -l`, SUID 바이너리 탐색 및 `/etc/passwd`·모의 `/etc/shadow` 접근
    4. `Data Exfiltration` (Class 5): 민감 더미 파일 압축 후 외부(로컬 더미 리스너) 소켓 전송
- **기하 도형 기반 3D 렌더링 최적화 (60 FPS 유지):**
  - 500~1,500개 노드 규모에서 Draw Call 병목을 방지하기 위해 모든 노드를 `THREE.InstancedMesh` 또는 Custom Shader `THREE.Points` 기반 경량 기하 도형으로 렌더링:
    - **`Process`:** 원형 점 (Circle)
    - **`File`:** 정사각형 (Square)
    - **`Socket`:** 마름모 (Diamond)
- **정상 vs. 악성 경로 시각화:**
  - 정상 노드와 엣지는 낮은 불투명도(`0.15~0.25`)의 무채색 톤으로 배치하여 시각적 노이즈 최소화.
  - 탐지된 악성 노드는 **6-Class 공격 단계별 네온 컬러 하이라이팅**을 적용하고, 공격 경로(`is_attack_path: true`) 엣지 위로 **방향성 파티클(Directional Particles)**이 흐르도록 구현.
- **단일 아이콘 연동 Inspector HUD:**
  - 특정 노드 마우스 호버(`Raycaster`) 시 해당 객체의 실시간 상세 정보(`cmdline`, `path`, `anomaly_score`, Conformal $p$-value, `attack_class`)와 함께 해당 엔티티 타입의 `256×256 PNG` 아이콘(`web/images/process.png`, `file.png`, `socket.png`)을 오버레이 HUD에 표시하여, 한 화면에 최대 **1개의 아이콘**만 렌더링되도록 제한.

### ☐ Step 11: [선택] CAM-LDS (2026) 통합 및 도메인 강건성 검증
- **데이터셋 통합 및 피처 스펙 정합:**
  - `dataset/` 보존 원칙을 유지한 채 CAM-LDS를 추가하고, `configs/default.yaml`에서 `include_supplementary_logs: false`로 설정해 순정 `auditd` 파서 단독 구동.
  - `configs/taxonomy.yaml`에 CAM-LDS의 **13 MITRE ATT&CK Tactics / 81 Techniques** 매핑 테이블을 추가하고, `MODEL.md` 기준 엔티티별 고유 입력 차원(`process` 68-dim, `file` 36-dim, `socket` 20-dim, `edge` 2-dim $\rightarrow$ 128-dim 잠재 공간 투영)으로 `HeteroData` 일관 인코딩.
- **2단계 도메인 전이 검증:**
  - **Zero-Shot Anomaly Transfer:** AIT-LDS 학습 가중치를 동결한 채 CAM-LDS 정상 구간에 Conformal Calibration만 재적합하여 비지도 복원 헤드($\mathcal{L}_{recon}$)의 Zero-shot AUROC 및 FPR 유지력 검증.
  - **Few-Shot Classification Fine-Tuning:** 엣지 분류 헤드를 6-Class에서 13-Tactic 출력층으로 교체·미세조정하여 타 도메인 적응 성능(Macro-F1) 입증.