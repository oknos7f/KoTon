# KoTon — Real-Time Provenance Graph Intrusion Detection System

## 1. 프로젝트 개요 및 현재 목표
고빈도 웹 로그 경고 누적으로 인한 보안 관리자의 인지 피로 완화와 네트워크 공격 발생 시 침입 경로 추적 및 감지 보조를 위해,
**KoTon**은 Linux 커널의 `auditd` 로그를 실시간으로 이종 Provenance Graph로 변환하여 Multi-Task Heterogeneous GNN을 통해 이상 행위(Anomaly)의 경로와 알려진 공격 단계를 동시에 탐지하여 직관적인 3D 구조도로 시각화하는 보안 관제 시스템

---

## 2. 데이터셋 명세, 보존 원칙

### 메인 데이터셋: AIT-LDS v2.0
Linux 네이티브 auditd 로그와 다중 클래스 GT을 제공하는 AIT-LDS v2.0을 메인 데이터셋으로 채택
- 선정 사유: 실제 엔터프라이즈 가상 환경에서 수집된 8개의 시나리오로 구성되며, OS 커널의 순정 auditd 로그를 그대로 포함함.
- 공격 클래스 다양성: 단순 이진 탐지를 넘어 Scan(Nmap/Dirb), Webshell, Privilege Escalation, Cracking 등 알려진 다중 공격 단계 라벨을 모두 포함함.

### 추후 확장 데이터셋: CAM-LDS (2026)
- 7개 Linux 공격 시나리오, 순정 `auditd` 로그, **13개 MITRE ATT&CK Tactics** 및 **81개 Techniques** 라벨 제공.
- 추후 코드 수정 없이 결합할 수 있도록 분리 설계함.

---

## 3. 이종 Feature 인코딩
노드 타입별 고유 차원을 유지한 뒤 모델 입력단의 타입별 선형 투영(`68/36/20 -> 128-dim`) + `LayerNorm` + `GELU`으로 공통 128차원 잠재 공간에 정합시킨다. 절대 타임스탬프, 호스트명, 임시 PID는 Data Leakage 방지를 위해 Feature에서 제외.

| 엔티티 타입 | 차원 | 세부 피쳐 |
| :--- | :---: | :--- |
| **`process` (Node)** | **68** | • **Path HFH (`16-dim`):** `exe` 경로의 계층적 해싱(Hierarchical Feature Hashing)<br>• **Cmdline Embedding (`48-dim`):** Train Split `cmdline` 전용 학습 FastText 벡터<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), cmdline_shannon_entropy, uid==0_flag]` |
| **`file` (Node)** | **36** | • **Path HFH (`32-dim`):** `CWD` 결합 및 Sanitizer 처리된 정규 절대경로 HFH<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), path_depth, is_sensitive_path_flag]` |
| **`socket` (Node)** | **20** | • **Subnet HFH (`12-dim`):** IPv4/IPv6 서브넷 프리픽스(A/B/C/D 클래스) 계층 해싱<br>• **Port Encoding (`4-dim`):** 포트 대역 One-hot(Well-known/Registered/Dynamic `3-dim`) + 정규화 포트(`port/65535`, `1-dim`)<br>• **Local Structural (`4-dim`):** `[log1p(in_deg), log1p(out_deg), is_rfc1918_private, log1p(window_conn_freq)]` |
| **Edge** | **2** | • **Temporal & Compression (`2-dim`):** `[log1p(delta_t), log1p(cpr_count)]` (동일 주체 직전 이벤트 대비 경과 시간 및 CPR 누적 횟수)<br>• 대상 관계(10종): 정방향 5종(`execute`, `read`, `write`, `fork`, `connect`) + 역방향 5종(`rev_*`) |

---

## 4. 개발 환경

- **OS / Runtime:** Windows + WSL2 (라이브 `auditd` 시연용 Ubuntu VM 연동)
- **데이터셋 격리:** `dataset/` 읽기 전용 마운트
- **가상환경:** ([`requirements.txt`](./requirements.txt)) 참고

---

## 5. 단계별 상세 구현 진행도

### ☑ Step 1: 환경 설정 및 데이터셋 분석
### ☑ Step 2: `auditd` 멀티라인 병합 파서 제작 및 타임스탬프 일치
### ☑ Step 3: Feature 인코딩, 라벨 매핑
### ☑ Step 4: 모델 구현 및 무결성 검증 ([모델 상세 정보](./MODEL.md))
- `verify_pipeline.py` 통과: **Val Macro-F1 `0.8721`**, **Weighted-F1 `0.9841`** 달성.
### ☑ Step 5: 프론트엔드 연동을 위한 모델 인터페이스 모듈화
### ☑ Step 6: Ubuntu VM 기반 실시간 `auditd` 수집 및 스트리밍 추론 엔진 제작
### ☑ Step 7: Flask-SocketIO + Three.js 3D 보안 관제 및 Dual-Window 데모 ([웹 상세 정보](./WEB.md))
### ☑ Step 8: 라이브 공격 시나리오 스크립트 및 고성능 3D 비주얼 고도화
### ☑ Step 9: AIT-LDS v2.0 전체 시나리오 추가 확장 및 모델 최적화 ([모델 상세 정보](./MODEL.md))
- **LOOCV Macro-F1 `0.7491 ± 0.0336`**, **Weighted-F1 `0.9838 ± 0.0086`**, **고위험 관계 Recall `0.9173`** 달성.
- **Val Macro-F1 `0.8268`**, **Weighted-F1 `0.9940`**
### ☑ Step 10: `false_positive_test` 및 VM 정상 구간 기반 임계치 고도화 ([웹 상세 정보](./WEB.md))

### ☐ Step 11: [선택] CAM-LDS (2026) 통합 및 도메인 강건성 검증
- **데이터셋 통합 및 피처 스펙 정합:**
  - `include_supplementary_logs: false` 설정으로 순정 `auditd` 파서를 단독 구동하고, `configs/taxonomy.yaml`에 CAM-LDS **13 Tactics / 81 Techniques** 매핑 추가(`MODEL.md` 기준 엔티티별 68/36/20-dim, 엣지 2-dim 유지).
- **도메인 전이 검증:**
  - CAM-LDS 정상 구간 대상 Conformal Calibration 재적합 기반 **Zero-Shot Anomaly Transfer** 및 13-Tactic 분류 헤드 **Few-Shot Fine-Tuning** 성능 검증.

---

## 6. 라이브 시연 실행 방법 (WSL2)
1. Window A — 백엔드 + 3D 관제 대시보드 실행 (http://127.0.0.1:5000):
   ```bash
   cd /mnt/c/Users/dongmin/Desktop/KoTon
   scripts/start_ubuntu_vm.sh
   KOTON_PROFILE_MODE=vm .venv/bin/python web/app.py
   ```
2. Window B — 격리 Ubuntu VM 접속 및 5단계 시나리오 실행:
   ```bash
   ssh -i ~/.koton_vm/id_ed25519 -p 2222 ubuntu@127.0.0.1
   /home/ubuntu/demo_attack.sh
   ```