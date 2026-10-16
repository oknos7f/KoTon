# KoTon (코톤): Real-Time GNN Provenance Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Heterogeneous%20GNN%20(HGT%2FRGCN)-orange.svg)](https://pyg.org/)
[![DARPA TC](https://img.shields.io/badge/DARPA%20TC-Chronological%20Validation-green.svg)](https://github.com/darpa-i2o/transparent-computing)
[![Low FPR](https://img.shields.io/badge/FPR-%E2%89%A4%200.1%25%20(Calibrated)-red.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

KoTon(코톤)은 **이종 그래프 신경망(Heterogeneous GNN - HGT / RGCN)** 기반 실시간 Linux 호스트 침입 탐지 및 인과 관계 포렌식 시스템입니다. WSL2 내부의 Linux `auditd` 로그를 스트리밍 형태로 수집·분석하여 Ultra-low FPR(목표 FPR &le; 0.1%) 수준으로 악성 행위를 탐지하고, WebGL 기반 **3D Force-Directed Causal Graph**로 공격 킬체인을 실시간 시각화합니다.

---

## 1. System Architecture Overview

```
[ WSL2 Linux Environment ]
│
├── 1. Linux auditd (System Call Stream: execve, connect, openat, fork)
│      │
│      ▼
├── 2. Provenance Graph Engine (PyTorch Geometric HeteroData)
│      ├── Temporal Sliding Window (30s duration, 2s slide)
│      └── Stateful Entity Cache (Process, File, Socket persistence)
│      │
│      ▼
├── 3. HGT / RGCN Anomaly Detector
│      ├── DARPA TC Chronological Training (No Temporal Leakage)
│      ├── Weighted Focal Loss (Penalty on False Alarms)
│      ├── Low-FPR Automated Threshold Calibration (FPR <= 0.1%)
│      └── Kill-Chain Causal Attention Extraction
│      │
│      ▼ (Server-Sent Events: SSE)
└── 4. Flask Demo Backend Server
       │
       ▼ (http://localhost:5000)

[ Windows Web Browser ]
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Window 1: Target App    │ Window 2: Web Shell     │ Window 3: SOC Dashboard │
│ ─────────────────────── │ ─────────────────────── │ ─────────────────────── │
│ • File Upload Board     │ • RCE Commands          │ • 3D Force Graph        │
│ • User Login & Profile  │ • Credential Dumping    │ • Real-time SSE Feed    │
│ • Normal/Vuln Vectors   │ • Dropper & PrivEsc     │ • Glowing Red KillChain │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 2. 3-Window Live Demo Setup

| Window | URL | Role | Key Features |
| :--- | :--- | :--- | :--- |
| **통합 분할 화면** | `http://localhost:5000/split` | **3-in-1 Unified Split** | 3개 창을 단일 모니터에 나란히 배치하여 실시간 상호작용 모니터링 |
| **메인 허브** | `http://localhost:5000/` | **Navigation Hub** | 시스템 사양 및 각 창 개별 열기 포털 |
| **창 1 (방어자/서비스)** | `http://localhost:5000/app` | **Victim Web App** | 사용자 로그인, 프로필 관리, 파일 업로드 게시판, 정상 웹 트래픽 및 취약점 업로드 시뮬레이션 |
| **창 2 (공격자 터미널)** | `http://localhost:5000/terminal` | **Attacker WebShell** | Kali Linux 웹셸: 1. 정찰(`whoami`, `id`), 2. 자격증명 탈취(`cat /etc/passwd`), 3. 드로퍼(`curl stage2.elf`), 4. 권한 상승(SUID/Reverse Shell) |
| **창 3 (관제 대시보드)** | `http://localhost:5000/dashboard` | **SOC 3D Causal Graph** | Three.js WebGL 다크 모드 3D 그래프: Process(🔵 Cyan), File(🟢 Green), Socket(🟠 Orange), 붉은색 광원 펄스(🔴) 및 인과 킬체인 파티클 애니메이션, MITRE ATT&CK 포렌식 드로어, 실시간 민감도 슬라이더 |

---

## 3. Quickstart Guide

### 1. 가상환경 활성화 (Virtual Environment)
```bash
# Windows PowerShell
.\.venv\Scripts\activate

# Linux / WSL2
source .venv/bin/activate
```

### 2. GNN 모델 학습 및 FPR 캘리브레이션 (DARPA TC 기준)
엄격한 시간순(Chronological) 분할로 시간적 정보 누출(Temporal Leakage)을 차단하고, 목표 FPR &le; 0.1% 임계값을 자동 도출합니다.
```bash
# HGT (Heterogeneous Graph Transformer) 학습
python scripts/train_darpa_tc.py --arch hgt --epochs 5 --batch-size 16

# RGCN (Relational GCN) 학습
python scripts/train_darpa_tc.py --arch rgcn --epochs 5 --batch-size 16
```

### 3. 실시간 3화면 데모 서버 실행
```bash
# Windows (Python 또는 배치 파일)
python scripts/run_demo.py
# 또는
.\scripts\run_demo.bat

# Linux / WSL2 (auditd 자동 연동)
bash scripts/run_demo.sh
```

브라우저에서 `http://localhost:5000/split`에 접속하여 3개의 화면을 한눈에 보면서:
1. 창 1에서 "+ Simulate Benign Web Traffic" 클릭 &rarr; 정상 웹 트래픽 유입 (FPR 0.00% 유지)
2. 창 2에서 `>> RUN FULL KILL-CHAIN <<` 클릭 또는 개별 명령어 입력
3. 창 3에서 20ms 이내에 실시간 붉은색 킬체인 점등 및 MITRE ATT&CK 경보 확인!

### 4. 전체 파이프라인 자동화 테스트 실행
```bash
python tests/test_e2e_pipeline.py
```

---

## 4. Core Technical Components

### 4.1 Provenance Graph Engine
- **Entity & Relation Representation**:
  - **Process**: PID, Binary Path, Command Line, UID (16-dim feature vector)
  - **File**: Filepath, Permissions, Extension, Directory Depth (12-dim feature vector)
  - **Network Socket**: Remote IP/Port, Ephemeral/C2 Port Flag (8-dim feature vector)
  - **Relations**: `(Process, fork, Process)`, `(Process, exec, Process)`, `(Process, read/write, File)`, `(Process, connect, NetworkSocket)`
- **Temporal Sliding Window & Entity Caching**:
  - 30초 윈도우 기반 실시간 그래프 스냅샷 생성. 장기 실행 프로세스 및 활성 소켓이 윈도우 경계에서 고립(Orphaned)되지 않도록 `MAX_CACHE_ENTITIES` 상태 캐시 유지.

### 4.2 Heterogeneous GNN (PyTorch Geometric)
- **Model Architectures**:
  - `HeteroProvenanceGNN`: 다형적 노드 및 엣지 어텐션을 학습하는 `HGTConv` 기반
  - `HeteroRGCNProvenanceGNN`: 관계형 메시지 패싱 기반 `HeteroConv` + `SAGEConv`
- **Low-FPR Optimization**:
  - **Weighted Focal Loss**: 대다수 정상 샘플의 그래디언트를 억제하고 False Alarm에 높은 페널티 부여
  - **Automated Threshold Calibration**: 검증 세트 기준 목표 FPR &le; 0.1%를 달성하는 최적 임계값 자동 산출
  - **Dynamic Sensitivity Slider**: 웹 UI 슬라이더로 런타임에 0.10 ~ 0.95 임계값 즉각 제어

### 4.3 Live Ingestion & SSE Pipeline
- **WSL2 Audit Ingest**: `LinuxAuditWatcher`가 `/var/log/audit/audit.log`를 실시간 tailing하여 시스템 콜(`execve`, `connect`, `openat`) 파싱.
- **Server-Sent Events (SSE)**: 지연 없이 그래프 업데이트, 어텐션 가중치, 위협 경보를 3D 프론트엔드로 브로드캐스트.
