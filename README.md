# KoTon — Real-Time Provenance Graph Intrusion Detection System

## 1. 프로젝트 개요 및 현재 목표
**KoTon**은 Linux 커널의 `auditd` 로그를 실시간 이종 Provenance Graph로 변환하고, Multi-Task Heterogeneous GNN을 통해 알 수 없는 이상 행위(Anomaly)와 알려진 공격 단계(Multi-Class)를 동시에 탐지하는 보안 관제 시스템이다.

---

## 2. 데이터셋 명세, 보존 원칙

### 메인 데이터셋: AIT-LDS v2.0
Linux 네이티브 auditd 로그와 다중 클래스 GT을 제공하는 AIT-LDS v2.0을 메인 데이터셋으로 채택
- 선정 사유: 실제 엔터프라이즈 가상 환경에서 수집된 8개의 시나리오로 구성되며, OS 커널의 순정 auditd 로그를 그대로 포함합니다.
- 공격 클래스 다양성: 단순 이진 탐지를 넘어 Reconnaissance (Nmap/Dirb 스캔), Initial Access (웹셸/익스플로잇), Privilege Escalation (권한 상승), Credential Access (패스워드 크래킹/브루트포스), Exfiltration (데이터 유출) 등 시연 임팩트가 확실한 다중 공격 단계(Multi-stage) 라벨을 모두 포함합니다.

### 추후 확장 데이터셋: CAM-LDS (2026)
- 7개 Linux 공격 시나리오, 순정 `auditd` 로그, **13개 MITRE ATT&CK Tactics** 및 **81개 Techniques** 라벨 제공.
- 추후 코드 수정 없이 결합할 수 있도록 분리 설계함.

---

## 3. 이종 Feature 인코딩
**노드 타입별 고유 차원(Type-Specific Input Dimensions)**을 유지한 뒤 모델 입력단의 타입별 선형 투영(`68/36/20 -> 128-dim`) + `LayerNorm` + `GELU`으로 공통 128차원 잠재 공간에 정합시킨다. 절대 타임스탬프, 호스트명, 임시 PID는 Data Leakage 방지를 위해 Feature에서 제외한다.

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
- **전체 시나리오 통합 전처리 (`data/` 읽기 전용 유지):**
  - `data/` 하위에 추가되는 AIT-LDS v2.0의 나머지 6개 시나리오(`russellmitchell`, `harrison`, `santos`, `shaw`, `wardbeck`, `wilson`)를 기존 파이프라인으로 일괄 전처리하여 `processed/<scenario_name>/`에 직렬화(`.pt`).
  - 시나리오 간 일반화 성능 검증을 위해 **LOOCV**, 시나리오 단위 Train/Val/Test 분할 적용.
- **하이퍼파라미터 및 손실 함수 최적화:**
  - `Optuna`를 통해 최적화.
  - 목표 성능 지표: 
  - 최적 가중치 저장, 추론 속도 최적화.

### ☐ Step 7: `false_positive_test` 분석 및 동적 임계치 산정 (`scripts/calibrate_threshold.py`)
- AIT-LDS v2.0의 순수 정상 구간(`false_positive_test`)에서 Anomaly Score($\mathcal{L}_{recon}$) 분포를 추출하라.
- 백분위수 및 $\mu + k\sigma$ 기반으로 **FPR < 1%**인 `default_threshold`와 데모 슬라이더용 조절 범위(`[min_threshold, max_threshold]`)를 산출해 `configs/threshold_profile.json`에 저장.

### ☐ Step 8: WSL2 기반 실시간 `auditd` 수집 및 추론 엔진 완성 (`src/collector/`, `src/serving/`)
- **WSL2 `auditd` 실시간 수집 데몬 (`src/collector/audit_stream.py`):**
  - 설치된 WSL2 환경에서 시스템 콜을 로깅하도록 `auditctl` 규칙을 구성.
  - audit log를 실시간으로 Tail하여 스트림으로 전달.
- **실시간 추론 엔진 고도화 (`src/serving/engine.py`):**
  - `KoTonInferenceEngine`에 **Sliding Window Graph Buffer** 및 **Online CPR**을 통합.
  - 이벤트 수신 $\rightarrow$ 인코딩 $\rightarrow$ 추론 $\rightarrow$ **Attack Subgraph 추출** $\rightarrow$ 직렬화까지의 End-to-End 지연 시간을 **500ms 이하**로 최적화.

### ☐ Step 9: Flask + Three.js 3D 보안 관제 및 Dual-Window 데모 (`web/`)
- **Flask-SocketIO 백엔드 (`web/app.py`):**
  - `Flask` + `Flask-SocketIO` (비동기 모드: `threading`)를 사용하여 `KoTonInferenceEngine`의 추론 결과를 WebSocket 이벤트(`graph_update`, `attack_alert`)로 프론트엔드에 실시간 반영.
  - **임계치 동적 상호작용 API:** 프론트엔드 슬라이더에서 전송된 신규 임계치(`update_threshold` 이벤트)를 수신하여 서버 재시작 없이 즉시 이상치 판정 기준과 `is_malicious` / `is_attack_path` 플래그를 재계산·반영.
- **Three.js 3D 프론트엔드 (`web/static/`, `web/templates/`):**
  - `Three.js` (또는 Three.js 기반 `3d-force-graph`)로 실시간 3D Provenance Cloud를 렌더링하라.
  - **Dual-Window 데모 레이아웃:** Window A에 3D 관제 대시보드(3D 그래프, 실시간 임계치 조절 슬라이더)를 배치하고, Window B에 공격자 WSL2 터미널 창을 병렬 배치하여 시연 가능하도록 구성.

### ☐ Step 10: 라이브 공격 시나리오 스크립트 및 3D 비주얼 고도화
- **재현 가능한 다중 단계 공격 스크립트 (`scripts/demo_attack.sh`):**
  - WSL2 환경에서 안전하게 실행 가능하며 AIT-LDS v2.0의 공격 패턴을 선택해 실제 실행하는 라이브 시연 스크립트 작성:
    1. `Scan`: 로컬 포트/디렉터리 스캔
    2. `Webshell / RCE`: 역방향 셸 또는 비정상 자식 프로세스 체인 생성
    3. `Privilege Escalation`: 권한 상승 시도 및 `/etc/passwd`·`/etc/shadow` 접근
    4. `Data Exfiltration`: 민감 파일 압축 및 외부 소켓 전송
- **3D 비주얼 고도화 (256x256 PNG 아이콘 기반):**
  - **아이콘 렌더링:** 모든 노드 타입(`Process`, `File`, `Socket`)은 `web/images/`에 배치된 **256×256 PNG 아이콘**(`process.png`, `file.png`, `socket.png`)을 `THREE.TextureLoader`로 사전 로드한 뒤 `THREE.Sprite`로 렌더링. 노드 수가 500~1,500개에 달해도 FPS 저하가 없도록 텍스처 인스턴스를 재사용.
- **정상 vs. 악성 경로 시각화:**
  - 정상 노드와 엣지는 낮은 불투명도의 회색 톤으로 배치하여 시각적 노이즈를 최소화.
  - 탐지된 악성 노드는 아이콘에 **공격 클래스별 컬러 하이라이팅**을 적용(노랑~보라 등 네온 계열)하고, 공격 경로 엣지 위로 **방향성 파티클(Directional Particles)**이 흐르도록 구현하라.
- **Inspector HUD:** 특정 노드 아이콘 마우스 호버 시 해당 객체의 `cmdline`, `path`, `anomaly_score`, `attack_class`를 표시하는 오버레이 HUD를 구현하라.

### ☐ Step 11: [선택] CAM-LDS 데이터셋 통합 및 강건성 검증
- `data/` 폴더에 **CAM-LDS** 데이터셋 추가, 기존 `data/` 보존 원칙을 유지한 채 `configs/taxonomy.yaml`에 CAM-LDS의 13 Tactics / 81 Techniques 매핑 테이블 추가.
- `auditd` 파서를 재사용하여 CAM-LDS를 동일한 84차원 `HeteroData`로 변환하고, AIT-LDS 학습 모델의 Zero-shot 전이 성능 측정 및 Fine-tuning을 통해 모델의 도메인 강건성 검증.