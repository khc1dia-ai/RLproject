# 따뜻한 아이스 아메리카노의 만족감 — 강화학습 카페 운영 시뮬레이션

경기도 시흥시 정왕역 1번 출구 인근 카페 매장을 모델로, **비 오는 날 포장 할인 폭(0% · 5% · 10%)**을 결정해 **30일 누적 주문별 이익**을 최대화하는 강화학습 프로젝트입니다.  
프로젝트의 모든 설계 기준, 가정값, 변경 이력은 [`project_spec.md`](./project_spec.md)를 기준으로 관리합니다.

## 1. 프로젝트 소개

- **매장:** 따뜻한 아이스 아메리카노의 만족감 (경기도 시흥시 정왕역 1번 출구 인근 카페, 영업 시간 매일 `07:00 ~ 22:00`)
- **반복되는 결정:** 비·눈 오는 날(`weather_score = 1`) 하루 한 번 포장 할인 폭 `0%`(0원), `5%`(100원), `10%`(200원) 중 하나를 선택
- **성공 척도(목적):** 30일 누적 주문별 이익 최대화  
  `실제 결제 매출(정상가 − 할인액) − 잔당 원가(정상가 기준 50%) − 배달 수수료(배달 매출의 30%)` (고정 임대료·고정급·공과금 제외)
- **핵심 트레이드오프:** 비 오는 날 할인을 높이면 배달 수요가 이익률이 높은 포장으로 이동하지만, 하루 처리 한도(`265잔`, 가정)를 크게 넘기면 제시간 제공률과 매장 만족도가 떨어져 다음 날 손님이 줄어듭니다.
- **시각화:** [`public/index.html`](./public/index.html)에서 정왕역 1번 출구 주변 동네와 매장 내부를 아이소메트릭 화면으로 재생하며 30일 운영 결과를 눈으로 확인할 수 있습니다.

## 2. 다른 PC에서 설치하는 방법 (`uv sync`)

1. **사전 준비**
   - **Python 3.12** (`pyproject.toml`에 시스템 파이썬 우선·없으면 `uv` 자동 설치 설정이 들어 있습니다.)
   - **uv** 설치 (Windows PowerShell 기준):
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
2. **저장소 내려받기 및 패키지 설치**
   ```powershell
   git clone https://github.com/khc1dia-ai/RLproject.git
   cd RLproject
   uv sync
   ```
   - `uv sync`를 실행하면 `pyproject.toml`과 `uv.lock`에 맞춰 `.venv` 가상 환경과 필요한 라이브러리(`gymnasium`, `stable-baselines3`, `torch`, `duckdb`, `openmeteo-requests`, `holidays` 등)가 자동으로 설치됩니다.
   - 참고: `pyproject.toml`은 NVIDIA CUDA 12.8용 PyTorch(`pytorch-cu128`)를 기준으로 설정되어 있습니다.

## 3. 실행 방법

### 1) 메뉴 기반 실행 (`run.bat` / `run.py`)
- **더블클릭 실행 (Windows):** 프로젝트 폴더의 `run.bat`을 더블클릭한 뒤 원하는 메뉴 번호를 입력합니다.
- **터미널 한 줄 실행:**
  ```powershell
  .\run.bat <번호>
  ```
  또는
  ```powershell
  uv run python run.py <번호>
  ```
- **지원 메뉴 (`run.py` 기준):**
  - `1`: **가상 데이터 생성과 환경 타당성 검증** — 성향 무결성 검사(`check.py`) → 환경 규격 검사(`check_env`) → 시험 규칙 4종 36판 평가(`simulate.py`) → `results/sanity_check.md` 저장
  - `2`: **정책 학습 (시험 학습 - 1분 안쪽)** — `20,480`스텝 시험 학습(`train.py --mode test`, 본 학습 모델 `v1.zip`은 덮어쓰지 않음)
  - `3`: **정책 학습 (본 학습 - 307,200스텝)** — `PPO` / `CPU`로 `2016~2024년` 본 학습(`train.py --mode main`) → `results/models/v1.zip`, `results/learning_curve_v1.png/.csv`, `results/experience_sample_v1.csv` 저장
  - `4`: **학습된 정책 평가 (v1 vs 시험 규칙 4종)** — 2025년 36판 기준 비교 평가(`simulate.py --eval-v1`) → `results/summary_v1.md`, `results/logs/`, `public/data/` 저장
  - `0`: 종료
- 모든 실행(`1~4`)은 실행별 스냅샷(`results/runs/`)과 누적 이력(`results/run_history.csv`)을 같은 방식으로 저장합니다.

### 2) 매장 재생 화면(`public/index.html`) 확인
로컬 웹 서버를 띄운 뒤 브라우저에서 열어 확인합니다.
```powershell
uv run python -m http.server 8000 --directory public
```
브라우저 주소창에 `http://localhost:8000/` 을 입력하면 매장 재생 화면이 열립니다.

## 4. 폴더 구성

```text
c:\Myproject\
├── AGENTS.md                  # 프로젝트 작업 규칙 및 코딩 AI 사용 원칙
├── project_spec.md            # 설계 기준 문서 (매장 정의·반응 규칙·성향·설계 카드·변경 기록)
├── screen_design.md           # 매장 재생 화면 디자인 기준
├── README.md                  # 프로젝트 안내 문서
├── pyproject.toml / uv.lock   # 파이썬 의존성 및 고정 버전 정보
├── config.py                  # 매장·수요·만족도·평가 설정값
├── reward.py                  # 보상 함수 v1 (하루 주문별 이익 ÷ 100,000원, 가정)
├── env.py                     # Gymnasium 가상 매장 환경 (CafeDiscountEnv)
├── train.py                   # PPO 학습 스크립트 (시험 학습·본 학습 및 곡선·경험 저장)
├── simulate.py                # 시험 규칙 4종·v1 정책 비교 평가 및 재생 JSON 생성
├── check.py                   # 고객 성향 데이터 무결성·평균 검증
├── run.bat / run.py           # 메뉴 기반 일괄 실행기 (1~4번) 및 스냅샷 저장
├── fetch_personas.py          # 고객 페르소나 표본 추출 스크립트
├── fetch_external_factors.py  # 과거 날씨·공휴일 수집 스크립트
├── data/                      # 가상 데이터 (personas.csv, customer_traits.csv, external_factors.csv)
├── public/
│   ├── index.html             # 아이소메트릭 매장 재생 화면 (단일 HTML, Canvas 2D)
│   └── data/                  # 화면 재생용 JSON (manifest, personas_summary, replay_*.json)
└── results/
    ├── sanity_check.md        # 시험 규칙 4종 비교 요약표
    ├── summary_v1.md          # v1 정책 vs 시험 규칙 4종 2025년 평가 요약표
    ├── learning_curve_v1.png  # v1 본 학습 곡선 그래프 (및 learning_curve_v1.csv)
    ├── experience_sample_v1.csv # 학습 중 처음 1,000스텝 경험 표본
    ├── run_history.csv        # 실행 이력 누적표
    ├── models/                # 학습된 정책 모델 (v1.zip)
    ├── logs/                  # 규칙·정책별 일자 로그 CSV (Git 포함)
    ├── screenshots/           # 화면 확인 스크린샷
    └── runs/                  # 실행별 스냅샷 폴더 (Git 제외)
```

> **보안 안내 (`.gitignore` 제외 대상):** `.venv/`, 파이썬 임시 폴더(`__pycache__/` 등), `results/runs/`, 실제 매장 데이터 원본 폴더(`data/raw/`, `data/real/` 등), API 키·비밀번호 파일(`.env` 등)은 Git에 올라가지 않습니다.

## 5. 데이터 출처

- **고객 페르소나 표본 (`data/personas.csv`):** Hugging Face [`nvidia/Nemotron-Personas-Korea`](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)  
  (`district = '경기-시흥시'`, 연령 `20~49세`, 직장인·학생·영화 관심 조건 4,126명 중 시드 `42`로 100명 추출)
- **과거 날씨 데이터 (`data/external_factors.csv`):** [Open-Meteo 과거 날씨 API](https://archive-api.open-meteo.com/v1/archive)  
  (경기도 시흥시 정왕역 일대 위도 `37.3517`, 경도 `126.7429`, `2016-01-01 ~ 2025-12-31`)
- **한국 공휴일 데이터 (`data/external_factors.csv`):** Python [`holidays`](https://pypi.org/project/holidays/) 라이브러리 (`holidays.KR`, 2016~2025년)
- **코딩 AI 사용 원칙 (`AGENTS.md`):** [`forrestchang/andrej-karpathy-skills`](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md)
