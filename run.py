"""Interactive runner for the cafe reinforcement-learning project."""

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from stable_baselines3.common.env_checker import check_env

import check
import env as env_module
import simulate
import train

# 경로·이름 설정 (검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다)
RUNS_DIR = Path("results/runs")
HISTORY_PATH = Path("results/run_history.csv")
HISTORY_COLUMNS = ["date", "run_type", "success_metric", "memo"]
SANITY_RUN_NAME = "sanity_check"
TRAIN_TEST_RUN_NAME = "train_v1_test"
TRAIN_MAIN_RUN_NAME = "train_v1_main"
EVAL_V1_RUN_NAME = "eval_v1"

ENV_ASSUMPTION_NAMES = (
    "BASE_BUY_RATE",
    "RAINY_DISCOUNT_SHIFT_FROM_DELIVERY",
    "RAINY_DISCOUNT_SHIFT_FROM_NONBUY",
    "BUSY_CINEMA_DAILY_PROB",
    "PERSONA_SCALE_FACTOR",
)
SNAPSHOT_FILES = (Path("config.py"), Path("reward.py"))


def make_unique_run_dir(name: str) -> tuple[str, Path]:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    base = f"{now.strftime('%Y%m%d_%H%M%S')}_{name}"
    run_dir = RUNS_DIR / base
    seq = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{base}_{seq:02d}"
        seq += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return date_str, run_dir


def ensure_virtual_data() -> None:
    if not Path("data/personas.csv").exists():
        print("[데이터] data/personas.csv가 없어 fetch_personas.py를 실행합니다.")
        subprocess.run([sys.executable, "fetch_personas.py"], check=True)
    if not Path("data/external_factors.csv").exists():
        print("[데이터] data/external_factors.csv가 없어 fetch_external_factors.py를 실행합니다.")
        subprocess.run([sys.executable, "fetch_external_factors.py"], check=True)
    if not Path("data/customer_traits.csv").exists():
        raise FileNotFoundError(
            "data/customer_traits.csv가 없습니다. 고객 성향 데이터를 먼저 준비해 주세요."
        )


def save_common_snapshot(run_dir: Path) -> None:
    # env.py 가정값 저장
    assumptions = {
        name: getattr(env_module, name)
        for name in ENV_ASSUMPTION_NAMES
        if hasattr(env_module, name)
    }
    (run_dir / "env_assumptions.json").write_text(
        json.dumps(assumptions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # config.py 및 reward.py 사본 저장
    for src in SNAPSHOT_FILES:
        if src.exists():
            shutil.copy2(src, run_dir / src.name)


def save_run_snapshot(run_dir: Path) -> None:
    # 로그와 요약 사본 저장
    if simulate.LOGS_DIR.exists():
        shutil.copytree(simulate.LOGS_DIR, run_dir / "logs")
    if simulate.SUMMARY_PATH.exists():
        shutil.copy2(simulate.SUMMARY_PATH, run_dir / "sanity_check.md")
    save_common_snapshot(run_dir)


def save_train_snapshot(run_dir: Path, summary: dict) -> None:
    for key in ("curve_png_path", "curve_csv_path", "experience_csv_path", "model_zip_path"):
        src = Path(summary[key])
        if src.exists():
            shutil.copy2(src, run_dir / src.name)
    save_common_snapshot(run_dir)


def save_eval_v1_snapshot(run_dir: Path) -> None:
    if simulate.LOGS_DIR.exists():
        shutil.copytree(simulate.LOGS_DIR, run_dir / "logs")
    if simulate.SUMMARY_V1_PATH.exists():
        shutil.copy2(simulate.SUMMARY_V1_PATH, run_dir / simulate.SUMMARY_V1_PATH.name)
    if simulate.MODEL_V1_PATH.exists():
        shutil.copy2(simulate.MODEL_V1_PATH, run_dir / simulate.MODEL_V1_PATH.name)
    save_common_snapshot(run_dir)


def append_run_history(
    date_str: str, run_type: str, success_metric: str, memo: str
) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HISTORY_COLUMNS)
        writer.writerow([date_str, run_type, success_metric, memo])


def run_data_and_sanity_check() -> None:
    print("\n[1/4] 가상 데이터 존재 여부 및 성향 데이터 무결성 검사...")
    ensure_virtual_data()
    if check.main() != 0:
        raise RuntimeError("check.py 검사에서 오류가 발견되었습니다.")

    print("\n[2/4] Gymnasium 환경 규격 검사(check_env)...")
    check_env(env_module.CafeDiscountEnv(), warn=True)
    print("check_env: PASS")

    print("\n[3/4] 시험 규칙 4종 시뮬레이션 실행(simulate.py)...")
    results = simulate.main()

    by_key = {r["key"]: r["avg_30d_profit"] for r in results}
    metric_summary = (
        f"custom={by_key['custom_busy_aware']:,.0f}원 | "
        f"do_nothing={by_key['do_nothing']:,.0f}원 | "
        f"always_max={by_key['always_max']:,.0f}원"
    )

    print("\n[4/4] 실행 스냅샷 폴더 및 이력(run_history.csv) 저장...")
    date_str, run_dir = make_unique_run_dir(SANITY_RUN_NAME)
    save_run_snapshot(run_dir)
    append_run_history(
        date_str=date_str,
        run_type="1. 가상 데이터 생성과 환경 타당성 검증",
        success_metric=metric_summary,
        memo=f"folder={run_dir.as_posix()} | check=PASS | check_env=PASS",
    )
    print(f"\n완료! 실행 기록 폴더: {run_dir}")
    print(f"이력 파일: {HISTORY_PATH}")


def run_train_menu(mode: str) -> None:
    is_test = mode == "test"
    steps = train.TEST_TIMESTEPS if is_test else train.MAIN_TIMESTEPS
    label = (
        "2. 정책 학습 (시험 학습 v1)" if is_test else "3. 정책 학습 (본 학습 v1)"
    )
    run_name = TRAIN_TEST_RUN_NAME if is_test else TRAIN_MAIN_RUN_NAME

    print(f"\n[1/2] {label} 실행 ({steps:,} 스텝)...")
    summary = train.run_training(mode=mode, total_timesteps=steps)

    print("\n[2/2] 실행 스냅샷 폴더 및 이력(run_history.csv) 저장...")
    date_str, run_dir = make_unique_run_dir(run_name)
    save_train_snapshot(run_dir, summary)
    metric_summary = (
        f"train_last50={summary['last_avg_profit_won']:,.0f}원 | "
        f"steps={summary['actual_timesteps']:,} | "
        f"time={summary['elapsed_sec']:.1f}s"
    )
    append_run_history(
        date_str=date_str,
        run_type=label,
        success_metric=metric_summary,
        memo=(
            f"folder={run_dir.as_posix()} | algo={summary['algorithm']} | "
            f"device={summary['device']}"
        ),
    )
    print(f"\n완료! 실행 기록 폴더: {run_dir}")
    print(f"이력 파일: {HISTORY_PATH}")


def run_eval_v1_menu() -> None:
    print("\n[1/2] 학습된 v1 정책과 시험 규칙 4종 비교 평가 (2025년 36판)...")
    results = simulate.evaluate_v1()
    by_key = {r["key"]: float(r["avg_30d_profit"]) for r in results}
    metric_summary = (
        f"ppo_v1={by_key['ppo_v1']:,.0f}원 | "
        f"custom={by_key['custom_busy_aware']:,.0f}원 | "
        f"do_nothing={by_key['do_nothing']:,.0f}원"
    )

    print("\n[2/2] 실행 스냅샷 폴더 및 이력(run_history.csv) 저장...")
    date_str, run_dir = make_unique_run_dir(EVAL_V1_RUN_NAME)
    save_eval_v1_snapshot(run_dir)
    append_run_history(
        date_str=date_str,
        run_type="4. 학습된 정책 평가 (v1)",
        success_metric=metric_summary,
        memo=f"folder={run_dir.as_posix()} | model={simulate.MODEL_V1_PATH.as_posix()}",
    )
    print(f"\n완료! 실행 기록 폴더: {run_dir}")
    print(f"이력 파일: {HISTORY_PATH}")


def main() -> None:
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        print("=== 강화학습 프로젝트 실행 메뉴 ===")
        print("1. 가상 데이터 생성과 환경 타당성 검증")
        print("2. 정책 학습 (시험 학습 - 1분 안쪽)")
        print("3. 정책 학습 (본 학습 - 307,200스텝)")
        print("4. 학습된 정책 평가 (v1 vs 시험 규칙 4종)")
        print("0. 종료")
        choice = input("번호를 입력하세요: ").strip()

    if choice == "1":
        run_data_and_sanity_check()
    elif choice == "2":
        run_train_menu("test")
    elif choice == "3":
        run_train_menu("main")
    elif choice == "4":
        run_eval_v1_menu()
    elif choice == "0":
        print("종료합니다.")
    else:
        print(f"지원하지 않는 번호입니다: {choice}")


if __name__ == "__main__":
    main()
