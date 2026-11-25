"""PPO training script for CafeDiscountEnv (v1 reward)."""

import argparse
import csv
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import config
from env import CafeDiscountEnv
import reward as reward_module

# 학습 설정 및 출력 경로 (가정값은 검토와 수정이 쉽도록 맨 위에 모아 둡니다)
ALGORITHM_NAME = "PPO"
DEVICE = "cpu"
POLICY_NAME = "MlpPolicy"

TEST_TIMESTEPS = 20_480  # 가정 (1분 안쪽 시험 학습용: PPO 10회 갱신, 약 682판)
MAIN_TIMESTEPS = 307_200  # 가정 (본 학습 기본값: 2,048 × 150회, 실측 기준 약 3분 25초로 10분 안쪽)
EXPERIENCE_SAMPLE_STEPS = 1_000
ROLLING_WINDOW_EPISODES = 50  # 가정 (학습 곡선 이동평균 판 수)

PPO_N_STEPS = 2_048  # 가정 (PPO 기본값)
PPO_BATCH_SIZE = 64  # 가정 (PPO 기본값)
PPO_LEARNING_RATE = 3e-4  # 가정 (PPO 기본값)

CURVE_PNG_PATH = Path("results/learning_curve_v1.png")
CURVE_CSV_PATH = Path("results/learning_curve_v1.csv")
MODEL_PATH = Path("results/models/v1")
EXPERIENCE_CSV_PATH = Path("results/experience_sample_v1.csv")


class TrainingRecordWrapper(gym.Wrapper):
    """학습 중 판별 성적과 처음 1,000스텝 경험을 기록하는 래퍼."""

    def __init__(self, env: CafeDiscountEnv, max_experience_steps: int) -> None:
        super().__init__(env)
        self.max_experience_steps = max_experience_steps
        self.start_time = time.perf_counter()
        self.total_steps = 0
        self.episode_idx = 0
        self.current_start_date = ""
        self.last_obs: np.ndarray | None = None
        self.ep_reward = 0.0
        self.ep_profit = 0.0
        self.episode_records: list[dict[str, Any]] = []
        self.experience_rows: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        start_date = str(info.get("start_date", ""))
        if not (
            config.TRAIN_EPISODE_START_MIN
            <= start_date
            <= config.TRAIN_EPISODE_START_MAX
        ):
            raise ValueError(f"학습 기간 밖 시작일이 선택되었습니다: {start_date}")
        self.current_start_date = start_date
        self.last_obs = obs.copy()
        self.ep_reward = 0.0
        self.ep_profit = 0.0
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_obs = self.last_obs
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_steps += 1
        self.ep_reward += float(reward)
        self.ep_profit += float(info["daily_profit"])

        if prev_obs is not None and len(self.experience_rows) < self.max_experience_steps:
            self.experience_rows.append(
                {
                    "step": self.total_steps,
                    "episode": self.episode_idx + 1,
                    "date": info["date"],
                    "s_sat_norm": round(float(prev_obs[0]), 4),
                    "s_days_left_norm": round(float(prev_obs[1]), 4),
                    "s_weather": int(round(float(prev_obs[2]))),
                    "s_weekend": int(round(float(prev_obs[3]))),
                    "s_holiday": int(round(float(prev_obs[4]))),
                    "s_busy_cinema": int(round(float(prev_obs[5]))),
                    "action": int(action),
                    "applied_discount_rate": round(
                        float(info["applied_discount_rate"]), 4
                    ),
                    "reward": round(float(reward), 6),
                    "next_sat_norm": round(float(next_obs[0]), 4),
                    "next_days_left_norm": round(float(next_obs[1]), 4),
                    "next_weather": int(round(float(next_obs[2]))),
                    "next_weekend": int(round(float(next_obs[3]))),
                    "next_holiday": int(round(float(next_obs[4]))),
                    "next_busy_cinema": int(round(float(next_obs[5]))),
                    "terminated": int(bool(terminated)),
                    "daily_profit_won": round(float(info["daily_profit"]), 1),
                    "total_orders": round(float(info["total_orders"]), 1),
                    "ontime_rate": round(float(info["ontime_rate"]), 4),
                }
            )

        self.last_obs = next_obs.copy()

        if terminated or truncated:
            self.episode_idx += 1
            self.episode_records.append(
                {
                    "episode": self.episode_idx,
                    "timesteps": self.total_steps,
                    "start_date": self.current_start_date,
                    "episode_reward": round(self.ep_reward, 6),
                    "episode_profit_won": round(self.ep_profit, 1),
                    "final_satisfaction": round(float(info["satisfaction"]), 1),
                    "elapsed_sec": round(time.perf_counter() - self.start_time, 3),
                }
            )

        return next_obs, reward, terminated, truncated, info


def save_experience_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curve_csv(
    records: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rewards = np.array([r["episode_reward"] for r in records], dtype=np.float64)
    profits = np.array([r["episode_profit_won"] for r in records], dtype=np.float64)

    enriched: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        lo = max(0, i - ROLLING_WINDOW_EPISODES + 1)
        row = dict(rec)
        row["rolling_mean_reward"] = round(float(np.mean(rewards[lo : i + 1])), 6)
        row["rolling_mean_profit_won"] = round(float(np.mean(profits[lo : i + 1])), 1)
        enriched.append(row)

    fieldnames = [
        "episode",
        "timesteps",
        "start_date",
        "episode_reward",
        "episode_profit_won",
        "rolling_mean_reward",
        "rolling_mean_profit_won",
        "final_satisfaction",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    return enriched


def save_learning_curve_png(
    enriched: list[dict[str, Any]],
    path: Path,
    mode_label: str,
    total_timesteps: int,
    elapsed_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    xs = [r["timesteps"] for r in enriched]
    ep_rewards = [r["episode_reward"] for r in enriched]
    roll_rewards = [r["rolling_mean_reward"] for r in enriched]
    ep_profits = [r["episode_profit_won"] / 10_000.0 for r in enriched]
    roll_profits = [r["rolling_mean_profit_won"] / 10_000.0 for r in enriched]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"PPO 학습 곡선 ({reward_module.REWARD_VERSION}, {mode_label} | "
        f"{total_timesteps:,} 스텝, {elapsed_sec:.1f}초, {DEVICE.upper()})",
        fontsize=12,
    )

    ax1.plot(xs, ep_rewards, color="#93c5fd", alpha=0.45, linewidth=0.8, label="판별 보상")
    ax1.plot(
        xs,
        roll_rewards,
        color="#1d4ed8",
        linewidth=2.0,
        label=f"최근 {ROLLING_WINDOW_EPISODES}판 이동평균",
    )
    ax1.set_ylabel("30일 누적 보상 (1 = 10만 원)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.plot(xs, ep_profits, color="#86efac", alpha=0.45, linewidth=0.8, label="판별 이익")
    ax2.plot(
        xs,
        roll_profits,
        color="#15803d",
        linewidth=2.0,
        label=f"최근 {ROLLING_WINDOW_EPISODES}판 이동평균",
    )
    ax2.set_xlabel("학습 스텝 수")
    ax2.set_ylabel("30일 누적 이익 (만 원)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _output_paths(mode: str) -> tuple[Path, Path, Path, Path]:
    if mode == "test":
        return (
            Path("results/learning_curve_v1_test.png"),
            Path("results/learning_curve_v1_test.csv"),
            Path("results/models/v1_test"),
            Path("results/experience_sample_v1_test.csv"),
        )
    return CURVE_PNG_PATH, CURVE_CSV_PATH, MODEL_PATH, EXPERIENCE_CSV_PATH


def run_training(mode: str, total_timesteps: int) -> dict[str, Any]:
    curve_png_path, curve_csv_path, model_path, exp_csv_path = _output_paths(mode)
    base_env = CafeDiscountEnv(start_date=None)
    wrapped_env = TrainingRecordWrapper(
        base_env, max_experience_steps=EXPERIENCE_SAMPLE_STEPS
    )

    model = PPO(
        POLICY_NAME,
        wrapped_env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        seed=config.TRAIN_SEED,
        device=DEVICE,
        verbose=0,
    )

    t0 = time.perf_counter()
    wrapped_env.start_time = t0
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed_sec = time.perf_counter() - t0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    save_experience_csv(wrapped_env.experience_rows, exp_csv_path)
    enriched = save_learning_curve_csv(wrapped_env.episode_records, curve_csv_path)
    mode_label = "시험 학습" if mode == "test" else "본 학습"
    save_learning_curve_png(
        enriched, curve_png_path, mode_label, wrapped_env.total_steps, elapsed_sec
    )

    actual_steps = wrapped_env.total_steps
    steps_per_sec = actual_steps / elapsed_sec if elapsed_sec > 0 else 0.0
    ms_per_step = (elapsed_sec / actual_steps * 1000.0) if actual_steps > 0 else 0.0

    first_avg_profit = (
        float(
            np.mean(
                [
                    r["episode_profit_won"]
                    for r in enriched[: min(ROLLING_WINDOW_EPISODES, len(enriched))]
                ]
            )
        )
        if enriched
        else 0.0
    )
    last_avg_profit = (
        float(
            np.mean(
                [
                    r["episode_profit_won"]
                    for r in enriched[-min(ROLLING_WINDOW_EPISODES, len(enriched)) :]
                ]
            )
        )
        if enriched
        else 0.0
    )

    summary = {
        "mode": mode,
        "algorithm": ALGORITHM_NAME,
        "device": DEVICE,
        "train_period": f"{config.TRAIN_DATA_START_DATE} ~ {config.TRAIN_DATA_END_DATE}",
        "requested_timesteps": total_timesteps,
        "actual_timesteps": actual_steps,
        "episodes": len(enriched),
        "elapsed_sec": elapsed_sec,
        "steps_per_sec": steps_per_sec,
        "ms_per_step": ms_per_step,
        "first_avg_profit_won": first_avg_profit,
        "last_avg_profit_won": last_avg_profit,
        "experience_rows": len(wrapped_env.experience_rows),
        "model_zip_path": str(model_path.with_suffix(".zip")),
        "curve_png_path": str(curve_png_path),
        "curve_csv_path": str(curve_csv_path),
        "experience_csv_path": str(exp_csv_path),
    }

    print(f"=== {mode_label} 완료 ({ALGORITHM_NAME} / {DEVICE.upper()}) ===")
    print(f"- 학습 기간: {summary['train_period']} (시드 {config.TRAIN_SEED})")
    print(f"- 실행 스텝: {actual_steps:,} 스텝 ({len(enriched):,}판)")
    print(
        f"- 걸린 시간: {elapsed_sec:.2f}초 "
        f"(스텝당 {ms_per_step:.3f}ms, 초당 {steps_per_sec:,.0f}스텝)"
    )
    print(
        f"- 처음 {min(ROLLING_WINDOW_EPISODES, len(enriched))}판 평균 30일 이익: "
        f"{first_avg_profit:,.0f}원"
    )
    print(
        f"- 마지막 {min(ROLLING_WINDOW_EPISODES, len(enriched))}판 평균 30일 이익: "
        f"{last_avg_profit:,.0f}원"
    )
    print(f"- 저장 모델: {model_path}.zip")
    print(f"- 학습 곡선: {curve_png_path}, {curve_csv_path}")
    print(
        f"- 경험 표본: {exp_csv_path} ({len(wrapped_env.experience_rows):,}스텝)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CafeDiscountEnv PPO 학습 스크립트")
    parser.add_argument(
        "--mode",
        choices=("test", "main"),
        default="test",
        help="test: 1분 안쪽 시험 학습, main: 본 학습",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="학습 스텝 수 직접 지정 (생략 시 모드 기본값 사용)",
    )
    args = parser.parse_args()
    steps = (
        args.timesteps
        if args.timesteps is not None
        else (TEST_TIMESTEPS if args.mode == "test" else MAIN_TIMESTEPS)
    )
    run_training(mode=args.mode, total_timesteps=steps)


if __name__ == "__main__":
    main()
