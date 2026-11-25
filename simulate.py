"""Run baseline policy sanity checks on CafeDiscountEnv and export replay JSON."""

import csv
from datetime import date as dt_date
import json
from pathlib import Path
from typing import Callable

import numpy as np

import config
from env import CafeDiscountEnv

# 숫자·경로 설정 (검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다)
RANDOM_POLICY_SEED_OFFSET = 1000  # 가정 (환경 시드와 분리한 무작위 행동 시드)
CUSTOM_RULE_LOW_SAT_CUTOFF = 50.0  # 가정 (혼잡도 반영 규칙: 만족도 50 미만이면 할인 중단)
LOGS_DIR = Path("results/logs")
SUMMARY_PATH = Path("results/sanity_check.md")
PUBLIC_DATA_DIR = Path("public/data")
PERSONAS_PATH = Path("data/personas.csv")
TRAITS_PATH = Path("data/customer_traits.csv")
EXTERNAL_PATH = Path("data/external_factors.csv")
WEEKDAY_LABELS = ("월", "화", "수", "목", "금", "토", "일")


def policy_do_nothing(obs: np.ndarray, rng: np.random.Generator) -> int:
    return 0


def policy_always_max(obs: np.ndarray, rng: np.random.Generator) -> int:
    return 2


def policy_random(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(config.TAKEOUT_DISCOUNT_RATES)))


def policy_custom_busy_aware(obs: np.ndarray, rng: np.random.Generator) -> int:
    """가정 규칙: 비 오는 날 평소 10%, 혼잡 요인 1개면 5%, 2개 이상이거나 만족도 50 미만이면 0%."""
    sat, _days_left, weather, is_wknd, is_hol, is_cin = obs
    if weather < 0.5:
        return 0
    busy_count = int(is_wknd > 0.5) + int(is_hol > 0.5) + int(is_cin > 0.5)
    if sat < CUSTOM_RULE_LOW_SAT_CUTOFF or busy_count >= 2:
        return 0
    if busy_count == 1:
        return 1
    return 2


POLICIES: list[tuple[str, str, Callable[[np.ndarray, np.random.Generator], int]]] = [
    ("do_nothing", "아무것도 안 함 (항상 0%)", policy_do_nothing),
    ("always_max", "매일 최대로 행동 (비 오는 날 10%)", policy_always_max),
    ("random", "무작위 (0%·5%·10% 무작위)", policy_random),
    (
        "custom_busy_aware",
        "내가 정한 규칙 (혼잡도·만족도 반영, 가정)",
        policy_custom_busy_aware,
    ),
]


def load_weather_lookup() -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    with EXTERNAL_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lookup[r["date"]] = {
                "temp_max_c": float(r["temp_max_c"]),
                "temp_min_c": float(r["temp_min_c"]),
                "precipitation_mm": float(r["precipitation_mm"]),
                "snowfall_cm": float(r["snowfall_cm"]),
            }
    return lookup


def export_personas_summary() -> None:
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PERSONAS_PATH.open(encoding="utf-8") as f:
        personas = list(csv.DictReader(f))
    with TRAITS_PATH.open(encoding="utf-8") as f:
        traits = {r["uuid"]: r for r in csv.DictReader(f)}

    summary = []
    for idx, p in enumerate(personas, start=1):
        tr = traits.get(p["uuid"], {})
        summary.append(
            {
                "customer_no": idx,
                "sex": p["sex"],
                "age": int(p["age"]),
                "occupation": p["occupation"],
                "family_type": p["family_type"],
                "takeout_preference": float(tr.get("takeout_preference", 0.55)),
                "cinema_visit_sensitivity": float(
                    tr.get("cinema_visit_sensitivity", 1.0)
                ),
            }
        )

    out_path = PUBLIC_DATA_DIR / "personas_summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def evaluate_policy(
    key: str,
    label: str,
    policy_fn: Callable[[np.ndarray, np.random.Generator], int],
    weather_lookup: dict[str, dict[str, float]] | None = None,
) -> dict[str, float | str]:
    if weather_lookup is None:
        weather_lookup = load_weather_lookup()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{key}.csv"

    ep_profits: list[float] = []
    ep_avg_orders: list[float] = []
    ep_avg_ontime: list[float] = []
    ep_final_sat: list[float] = []
    ep_low_sat_days: list[int] = []
    ep_rainy_discount_rates: list[float] = []
    replay_episodes: list[dict] = []

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_start",
                "seed",
                "date",
                "target",
                "state",
                "action",
                "response",
                "profit",
            ]
        )

        for start_date in config.EVAL_START_DATES:
            env = CafeDiscountEnv(start_date=start_date)
            for seed in config.EVAL_SEEDS:
                obs, _ = env.reset(seed=seed)
                policy_rng = np.random.default_rng(seed + RANDOM_POLICY_SEED_OFFSET)

                total_profit = 0.0
                total_reward = 0.0
                orders_sum = 0.0
                ontime_sum = 0.0
                low_sat_days = 0
                day_records: list[dict] = []
                day_num = 0
                done = False
                info: dict = {}

                while not done:
                    day_num += 1
                    sat, days_left, weather, is_wknd, is_hol, is_cin = obs
                    action = policy_fn(obs, policy_rng)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    total_profit += info["daily_profit"]
                    total_reward += float(reward)
                    orders_sum += info["total_orders"]
                    ontime_sum += info["ontime_rate"]
                    if info["satisfaction"] < config.LOW_SATISFACTION_THRESHOLD:
                        low_sat_days += 1
                    if weather > 0.5:
                        ep_rainy_discount_rates.append(info["applied_discount_rate"])

                    d_str = info["date"]
                    wd = WEEKDAY_LABELS[dt_date.fromisoformat(d_str).weekday()]
                    w_extra = weather_lookup[d_str]

                    day_records.append(
                        {
                            "day": day_num,
                            "date": d_str,
                            "weekday": wd,
                            "weather_score": int(weather),
                            "temp_max_c": w_extra["temp_max_c"],
                            "temp_min_c": w_extra["temp_min_c"],
                            "precipitation_mm": w_extra["precipitation_mm"],
                            "snowfall_cm": w_extra["snowfall_cm"],
                            "is_weekend": int(is_wknd),
                            "is_holiday": int(is_hol),
                            "is_busy_cinema": int(is_cin),
                            "action": int(action),
                            "discount_rate": round(info["applied_discount_rate"], 4),
                            "normal_takeout_orders": round(
                                info["normal_takeout_orders"], 1
                            ),
                            "discounted_takeout_orders": round(
                                info["discounted_takeout_orders"], 1
                            ),
                            "delivery_orders": round(info["delivery_orders"], 1),
                            "total_orders": round(info["total_orders"], 1),
                            "ontime_rate": round(info["ontime_rate"], 4),
                            "satisfaction_before": round(float(sat), 1),
                            "satisfaction_after": round(info["satisfaction"], 1),
                            "daily_profit": round(info["daily_profit"], 1),
                            "reward": round(float(reward), 1),
                        }
                    )

                    state_str = (
                        f"sat={sat:.1f}|days={int(days_left)}|rain={int(weather)}"
                        f"|wknd={int(is_wknd)}|hol={int(is_hol)}|cin={int(is_cin)}"
                    )
                    action_str = (
                        f"{round(info['applied_discount_rate'] * 100)}%(act={action})"
                    )
                    response_str = (
                        f"orders={info['total_orders']:.1f}"
                        f"|ontime={info['ontime_rate'] * 100:.1f}%"
                        f"|next_sat={info['satisfaction']:.1f}"
                    )
                    writer.writerow(
                        [
                            start_date,
                            seed,
                            d_str,
                            "store_all",
                            state_str,
                            action_str,
                            response_str,
                            f"{info['daily_profit']:.1f}",
                        ]
                    )
                    obs = next_obs

                ep_profits.append(total_profit)
                ep_avg_orders.append(orders_sum / config.EPISODE_DAYS)
                ep_avg_ontime.append(ontime_sum / config.EPISODE_DAYS)
                ep_final_sat.append(info["satisfaction"])
                ep_low_sat_days.append(low_sat_days)
                replay_episodes.append(
                    {
                        "episode_id": f"{start_date}_s{seed}",
                        "start_date": start_date,
                        "seed": seed,
                        "total_profit": round(total_profit, 1),
                        "total_reward": round(total_reward, 1),
                        "avg_orders": round(orders_sum / config.EPISODE_DAYS, 1),
                        "avg_ontime_rate": round(ontime_sum / config.EPISODE_DAYS, 4),
                        "final_satisfaction": round(info["satisfaction"], 1),
                        "low_sat_days": low_sat_days,
                        "days": day_records,
                    }
                )

    replay_path = PUBLIC_DATA_DIR / f"replay_{key}.json"
    replay_payload = {
        "policy_key": key,
        "policy_label": label,
        "episodes": replay_episodes,
    }
    replay_path.write_text(
        json.dumps(replay_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    return {
        "key": key,
        "label": label,
        "avg_30d_profit": float(np.mean(ep_profits)),
        "avg_daily_orders": float(np.mean(ep_avg_orders)),
        "avg_ontime_pct": float(np.mean(ep_avg_ontime) * 100.0),
        "avg_final_sat": float(np.mean(ep_final_sat)),
        "avg_low_sat_days": float(np.mean(ep_low_sat_days)),
        "avg_rainy_discount_pct": float(np.mean(ep_rainy_discount_rates) * 100.0)
        if ep_rainy_discount_rates
        else 0.0,
    }


def export_manifest() -> None:
    manifest = {
        "store_name": config.STORE_NAME,
        "business_type": config.BUSINESS_TYPE,
        "open_hour": config.OPEN_HOUR,
        "close_hour": config.CLOSE_HOUR,
        "episode_days": config.EPISODE_DAYS,
        "capacity_limit": config.DAILY_CAPACITY_LIMIT,
        "low_sat_threshold": config.LOW_SATISFACTION_THRESHOLD,
        "baseline_key": "custom_busy_aware",
        "reward_formula": "임시 : 하루 주문별 이익(원)",
        "records": [
            {"key": k, "label": lbl, "file": f"data/replay_{k}.json"}
            for k, lbl, _ in POLICIES
        ],
    }
    (PUBLIC_DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> list[dict[str, float | str]]:
    weather_lookup = load_weather_lookup()
    export_personas_summary()
    results = [
        evaluate_policy(k, lbl, fn, weather_lookup) for k, lbl, fn in POLICIES
    ]
    export_manifest()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 가상 매장 시험 규칙 점검 결과 (Sanity Check)",
        "",
        f"- **평가 기준:** 2025년 비겹침 시작일 {len(config.EVAL_START_DATES)}개 × 난수 시드 {len(config.EVAL_SEEDS)}개(`{config.EVAL_SEEDS}`) = 규칙별 총 {len(config.EVAL_START_DATES) * len(config.EVAL_SEEDS)}판 평균 (한 판 {config.EPISODE_DAYS}일)",
        "- **내가 정한 규칙(`custom_busy_aware`, 가정):** 비 오는 날 평소 10% 할인, 주말·공휴일·영화관 중 1개 겹치면 5% 할인, 2개 이상 겹치거나 만족도 50 미만이면 0% 할인",
        "",
        f"| 시험 규칙 | 30일 누적 이익(원, 성공 척도) | 하루 평균 주문 수(잔) | 평균 제시간 제공률(%) | 최종 만족도(점) | 만족도 {config.LOW_SATISFACTION_THRESHOLD} 미만 일수(일) | 비 오는 날 평균 할인(%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['avg_30d_profit']:,.0f} | {r['avg_daily_orders']:.1f} "
            f"| {r['avg_ontime_pct']:.2f}% | {r['avg_final_sat']:.1f} "
            f"| {r['avg_low_sat_days']:.2f} | {r['avg_rainy_discount_pct']:.2f}% |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[5:]))
    print(f"\n저장 완료: {SUMMARY_PATH}, {LOGS_DIR}/*.csv 및 {PUBLIC_DATA_DIR}/*.json")
    return results


if __name__ == "__main__":
    main()
