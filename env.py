"""Gymnasium environment for the cafe rainy-day takeout discount simulation."""

import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config
import reward as reward_module

# 반응 규칙 가정값 묶음 (config.py와 겹치지 않는 값만 모아 둡니다)
BASE_BUY_RATE = 0.85  # 규칙 1 (가정: 맑음·비 기본 미구매 15% 고정, 총 구매율 85%)
RAINY_DISCOUNT_SHIFT_FROM_DELIVERY = {
    0.0: 0.0,
    0.05: 0.05,
    0.10: 0.15,
}  # 규칙 2 (가정: 비 오는 날 할인 시 배달→포장 기준 이동 폭)
RAINY_DISCOUNT_SHIFT_FROM_NONBUY = {
    0.0: 0.0,
    0.05: 0.05,
    0.10: 0.14,
}  # 규칙 2 (가정: 비 오는 날 할인 시 미구매→포장 기준 이동 폭)
BUSY_CINEMA_DAILY_PROB = 0.20  # 규칙 6 (가정: 영화관 관람객 많은 날 매일 20% 확률)
PERSONA_SCALE_FACTOR = 3.0  # 3절 합산 규칙 (기준 잠재 고객 300명 ÷ 페르소나 100명)

TRAITS_PATH = Path("data/customer_traits.csv")
EXTERNAL_PATH = Path("data/external_factors.csv")


class CafeDiscountEnv(gym.Env):
    """경기도 시흥시 정왕역 카페의 하루 단위 포장 할인 가상 매장 환경."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        start_date: str | None = None,
        traits_path: Path = TRAITS_PATH,
        external_path: Path = EXTERNAL_PATH,
    ) -> None:
        super().__init__()
        self.fixed_start_date = start_date

        self._load_traits(traits_path)
        self._load_external_factors(external_path)

        # 상태(관측) 6개 값 (모두 0~1 범위로 맞춤):
        # [satisfaction/100, days_left/30, weather_score, is_weekend, is_holiday, is_busy_cinema]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        # 행동: 선택지 3개 (0: 0% 할인, 1: 5% 할인, 2: 10% 할인)
        self.action_space = spaces.Discrete(len(config.TAKEOUT_DISCOUNT_RATES))

        self.start_idx = 0
        self.day_offset = 0
        self.satisfaction = float(config.SATISFACTION_INITIAL)
        self.current_busy_cinema = 0

    def _load_traits(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.takeout_pref = np.array(
            [float(r["takeout_preference"]) for r in rows], dtype=np.float64
        )
        self.rain_shift = np.array(
            [float(r["rain_shift_rate"]) for r in rows], dtype=np.float64
        )
        self.discount_sens = np.array(
            [float(r["discount_sensitivity"]) for r in rows], dtype=np.float64
        )
        self.low_sat_churn = np.array(
            [float(r["low_sat_churn_prob"]) for r in rows], dtype=np.float64
        )
        self.holiday_sens = np.array(
            [float(r["holiday_visit_sensitivity"]) for r in rows], dtype=np.float64
        )
        self.cinema_sens = np.array(
            [float(r["cinema_visit_sensitivity"]) for r in rows], dtype=np.float64
        )

    def _load_external_factors(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        self.external_rows = [
            {
                "date": r["date"],
                "weather_score": int(r["weather_score"]),
                "is_weekend": int(r["is_weekend"]),
                "is_holiday": int(r["is_holiday"]),
            }
            for r in raw
        ]
        self.date_to_idx = {
            row["date"]: i for i, row in enumerate(self.external_rows)
        }
        self.train_start_indices = [
            i
            for i, row in enumerate(self.external_rows)
            if config.TRAIN_EPISODE_START_MIN
            <= row["date"]
            <= config.TRAIN_EPISODE_START_MAX
        ]
        if not self.train_start_indices:
            raise ValueError("학습 기간에 해당하는 시작일이 external_factors.csv에 없습니다.")

    def _get_obs(self) -> np.ndarray:
        days_left = config.EPISODE_DAYS - self.day_offset
        sat_norm = self.satisfaction / float(config.SATISFACTION_MAX)
        if days_left <= 0:
            return np.array([sat_norm, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        row = self.external_rows[self.start_idx + self.day_offset]
        return np.array(
            [
                sat_norm,
                float(days_left) / float(config.EPISODE_DAYS),
                float(row["weather_score"]),
                float(row["is_weekend"]),
                float(row["is_holiday"]),
                float(self.current_busy_cinema),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        req_start = (
            options.get("start_date") if options and "start_date" in options else None
        )
        target_start = req_start if req_start is not None else self.fixed_start_date

        if target_start is not None:
            if target_start not in self.date_to_idx:
                raise ValueError(f"시작일을 찾을 수 없습니다: {target_start}")
            idx = self.date_to_idx[target_start]
            if idx + config.EPISODE_DAYS > len(self.external_rows):
                raise ValueError(
                    f"시작일 {target_start}부터 {config.EPISODE_DAYS}일 데이터가 부족합니다."
                )
            self.start_idx = idx
        else:
            pick = int(self.np_random.integers(0, len(self.train_start_indices)))
            self.start_idx = self.train_start_indices[pick]

        self.day_offset = 0
        self.satisfaction = float(config.SATISFACTION_INITIAL)
        self.current_busy_cinema = int(
            self.np_random.random() < BUSY_CINEMA_DAILY_PROB
        )

        info = {"start_date": self.external_rows[self.start_idx]["date"]}
        return self._get_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        row = self.external_rows[self.start_idx + self.day_offset]
        weather_score = row["weather_score"]
        is_weekend = row["is_weekend"]
        is_holiday = row["is_holiday"]
        is_busy_cinema = self.current_busy_cinema

        # 맑은 날은 어떤 선택지를 고르든 정상가(0%) 자동 적용
        if weather_score == config.RAINY_WEATHER_SCORE:
            discount_rate = config.TAKEOUT_DISCOUNT_RATES[int(action)]
        else:
            discount_rate = 0.0

        # 규칙 6 & 규칙 5: 페르소나별 방문 가중치
        visit_weight = (
            1.0
            + is_weekend * config.WEEKEND_CUSTOMER_BONUS_RATE * self.holiday_sens
            + is_holiday * config.HOLIDAY_CUSTOMER_BONUS_RATE * self.holiday_sens
            + is_busy_cinema * config.BUSY_CINEMA_CUSTOMER_BONUS_RATE * self.cinema_sens
        )
        if self.satisfaction < config.LOW_SATISFACTION_THRESHOLD:
            visit_weight = visit_weight * (1.0 - self.low_sat_churn)

        # 규칙 1: 날씨별 기본 포장·배달 구매 확률
        if weather_score == config.RAINY_WEATHER_SCORE:
            p_takeout = np.clip(self.takeout_pref - self.rain_shift, 0.0, BASE_BUY_RATE)
        else:
            p_takeout = self.takeout_pref.copy()
        p_delivery = BASE_BUY_RATE - p_takeout

        # 규칙 2: 비 오는 날 할인 제안 대상(30%)의 포장·배달 이동
        offer_ratio = config.DISCOUNT_OFFER_RATIO if discount_rate > 0.0 else 0.0
        if offer_ratio > 0.0:
            shift_del = np.minimum(
                p_delivery,
                RAINY_DISCOUNT_SHIFT_FROM_DELIVERY[discount_rate] * self.discount_sens,
            )
            p_nonbuy = np.clip(1.0 - p_takeout - p_delivery, 0.0, 1.0)
            shift_non = np.minimum(
                p_nonbuy,
                RAINY_DISCOUNT_SHIFT_FROM_NONBUY[discount_rate] * self.discount_sens,
            )
            p_takeout_disc = np.clip(p_takeout + shift_del + shift_non, 0.0, 1.0)
            p_delivery_disc = np.clip(p_delivery - shift_del, 0.0, 1.0)
        else:
            p_takeout_disc = p_takeout
            p_delivery_disc = p_delivery

        # 규칙 7 & 3절 합산 방식: 100명 합산 × 3배 × 하루 우연 변동
        noise = float(
            self.np_random.uniform(
                config.DAILY_CUSTOMER_NOISE_MIN, config.DAILY_CUSTOMER_NOISE_MAX
            )
        )
        scale = PERSONA_SCALE_FACTOR * (1.0 + noise)

        normal_takeout_orders = float(
            (1.0 - offer_ratio) * np.sum(visit_weight * p_takeout) * scale
        )
        discounted_takeout_orders = float(
            offer_ratio * np.sum(visit_weight * p_takeout_disc) * scale
        )
        delivery_orders = float(
            (
                (1.0 - offer_ratio) * np.sum(visit_weight * p_delivery)
                + offer_ratio * np.sum(visit_weight * p_delivery_disc)
            )
            * scale
        )
        total_orders = (
            normal_takeout_orders + discounted_takeout_orders + delivery_orders
        )

        # 규칙 3: 주문별 이익 (성공 척도와 동일, 할인액·만족도 이중 차감 없음)
        takeout_cost = config.TAKEOUT_PRICE_WON * config.TAKEOUT_COST_RATE
        normal_takeout_profit = config.TAKEOUT_PRICE_WON - takeout_cost
        discounted_takeout_profit = (
            config.TAKEOUT_PRICE_WON * (1.0 - discount_rate) - takeout_cost
        )
        delivery_profit = config.DELIVERY_PRICE_WON * (
            1.0 - config.DELIVERY_COST_RATE - config.DELIVERY_FEE_RATE
        )
        daily_profit = (
            normal_takeout_orders * normal_takeout_profit
            + discounted_takeout_orders * discounted_takeout_profit
            + delivery_orders * delivery_profit
        )

        # 규칙 4: 하루 처리 한도 대비 제시간 제공률
        if total_orders <= config.DAILY_CAPACITY_LIMIT:
            ontime_rate = 1.0
        else:
            ontime_rate = config.DAILY_CAPACITY_LIMIT / total_orders

        # 규칙 5: 만족도 갱신 (0~100 제한)
        if ontime_rate >= config.ONTIME_RATE_HIGH_THRESHOLD:
            sat_delta = config.SATISFACTION_DELTA_HIGH
        elif ontime_rate >= config.ONTIME_RATE_LOW_THRESHOLD:
            sat_delta = config.SATISFACTION_DELTA_MID
        else:
            sat_delta = config.SATISFACTION_DELTA_LOW
        self.satisfaction = float(
            np.clip(
                self.satisfaction + sat_delta,
                config.SATISFACTION_MIN,
                config.SATISFACTION_MAX,
            )
        )

        # 보상 계산 (reward.py v1 사용)
        reward = reward_module.compute_reward(daily_profit)

        self.day_offset += 1
        terminated = self.day_offset >= config.EPISODE_DAYS
        truncated = False
        if not terminated:
            self.current_busy_cinema = int(
                self.np_random.random() < BUSY_CINEMA_DAILY_PROB
            )

        info = {
            "date": row["date"],
            "applied_discount_rate": discount_rate,
            "normal_takeout_orders": normal_takeout_orders,
            "discounted_takeout_orders": discounted_takeout_orders,
            "delivery_orders": delivery_orders,
            "total_orders": total_orders,
            "ontime_rate": ontime_rate,
            "satisfaction": self.satisfaction,
            "daily_profit": daily_profit,
        }
        return self._get_obs(), reward, terminated, truncated, info
