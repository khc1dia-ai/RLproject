"""Fetch 2016-2025 daily weather, weekend, and Korean holiday data."""

import csv
from datetime import date, datetime, timezone
import math
from pathlib import Path

import holidays
import openmeteo_requests
import requests_cache
from retry_requests import retry

# 숫자·기준 설정 (검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다)
STORE_LATITUDE = 37.3517  # 경기도 시흥시 정왕역 일대
STORE_LONGITUDE = 126.7429
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Asia/Seoul"
EXPECTED_DAYS = 3653

RAIN_THRESHOLD_MM = 1.0  # 가정 (대안 B: 하루 강수량 1.0mm 이상)
SNOW_THRESHOLD_CM = 0.5  # 가정 (대안 B: 하루 적설량 0.5cm 이상)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_PATH = Path("data/external_factors.csv")
OUTPUT_COLUMNS = [
    "date",
    "temp_max_c",
    "temp_min_c",
    "precipitation_mm",
    "snowfall_cm",
    "weather_score",
    "is_weekend",
    "is_holiday",
]


def main() -> None:
    cache_session = requests_cache.CachedSession(backend="memory")
    retry_session = retry(cache_session, retries=5, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": STORE_LATITUDE,
        "longitude": STORE_LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
        ],
        "timezone": TIMEZONE,
    }
    responses = client.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]
    daily = response.Daily()

    utc_offset = response.UtcOffsetSeconds()
    start_ts = daily.Time()
    end_ts = daily.TimeEnd()
    step = daily.Interval()
    dates = [
        datetime.fromtimestamp(ts + utc_offset, tz=timezone.utc).date()
        for ts in range(start_ts, end_ts, step)
    ]

    temp_max = daily.Variables(0).ValuesAsNumpy()
    temp_min = daily.Variables(1).ValuesAsNumpy()
    precip = daily.Variables(2).ValuesAsNumpy()
    snow = daily.Variables(3).ValuesAsNumpy()

    assert len(dates) == EXPECTED_DAYS, f"일수 불일치: {len(dates)}"
    assert dates[0] == date(2016, 1, 1) and dates[-1] == date(2025, 12, 31), (
        f"날짜 범위 불일치: {dates[0]} ~ {dates[-1]}"
    )

    kr_holidays = holidays.KR(years=range(2016, 2026))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rainy_days = 0
    weekend_days = 0
    holiday_days = 0

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for i, d in enumerate(dates):
            tmax = float(temp_max[i])
            tmin = float(temp_min[i])
            pr = float(precip[i])
            sn = float(snow[i])
            if any(math.isnan(x) for x in (tmax, tmin, pr, sn)):
                raise RuntimeError(f"결측치 발견: {d}")

            w_score = 1 if (pr >= RAIN_THRESHOLD_MM or sn >= SNOW_THRESHOLD_CM) else 0
            is_wknd = 1 if d.weekday() >= 5 else 0
            is_hol = 1 if d in kr_holidays else 0

            rainy_days += w_score
            weekend_days += is_wknd
            holiday_days += is_hol

            writer.writerow(
                [
                    d.isoformat(),
                    f"{tmax:.1f}",
                    f"{tmin:.1f}",
                    f"{pr:.2f}",
                    f"{sn:.2f}",
                    w_score,
                    is_wknd,
                    is_hol,
                ]
            )

    print(
        f"{OUTPUT_PATH} 저장 완료: {len(dates)}일 "
        f"(궂은 날 {rainy_days}일={rainy_days / len(dates) * 100:.1f}%, "
        f"주말 {weekend_days}일, 공휴일 {holiday_days}일)"
    )


if __name__ == "__main__":
    main()
