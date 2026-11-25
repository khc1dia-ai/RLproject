"""Fetch 100 matching personas from Hugging Face nvidia/Nemotron-Personas-Korea."""

import csv
from pathlib import Path
import random

import duckdb

# 숫자·조건 설정 (검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다)
PARQUET_COUNT = 9
TARGET_DISTRICT = "경기-시흥시"
AGE_MIN = 20  # 가정 (20~40대)
AGE_MAX = 49  # 가정 (20~40대)
SAMPLE_SIZE = 100
RANDOM_SEED = 42  # 가정

BASE_URL = (
    "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/{index:04d}.parquet"
)
OUTPUT_PATH = Path("data/personas.csv")
OUTPUT_COLUMNS = [
    "uuid",
    "sex",
    "age",
    "province",
    "district",
    "occupation",
    "education_level",
    "marital_status",
    "family_type",
    "housing_type",
    "persona",
    "culinary_persona",
    "hobbies_and_interests",
    "arts_persona",
]


def main() -> None:
    urls = [BASE_URL.format(index=i) for i in range(PARQUET_COUNT)]
    selected_cols = ", ".join(OUTPUT_COLUMNS)
    query = f"""
        SELECT {selected_cols}
        FROM read_parquet(?)
        WHERE district = ?
          AND age BETWEEN ? AND ?
          AND (
              occupation != '무직'
              OR (
                  occupation = '무직'
                  AND (
                      persona LIKE '%학생%'
                      OR professional_persona LIKE '%학생%'
                      OR career_goals_and_ambitions LIKE '%학생%'
                  )
              )
              OR hobbies_and_interests LIKE '%영화%'
              OR arts_persona LIKE '%영화%'
          )
        ORDER BY uuid
    """

    con = duckdb.connect()
    rows = con.execute(query, [urls, TARGET_DISTRICT, AGE_MIN, AGE_MAX]).fetchall()
    con.close()

    if len(rows) < SAMPLE_SIZE:
        raise RuntimeError(f"조건에 맞는 인원이 부족합니다: {len(rows)}명")

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(rows, SAMPLE_SIZE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(sampled)

    print(f"후보 {len(rows)}명 중 {len(sampled)}명을 {OUTPUT_PATH}에 저장했습니다 (seed={RANDOM_SEED}).")


if __name__ == "__main__":
    main()
