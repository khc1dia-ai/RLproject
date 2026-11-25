"""Validate data/customer_traits.csv against project_spec.md Section 3."""

import csv
from pathlib import Path
import re

# 숫자·기준 설정 (검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다)
EXPECTED_COUNT = 100
PERSONAS_PATH = Path("data/personas.csv")
TRAITS_PATH = Path("data/customer_traits.csv")
VALID_CONFIDENCE = {"높음", "보통", "낮음"}

# (최솟값, 최댓값, 목표 평균, 허용 오차) — 범위와 허용 오차는 가정
TRAIT_SPECS = {
    "takeout_preference": (0.25, 0.85, 0.55, 0.02),  # 가정
    "rain_shift_rate": (0.00, 0.50, 0.25, 0.02),  # 가정
    "discount_sensitivity": (0.0, 2.0, 1.00, 0.05),  # 가정
    "low_sat_churn_prob": (0.00, 0.40, 0.20, 0.02),  # 가정
    "holiday_visit_sensitivity": (0.0, 2.0, 1.00, 0.05),  # 가정
    "cinema_visit_sensitivity": (0.0, 2.0, 1.00, 0.05),  # 가정
}

AGE_SEX_ONLY_PATTERN = re.compile(
    r"^(?:[\s,.;:/()·\-]+|\d+대|\d+세|남자|여자|남성|여성|청년|중년|나이|성별)+$"
)


def main() -> int:
    with PERSONAS_PATH.open(encoding="utf-8") as f:
        personas = list(csv.DictReader(f))
    with TRAITS_PATH.open(encoding="utf-8") as f:
        traits = list(csv.DictReader(f))

    errors: list[str] = []

    # 1. 인원 수 및 번호·uuid 일치 검사
    if len(traits) != EXPECTED_COUNT:
        errors.append(f"인원 수 불일치: {len(traits)}명 (기대 {EXPECTED_COUNT}명)")

    missing_nos = sorted(
        set(range(1, EXPECTED_COUNT + 1)) - {int(r["customer_no"]) for r in traits}
    )
    if missing_nos:
        errors.append(f"빠진 고객 번호: {missing_nos}")

    uuid_mismatches = []
    for i, r in enumerate(traits):
        no = int(r["customer_no"])
        if 1 <= no <= len(personas) and r["uuid"] != personas[no - 1]["uuid"]:
            uuid_mismatches.append(no)
        if no != i + 1:
            errors.append(f"번호 순서 불일치: {i + 1}번째 줄에 {no}번")
    if uuid_mismatches:
        errors.append(f"uuid 불일치 번호: {uuid_mismatches}")

    # 2. 범위, 확신도, 근거 빈 줄, 나이·성별뿐인 근거 검사
    range_violations = []
    bad_confidence = []
    empty_evidence = []
    age_sex_only = []

    for r in traits:
        no = int(r["customer_no"])
        for col, (lo, hi, _, _) in TRAIT_SPECS.items():
            val = float(r[col])
            if not (lo <= val <= hi):
                range_violations.append((no, col, val, lo, hi))
        if r.get("confidence", "") not in VALID_CONFIDENCE:
            bad_confidence.append((no, r.get("confidence", "")))
        ev = r.get("evidence", "").strip()
        if not ev:
            empty_evidence.append(no)
        elif AGE_SEX_ONLY_PATTERN.match(ev):
            age_sex_only.append(no)

    if range_violations:
        errors.append(f"범위 초과 {len(range_violations)}건: {range_violations[:5]}")
    if bad_confidence:
        errors.append(f"확신도 오류 {len(bad_confidence)}건: {bad_confidence[:5]}")
    if empty_evidence:
        errors.append(f"근거 빈 줄 번호: {empty_evidence}")
    if age_sex_only:
        errors.append(f"나이·성별뿐인 근거 번호: {age_sex_only}")

    # 3. 칸별 목표 평균 허용 범위 검사
    print("=== 1. 기본 무결성 검사 ===")
    print(f"- 총 인원: {len(traits)} / {EXPECTED_COUNT}")
    print(f"- 범위 위반: {len(range_violations)}건")
    print(f"- 근거 빈 줄: {len(empty_evidence)}건")
    print(f"- 나이·성별뿐인 근거: {len(age_sex_only)}건")
    adjusted_rows = [
        int(r["customer_no"]) for r in traits if r.get("adjusted", "").strip()
    ]
    print(f"- 평균 보정(adjusted) 기록 고객: {len(adjusted_rows)}명")

    print("\n=== 2. 칸별 평균 vs 목표 평균 ===")
    out_of_tol = []
    for col, (lo, hi, target, tol) in TRAIT_SPECS.items():
        vals = [float(r[col]) for r in traits]
        mean = sum(vals) / len(vals)
        diff = mean - target
        ok = abs(diff) <= tol + 1e-9
        status = "PASS" if ok else "OUT"
        if not ok:
            out_of_tol.append((col, mean, target, diff, tol))
        print(
            f"- {col:26s}: 평균 {mean:.4f} | 목표 {target:.2f} (±{tol:.2f}) | 차이 {diff:+.4f} -> {status}"
        )

    if out_of_tol:
        errors.append(
            "목표 평균 허용 범위 초과 칸: "
            + ", ".join(f"{c}({m:.3f} vs {t:.2f})" for c, m, t, _, _ in out_of_tol)
        )

    print("\n=== 3. 최종 판정 ===")
    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
