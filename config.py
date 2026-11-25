"""Configuration for the cafe reinforcement-learning simulation."""

# 숫자 설정은 검토와 수정이 쉽도록 파일 맨 위에 모아 둡니다.
EPISODE_DAYS = 30
OPEN_HOUR = 7
CLOSE_HOUR = 22
TRAIN_DATA_START_DATE = "2016-01-01"
TRAIN_DATA_END_DATE = "2024-12-31"
TRAIN_EPISODE_START_MIN = "2016-01-01"
TRAIN_EPISODE_START_MAX = "2024-12-02"  # 30일 에피소드가 2024-12-31 안에 종료되는 마지막 시작일
TRAIN_SEED = 42  # 가정
EVAL_START_DATES = (
    "2025-01-01",
    "2025-01-31",
    "2025-03-02",
    "2025-04-01",
    "2025-05-01",
    "2025-05-31",
    "2025-06-30",
    "2025-07-30",
    "2025-08-29",
    "2025-09-28",
    "2025-10-28",
    "2025-11-27",
)
EVAL_SEEDS = (100, 101, 102)  # 가정
AMERICANO_UNITS_PER_ORDER = 1
TAKEOUT_PRICE_WON = 2_000
DELIVERY_PRICE_WON = 2_500
TAKEOUT_COST_RATE = 0.5
DELIVERY_COST_RATE = 0.5
TAKEOUT_FEE_RATE = 0.0
DELIVERY_FEE_RATE = 0.3

# 비 오는 날 포장 할인 결정 설정
CLEAR_WEATHER_SCORE = 0
RAINY_WEATHER_SCORE = 1
TAKEOUT_DISCOUNT_RATES = (0.0, 0.05, 0.10)  # 가정
DISCOUNT_OFFER_RATIO = 1.00  # 가정 (비 오는 날 전체 고객에게 할인 안내)
DAILY_CAPACITY_LIMIT = 265  # 가정 (하루 처리 한도)

# 잠재 고객 및 달력·변동 설정
BASE_POTENTIAL_CUSTOMERS = 300  # 가정
WEEKEND_CUSTOMER_BONUS_RATE = 0.20  # 가정
HOLIDAY_CUSTOMER_BONUS_RATE = 0.30  # 가정
BUSY_CINEMA_CUSTOMER_BONUS_RATE = 0.20  # 가정
DAILY_CUSTOMER_NOISE_MIN = -0.10  # 가정
DAILY_CUSTOMER_NOISE_MAX = 0.10  # 가정

# 날씨·할인별 구매 전환율 설정
CLEAR_TAKEOUT_CONVERSION_RATE = 0.55  # 가정
CLEAR_DELIVERY_CONVERSION_RATE = 0.30  # 가정
RAINY_TAKEOUT_CONVERSION_RATE = 0.30  # 가정
RAINY_DELIVERY_CONVERSION_RATE = 0.55  # 가정
RAINY_DISCOUNT_TAKEOUT_CONVERSION_RATES = {
    0.0: 0.30,
    0.05: 0.40,
    0.10: 0.59,
}  # 가정

# 만족도 누적 및 다음 날 수요 반영 설정
SATISFACTION_MIN = 0  # 가정
SATISFACTION_MAX = 100  # 가정
SATISFACTION_INITIAL = 55  # 가정
ONTIME_RATE_HIGH_THRESHOLD = 0.90  # 가정
ONTIME_RATE_LOW_THRESHOLD = 0.80  # 가정
SATISFACTION_DELTA_HIGH = 2  # 가정
SATISFACTION_DELTA_MID = 0  # 가정
SATISFACTION_DELTA_LOW = -10  # 가정
LOW_SATISFACTION_THRESHOLD = 45  # 가정
LOW_SATISFACTION_CUSTOMER_PENALTY_RATE = -0.20  # 가정

# 주요 고객 연령대
CUSTOMER_AGE_MIN = 20  # 가정
CUSTOMER_AGE_MAX = 40  # 가정

# 매장 기본 정보
STORE_NAME = "따뜻한 아이스 아메리카노의 만족감"
BUSINESS_TYPE = "카페"
LOCATION_CITY = "경기도 시흥시"
LOCATION_DESCRIPTION = "지하철 4호선 정왕역 1번 출구 롯데시네마 옆 건물"
TRADE_AREA = "역세권·영화관 인접 상권"
PRIMARY_CUSTOMERS = (
    f"{CUSTOMER_AGE_MIN}~{CUSTOMER_AGE_MAX}대 직장인·학생·영화관 방문객"
)  # 가정

# 시뮬레이션 결정 및 행동 후보
DECISION_UNIT = "하루"
DECISION_INTERVAL = "비 오는 날(날씨 점수 1) 하루 한 번"
DISCOUNT_TARGET_RULE = (
    "잠재 고객 중 무작위 30%에게 주문 전 제안, 제안받은 고객이 포장 주문 시 적용"
)  # 가정
ACTION_CANDIDATES = {
    f"takeout_discount_{round(rate * 100)}pct": {
        "discount_rate": rate,
        "discount_won": round(TAKEOUT_PRICE_WON * rate),
        "discounted_takeout_price_won": round(TAKEOUT_PRICE_WON * (1 - rate)),
    }
    for rate in TAKEOUT_DISCOUNT_RATES
}  # 가정

# 주문 유형 (실제 구매 흐름에서 고객이 선택)
ORDER_CHANNELS = {
    "takeout": {
        "product": "아메리카노",
        "units_per_order": AMERICANO_UNITS_PER_ORDER,
        "price_won": TAKEOUT_PRICE_WON,
        "cost_rate": TAKEOUT_COST_RATE,
        "fee_rate": TAKEOUT_FEE_RATE,
    },
    "delivery": {
        "product": "아메리카노",
        "units_per_order": AMERICANO_UNITS_PER_ORDER,
        "price_won": DELIVERY_PRICE_WON,
        "cost_rate": DELIVERY_COST_RATE,
        "fee_rate": DELIVERY_FEE_RATE,
    },
}

# 성공 척도 및 비용 처리 기준
GAINS = ["주문별 매출"]
LOSSES = ["잔당 원가(재료·컵·포장재)", "배달 수수료"]
EXCLUDED_COSTS = ["고정 임대료", "고정급", "공과금", "확인되지 않은 추가 처리비"]
SUCCESS_METRIC = "30일 누적 주문별 이익 최대화"
SUCCESS_METRIC_DESCRIPTION = (
    "매출 − 잔당 원가 − 배달 수수료 (대기시간·만족도는 금액 벌점으로 차감하지 않음)"
)
DELIVERY_FEE_TREATMENT = "배달 매출의 30%를 잔당 원가와 별도로 차감"
EXTERNAL_FACTORS = [
    "날씨(비·기온)",
    "요일(평일·주말)",
    "공휴일",
    "인접 영화관 관람객·상영 일정",
]
