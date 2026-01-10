"""
프로젝트 설정 및 경로 관리
"""
from pathlib import Path
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "tables"

# 디렉토리 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Need 변수 목록
NEED_VARS = [
    'suicide_rate',
    'depression_experience_rate',
    'perceived_stress_rate',
    'high_risk_drinking_rate',
    'unmet_medical_need_rate',
    'unemployment_rate',
    'elderly_population_rate',
    'old_dependency_ratio',
    'single_households',
    'basic_livelihood_recipients'
]

# Supply 변수 목록
SUPPLY_VARS = [
    'welfare_budget_per_capita',
    'public_sports_facilities_count',
    'parks_count',
    'libraries_count',
    'medical_institutions_count',
    'health_promotion_centers_count',
    'elderly_leisure_welfare_facilities_count',
    'in_home_elderly_welfare_facilities_count',
    'cultural_satisfaction'
]

# Need 가중치 (총합 100%)
WEIGHTS_NEED = {
    'suicide_rate_norm': 0.12,
    'depression_experience_rate_norm': 0.09,
    'perceived_stress_rate_norm': 0.07,
    'high_risk_drinking_rate_norm': 0.07,
    'elderly_population_rate_norm': 0.10,
    'single_households_norm': 0.08,
    'basic_livelihood_recipients_norm': 0.07,
    'unemployment_rate_norm': 0.15,
    'unmet_medical_need_rate_norm': 0.14,
    'old_dependency_ratio_norm': 0.11
}

# Supply 가중치 (총합 100%)
WEIGHTS_SUPPLY = {
    'health_promotion_centers_count_norm': 0.20,
    'medical_institutions_count_norm': 0.20,
    'elderly_leisure_welfare_facilities_count_norm': 0.15,
    'in_home_elderly_welfare_facilities_count_norm': 0.15,
    'parks_count_norm': 0.10,
    'libraries_count_norm': 0.07,
    'public_sports_facilities_count_norm': 0.07,
    'cultural_satisfaction_norm': 0.03,
    'welfare_budget_per_capita_norm': 0.03
}

# 정책 시뮬레이션 시나리오
POLICY_SCENARIO = {
    "welfare_budget_per_capita": ("pct", 0.10),
    "cultural_satisfaction": ("add", 0.20),
    "parks_count": ("add", 10),
    "libraries_count": ("add", 2),
    "public_sports_facilities_count": ("add", 2),
    "medical_institutions_count": ("add", 20),
    "health_promotion_centers_count": ("add", 1),
    "elderly_leisure_welfare_facilities_count": ("add", 10),
    "in_home_elderly_welfare_facilities_count": ("add", 5),
}