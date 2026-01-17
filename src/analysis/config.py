"""
config.py

프로젝트 설정 및 경로 관리

역할 요약:
- 프로젝트 전반에서 공통으로 쓰이는
  ① 경로(Path)
  ② 변수 목록(Need / Supply)
  ③ 지수 계산 가중치
  ④ 정책 시뮬레이션 시나리오
를 한 곳에 모아 관리하는 설정 파일

→ 분석 로직과 '정책적 판단 기준'을 분리하기 위한 핵심 파일
"""
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================
# 1. 시각화 환경 설정 (한글 깨짐 방지)
# =====================================================
# matplotlib에서 한글이 깨지는 문제를 방지하기 위해
# 시스템에 설치된 'Malgun Gothic' 폰트를 기본 폰트로 지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스(-) 기호가 네모(□)로 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 2. 프로젝트 경로 설정
# =====================================================
# 이 파일(config.py)의 위치를 기준으로
# 상위 디렉토리를 타고 올라가 프로젝트 루트(BASE_DIR)를 계산


BASE_DIR = Path(__file__).resolve().parents[2]

# 전처리 완료된 입력 데이터가 저장된 디렉토리
# (정규화, 지수 계산 직전 단계의 tidy 데이터 등)
DATA_DIR = BASE_DIR / "data" / "processed"

# 분석 결과(테이블)를 저장할 출력 디렉토리
# 예: feature importance, SHAP 결과, 사각지대 랭킹 등
OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "tables"

# 출력 디렉토리가 없으면 자동 생성
# parents=True → 중간 폴더까지 함께 생성
# exist_ok=True → 이미 존재해도 에러 발생 안 함
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 3. Need 변수 목록 (취약도 측면 지표)
# =====================================================
# 지역 주민의 정신건강 위험 및 사회적 취약성을 나타내는 변수들
#
# 원칙:
# - 개인 수준 위험의 '집계 결과'를 지역 단위로 표현
# - 직접적 정신건강 지표 + 구조적 사회 취약 지표를 혼합
NEED_VARS = [
    'suicide_rate',                    # 자살률 (가장 직접적인 결과 지표)
    'depression_experience_rate',      # 우울감 경험률
    'perceived_stress_rate',           # 스트레스 인지율
    'high_risk_drinking_rate',          # 고위험 음주율
    'unmet_medical_need_rate',          # 의료 미충족률
    'unemployment_rate',               # 실업률 (사회경제적 스트레스 요인)
    'elderly_population_rate',          # 고령 인구 비율
    'old_dependency_ratio',             # 노년부양비
    'single_households',                # 1인 가구 수 (사회적 고립 proxy)
    'basic_livelihood_recipients'       # 기초생활수급자 수 (경제적 취약)
]

# =====================================================
# 4. Supply 변수 목록 (인프라 / 완충 자원)
# =====================================================
# 지역 내 정신건강 및 삶의 질을 완충할 수 있는
# 공공·의료·문화·복지 인프라 지표들
#
# 특징:
# - 단순 의료 인프라뿐 아니라
#   "생활 환경 / 문화적 자원"까지 포함
SUPPLY_VARS = [
    'welfare_budget_per_capita',                 # 1인당 복지 예산
    'public_sports_facilities_count',            # 공공 체육시설 수
    'parks_count',                               # 공원 수
    'libraries_count',                           # 도서관 수
    'medical_institutions_count',                # 의료기관 수
    'health_promotion_centers_count',            # 건강증진센터 수
    'elderly_leisure_welfare_facilities_count',  # 노인 여가복지시설 수
    'in_home_elderly_welfare_facilities_count',  # 재가 노인복지시설 수
    'cultural_satisfaction'                      # 문화 만족도 (주관적 지표)
]

# =====================================================
# 5. Need 지수 가중치 (총합 = 1)
# =====================================================
# 정규화된 Need 변수들을 하나의 Need_Index로 합산할 때 사용
#
# 설계 원칙:
# - 자살률: 결과 변수이자 가장 중대한 지표 → 가장 높은 가중치
# - 정서적 요인(우울, 스트레스): 중간 수준 비중
# - 구조적 요인(고령화, 실업, 빈곤): 누적 위험 요인으로 반영
#
# ⚠️ 주의:
# - 이 가중치는 "진실"이 아니라
#   '정책적 판단을 반영한 가설적 설정'
# - 민감도 분석/대안 시나리오로 조정 가능
WEIGHTS_NEED = {
    'suicide_rate_norm': 0.25,
    'depression_experience_rate_norm': 0.125,
    'perceived_stress_rate_norm': 0.125,
    'high_risk_drinking_rate_norm': 0.05,
    'elderly_population_rate_norm': 0.05,
    'single_households_norm': 0.10,
    'basic_livelihood_recipients_norm': 0.125,
    'unemployment_rate_norm': 0.125,
    'unmet_medical_need_rate_norm': 0.10,
    'old_dependency_ratio_norm': 0.05
}

# =====================================================
# 6. Supply 지수 가중치 (총합 = 1)
# =====================================================
# 정규화된 Supply 변수들을 하나의 Supply_Index로 합산할 때 사용
#
# 설계 원칙:
# - 직접적인 의료/보건 인프라: 가장 높은 비중
# - 노인 복지 인프라: 고위험 집단 완충 자원으로 중요
# - 공원/문화/체육: 간접적·장기적 완충 요인 → 상대적으로 낮은 비중
#
# ⚠️ 해석 주의:
# - Supply_Index가 높다고 반드시 위험이 낮아지는 것은 아님
# - 이후 AI 진단 단계에서 "공급 대비 효과가 작동하지 않는 지역"을 탐색
WEIGHTS_SUPPLY = {
    'health_promotion_centers_count_norm': 0.20,
    'medical_institutions_count_norm': 0.20,
    'elderly_leisure_welfare_facilities_count_norm': 0.15,
    'in_home_elderly_welfare_facilities_count_norm': 0.15,
    'parks_count_norm': 0.04,
    'libraries_count_norm': 0.02,
    'public_sports_facilities_count_norm': 0.02,
    'cultural_satisfaction_norm': 0.07,
    'welfare_budget_per_capita_norm': 0.15
}

# =====================================================
# 7. 정책 시뮬레이션 시나리오
# =====================================================
# 가상 정책 개입이 있었을 때
# Supply 변수를 어떻게 변화시킬지를 정의한 시나리오
#
# 형식:
#   변수명: (적용 방식, 변화량)
#
# 적용 방식:
# - "pct": 기존 값 대비 비율 증가 (예: +10%)
# - "add": 절대값 증가 (예: 시설 +3개)
#
# 목적:
# - "이 지역에 이런 정책을 하면 Supply_Index가 얼마나 변할까?"
# - "그 변화가 AI 기준선 대비 Inefficiency를 얼마나 줄일까?"
#
# ⚠️ 실제 정책 효과가 아니라,
#   '정책 실험용 가상 시나리오'임을 명확히 해야 함
POLICY_SCENARIO = {
    "welfare_budget_per_capita": ("pct", 0.10),            # 복지 예산 10% 증가
    "cultural_satisfaction": ("add", 0.15),                # 문화 만족도 점수 상승
    "parks_count": ("add", 4),                              # 공원 4개 추가
    "libraries_count": ("add", 2),                          # 도서관 2개 추가
    "public_sports_facilities_count": ("add", 2),          # 체육시설 2개 추가
    "medical_institutions_count": ("add", 5),              # 의료기관 5개 추가
    "health_promotion_centers_count": ("add", 1),          # 건강증진센터 1개 추가
    "elderly_leisure_welfare_facilities_count": ("add", 8),
    "in_home_elderly_welfare_facilities_count": ("add", 4),
}
