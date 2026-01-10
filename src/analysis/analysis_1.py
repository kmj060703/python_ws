import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

#import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 파일 로드

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df_need = pd.read_csv(DATA_DIR / "need_tidy.csv")
df_supply = pd.read_csv(DATA_DIR / "supply_tidy.csv")

# 통합
df = df_need.merge(df_supply, on='district', how='inner')

print("=" * 60); print("📊 데이터 로드 완료"); print("=" * 60); print(f"Shape: {df.shape}"); print(f"구 개수: {len(df)}"); print(f"변수 개수: {len(df.columns) - 1}")
print("\n변수 목록:"); print("Need (11개):", df_need.columns.tolist()[1:]); print("Supply (9개):", df_supply.columns.tolist()[1:])
print("\n첫 5개 구:"); print(df.head()); print("\n기초 통계:"); print(df.describe()); print("\n결측치 확인:"); print(df.isnull().sum().sum()); print("\n✅ 데이터 준비 완료!")

# ===== 정규화 함수 =====

def normalize_to_100(series, direction='positive'):
    """
    0~100점으로 정규화

    direction:
    - 'positive': 높을수록 나쁨 (자살률, 우울감 등)
    - 'negative': 낮을수록 나쁨 (인프라 등)
    """
    scaler = MinMaxScaler(feature_range=(0, 100))

    if direction == 'positive':
        # 높을수록 100점
        normalized = scaler.fit_transform(series.values.reshape(-1, 1))
    else:
        # 낮을수록 100점 (역전)
        normalized = 100 - scaler.fit_transform(series.values.reshape(-1, 1))

    return normalized.flatten()

print("=" * 60); print("📏 Step 2: 변수 정규화"); print("=" * 60)

# Need 변수 정규화 (높을수록 위험 = 100점)
need_vars = [
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

df_need_norm = df[['district']].copy()

for var in need_vars:
    df_need_norm[f'{var}_norm'] = normalize_to_100(df[var], direction='positive')
    print(f"  ✅ {var:40s} → 정규화 완료")

# Supply 변수 정규화 (높을수록 좋음 = 역전 필요)
supply_vars = [
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

df_supply_norm = df[['district']].copy()

for var in supply_vars:
    df_supply_norm[f'{var}_norm'] = normalize_to_100(df[var], direction='negative')
    print(f"  ✅ {var:40s} → 정규화 완료")

print("\n정규화 결과 샘플:"); print(df_need_norm.head()); print("\n✅ Step 2 완료: 모든 변수 0~100점 변환")

# ===== Need Index 계산 =====

print("\n" + "=" * 60); print("📊 Step 3: Need Index 계산"); print("=" * 60)

# 가중치 설정
weights_need = {
    # 정신건강 직접 지표 (50%)
    'suicide_rate_norm': 0.12,
    'depression_experience_rate_norm': 0.09,
    'perceived_stress_rate_norm': 0.07,
    'high_risk_drinking_rate_norm': 0.07,

    # 사회경제적 취약성 (40%)
    'elderly_population_rate_norm': 0.10,
    'single_households_norm': 0.08,
    'basic_livelihood_recipients_norm': 0.07,
    'unemployment_rate_norm': 0.15,

    # 의료 접근성 (10%)
    'unmet_medical_need_rate_norm': 0.14,
    'old_dependency_ratio_norm': 0.11
}

print("가중치:")
total_weight = 0
for var, weight in weights_need.items():
    print(f"  {var:45s}: {weight:5.1%}")
    total_weight += weight

print(f"\n총 가중치 합: {total_weight:.1%}")

# Need Index 계산
df_need_norm['Need_Index'] = 0

for var, weight in weights_need.items():
    df_need_norm['Need_Index'] += df_need_norm[var] * weight

# 정렬
df_need_norm_sorted = df_need_norm.sort_values('Need_Index', ascending=False)

print("\n📈 Need Index TOP 10 (위험도 높은 순):"); print(df_need_norm_sorted[['district', 'Need_Index']].head(10).to_string(index=False))
print("\n📉 Need Index BOTTOM 5 (위험도 낮은 순):"); print(df_need_norm_sorted[['district', 'Need_Index']].tail(5).to_string(index=False))
print(f"\n평균: {df_need_norm['Need_Index'].mean():.2f}"); print(f"최대: {df_need_norm['Need_Index'].max():.2f}"); print(f"최소: {df_need_norm['Need_Index'].min():.2f}"); print("\n✅ Step 3 완료: Need Index")

# ===== Supply Index 계산 =====

print("\n" + "=" * 60); print("🏥 Step 4: Supply Index 계산"); print("=" * 60)

# 가중치 설정 (Supply는 낮을수록 문제)
weights_supply = {
    # 정신건강 직접 인프라 (40%)
    'health_promotion_centers_count_norm': 0.20,
    'medical_institutions_count_norm': 0.20,

    # 사회복지 인프라 (30%)
    'elderly_leisure_welfare_facilities_count_norm': 0.15,
    'in_home_elderly_welfare_facilities_count_norm': 0.15,

    # 삶의 질 인프라 (30%)
    'parks_count_norm': 0.10,
    'libraries_count_norm': 0.07,
    'public_sports_facilities_count_norm': 0.07,
    'cultural_satisfaction_norm': 0.03,
    'welfare_budget_per_capita_norm': 0.03
}

print("가중치:")
total_weight = 0
for var, weight in weights_supply.items():
    print(f"  {var:50s}: {weight:5.1%}")
    total_weight += weight

print(f"\n총 가중치 합: {total_weight:.1%}")

# Supply Index 계산 (낮을수록 문제 = 역전 필요)
df_supply_norm['Supply_Index'] = 0

for var, weight in weights_supply.items():
    df_supply_norm['Supply_Index'] += (100 - df_supply_norm[var]) * weight

# 정렬
df_supply_norm_sorted = df_supply_norm.sort_values('Supply_Index', ascending=False)

print("\n📈 Supply Index TOP 10 (인프라 부족한 순):"); print(df_supply_norm_sorted[['district', 'Supply_Index']].head(10).to_string(index=False)); print("\n📉 Supply Index BOTTOM 5 (인프라 풍부한 순):")
print(df_supply_norm_sorted[['district', 'Supply_Index']].tail(5).to_string(index=False)); print(f"\n평균: {df_supply_norm['Supply_Index'].mean():.2f}")
print(f"최대: {df_supply_norm['Supply_Index'].max():.2f}"); print(f"최소: {df_supply_norm['Supply_Index'].min():.2f}")
print("\n✅ Step 4 완료: Supply Index")

# ===== Gap Index 계산 =====

print("\n" + "=" * 60); print("🎯 Step 5: Gap Index 계산 (Need - Supply)"); print("=" * 60)

# 통합
df_final = df[['district']].copy()
df_final = df_final.merge(df_need_norm[['district', 'Need_Index']], on='district')
df_final = df_final.merge(df_supply_norm[['district', 'Supply_Index']], on='district')

# Gap 계산
df_final['Gap_Index'] = df_final['Need_Index'] - df_final['Supply_Index']

# 정렬
df_final_sorted = df_final.sort_values('Gap_Index', ascending=False)

print("\n🚨 Gap Index TOP 10 (정책 개입 최우선):"); print(df_final_sorted[['district', 'Need_Index', 'Supply_Index', 'Gap_Index']].head(10).to_string(index=False))
print("\n✅ Gap Index BOTTOM 5 (상대적 안정):"); print(df_final_sorted[['district', 'Need_Index', 'Supply_Index', 'Gap_Index']].tail(5).to_string(index=False))
print(f"\n평균 Gap: {df_final['Gap_Index'].mean():.2f}"); print(f"최대 Gap: {df_final['Gap_Index'].max():.2f}"); print(f"최소 Gap: {df_final['Gap_Index'].min():.2f}")

# 4사분면 분류
median_need = df_final['Need_Index'].median()
median_supply = df_final['Supply_Index'].median()

def classify_quadrant(row):
    if row['Need_Index'] >= median_need and row['Supply_Index'] >= median_supply:
        return 'D: 고위험 대응형'
    elif row['Need_Index'] >= median_need and row['Supply_Index'] < median_supply:
        return 'C: 심각 부족형 ⚠️'
    elif row['Need_Index'] < median_need and row['Supply_Index'] >= median_supply:
        return 'B: 양호형'
    else:
        return 'A: 과잉공급형'

df_final['Quadrant'] = df_final.apply(classify_quadrant, axis=1)

print("\n📊 4사분면 분류:"); print(df_final['Quadrant'].value_counts()) ; print("\n✅ Step 5 완료: Gap Index 및 분류")

# == 4사분면 시각화 (Gap Index 상위 10개만 라벨 표시) =======
plt.figure(figsize=(8, 8))

# 분면별 색상
color_map = {
    'A: 과잉공급형': '#4CAF50',
    'B: 양호형': '#2196F3',
    'C: 심각 부족형': '#F44336',
    'D: 고위험 대응형': '#FF9800'
}

# 산점도
for quad, color in color_map.items():
    subset = df_final[df_final['Quadrant'] == quad]
    plt.scatter(
        subset['Supply_Index'],
        subset['Need_Index'],
        label=quad,
        color=color,
        s=80,
        alpha=0.75
    )

# 중앙값 기준선 (4분면 나누는 선)
plt.axhline(median_need, color='black', linestyle='--', linewidth=1)
plt.axvline(median_supply, color='black', linestyle='--', linewidth=1)

# Gap Index 상위 10개 구 라벨링

top_districts = (
    df_final
    .sort_values('Gap_Index', ascending=False)
    .head(10)
)

for _, row in top_districts.iterrows():
    plt.annotate(
        row['district'],                           # 구 이름
        (row['Supply_Index'], row['Need_Index']),  # 좌표
        textcoords="offset points",
        xytext=(6, 6),
        ha='left',
        fontsize=10,
        fontweight='bold',
        color='black'
    )

# 라벨 & 제목
plt.xlabel("Supply Index (인프라 부족도)")
plt.ylabel("Need Index (위험도)")
plt.title("Need–Supply 기반 4사분면 분류 (Gap Index 상위 10개 강조)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# Need Index 순위 저장
# =========================

need_rank_df = (
    df_need_norm[['district', 'Need_Index']]
    .sort_values('Need_Index', ascending=False)
    .reset_index(drop=True)
)

need_rank_df['rank'] = need_rank_df.index + 1

need_rank_out = OUTPUT_DIR / "need_index_ranking.csv"
need_rank_df.to_csv(
    need_rank_out,
    index=False,
    encoding="utf-8-sig"
)

print("\n📊 Need Index 순위 CSV 저장 완료")
print(need_rank_df.head(10))
print(f"저장 위치: {need_rank_out}")

# =========================
# Supply Index 순위 저장
# =========================

supply_rank_df = (
    df_supply_norm[['district', 'Supply_Index']]
    .sort_values('Supply_Index', ascending=False)
    .reset_index(drop=True)
)

supply_rank_df['rank'] = supply_rank_df.index + 1

supply_rank_out = OUTPUT_DIR / "supply_index_ranking.csv"
supply_rank_df.to_csv(
    supply_rank_out,
    index=False,
    encoding="utf-8-sig"
)

print("\n🏥 Supply Index 순위 CSV 저장 완료")
print(supply_rank_df.head(10))
print(f"저장 위치: {supply_rank_out}")

# =========================
# Need 지표별 TOP 3 구 추출
# =========================

need_top3_rows = []

for var in need_vars:
    temp = (
        df[['district', var]]
        .sort_values(var, ascending=False)
        .head(3)
        .copy()
    )

    temp['need_variable'] = var
    temp['rank'] = range(1, 4)

    need_top3_rows.append(temp)

# 하나의 DataFrame으로 통합
need_top3_df = pd.concat(need_top3_rows, ignore_index=True)

# 컬럼 정리
need_top3_df = need_top3_df[
    ['need_variable', 'rank', 'district', var]
].rename(columns={var: 'raw_value'})

# 저장
need_top3_out = OUTPUT_DIR / "need_variables_top3_by_district.csv"
need_top3_df.to_csv(
    need_top3_out,
    index=False,
    encoding="utf-8-sig"
)

print("\n📌 Need 지표별 상위 3개 구 저장 완료")
print(need_top3_df.head(10))
print(f"저장 위치: {need_top3_out}")


# ===== 결과 저장 =====

# 디렉토리 없으면 자동 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_final.to_csv(
    OUTPUT_DIR / "mhvi_final_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60); print("💾 결과 저장 완료"); print("=" * 60); print(f"\n저장된 변수:"); print(f"  - district (구)") ; print(f"  - Need_Index (위험도)") 
print(f"  - Supply_Index (인프라 부족도)") ; print(f"  - Gap_Index (격차)") ; print(f"  - Quadrant (4사분면 분류)") ; print("\n🎉 분석 완료!") ; print("=" * 60)


# =========================
# 0) 데이터 로드 & 병합
# =========================

need = pd.read_csv(DATA_DIR / "need_tidy.csv")
supply = pd.read_csv(DATA_DIR / "supply_tidy.csv")

df = (
    need
    .merge(supply, on="district", how="inner")
    .sort_values("district")
    .reset_index(drop=True)
)

# =========================
# 1) X(환경) / y(타겟) 정의
# =========================
# 타겟: 자살률 (원하면 depression_experience_rate 등으로 바꿔도 됨)
target = "suicide_rate"

# 환경 변수(설명 변수): "구조/환경" + "공급(정책 레버)"
# - 구조/환경(단기 정책 레버는 아니지만 모델 설명력에 도움)
structural_vars = [
    "elderly_population_rate",
    "unemployment_rate",
    "old_dependency_ratio",
    "single_households",
    "basic_livelihood_recipients",
]

# - 정책 레버 후보 (시뮬레이션할 변수)
policy_levers = [
    "welfare_budget_per_capita",
    "parks_count",
    "libraries_count",
    "public_sports_facilities_count",
    "medical_institutions_count",
    "health_promotion_centers_count",
    "elderly_leisure_welfare_facilities_count",
    "in_home_elderly_welfare_facilities_count",
    "cultural_satisfaction",
]

X_cols = structural_vars + policy_levers

# 결측 처리(안전)
df = df.dropna(subset=[target] + X_cols).copy()

X = df[X_cols]
y = df[target]

# =========================
# 2) 모델 학습 + LOOCV 성능 확인
# =========================
rf = RandomForestRegressor(
    n_estimators=800,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,   # 소표본 과적합 완화
)

loo = LeaveOneOut()

# LOOCV 예측(각 구를 한 번씩 테스트로)
y_pred_loo = cross_val_predict(rf, X, y, cv=loo)

mae = mean_absolute_error(y, y_pred_loo)
rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
r2 = r2_score(y, y_pred_loo)

print("=== LOOCV 성능(자살률 예측) ===") ;print(f"MAE : {mae:.3f}"); print(f"RMSE: {rmse:.3f}") ;print(f"R^2 : {r2:.3f}")

# 최종 모델(전체 데이터로 재학습) -> 정책 시뮬레이션용
rf.fit(X, y)

# =========================
# 3) 정책 시나리오(레버 변화량) 설정
# =========================
# "현실적인 변화량"을 정해줘야 함.
# - 예산: +10%
# - 만족도: +0.2 (척도에 맞춰 조정)
# - 시설/개수: +10 (혹은 +1, +3 등으로 바꿔도 됨)
# 데이터 스케일에 맞춰 마음대로 조절 가능!
scenario = {
    "welfare_budget_per_capita": ("pct", 0.10),     # +10%
    "cultural_satisfaction": ("add", 0.20),         # +0.2점
    "parks_count": ("add", 10),
    "libraries_count": ("add", 2),
    "public_sports_facilities_count": ("add", 2),
    "medical_institutions_count": ("add", 20),
    "health_promotion_centers_count": ("add", 1),
    "elderly_leisure_welfare_facilities_count": ("add", 10),
    "in_home_elderly_welfare_facilities_count": ("add", 5),
}

# =========================
# 4) 구별 정책 추천 (Top-3)
# =========================
rows = []
for idx, row in df.iterrows():
    district = row["district"]
    x_base = row[X_cols].copy()

    # 현재 예측
    y_base = rf.predict(pd.DataFrame([x_base]))[0]

    effects = []
    for lever, (mode, val) in scenario.items():
        x_new = x_base.copy()

        if mode == "add":
            x_new[lever] = x_new[lever] + val
        elif mode == "pct":
            x_new[lever] = x_new[lever] * (1.0 + val)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y_new = rf.predict(pd.DataFrame([x_new]))[0]
        delta = y_new - y_base  # 음수면 개선(자살률 감소)

        effects.append((lever, delta, y_new))

    # 개선 큰 순(가장 음수부터)
    effects.sort(key=lambda t: t[1])

    # top-3 저장
    top3 = effects[:3]
    rows.append({
        "district": district,
        "pred_baseline": y_base,
        "rec1_lever": top3[0][0], "rec1_delta": top3[0][1],
        "rec2_lever": top3[1][0], "rec2_delta": top3[1][1],
        "rec3_lever": top3[2][0], "rec3_delta": top3[2][1],
    })

recommend_df = pd.DataFrame(rows).sort_values("rec1_delta")  # 1순위 개선 큰 구 먼저
print("\n=== 구별 정책 추천 TOP-3 (자살률 예측 감소 기준) ==="); print(recommend_df.head(10).to_string(index=False))


policy_out = OUTPUT_DIR / "policy_recommendations_rf.csv"

recommend_df.to_csv(
    policy_out,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n저장 완료: {policy_out}")
