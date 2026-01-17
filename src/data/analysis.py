# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =============================
# 경로 설정 (Git 레포 기준)
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(DATA_DIR, "mhvi_final_result.csv")

NEED_PATH = os.path.join(DATA_DIR, "need_tidy.csv")
SUPPLY_PATH = os.path.join(DATA_DIR, "supply_tidy.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# =============================
# 데이터 로드
# =============================

df_need = pd.read_csv(NEED_PATH)
df_supply = pd.read_csv(SUPPLY_PATH)

df = df_need.merge(df_supply, on="district", how="inner")

# =============================
# 정규화 함수
# =============================

def normalize_to_100(series, direction="positive"):
    scaler = MinMaxScaler(feature_range=(0, 100))
    values = series.values.reshape(-1, 1)

    if direction == "positive":
        norm = scaler.fit_transform(values)
    else:
        norm = 100 - scaler.fit_transform(values)

    return norm.flatten()

# =============================
# Step 1. Need 정규화
# =============================

need_vars = [
    "suicide_rate",
    "depression_experience_rate",
    "perceived_stress_rate",
    "high_risk_drinking_rate",
    "unmet_medical_need_rate",
    "unemployment_rate",
    "elderly_population_rate",
    "old_dependency_ratio",
    "single_households",
    "basic_livelihood_recipients",
]

df_need_norm = df[["district"]].copy()

for v in need_vars:
    df_need_norm[f"{v}_norm"] = normalize_to_100(df[v], "positive")

# =============================
# Step 2. Supply 정규화
# =============================

supply_vars = [
    "welfare_budget_per_capita",
    "public_sports_facilities_count",
    "parks_count",
    "libraries_count",
    "medical_institutions_count",
    "health_promotion_centers_count",
    "elderly_leisure_welfare_facilities_count",
    "in_home_elderly_welfare_facilities_count",
    "cultural_satisfaction",
]

df_supply_norm = df[["district"]].copy()

for v in supply_vars:
    df_supply_norm[f"{v}_norm"] = normalize_to_100(df[v], "negative")

# =============================
# Step 3. Need Index
# =============================

weights_need = {
    "suicide_rate_norm": 0.25,
    "depression_experience_rate_norm": 0.125,
    "perceived_stress_rate_norm": 0.125,
    "high_risk_drinking_rate_norm": 0.05,
    "basic_livelihood_recipients_norm": 0.125,
    "unemployment_rate_norm": 0.125,
    "single_households_norm": 0.10,
    "old_dependency_ratio_norm": 0.05,
    "elderly_population_rate_norm": 0.05,
    "unmet_medical_need_rate_norm": 0.10,
}

df_need_norm["Need_Index"] = 0
for k, w in weights_need.items():
    df_need_norm["Need_Index"] += df_need_norm[k] * w

# =============================
# Step 4. Supply Index
# =============================

weights_supply = {
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


df_supply_norm["Supply_Index"] = 0
for k, w in weights_supply.items():
    df_supply_norm["Supply_Index"] += (100 - df_supply_norm[k]) * w

# =============================
# Step 5. Gap + Quadrant
# =============================

df_final = (
    df[["district"]]
    .merge(df_need_norm[["district", "Need_Index"]], on="district")
    .merge(df_supply_norm[["district", "Supply_Index"]], on="district")
)

df_final["Gap_Index"] = df_final["Need_Index"] - df_final["Supply_Index"]

median_need = df_final["Need_Index"].median()
median_supply = df_final["Supply_Index"].median()

def classify(row):
    if row["Need_Index"] >= median_need and row["Supply_Index"] >= median_supply:
        return "D: 고위험 대응형"
    elif row["Need_Index"] >= median_need and row["Supply_Index"] < median_supply:
        return "C: 심각 부족형"
    elif row["Need_Index"] < median_need and row["Supply_Index"] >= median_supply:
        return "B: 양호형"
    else:
        return "A: 과잉공급형"

df_final["Quadrant"] = df_final.apply(classify, axis=1)

# =============================
# 저장
# =============================

df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("✅ mhvi_final_result.csv 생성 완료")
print(f"📁 경로: {OUTPUT_PATH}")
