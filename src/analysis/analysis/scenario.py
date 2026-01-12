import os
import numpy as np
import pandas as pd
from src.analysis.features.scaler import MinMaxScaler100


def run_policy_scenario_analysis(need_df, supply_df, final_df, config, output_path):
    district_col = config['keys']['district_col']
    parks_pct = config['scenario']['parks_pct']
    health_add = config['scenario']['health_center_add']
    medical_pct = config['scenario']['medical_pct']

    policies = [
        ('parks_count', lambda x: x * (1 + parks_pct), 'parks_count +10%'),
        ('health_promotion_centers_count', lambda x: x + health_add, 'health_promotion_centers_count +1'),
        ('medical_institutions_count', lambda x: x * (1 + medical_pct), 'medical_institutions_count +10%'),
    ]

    # 검증
    required_supply_cols = [
        'welfare_budget_per_capita', 'public_sports_facilities_count',
        'parks_count', 'libraries_count', 'medical_institutions_count',
        'health_promotion_centers_count', 'elderly_leisure_welfare_facilities_count',
        'in_home_elderly_welfare_facilities_count', 'cultural_satisfaction'
    ]
    missing = [c for c in required_supply_cols if c not in supply_df.columns]
    if missing:
        raise ValueError(f"[scenario] supply_df missing columns: {missing}")

    if district_col not in final_df.columns or 'Need_Index' not in final_df.columns:
        raise ValueError("[scenario] final_df must contain 'district' and 'Need_Index'")

    # 결측치 중앙값 대체
    supply_base = supply_df.copy()
    for c in required_supply_cols:
        if supply_base[c].isnull().any():
            supply_base[c] = supply_base[c].fillna(supply_base[c].median())

    # ✅ 1. baseline 스케일러 fit (이것이 모든 정책의 기준!)
    scaler_base = MinMaxScaler100()
    scaler_base.fit(supply_base[required_supply_cols], required_supply_cols)

    # baseline Supply_Index 계산
    supply_scaled_base = scaler_base.transform(supply_base[required_supply_cols], required_supply_cols)
    w_supply = config['index_weights']['supply']
    total_w = sum(w_supply.values())
    supply_before = np.zeros(len(supply_base))
    for feat, weight in w_supply.items():
        supply_before += supply_scaled_base[feat].to_numpy() * float(weight)
    supply_before = supply_before / total_w

    # district 순서 맞추기
    final_order = final_df[[district_col]].copy()
    supply_before_aligned = final_order.merge(
        pd.DataFrame({district_col: supply_base[district_col], "_supply_before": supply_before}),
        on=district_col, how='left'
    )["_supply_before"].to_numpy()

    need_idx = final_df['Need_Index'].to_numpy()
    gap_before = need_idx - supply_before_aligned

    # ✅ 2. 각 정책 적용: **baseline 스케일러 기준으로 transform만 사용** (재fit 금지!)
    results_all = []
    for col, transform_func, policy_name in policies:
        supply_policy = supply_base.copy()
        supply_policy[col] = transform_func(supply_policy[col])

        # ✅ 여기서 **재fit 하지 않음** — baseline의 min/max만 사용
        supply_scaled_policy = scaler_base.transform(supply_policy[required_supply_cols], required_supply_cols)

        supply_after = np.zeros(len(supply_policy))
        for feat, weight in w_supply.items():
            supply_after += supply_scaled_policy[feat].to_numpy() * float(weight)
        supply_after = supply_after / total_w

        gap_after = need_idx - supply_after
        gap_delta = gap_before - gap_after  # 양수면 개선

        for i, d in enumerate(final_df[district_col].to_list()):
            results_all.append({
                district_col: d,
                "policy": policy_name,
                "supply_before": float(supply_before_aligned[i]),
                "supply_after": float(supply_after[i]),
                "gap_before": float(gap_before[i]),
                "gap_after": float(gap_after[i]),
                "gap_delta": float(gap_delta[i]),
            })

    df_all = pd.DataFrame(results_all)

    # 구별 최적 정책
    idx = df_all.groupby(district_col)["gap_delta"].idxmax()
    df_best = df_all.loc[idx].copy()
    df_best = df_best.rename(columns={
        "policy": "best_policy",
        "gap_delta": "best_gap_delta",
        "supply_after": "best_supply_after",
        "gap_after": "best_gap_after",
    })
    df_best = df_best[[district_col, "best_policy", "best_gap_delta",
                       "supply_before", "best_supply_after", "gap_before", "best_gap_after"]]

    # 정책별 랭킹
    best_count = df_best["best_policy"].value_counts().reset_index()
    best_count.columns = ["policy", "best_count"]
    df_rank = df_all.groupby("policy")["gap_delta"].agg(
        mean_gap_delta="mean", median_gap_delta="median"
    ).reset_index().merge(best_count, on="policy", how="left").fillna(0)
    df_rank["best_count"] = df_rank["best_count"].astype(int)
    df_rank = df_rank.sort_values(["best_count", "mean_gap_delta"], ascending=[False, False])

    # 저장
    out_dir = os.path.dirname(output_path)
    if not out_dir:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    df_all.to_csv(os.path.join(out_dir, "policy_scenario_effect_all.csv"), index=False, encoding="utf-8-sig")
    df_best.to_csv(os.path.join(out_dir, "policy_scenario_best.csv"), index=False, encoding="utf-8-sig")
    df_rank.to_csv(os.path.join(out_dir, "policy_scenario_policy_ranking.csv"), index=False, encoding="utf-8-sig")

    # ✅ 추가: 각 구별 최적 정책 저장
    df_best_output = df_best[[district_col, "best_policy"]].copy()
    df_best_output.to_csv(os.path.join(out_dir, "best_policy_by_district.csv"), index=False, encoding="utf-8-sig")

    print(f"✅ Policy analysis results saved to: {out_dir}")
