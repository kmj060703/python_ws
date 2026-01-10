"""
Random Forest 기반 정책 효과 시뮬레이션
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import DATA_DIR, OUTPUT_DIR, POLICY_SCENARIO


def run_policy_simulation():
    """정책 시뮬레이션 실행"""
    
    # 데이터 로드
    need = pd.read_csv(DATA_DIR / "need_tidy.csv")
    supply = pd.read_csv(DATA_DIR / "supply_tidy.csv")
    
    df = (
        need.merge(supply, on="district", how="inner")
        .sort_values("district")
        .reset_index(drop=True)
    )
    
    # 타겟 및 설명변수 정의
    target = "suicide_rate"
    
    structural_vars = [
        "elderly_population_rate",
        "unemployment_rate",
        "old_dependency_ratio",
        "single_households",
        "basic_livelihood_recipients",
    ]
    
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
    
    # 결측 처리
    df = df.dropna(subset=[target] + X_cols).copy()
    
    X = df[X_cols]
    y = df[target]
    
    # 모델 학습 및 LOOCV
    rf = RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
    )
    
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(rf, X, y, cv=loo)
    
    mae = mean_absolute_error(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    r2 = r2_score(y, y_pred_loo)
    
    print("=" * 60)
    print("=== LOOCV 성능(자살률 예측) ===")
    print("=" * 60)
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")
    
    # 전체 데이터로 재학습
    rf.fit(X, y)
    
    # 구별 정책 추천
    rows = []
    for idx, row in df.iterrows():
        district = row["district"]
        x_base = row[X_cols].copy()
        
        # 현재 예측
        y_base = rf.predict(pd.DataFrame([x_base]))[0]
        
        effects = []
        for lever, (mode, val) in POLICY_SCENARIO.items():
            x_new = x_base.copy()
            
            if mode == "add":
                x_new[lever] = x_new[lever] + val
            elif mode == "pct":
                x_new[lever] = x_new[lever] * (1.0 + val)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            y_new = rf.predict(pd.DataFrame([x_new]))[0]
            delta = y_new - y_base  # 음수면 개선
            
            effects.append((lever, delta, y_new))
        
        # 개선 큰 순 정렬
        effects.sort(key=lambda t: t[1])
        
        # top-3 저장
        top3 = effects[:3]
        rows.append({
            "district": district,
            "pred_baseline": y_base,
            "rec1_lever": top3[0][0],
            "rec1_delta": top3[0][1],
            "rec2_lever": top3[1][0],
            "rec2_delta": top3[1][1],
            "rec3_lever": top3[2][0],
            "rec3_delta": top3[2][1],
        })
    
    recommend_df = pd.DataFrame(rows).sort_values("rec1_delta")
    
    print("\n" + "=" * 60)
    print("=== 구별 정책 추천 TOP-3 (자살률 예측 감소 기준) ===")
    print("=" * 60)
    print(recommend_df.head(10).to_string(index=False))
    
    # 저장
    policy_out = OUTPUT_DIR / "policy_recommendations_rf.csv"
    recommend_df.to_csv(policy_out, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {policy_out}")
    
    return recommend_df


if __name__ == "__main__":
    run_policy_simulation()