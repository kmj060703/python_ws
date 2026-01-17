"""
ai_diagnosis.py

RandomForest 기반 사각지대(공급 대비 과도한 위험) 진단
"""
import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import SUPPLY_VARS, OUTPUT_DIR


def run_ai_diagnosis(df, df_final):
    """
    A/B 유형(정상 작동 지역)에서 학습한
    '공급 → 위험 완화의 평균적 정책 효과'를 기준으로,
    실제 위험이 과도한 지역을 사각지대로 진단한다.
    """

    print("\n" + "=" * 60)
    print("🤖 AI 기반 사각지대 진단 (RandomForest)")
    print("=" * 60)

    # =====================================================
    # 0. 4사분면 분류 (Need / Supply 중앙값 기준)
    # =====================================================
    def assign_quadrant(row, median_need, median_supply):
        if row["Need_Index"] < median_need and row["Supply_Index"] >= median_supply:
            return "A"  # 과잉공급형
        elif row["Need_Index"] < median_need and row["Supply_Index"] < median_supply:
            return "B"  # 양호형
        elif row["Need_Index"] >= median_need and row["Supply_Index"] < median_supply:
            return "C"  # 심각부족형
        else:
            return "D"  # 고위험 대응형

    median_need = df_final["Need_Index"].median()
    median_supply = df_final["Supply_Index"].median()

    df_final["Quadrant"] = df_final.apply(
        assign_quadrant,
        axis=1,
        args=(median_need, median_supply)
    )

    # =====================================================
    # 1. A/B 유형 지역만 학습 데이터로 사용
    #    → "정책이 정상 작동한 평균적 패턴" 학습
    # =====================================================
    ab_districts = df_final.loc[
        df_final["Quadrant"].isin(["A", "B"]),
        "district"
    ]

    # 공급 변수 (원본 df 기준)
    df_train = df[df["district"].isin(ab_districts)]
    X_train = df_train[SUPPLY_VARS]

    # 타겟 위험도 (df_final 기준)
    y_train = df_final.loc[
        df_final["district"].isin(ab_districts),
        "Need_Index"
    ]

    # =====================================================
    # 2. RandomForest 학습
    # =====================================================
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # =====================================================
    # 3. 모든 지역에 대해 "정책 기준 위험도" 예측
    # =====================================================
    X_all = df[SUPPLY_VARS]
    df_final["Predicted_Need_by_Supply"] = model.predict(X_all)

    # =====================================================
    # 4. 사각지대 점수 (Inefficiency)
    # =====================================================
    df_final["Inefficiency"] = (
        df_final["Need_Index"] - df_final["Predicted_Need_by_Supply"]
    )

    # =====================================================
    # 5. 사각지대 순위 테이블
    # =====================================================
    df_ai = (
        df_final[[
            "district",
            "Quadrant",
            "Need_Index",
            "Supply_Index",
            "Predicted_Need_by_Supply",
            "Inefficiency"
        ]]
        .sort_values("Inefficiency", ascending=False)
        .reset_index(drop=True)
    )

    print("\n🚨 AI가 찾은 사각지대 TOP 10")
    print(df_ai.head(10).to_string(index=False))

    df_ai.to_csv(
        OUTPUT_DIR / "ai_blindspot_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("💾 ai_blindspot_ranking.csv 저장 완료")

    # =====================================================
    # 6. SHAP 기반 원인 분석
    # =====================================================
    print("\n🔍 SHAP 기반 원인 분석 시작")

    explainer = shap.Explainer(model, X_all)
    shap_values = explainer(X_all)

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=SUPPLY_VARS
    )

    shap_df["district"] = df["district"].values
    shap_df["Inefficiency"] = df_final["Inefficiency"].values
    shap_df["Quadrant"] = df_final["Quadrant"].values

    # 사각지대 의심 지역만 저장
    blindspots = shap_df[shap_df["Inefficiency"] > 0]

    blindspots.to_csv(
        OUTPUT_DIR / "ai_blindspot_shap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("💾 ai_blindspot_shap.csv 저장 완료")
    print("✅ AI 기반 사각지대 진단 완료")
    print("=" * 60)

    return df_final, model
