"""
ai_diagnosis.py

RandomForest 기반 사각지대(공급 대비 과도한 위험) 진단
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import SUPPLY_VARS, OUTPUT_DIR


def run_ai_diagnosis(df, df_final):
    """
    Supply 변수들로 Need_Index를 예측하고,
    실제 Need와의 차이로 '구조적 사각지대'를 계산한다.
    """

    print("\n" + "=" * 60)
    print("🤖 AI 기반 사각지대 진단 (RandomForest)")
    print("=" * 60)

    # ------------------------
    # 1. 입력 X (공급 변수들)
    # ------------------------
    X = df[SUPPLY_VARS]

    # ------------------------
    # 2. 타겟 y (위험도)
    # ------------------------
    y = df_final["Need_Index"]

    # ------------------------
    # 3. RandomForest 학습
    # ------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    # ------------------------
    # 4. 공급으로 예측한 Need
    # ------------------------
    df_final["Predicted_Need_by_Supply"] = model.predict(X)

    # ------------------------
    # 5. 사각지대 점수 (잔차)
    # ------------------------
    df_final["Inefficiency"] = (
        df_final["Need_Index"] - df_final["Predicted_Need_by_Supply"]
    )

    # ------------------------
    # 6. 사각지대 순위
    # ------------------------
    df_ai = (
        df_final[["district", "Need_Index", "Predicted_Need_by_Supply", "Inefficiency"]]
        .sort_values("Inefficiency", ascending=False)
        .reset_index(drop=True)
    )

    print("\n🚨 AI가 찾은 사각지대 TOP 10")
    print(df_ai.head(10).to_string(index=False))

    # ------------------------
    # 7. 결과 저장
    # ------------------------
    df_ai.to_csv(
        OUTPUT_DIR / "ai_blindspot_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n💾 ai_blindspot_ranking.csv 저장 완료")

    # ============================
    # 8. SHAP 분석
    # ============================
    print("\n🔍 SHAP 기반 원인 분석 시작")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)


    shap_df = pd.DataFrame(
    shap_values.values,
    columns=SUPPLY_VARS
    )

    shap_df["district"] = df["district"].values
    shap_df["Inefficiency"] = df_final["Inefficiency"].values

    # Inefficiency 양수 지역만 추출 (사각지대)
    blindspots = shap_df[shap_df["Inefficiency"] > 0]

    blindspots.to_csv(
        OUTPUT_DIR / "ai_blindspot_shap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("💾 ai_blindspot_shap.csv 저장 완료")


    return df_final, model
