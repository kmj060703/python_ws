"""
ai_diagnosis.py

RandomForest 기반 사각지대(공급 대비 과도한 위험) 진단

핵심 아이디어:
- 공급(Supply)이 충분한데도 위험(Need)이 과도하게 높은 지역이 있다면,
  "공급이 평균적으로 위험을 낮추는 정책 효과"가 그 지역에서는 제대로 작동하지 않았을 수 있다.
- 그래서 '정책이 정상 작동한 지역'에서 공급→위험의 평균적 관계를 학습하고,
  그 관계로부터 벗어난 지역을 사각지대로 진단한다.
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

    입력:
      - df: (원본 통합 데이터) district + supply 변수들(SUPPLY_VARS) 포함
            ※ 여기에서 '공급 변수'를 가져옴
      - df_final: (최종 지표 테이블) district + Need_Index + Supply_Index 포함
            ※ 여기에서 'Need/Supply 지수'와 사분면, 결과 컬럼을 생성함

    출력:
      - df_final: Quadrant, Predicted_Need_by_Supply, Inefficiency 컬럼이 추가된 결과
      - model: AB 지역으로 학습한 RandomForest 모델
    """

    print("\n" + "=" * 60)
    print("🤖 AI 기반 사각지대 진단 (RandomForest)")
    print("=" * 60)

    # =====================================================
    # 0. 4사분면 분류 (Need / Supply 중앙값 기준)
    # =====================================================
    # 목적:
    # - 자치구를 Need(취약도)와 Supply(인프라)의 상대적 수준에 따라 4가지 유형으로 분류
    # - 중앙값(median)을 기준으로 High/Low를 나눔 (표본 수가 작을 때 평균보다 덜 민감한 기준)
    #
    # A: Need 낮고, Supply 높음  → "과잉공급형" (위험 낮은데 공급은 많은 유형)
    # B: Need 낮고, Supply 낮음  → "양호형" (위험 낮고 공급도 낮음: 큰 문제 없음)
    # C: Need 높고, Supply 낮음  → "심각부족형" (위험 높은데 공급 부족)
    # D: Need 높고, Supply 높음  → "고위험 대응형" (위험도 높고 공급도 많은데 해결이 어려운 유형)
    
    def assign_quadrant(row, median_need, median_supply):
        # Need가 중앙값보다 낮고, Supply가 중앙값 이상이면 A
        if row["Need_Index"] < median_need and row["Supply_Index"] >= median_supply:
            return "A"  # 과잉공급형
        # Need 낮고 Supply도 낮으면 B
        elif row["Need_Index"] < median_need and row["Supply_Index"] < median_supply:
            return "B"  # 양호형
        # Need 높고 Supply 낮으면 C (가장 전형적인 "지원 부족" 문제)
        elif row["Need_Index"] >= median_need and row["Supply_Index"] < median_supply:
            return "C"  # 심각부족형
        # Need 높고 Supply도 높으면 D (공급이 있음에도 위험이 높음)
        else:
            return "D"  # 고위험 대응형

    # 중앙값 계산 (서울 25개 자치구 표본의 가운데 값)
    median_need = df_final["Need_Index"].median()
    median_supply = df_final["Supply_Index"].median()

    # 각 자치구에 대해 Quadrant(A/B/C/D) 라벨을 부여
    df_final["Quadrant"] = df_final.apply(
        assign_quadrant,
        axis=1,
        args=(median_need, median_supply)
    )

    # =====================================================
    # 1. A/B 유형 지역만 학습 데이터로 사용
    #    → "정책이 정상 작동한 평균적 패턴" 학습
    # =====================================================
    # 핵심 가정(탐색적 가정):
    # - A/B는 Need가 낮은 지역 → (원인 불문) '위험이 상대적으로 낮게 유지되는' 그룹
    # - 이 그룹에서 "공급이 어떤 조합이면 평균적으로 Need가 낮게 나타나는가"를 학습
    #
    # 즉, AB 그룹을 이용해
    #   공급(SUPPLY_VARS) → 기대되는(평균적) Need 수준
    # 을 모델로 학습한다.
    #
    # 그리고 전체 지역에 대해
    #   실제 Need - (공급으로 기대되는 Need)
    # 를 계산하면,
    # "공급 대비 Need가 과도한 지역" = 사각지대 후보가 된다.
    ab_districts = df_final.loc[
        df_final["Quadrant"].isin(["A", "B"]),
        "district"
    ]

    # 학습용 입력 X_train:
    # - 원본 df에서 AB 자치구만 필터링
    # - 공급 변수(SUPPLY_VARS)만 사용
    df_train = df[df["district"].isin(ab_districts)]
    X_train = df_train[SUPPLY_VARS]

    # 학습용 타겟 y_train:
    # - df_final에서 AB 자치구만 필터링
    # - Need_Index를 타겟으로 사용
    #
    # ⚠️ 주의:
    # - X_train은 df(원본), y_train은 df_final(가공 결과)에서 가져오므로
    #   "district 기준으로 행 순서/정렬이 완전히 동일"하다는 전제가 깔려 있음.
    #   (만약 순서가 다르면 X와 y가 매칭이 꼬일 수 있음)
    y_train = df_final.loc[
        df_final["district"].isin(ab_districts),
        "Need_Index"
    ]

    # =====================================================
    # 2. RandomForest 학습
    # =====================================================
    # RandomForest 사용 이유(직관):
    # - 선형 모델이 놓치기 쉬운 비선형 관계/상호작용(예: 특정 인프라 조합) 포착 가능
    # - 단, 표본이 작아 과적합 위험이 있으므로 깊이 제한 등 튜닝이 중요할 수 있음
    model = RandomForestRegressor(
        n_estimators=300,   # 트리 개수 (많을수록 안정적이지만 과적합은 깊이에서 주로 발생)
        max_depth=6,        # 트리 깊이 제한 (너무 깊으면 n=25에서 과적합 쉽게 발생)
        random_state=42
    )
    model.fit(X_train, y_train)

    # =====================================================
    # 3. 모든 지역에 대해 "정책 기준 위험도" 예측
    # =====================================================
    # 이제 AB에서 학습한 "공급→Need의 평균적 패턴"을
    # 전체 자치구에 적용한다.
    #
    # X_all: 전체 자치구의 공급 변수 행렬
    X_all = df[SUPPLY_VARS]

    # Predicted_Need_by_Supply:
    # - "이 공급 수준이라면 평균적으로 이 정도 Need가 나와야 한다"는
    #   모델의 기대값(정책이 정상 작동했을 때의 기준선 같은 것)
    df_final["Predicted_Need_by_Supply"] = model.predict(X_all)

    # =====================================================
    # 4. 사각지대 점수 (Inefficiency)
    # =====================================================
    # Inefficiency = 실제 Need - 공급으로 기대되는 Need
    #
    # 해석:
    # - 양수(>0): 공급 수준에 비해 Need가 "과도하게 높음"
    #           → 정책/인프라가 평균적 기대만큼 효과를 내지 못했을 가능성(사각지대 후보)
    # - 음수(<0): 공급 대비 Need가 낮음
    #           → 공급이 효과적으로 작동했거나, 다른 요인으로 위험이 낮은 지역
    df_final["Inefficiency"] = (
        df_final["Need_Index"] - df_final["Predicted_Need_by_Supply"]
    )

    # =====================================================
    # 5. 사각지대 순위 테이블
    # =====================================================
    # 사각지대(양수) 후보를 높은 순서대로 보기 위해
    # Inefficiency 내림차순으로 정렬한 표를 만든다.
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

    # 순위 테이블 저장:
    # OUTPUT_DIR는 config에서 Path 객체로 정의되어 있다고 가정 (OUTPUT_DIR / "파일명" 사용)
    df_ai.to_csv(
        OUTPUT_DIR / "ai_blindspot_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("💾 ai_blindspot_ranking.csv 저장 완료")

    # =====================================================
    # 6. SHAP 기반 원인 분석
    # =====================================================
    # 목적:
    # - "왜 이 지역이 Inefficiency가 큰가?"를 공급 변수 관점에서 해석 가능하게 만들기
    # - SHAP은 각 변수의 기여도를 지역(샘플)별로 분해해준다.
    #
    # 주의:
    # - SHAP은 '인과'가 아니라 '모델 내부에서 예측에 어떻게 기여했는지'를 보여준다.
    # - 즉, "이 변수가 자살/위험을 만든다"가 아니라
    #   "이 변수가 모델의 예측을 이렇게 밀어올렸다/내렸다" 수준의 해석이 안전하다.
    print("\n🔍 SHAP 기반 원인 분석 시작")

    # shap.Explainer(model, X_all):
    # - 트리 모델이므로 내부적으로 TreeExplainer를 사용하게 될 가능성이 큼
    # - X_all은 background/데이터 분포 정보로 활용될 수 있음
    explainer = shap.Explainer(model, X_all)
    shap_values = explainer(X_all)

    # shap_values.values: (샘플 수, 변수 수) 형태의 SHAP 값 행렬
    # 이를 DataFrame으로 만들어 변수명(SUPPLY_VARS)을 컬럼으로 붙인다.
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=SUPPLY_VARS
    )

    # 각 행(자치구)에 대해:
    # - district 라벨을 붙여서 어떤 자치구의 SHAP인지 식별 가능하게 함
    # - Inefficiency(사각지대 점수)를 붙여서 "사각지대일수록 어떤 변수 패턴이 있나" 탐색 가능
    # - Quadrant(A/B/C/D)도 같이 저장하여 유형별 비교 가능
    shap_df["district"] = df["district"].values
    shap_df["Inefficiency"] = df_final["Inefficiency"].values
    shap_df["Quadrant"] = df_final["Quadrant"].values

    # 사각지대 의심 지역만 저장:
    # Inefficiency > 0인 지역(공급 대비 Need가 과도한 지역)만 필터링
    blindspots = shap_df[shap_df["Inefficiency"] > 0]

    # SHAP 결과 저장:
    # 이 파일은 "사각지대 후보 지역들의 변수 기여도" 테이블이라고 보면 됨
    blindspots.to_csv(
        OUTPUT_DIR / "ai_blindspot_shap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("💾 ai_blindspot_shap.csv 저장 완료")
    print("✅ AI 기반 사각지대 진단 완료")
    print("=" * 60)

    # 최종 결과(df_final)와 학습 모델을 반환
    return df_final, model
