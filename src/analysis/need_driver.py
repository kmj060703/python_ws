# src/analysis/need_driver.py
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    NEED_VARS,
    WEIGHTS_NEED,
)


# =====================================================
# 정책 방향 매핑
# =====================================================
POLICY_MAP = {
    "suicide_rate": [
        "고위험군 조기발견(게이트키퍼/위기신호 모니터링) 및 정신건강센터 기능 강화",
        "24시간 위기상담 접근성 확대 및 정신응급 연계 프로토콜 정교화",
    ],
    "depression_experience_rate": [
        "우울 고위험군 선별검사 및 상담·치료 연계 강화",
        "생활권 기반 찾아가는 심리 지원 서비스 확대",
    ],
    "perceived_stress_rate": [
        "직장·생활 스트레스 완화 프로그램 확대",
        "여가·문화·커뮤니티 인프라 접근성 개선",
    ],
    "high_risk_drinking_rate": [
        "고위험 음주군 대상 중독 상담 및 가족 연계 프로그램",
    ],
    "unmet_medical_need_rate": [
        "야간·주말 의료 접근성 개선 및 취약계층 진료 연계",
    ],
    "unemployment_rate": [
        "구직·재취업 지원과 정신건강 서비스 결합",
    ],
    "elderly_population_rate": [
        "노인 고립 예방 및 지역 기반 돌봄 강화",
    ],
    "old_dependency_ratio": [
        "돌봄 부담 완화를 위한 공공·재가 돌봄 확대",
    ],
    "single_households": [
        "1인가구 사회적 고립 예방 커뮤니티 정책 강화",
    ],
    "basic_livelihood_recipients": [
        "경제취약계층 대상 정신건강·복지 통합 지원",
    ],
}

# =====================================================
# 유틸
# =====================================================
def _minmax_0_100(col: pd.Series) -> pd.Series:
    mn, mx = col.min(), col.max()
    if mx == mn:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return 100.0 * (col - mn) / (mx - mn)


# =====================================================
# 핵심 분석 함수
# =====================================================
def run_need_driver_analysis(df_need_norm: pd.DataFrame) -> pd.DataFrame:
    """
    입력:
      - district
      - *_norm need 변수들

    출력:
      - district
      - top1_factor, top2_factor, top3_factor
      - policy_direction_1~3
    """

    DISTRICT_COL = "district"
    NEED_FEATURES = [f"{c}_norm" for c in NEED_VARS]

    # --- validate (변수명 수정: need_df_norm -> df_need_norm)
    missing = [c for c in [DISTRICT_COL] + NEED_FEATURES if c not in df_need_norm.columns]
    if missing:
        raise ValueError(f"[need_driver] missing columns: {missing}")

    df = df_need_norm[[DISTRICT_COL] + NEED_FEATURES].copy()

    # --- 결측 처리 (중앙값)
    for c in NEED_FEATURES:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    # --- 0~100 정규화
    scaled = pd.DataFrame({DISTRICT_COL: df[DISTRICT_COL]})
    for c in NEED_FEATURES:
        scaled[c] = _minmax_0_100(df[c])

    # --- Need Index 기여 점수 계산
    # 가중치 키 수정: WEIGHTS_NEED는 이미 '_norm'이 붙은 키를 가지고 있음
    total_w = float(sum(WEIGHTS_NEED.get(c, 0) for c in NEED_FEATURES))
    if total_w == 0:
        raise ValueError("[need_driver] total weight is zero")

    contrib_score = pd.DataFrame({DISTRICT_COL: df[DISTRICT_COL]})
    for c in NEED_FEATURES:
        w = WEIGHTS_NEED.get(c, 0)
        contrib_score[c] = scaled[c] * w / total_w

    contrib_score["Need_Index"] = contrib_score[NEED_FEATURES].sum(axis=1)

    # --- 기여 비율(%)
    contrib_share = contrib_score.copy()
    for c in NEED_FEATURES:
        contrib_share[c] = np.where(
            contrib_score["Need_Index"] > 0,
            100.0 * contrib_score[c] / contrib_score["Need_Index"],
            0.0,
        )

    # --- long format
    df_long = (
        contrib_score[[DISTRICT_COL, "Need_Index"] + NEED_FEATURES]
        .melt(id_vars=[DISTRICT_COL, "Need_Index"],
              var_name="need_factor",
              value_name="contrib_score")
        .merge(
            contrib_share[[DISTRICT_COL] + NEED_FEATURES]
            .melt(id_vars=[DISTRICT_COL],
                  var_name="need_factor",
                  value_name="contrib_share_pct"),
            on=[DISTRICT_COL, "need_factor"],
            how="left",
        )
    )

    # --- 구별 Top3
    top3 = (
        df_long.sort_values([DISTRICT_COL, "contrib_score"], ascending=[True, False])
        .groupby(DISTRICT_COL)
        .head(3)
        .copy()
    )

    # --- 정책 제안 생성 (변수명 정리)
    rec_rows = []
    for d, g in top3.groupby(DISTRICT_COL):
        # '_norm' 제거하여 원본 변수명으로 매핑
        factors_raw = g["need_factor"].tolist()
        factors = [f.replace("_norm", "") for f in factors_raw]
        
        recs = []
        for f in factors:
            recs.extend(POLICY_MAP.get(f, ["(정책 방향 매핑 필요)"]))

        recs_clean = list(dict.fromkeys(recs))  # 중복 제거

        rec_rows.append({
            DISTRICT_COL: d,
            "top1_factor": factors[0] if len(factors) > 0 else "",
            "top2_factor": factors[1] if len(factors) > 1 else "",
            "top3_factor": factors[2] if len(factors) > 2 else "",
            "policy_direction_1": recs_clean[0] if len(recs_clean) > 0 else "",
            "policy_direction_2": recs_clean[1] if len(recs_clean) > 1 else "",
            "policy_direction_3": recs_clean[2] if len(recs_clean) > 2 else "",
        })

    return pd.DataFrame(rec_rows)


# =====================================================
# 실행 진입점
# =====================================================
def main():
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_path = BASE_DIR / "data" / "processed" / "mhvi_final_result.csv"
    output_dir = BASE_DIR / "data" / "outputs" / "recommend_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    result = run_need_driver_analysis(df)

    result.to_csv(
        output_dir / "need_policy_recommendation_by_district.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 정책 제안 결과 생성 완료")
    print(f"📁 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()