# src/analysis/analysis/need_driver.py
import os
import numpy as np
import pandas as pd


NEED_FEATURES = [
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

# ✅ 변수별 정책 방향(너희 프로젝트 톤에 맞게 "방향" 중심으로)
POLICY_MAP = {
    "suicide_rate": [
        "고위험군 조기발견(게이트키퍼/위기신호 모니터링) + 정신건강센터 기능 강화",
        "위기상담 접근성(24시간 통합 상담 핫라인 홍보 등) 확장 + 응급연계(정신응급/119/병원) 프로토콜 정교화",
    ],
    "depression_experience_rate": [
        "우울 고위험군 선별검사(학교/직장/동주민센터 연계) + 상담 바우처/치료연계 강화",
        "정신건강 인식개선 캠페인 + 상담 문턱(비용/낙인) 낮추는 서비스 확대",
        "생활권 기반 심리 지원 : 일상 공간으로 찾아가는 상담 서비스"
    ],
    "perceived_stress_rate": [
        "직장/생활 스트레스 완화 프로그램(마음건강 교육, 스트레스 관리 코칭) 확대",
        "휴식/여가 인프라(문화·체육·공원) 접근성 개선 + 커뮤니티 활동 촉진",
    ],
    "high_risk_drinking_rate": [
        "고위험 음주군 대상 중독상담 연계",
        "가족 상담, 직업 연계 프로그램 병행",
    ],
    "unmet_medical_need_rate": [
        "의료 접근성(야간·주말·이동/예약) 개선 + 취약층 진료비 지원/연계",
        "지역 내 1차의료-정신건강 서비스 연계(동네의원/보건소/센터) 강화",
    ],
    "unemployment_rate": [
        "구직·재취업 지원(훈련/매칭/상담) 강화 + 청년/중장년 타깃 프로그램 분리",
        "실업 스트레스 완화 위한 마음건강 패키지(상담+일자리 서비스) 결합",
    ],
    "elderly_population_rate": [
        "노인 고립·우울 예방: 방문/모니터링 + 여가·커뮤니티 프로그램 강화",
        "만성질환·정신건강 통합관리(보건소/센터 연계) 강화",
    ],
    "old_dependency_ratio": [
        "돌봄 부담 완화: 재가/방문 돌봄, 공공 돌봄 확대",
        "고령가구 대상 지역기반 안전망(안부확인, 커뮤니티케어) 촘촘히",
    ],
    "single_households": [
        "1인가구 고립 예방: 커뮤니티 프로그램/모임/공유공간 활성화",
        "위기 신호(단절·경제·정서) 조기 발견 위한 생활접점 연계 강화",
    ],
    "basic_livelihood_recipients": [
        "경제취약층 대상 마음건강+복지 연계",
        "주거·생계 불안 완화 지원과 정신건강 서비스 동시 제공",
    ],
}


def _minmax_0_100(col: pd.Series) -> pd.Series:
    mn = col.min()
    mx = col.max()
    if mx == mn:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return 100.0 * (col - mn) / (mx - mn)


def run_need_driver_analysis(need_df: pd.DataFrame, config: dict, output_dir: str):
    district_col = config["keys"]["district_col"]
    weights = config["index_weights"]["need"]

    # --- validate
    missing = [c for c in [district_col] + NEED_FEATURES if c not in need_df.columns]
    if missing:
        raise ValueError(f"[need_driver] need_df missing columns: {missing}")

    os.makedirs(output_dir, exist_ok=True)

    df = need_df[[district_col] + NEED_FEATURES].copy()

    # --- 결측 처리(중앙값)
    for c in NEED_FEATURES:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    # --- 0~100 정규화
    scaled = pd.DataFrame({district_col: df[district_col]})
    for c in NEED_FEATURES:
        scaled[c] = _minmax_0_100(df[c])

    # --- Need_Index(가중평균)
    total_w = float(sum(weights.get(c, 0) for c in NEED_FEATURES))
    if total_w == 0:
        raise ValueError("[need_driver] total need weight is zero")

    contrib_score = pd.DataFrame({district_col: df[district_col]})
    for c in NEED_FEATURES:
        w = float(weights.get(c, 0))
        contrib_score[c] = scaled[c] * w / total_w  # Need_Index에 기여하는 "점수" 기여분

    need_index = contrib_score[NEED_FEATURES].sum(axis=1)
    contrib_score["Need_Index_recomputed"] = need_index

    # --- 비율(%) 기여도: 구별 Need_Index에서 각 요인이 차지하는 비중
    contrib_share = contrib_score.copy()
    for c in NEED_FEATURES:
        contrib_share[c] = np.where(
            contrib_score["Need_Index_recomputed"] > 0,
            100.0 * contrib_score[c] / contrib_score["Need_Index_recomputed"],
            0.0,
        )

    # --- long format (UI나 리포트에 제일 쓰기 좋음)
    df_long = (
        contrib_score[[district_col, "Need_Index_recomputed"] + NEED_FEATURES]
        .melt(id_vars=[district_col, "Need_Index_recomputed"], var_name="need_factor", value_name="contrib_score")
        .merge(
            contrib_share[[district_col] + NEED_FEATURES]
            .melt(id_vars=[district_col], var_name="need_factor", value_name="contrib_share_pct"),
            on=[district_col, "need_factor"],
            how="left",
        )
    )
    df_long["rank_in_district"] = df_long.groupby(district_col)["contrib_score"].rank(ascending=False, method="min")

    df_long.to_csv(os.path.join(output_dir, "need_driver_by_district_long.csv"), index=False, encoding="utf-8-sig")

    # --- 구별 TOP3
    top3 = (
        df_long.sort_values([district_col, "contrib_score"], ascending=[True, False])
        .groupby(district_col)
        .head(3)
        .copy()
    )
    top3.to_csv(os.path.join(output_dir, "need_driver_top3_by_district.csv"), index=False, encoding="utf-8-sig")

    # --- 전체 요약(평균 기여도/Top1 빈도)
    top1 = top3.groupby(district_col).head(1)
    summary = (
        df_long.groupby("need_factor")
        .agg(
            mean_contrib_score=("contrib_score", "mean"),
            mean_contrib_share_pct=("contrib_share_pct", "mean"),
        )
        .reset_index()
    )
    top1_count = top1["need_factor"].value_counts().rename_axis("need_factor").reset_index(name="top1_count")
    summary = summary.merge(top1_count, on="need_factor", how="left").fillna({"top1_count": 0})
    summary = summary.sort_values(["top1_count", "mean_contrib_score"], ascending=[False, False])
    summary.to_csv(os.path.join(output_dir, "need_driver_summary_overall.csv"), index=False, encoding="utf-8-sig")

    # --- 구별 정책 제안(Top3 요인을 문장으로 매핑)
    rec_rows = []
    for d, g in top3.groupby(district_col):
        factors = g["need_factor"].tolist()
        recs = []
        for f in factors:
            recs.extend(POLICY_MAP.get(f, ["(정책 방향 매핑 필요)"]))
        # 중복 제거 + 길이 제한(너무 길면 UI에서 보기 힘듦)
        recs_clean = []
        seen = set()
        for r in recs:
            if r not in seen:
                seen.add(r)
                recs_clean.append(r)

        rec_rows.append({
            district_col: d,
            "top1_factor": factors[0] if len(factors) > 0 else "",
            "top2_factor": factors[1] if len(factors) > 1 else "",
            "top3_factor": factors[2] if len(factors) > 2 else "",
            "policy_direction_1": recs_clean[0] if len(recs_clean) > 0 else "",
            "policy_direction_2": recs_clean[1] if len(recs_clean) > 1 else "",
            "policy_direction_3": recs_clean[2] if len(recs_clean) > 2 else "",
        })

    df_rec = pd.DataFrame(rec_rows)
    df_rec.to_csv(os.path.join(output_dir, "need_policy_recommendation_by_district.csv"), index=False, encoding="utf-8-sig")

    return {
        "long": df_long,
        "top3": top3,
        "summary": summary,
        "recommendation": df_rec,
    }
