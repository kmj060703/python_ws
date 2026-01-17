"""
index_calculator.py

Need, Supply, Gap Index 계산

역할 요약:
- 정규화된 Need / Supply 변수들을 가중합하여
  각각 Need_Index, Supply_Index를 계산
- 두 지수의 차이를 통해 Gap_Index를 정의
- Gap_Index 및 Need/Supply 분포를 기반으로 4사분면 분류 수행
- 정책 보고 및 시각화를 위한 순위 테이블 저장

이 파일은:
'데이터 → 의미 있는 지수 → 정책적 해석 단위'
로 변환하는 핵심 단계다.
"""
import pandas as pd
from config import WEIGHTS_NEED, WEIGHTS_SUPPLY, OUTPUT_DIR


def calculate_need_index(df_need_norm):
    """

    Need Index 계산

    MHVI – Need Index 순위

    목적:
    - 각 지역의 구조적 정신건강 위험 수준을 하나의 지수로 요약
    - 개별 위험 지표들을 단순 나열이 아니라,
      '정책적으로 중요하다고 판단한 비중'에 따라 통합

    계산식:
    Need_Index = Σ (정규화된 Need 변수 × 해당 가중치)

    해석:
    - Need_Index가 높을수록:
        → 정신건강 취약도가 높고
        → 구조적 위험이 누적되어 있으며
        → 정책 개입의 필요성이 큼
    - 0~100 스케일에서 상대적 비교 지표

    순위:
    - 1위 = 가장 취약한 지역
    """

    # -----------------------------------------------------
    # 1. Need_Index 초기화
    # -----------------------------------------------------
    # 기존 df_need_norm에는 *_norm 컬럼들만 존재
    # 여기에 새로운 집계 지표인 Need_Index를 추가
    df_need_norm['Need_Index'] = 0

    # -----------------------------------------------------
    # 2. 가중합 계산
    # -----------------------------------------------------
    # WEIGHTS_NEED:
    #   key   → 정규화된 변수명 (예: suicide_rate_norm)
    #   value → 해당 변수의 정책적 중요도 가중치
    #
    # 모든 가중치의 합은 1 → 지수 해석이 직관적
    for var, weight in WEIGHTS_NEED.items():
        df_need_norm['Need_Index'] += df_need_norm[var] * weight

    # -----------------------------------------------------
    # 3. 정렬 (출력/확인용)
    # -----------------------------------------------------
    # 실제 반환값에는 정렬을 강제하지 않지만,
    # 상위 위험 지역을 빠르게 확인하기 위한 용도
    df_sorted = df_need_norm.sort_values('Need_Index', ascending=False)

    print("\n✅ Need Index 계산 완료")

    # Need_Index가 추가된 DataFrame 반환
    return df_need_norm


def calculate_supply_index(df_supply_norm):
    """

    Supply Index 계산

    MHVI – 공급 결핍(Supply Deficit) 지수 순위

    목적:
    - 지역별 정신건강 인프라 및 서비스의 '부족 정도'를 하나의 지수로 요약
    - 단순한 인프라 나열이 아니라,
      정책적으로 중요하다고 판단한 인프라에 더 큰 비중 부여

    계산식:
    Supply_Index = Σ (정규화된 Supply 변수 × 해당 가중치)

    해석:
    - Supply_Index가 높을수록:
        → 해당 지역의 지원 체계가 상대적으로 부족함
        → 동일한 위험 수준이라면 더 취약한 상태
    - '공급이 많다/적다'가 아니라
      '공급 결핍의 상대적 정도'를 표현하는 지표

    순위:
    - 1위 = 가장 지원이 부족한 지역
    """

    print("\n" + "=" * 60)
    print("🏥 Supply Index 계산")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Supply_Index 초기화
    # -----------------------------------------------------
    df_supply_norm['Supply_Index'] = 0

    # -----------------------------------------------------
    # 2. 가중합 계산
    # -----------------------------------------------------
    # WEIGHTS_SUPPLY에 정의된 가중치에 따라
    # 의료/보건/노인/문화 인프라를 종합적으로 반영
    for var, weight in WEIGHTS_SUPPLY.items():
        df_supply_norm['Supply_Index'] += df_supply_norm[var] * weight

    # -----------------------------------------------------
    # 3. 정렬 (확인용)
    # -----------------------------------------------------
    df_sorted = df_supply_norm.sort_values('Supply_Index', ascending=False)

    print("\n✅ Supply Index 계산 완료")

    return df_supply_norm


def calculate_gap_index(df, df_need_norm, df_supply_norm):
    """
    Gap Index 계산 및 4사분면 분류

    목적:
    - '위험(Need)'과 '지원 부족(Supply)'을 동시에 고려하여
      정책 개입 우선순위를 도출
    - 단일 지수로 보기 어려운 두 축을
      Gap_Index와 4사분면 구조로 해석 가능하게 만듦

    정의:
    Gap_Index = Need_Index - Supply_Index

    의미:
    - Need_Index   : 해당 지역의 정신건강 위험 수준
    - Supply_Index : 해당 지역의 지원 결핍 수준
    - Gap_Index    : 위험 대비 방치 정도
        • 값이 클수록:
            → 위험은 큰데
            → 지원은 부족한 지역
            → 정책 개입이 시급
    """

    print("\n" + "=" * 60)
    print("🎯 Gap Index 계산 (Need - Supply)")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Need / Supply 지수 통합
    # -----------------------------------------------------
    # district를 기준으로
    # Need_Index와 Supply_Index를 하나의 테이블로 결합
    df_final = df[['district']].copy()
    df_final = df_final.merge(
        df_need_norm[['district', 'Need_Index']],
        on='district'
    )
    df_final = df_final.merge(
        df_supply_norm[['district', 'Supply_Index']],
        on='district'
    )

    # -----------------------------------------------------
    # 2. Gap_Index 계산
    # -----------------------------------------------------
    df_final['Gap_Index'] = (
        df_final['Need_Index'] - df_final['Supply_Index']
    )

    # -----------------------------------------------------
    # 3. Gap 기준 정렬 (확인용)
    # -----------------------------------------------------
    df_sorted = df_final.sort_values('Gap_Index', ascending=False)

    # -----------------------------------------------------
    # 4. 4사분면 분류
    # -----------------------------------------------------
    # 중앙값(median)을 기준으로 High / Low 구분
    # → 소표본에서 평균보다 안정적인 기준
    median_need = df_final['Need_Index'].median()
    median_supply = df_final['Supply_Index'].median()

    # 4사분면 분류 규칙:
    # C: Need 높음 & Supply 부족 → 가장 시급한 개입 대상
    # D: Need 높음 & Supply 높음 → 대응 중이나 위험 지속
    # B: Need 낮음 & Supply 부족 → 잠재 위험군
    # A: Need 낮음 & Supply 충분 → 비교적 안정
    def classify_quadrant(row):
        if row['Need_Index'] >= median_need and row['Supply_Index'] < median_supply:
            return 'C'
        elif row['Need_Index'] >= median_need and row['Supply_Index'] >= median_supply:
            return 'D'
        elif row['Need_Index'] < median_need and row['Supply_Index'] < median_supply:
            return 'B'
        else:
            return 'A'

    df_final['Quadrant'] = df_final.apply(
        classify_quadrant,
        axis=1
    )

    print("\n📊 4사분면 분류:")
    print(df_final['Quadrant'].value_counts())
    print("\n✅ Gap Index 및 분류 완료")

    return df_final, median_need, median_supply


def save_rankings(df, df_need_norm, df_supply_norm):
    """
    순위 테이블 저장 함수

    생성 파일:
    1) Need Index 순위
    2) Supply Index 순위
    3) 지역별 Need 상위 3개 위험 요인

    목적:
    - 정책 보고서, 대시보드, 지도 시각화에 바로 사용할 수 있는
      '정리된 결과 테이블' 생성
    """
    from config import NEED_VARS, OUTPUT_DIR
    import pandas as pd

    # =========================
    # 1. Need Index 순위
    # =========================
    # Need_Index가 높은 지역부터 순위 부여
    need_rank_df = (
        df_need_norm[['district', 'Need_Index']]
        .sort_values('Need_Index', ascending=False)
        .reset_index(drop=True)
    )
    need_rank_df['rank'] = need_rank_df.index + 1

    need_rank_df.to_csv(
        OUTPUT_DIR / "need_index_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # 2. Supply Index 순위
    # =========================
    # Supply_Index가 높은 지역 = 지원이 더 부족한 지역
    supply_rank_df = (
        df_supply_norm[['district', 'Supply_Index']]
        .sort_values('Supply_Index', ascending=False)
        .reset_index(drop=True)
    )
    supply_rank_df['rank'] = supply_rank_df.index + 1

    supply_rank_df.to_csv(
        OUTPUT_DIR / "supply_index_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # 3. 지역별 NEED 상위 3개 지표
    # =========================
    """
    MHVI – 지역별 주요 위험 요인 상위 3개

    목적:
    - 각 지역의 Need_Index를 구성하는 핵심 위험 요인을 식별
    - "왜 이 지역의 Need가 높은가?"에 대한 설명 가능성 확보

    컬럼 설명:
    - district      : 자치구
    - rank          : 해당 지역 내 위험 요인 순위
    - need_variable : 위험 지표 이름 (정규화 이전 변수명)
    - score         : 정규화 점수 (0~100, 높을수록 심각)
    """

    rows = []
    for _, row in df_need_norm.iterrows():
        district = row['district']

        # Need_Index를 구성하는 개별 정규화 변수 점수 추출
        scores = {
            var.replace('_norm', ''): row[var]
            for var in df_need_norm.columns
            if var.endswith('_norm') and var != 'Need_Index'
        }

        # 점수가 높은 순으로 상위 3개 선택
        top3 = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # 지역 × 순위 × 변수 형태로 행 구성
        for rank, (var, score) in enumerate(top3, start=1):
            rows.append({
                'district': district,
                'rank': rank,
                'need_variable': var,
                'score': score
            })

    need_top3_df = pd.DataFrame(rows)
    need_top3_df.to_csv(
        OUTPUT_DIR / "district_need_top3.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("📊 순위 테이블 저장 완료")
