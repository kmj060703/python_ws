"""
index_calculator.py

Need, Supply, Gap Index 계산
"""
import pandas as pd
from config import WEIGHTS_NEED, WEIGHTS_SUPPLY, OUTPUT_DIR


def calculate_need_index(df_need_norm):
   
    """

    Need Index 계산

    MHVI – Need Index 순위
    목적: 각 지역의 구조적 정신건강 위험 수준을 측정
    계산식: Need_Index = 정규화된 위험 지표들의 가중합 (0–100 척도)
    해석: Need_Index가 높을수록 해당 지역의 취약도와 정책 개입 필요성이 큼
    순위: 1위 = 가장 취약한 지역
    
    """
    # Need Index 계산
    df_need_norm['Need_Index'] = 0
    for var, weight in WEIGHTS_NEED.items():
        df_need_norm['Need_Index'] += df_need_norm[var] * weight
    
    # 정렬
    df_sorted = df_need_norm.sort_values('Need_Index', ascending=False)
    
    print("\n📈 Need Index TOP 10 (위험도 높은 순):")
    print(df_sorted[['district', 'Need_Index']].head(10).to_string(index=False))
    print("\n📉 Need Index BOTTOM 5 (위험도 낮은 순):")
    print(df_sorted[['district', 'Need_Index']].tail(5).to_string(index=False))
    print(f"\n평균: {df_need_norm['Need_Index'].mean():.2f}")
    print(f"최대: {df_need_norm['Need_Index'].max():.2f}")
    print(f"최소: {df_need_norm['Need_Index'].min():.2f}")
    print("\n✅ Need Index 계산 완료")
    
    return df_need_norm


def calculate_supply_index(df_supply_norm):

    """

    Supply Index 계산

    MHVI – 공급 결핍(Supply Deficit) 지수 순위
    목적: 지역별 정신건강 인프라 및 서비스의 부족 정도를 측정
    계산식: Supply_Index = (100 - 정규화된 공급 지표)의 가중합
    해석: Supply_Index가 높을수록 해당 지역의 서비스 제공 수준이 더 부족함
    순위: 1위 = 가장 지원이 부족한 지역

    """
    print("\n" + "=" * 60); print("🏥 Supply Index 계산"); print("=" * 60); 

    df_supply_norm['Supply_Index'] = 0
    for var, weight in WEIGHTS_SUPPLY.items():
        df_supply_norm['Supply_Index'] += df_supply_norm[var] * weight
    
    # 정렬
    df_sorted = df_supply_norm.sort_values('Supply_Index', ascending=False)
    
    print("\n📈 Supply Index TOP 10 (인프라 풍부한 순):")
    print(df_sorted[['district', 'Supply_Index']].head(10).to_string(index=False))
    print("\n📉 Supply Index BOTTOM 5 (인프라 부족한 순):")
    print(df_sorted[['district', 'Supply_Index']].tail(5).to_string(index=False))
    print(f"\n평균: {df_supply_norm['Supply_Index'].mean():.2f}")
    print(f"최대: {df_supply_norm['Supply_Index'].max():.2f}")
    print(f"최소: {df_supply_norm['Supply_Index'].min():.2f}")
    print("\n✅ Supply Index 계산 완료")
    
    return df_supply_norm


def calculate_gap_index(df, df_need_norm, df_supply_norm):
    
    """
    Gap Index 계산 및 4사분면 분류
    
    
    지역별 정책 개입 우선순위를 계산하는 Gap Index와 4사분면 분류 함수

    Gap Index = Need_Index - Supply_Index

    의미:
    - Need_Index   : 해당 지역의 정신건강 위험 수준
    - Supply_Index : 해당 지역의 정신건강 인프라 결핍 수준
    - Gap_Index    : '위험 대비 방치 정도'
                     → 위험은 큰데 지원이 부족할수록 값이 커짐
                     → 정책 개입이 시급한 지역을 의미
    
    """

    print("\n" + "=" * 60); print("🎯 Gap Index 계산 (Need - Supply)"); print("=" * 60)
    
    # 통합
    df_final = df[['district']].copy()
    df_final = df_final.merge(df_need_norm[['district', 'Need_Index']], on='district')
    df_final = df_final.merge(df_supply_norm[['district', 'Supply_Index']], on='district')
    
    # Gap 계산
    df_final['Gap_Index'] = df_final['Need_Index'] - df_final['Supply_Index']
    
    # 정렬
    df_sorted = df_final.sort_values('Gap_Index', ascending=False)
    
    print("\n🚨 Gap Index TOP 10 (정책 개입 최우선):")
    print(df_sorted[['district', 'Need_Index', 'Supply_Index', 'Gap_Index']].head(10).to_string(index=False))
    print("\n✅ Gap Index BOTTOM 5 (상대적 안정):")
    print(df_sorted[['district', 'Need_Index', 'Supply_Index', 'Gap_Index']].tail(5).to_string(index=False))
    print(f"\n평균 Gap: {df_final['Gap_Index'].mean():.2f}")
    print(f"최대 Gap: {df_final['Gap_Index'].max():.2f}")
    print(f"최소 Gap: {df_final['Gap_Index'].min():.2f}")
    
    # 4사분면 분류
    median_need = df_final['Need_Index'].median()
    median_supply = df_final['Supply_Index'].median()
    
    # def classify_quadrant(row):
    #     if row['Need_Index'] >= median_need and row['Supply_Index'] >= median_supply:
    #         return 'D: 고위험 대응형'
    #     elif row['Need_Index'] >= median_need and row['Supply_Index'] < median_supply:
    #         return 'C: 심각 부족형 ⚠️'
    #     elif row['Need_Index'] < median_need and row['Supply_Index'] >= median_supply:
    #         return 'B: 양호형'
    #     else:
    #         return 'A: 과잉공급형'

    def classify_quadrant(row):
        if row['Need_Index'] >= median_need and row['Supply_Index'] < median_supply:
            return 'C'
        elif row['Need_Index'] >= median_need and row['Supply_Index'] >= median_supply:
            return 'D'
        elif row['Need_Index'] < median_need and row['Supply_Index'] < median_supply:
            return 'B'
        else:
            return 'A'

    
    df_final['Quadrant'] = df_final.apply(classify_quadrant, axis=1)
    
    print("\n📊 4사분면 분류:")
    print(df_final['Quadrant'].value_counts())
    print("\n✅ Gap Index 및 분류 완료")
    
    return df_final, median_need, median_supply

def save_rankings(df, df_need_norm, df_supply_norm):
    """
    1) Need Index 순위
    2) Supply Index 순위
    3) 구별 NEED 상위 3개 지표
    """
    from config import NEED_VARS, OUTPUT_DIR
    import pandas as pd

    # =========================
    # Need Index 순위
    # =========================
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
    # Supply Index 순위
    # =========================
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
    # 구별 NEED 상위 3개 지표
    # =========================

    """
    
    MHVI – 지역별 주요 위험 요인 상위 3개
    목적: 각 지역의 Need Index를 구성하는 핵심 위험 요인을 식별
    need_variable: 원본 위험 지표 이름 (정규화 이전 변수)
    score: 정규화된 값 (0–100), 값이 클수록 해당 요인이 더 심각함
    rank: 1위 = 해당 지역에서 가장 큰 영향을 미치는 위험 요인
    
    """
    rows = []
    for _, row in df_need_norm.iterrows():
        district = row['district']

        scores = {
            var.replace('_norm', ''): row[var]
            for var in df_need_norm.columns
            if var.endswith('_norm') and var != 'Need_Index'
        }

        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

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
