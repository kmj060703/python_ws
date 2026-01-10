"""
Need, Supply, Gap Index 계산
"""
import pandas as pd
from config import WEIGHTS_NEED, WEIGHTS_SUPPLY, OUTPUT_DIR


def calculate_need_index(df_need_norm):
    """Need Index 계산"""
    print("\n" + "=" * 60)
    print("📊 Need Index 계산")
    print("=" * 60)
    
    print("가중치:")
    total_weight = 0
    for var, weight in WEIGHTS_NEED.items():
        print(f"  {var:45s}: {weight:5.1%}")
        total_weight += weight
    print(f"\n총 가중치 합: {total_weight:.1%}")
    
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
    """Supply Index 계산"""
    print("\n" + "=" * 60)
    print("🏥 Supply Index 계산")
    print("=" * 60)
    
    print("가중치:")
    total_weight = 0
    for var, weight in WEIGHTS_SUPPLY.items():
        print(f"  {var:50s}: {weight:5.1%}")
        total_weight += weight
    print(f"\n총 가중치 합: {total_weight:.1%}")
    
    # Supply Index 계산 (낮을수록 문제 = 역전)
    df_supply_norm['Supply_Index'] = 0
    for var, weight in WEIGHTS_SUPPLY.items():
        df_supply_norm['Supply_Index'] += (100 - df_supply_norm[var]) * weight
    
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
    """Gap Index 계산 및 4사분면 분류"""
    print("\n" + "=" * 60)
    print("🎯 Gap Index 계산 (Need - Supply)")
    print("=" * 60)
    
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
    
    def classify_quadrant(row):
        if row['Need_Index'] >= median_need and row['Supply_Index'] >= median_supply:
            return 'D: 고위험 대응형'
        elif row['Need_Index'] >= median_need and row['Supply_Index'] < median_supply:
            return 'C: 심각 부족형 ⚠️'
        elif row['Need_Index'] < median_need and row['Supply_Index'] >= median_supply:
            return 'B: 양호형'
        else:
            return 'A: 과잉공급형'
    
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
