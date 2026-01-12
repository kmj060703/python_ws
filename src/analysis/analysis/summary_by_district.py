import pandas as pd
import numpy as np
from src.analysis.io.loaders import load_need_data, load_supply_data
from src.analysis.utils.config import load_config

def main():
    # 1. 로드
    config = load_config('configs/analysis.yaml')
    df_imp = pd.read_csv('outputs/model_feature_importance.csv')
    need_df = load_need_data(config['paths']['need_csv'])
    supply_df = load_supply_data(config['paths']['supply_csv'])

    # 2. need/supply 컬럼 분리
    need_features = [
        'suicide_rate', 'depression_experience_rate', 'perceived_stress_rate',
        'high_risk_drinking_rate', 'unmet_medical_need_rate', 'unemployment_rate',
        'elderly_population_rate', 'old_dependency_ratio', 'single_households',
        'basic_livelihood_recipients'
    ]
    supply_features = [
        'welfare_budget_per_capita', 'public_sports_facilities_count',
        'parks_count', 'libraries_count', 'medical_institutions_count',
        'health_promotion_centers_count', 'elderly_leisure_welfare_facilities_count',
        'in_home_elderly_welfare_facilities_count', 'cultural_satisfaction'
    ]

    # 3. feature 타입 구분 (need/supply)
    df_imp['type'] = df_imp['feature'].apply(
        lambda x: 'need' if x in need_features else ('supply' if x in supply_features else 'other')
    )

    # 4. LinearRegression만 사용 (RF는 방향 없음, 불안정)
    df_lr = df_imp[df_imp['model'] == 'LinearRegression'].copy()

    # 5. 각 구별로 가장 큰 영향을 주는 변수 찾기 (절대값 기준)
    # → need/supply 모두 포함, 단, 각 구는 한 변수만 선택
    # → 여기서는 "자살률에 가장 큰 영향을 주는 변수"를 구별로 추출
    df_lr['abs_value'] = np.abs(df_lr['value'])
    top_per_district = df_lr.loc[df_lr.groupby('feature')['abs_value'].idxmax()]

    # ❌ 위는 변수별 최대값. 우리가 원하는 건: **구별로 가장 큰 변수**

    # ✅ 우리가 원하는 것: 각 **구**에 대해, **자살률 예측에 가장 큰 계수를 가진 변수 1개**
    # 그런데... LinearRegression은 **전체 25개 구를 한 번에 학습**했고,  
    # **계수는 전국 평균**임 → 구별로 변수 중요도가 다를 수 없음

    # ⚠️ 여기서 진짜 문제:  
    # **LinearRegression은 전국 평균 계수 하나만 줌 → 구별 차이는 없음**

    # ✅ 그러면?  
    # **“전국에서 자살률에 가장 큰 영향을 주는 변수 5개”를 뽑고,**  
    # **그 변수들 중에서 각 구의 실제 값이 높은 순으로 정렬**

    # ✅ 전략 변경:  
    # 1. 전국 기준으로 가장 영향력 큰 need/supply 변수 5개 선정  
    # 2. 각 구별로 그 변수들의 실제 값 순위 매기기  
    # 3. “이 구는 ‘고령화’가 가장 심각하고, ‘경제적 취약’도 높음”처럼 요약

    # 6. 전국 기준 상위 5개 변수 (절대값 기준)
    top5_features = df_lr.nlargest(5, 'abs_value')['feature'].tolist()

    # 7. 각 구의 top5 변수 실제 값
    df_values = need_df[['district'] + need_features].merge(
        supply_df[['district'] + supply_features], on='district'
    )

    # 8. 각 구별로 top5 변수의 값을 기준으로 점수 매기기 (순위)
    df_ranked = df_values[['district']].copy()
    for feat in top5_features:
        df_ranked[feat + '_rank'] = df_values[feat].rank(method='dense', ascending=False)

    # 9. 각 구별로 top5 변수 순위 중 가장 낮은(=가장 심각한) 순위 1개 선택
    rank_cols = [f + '_rank' for f in top5_features]
    df_ranked['dominant_factor'] = df_ranked[rank_cols].idxmin(axis=1).str.replace('_rank', '')
    df_ranked['dominant_rank'] = df_ranked[rank_cols].min(axis=1)

    # 10. 최종 요약: 구별로 가장 심각한 변수 1개 + 실제값 + 방향
    df_summary = df_ranked[['district', 'dominant_factor']].merge(
        df_lr[['feature', 'value', 'direction']], left_on='dominant_factor', right_on='feature', how='left'
    ).drop('feature', axis=1)

    # 11. 실제값 추가
    df_summary['value'] = df_summary.apply(
        lambda row: df_values[df_values['district'] == row['district']][row['dominant_factor']].iloc[0], axis=1
    )

    # 12. type 추가
    df_summary['type'] = df_summary['dominant_factor'].apply(
        lambda x: 'need' if x in need_features else ('supply' if x in supply_features else 'other')
    )

    # 13. 정렬
    df_summary = df_summary[['district', 'dominant_factor', 'value', 'direction', 'type']]
    df_summary = df_summary.sort_values('district')

    # 14. 저장
    output_dir = 'data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(output_dir, 'summary_by_district.csv'), index=False, encoding='utf-8-sig')

    print("✅ summary_by_district.csv 생성 완료: 각 구별로 자살률에 가장 큰 영향을 주는 변수 1개 표시")

if __name__ == "__main__":
    import os
    main()
