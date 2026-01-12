import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.analysis.features.scaler import MinMaxScaler100
from src.features.index_builder import build_need_index, build_supply_index

def run_sensitivity_analysis(need_df, supply_df, final_df, config, output_path):
    district_col = config['keys']['district_col']
    
    # Baseline: 계산된 값만 사용
    scaler = MinMaxScaler100()
    baseline_need = build_need_index(need_df, scaler, config['index_weights']['need'])
    baseline_supply = build_supply_index(supply_df, scaler, config['index_weights']['supply'])
    baseline_gap = baseline_need - baseline_supply

    # ✅ 기준 top10은 baseline_gap 기준으로 계산
    baseline_top10 = pd.Series(baseline_gap, index=need_df[district_col]).sort_values(ascending=False).head(10).index.tolist()

    scenarios = [
        ('baseline', 0),
        ('need_plus10', 0.1),
        ('need_minus10', -0.1),
        ('need_plus20', 0.2),
        ('need_minus20', -0.2),
        ('supply_plus10', 0.1),
        ('supply_minus10', -0.1),
        ('supply_plus20', 0.2),
        ('supply_minus20', -0.2),
    ]

    results = []

    for scenario_name, delta in scenarios:
        if 'need' in scenario_name:
            adjusted_weights = {k: v * (1 + delta) for k, v in config['index_weights']['need'].items()}
            adjusted_need = build_need_index(need_df, scaler, adjusted_weights)
            adjusted_supply = baseline_supply
        elif 'supply' in scenario_name:
            adjusted_weights = {k: v * (1 + delta) for k, v in config['index_weights']['supply'].items()}
            adjusted_supply = build_supply_index(supply_df, scaler, adjusted_weights)
            adjusted_need = baseline_need
        else:
            adjusted_need = baseline_need
            adjusted_supply = baseline_supply

        adjusted_gap = adjusted_need - adjusted_supply
        adjusted_top10 = pd.Series(adjusted_gap, index=need_df[district_col]).sort_values(ascending=False).head(10).index.tolist()

        unchanged_top10_count = len(set(baseline_top10) & set(adjusted_top10))
        corr, _ = spearmanr(baseline_gap, adjusted_gap)

        gap_diff = np.abs(baseline_gap - adjusted_gap)
        largest_change_district = need_df.iloc[gap_diff.idxmax()][district_col]
        largest_change_value = gap_diff.max()

        notes = f"Largest gap change: {largest_change_district} ({largest_change_value:.2f})"

        results.append({
            'scenario': scenario_name,
            'unchanged_top10_count': unchanged_top10_count,
            'rank_correlation': corr,
            'notes': notes
        })

    pd.DataFrame(results).to_csv(output_path, index=False)
