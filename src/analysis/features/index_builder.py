import pandas as pd
import numpy as np

def build_need_index(need_df, scaler, weights):
    need_features = [
        'suicide_rate', 'depression_experience_rate', 'perceived_stress_rate',
        'high_risk_drinking_rate', 'unmet_medical_need_rate', 'unemployment_rate',
        'elderly_population_rate', 'old_dependency_ratio', 'single_households',
        'basic_livelihood_recipients'
    ]
    
    missing = [col for col in need_features if col not in need_df.columns]
    if missing:
        raise ValueError(f"Missing need features: {missing}")
    
    for col in need_features:
        median_val = need_df[col].median()
        if need_df[col].isnull().any():
            need_df[col] = need_df[col].fillna(median_val)
    
    need_scaled = scaler.fit_transform(need_df[need_features], need_features)
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight for need index cannot be zero")
    
    weighted_sum = np.zeros(len(need_df))
    for feature, weight in weights.items():
        if feature not in need_scaled.columns:
            raise ValueError(f"Feature '{feature}' not found in need data")
        weighted_sum += need_scaled[feature] * weight
    
    need_index = weighted_sum / total_weight
    return need_index

def build_supply_index(supply_df, scaler, weights):
    supply_features = [
        'welfare_budget_per_capita', 'public_sports_facilities_count',
        'parks_count', 'libraries_count', 'medical_institutions_count',
        'health_promotion_centers_count', 'elderly_leisure_welfare_facilities_count',
        'in_home_elderly_welfare_facilities_count', 'cultural_satisfaction'
    ]
    
    missing = [col for col in supply_features if col not in supply_df.columns]
    if missing:
        raise ValueError(f"Missing supply features: {missing}")
    
    for col in supply_features:
        median_val = supply_df[col].median()
        if supply_df[col].isnull().any():
            supply_df[col] = supply_df[col].fillna(median_val)
    
    supply_scaled = scaler.fit_transform(supply_df[supply_features], supply_features)
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight for supply index cannot be zero")
    
    weighted_sum = np.zeros(len(supply_df))
    for feature, weight in weights.items():
        if feature not in supply_scaled.columns:
            raise ValueError(f"Feature '{feature}' not found in supply data")
        weighted_sum += supply_scaled[feature] * weight
    
    supply_index = weighted_sum / total_weight
    return supply_index
