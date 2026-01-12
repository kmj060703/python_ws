# src/analysis/model_based.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def run_model_based_analysis(need_df, supply_df, config, output_path):
    # Prepare features: all need features + selected supply features
    need_features = [
        'suicide_rate', 'depression_experience_rate', 'perceived_stress_rate',
        'high_risk_drinking_rate', 'unmet_medical_need_rate', 'unemployment_rate',
        'elderly_population_rate', 'old_dependency_ratio', 'single_households',
        'basic_livelihood_recipients'
    ]
    
    # Use only numeric supply features specified in config, or all if not specified
    supply_features = config.get('model_features', [
        'welfare_budget_per_capita', 'public_sports_facilities_count',
        'parks_count', 'libraries_count', 'medical_institutions_count',
        'health_promotion_centers_count', 'elderly_leisure_welfare_facilities_count',
        'in_home_elderly_welfare_facilities_count', 'cultural_satisfaction'
    ])
    
    # Merge need and supply
    district_col = config['keys']['district_col']
    df = need_df[need_features + [district_col]].merge(
        supply_df[supply_features + [district_col]], on=district_col
    )
    
    # Target
    target_col = config['modeling']['target_col']
    y = df[target_col]
    
    # Features (all except target and district)
    X = df.drop([district_col, target_col], axis=1)
    
    # Check for missing values and impute
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_imputed, y)
    lr_importance = pd.DataFrame({
        'model': 'LinearRegression',
        'target': target_col,
        'feature': X.columns,
        'value': lr.coef_,
        'direction': np.where(lr.coef_ > 0, '+', '-')
    })
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=config['modeling']['random_state'])
    rf.fit(X_imputed, y)
    rf_importance = pd.DataFrame({
        'model': 'RandomForestRegressor',
        'target': target_col,
        'feature': X.columns,
        'value': rf.feature_importances_,
        'direction': ''  # RF doesn't have direction
    })
    
    # Combine results
    result = pd.concat([lr_importance, rf_importance], ignore_index=True)
    result.to_csv(output_path, index=False)
