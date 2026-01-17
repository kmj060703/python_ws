# src/analysis/model_based.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os


def run_model_based_analysis(need_df, supply_df, config, output_path):
    print("=" * 50)
    print("분석 시작...")
    print("=" * 50)
    
    # Prepare features
    need_features = [
        'suicide_rate', 'depression_experience_rate', 'perceived_stress_rate',
        'high_risk_drinking_rate', 'unmet_medical_need_rate', 'unemployment_rate',
        'elderly_population_rate', 'old_dependency_ratio', 'single_households',
        'basic_livelihood_recipients'
    ]
    
    supply_features = config.get('model_features', [
        'welfare_budget_per_capita', 'public_sports_facilities_count',
        'parks_count', 'libraries_count', 'medical_institutions_count',
        'health_promotion_centers_count', 'elderly_leisure_welfare_facilities_count',
        'in_home_elderly_welfare_facilities_count', 'cultural_satisfaction'
    ])
    
    print(f"\nNeed features: {need_features}")
    print(f"Supply features: {supply_features}")
    
    # Merge
    district_col = config['keys']['district_col']
    print(f"\nDistrict column: {district_col}")
    print(f"Need df shape: {need_df.shape}")
    print(f"Supply df shape: {supply_df.shape}")
    
    # 컬럼 존재 확인
    print(f"\nNeed df columns: {need_df.columns.tolist()}")
    print(f"Supply df columns: {supply_df.columns.tolist()}")
    
    df = need_df[need_features + [district_col]].merge(
        supply_df[supply_features + [district_col]], on=district_col
    )
    print(f"\nMerged df shape: {df.shape}")
    print(f"Merged df columns: {df.columns.tolist()}")
    
    # Target
    target_col = config['modeling']['target_col']
    print(f"\nTarget column: {target_col}")
    
    y = df[target_col]
    X = df.drop([district_col, target_col], axis=1)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X columns: {X.columns.tolist()}")
    
    # 결측치 확인
    print(f"\n결측치 개수:\n{X.isnull().sum()}")
    
    # Impute
    print("\n결측치 처리 중...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    print("결측치 처리 완료")
    
    # Scaling
    print("\nMin-Max Scaling 중...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Scaling 완료")
    
    # Linear Regression
    print("\nLinear Regression 학습 중...")
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    print(f"Linear Regression 완료. R^2 score: {lr.score(X_scaled, y):.4f}")
    
    lr_importance = pd.DataFrame({
        'model': 'LinearRegression',
        'target': target_col,
        'feature': X.columns,
        'value': lr.coef_,
        'direction': np.where(lr.coef_ > 0, '+', '-')
    })
    
    # Random Forest
    print("\nRandom Forest 학습 중...")
    rf = RandomForestRegressor(
        n_estimators=100, 
        random_state=config['modeling']['random_state'],
        max_depth=10,
        min_samples_split=5
    )
    rf.fit(X_scaled, y)
    print(f"Random Forest 완료. R^2 score: {rf.score(X_scaled, y):.4f}")
    
    rf_importance = pd.DataFrame({
        'model': 'RandomForestRegressor',
        'target': target_col,
        'feature': X.columns,
        'value': rf.feature_importances_,
        'direction': ''
    })
    
    # Combine
    print("\n결과 결합 중...")
    result = pd.concat([lr_importance, rf_importance], ignore_index=True)
    result['abs_value'] = result['value'].abs()
    result = result.sort_values(['model', 'abs_value'], ascending=[True, False])
    result = result.drop('abs_value', axis=1)
    
    print(f"\n결과 shape: {result.shape}")
    print(f"결과 head:\n{result.head()}")
    
    # 저장
    print(f"\n결과 저장 시도: {output_path}")
    print(f"디렉토리 존재 여부: {os.path.exists(os.path.dirname(output_path))}")
    
    try:
        result.to_csv(output_path, index=False)
        print(f"✓ 저장 성공: {output_path}")
        print(f"✓ 파일 존재 확인: {os.path.exists(output_path)}")
        print(f"✓ 파일 크기: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"✗ 저장 실패: {e}")
    
    # 콘솔 출력
    print("\n" + "=" * 50)
    print("=== Linear Regression - Top 5 Features ===")
    lr_importance['abs_value'] = lr_importance['value'].abs()
    lr_top5 = lr_importance.nlargest(5, 'abs_value')[['feature', 'value', 'direction']]
    print(lr_top5.to_string(index=False))
    
    print("\n=== Random Forest - Top 5 Features ===")
    rf_top5 = rf_importance.nlargest(5, 'value')[['feature', 'value']]
    print(rf_top5.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    try:
        print("스크립트 시작...")
        
        from src.analysis.utils.config import load_config
        
        print("Config 로딩 중...")
        config = load_config("configs/analysis.yaml")
        print("Config 로딩 완료")
        
        print(f"\n설정 내용:")
        print(f"- need_csv: {config['paths']['need_csv']}")
        print(f"- supply_csv: {config['paths']['supply_csv']}")
        print(f"- output_dir: {config['paths']['output_dir']}")
        
        print("\n데이터 로딩 중...")
        need_df = pd.read_csv(config['paths']['need_csv'])
        supply_df = pd.read_csv(config['paths']['supply_csv'])
        print("데이터 로딩 완료")
        
        output_path = os.path.join(
            config['paths']['output_dir'],
            "feature_importance.csv"
        )
        
        print(f"\n출력 경로: {output_path}")
        
        # 출력 디렉토리 생성
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
        print(f"출력 디렉토리 생성 완료: {config['paths']['output_dir']}")
        
        # 분석 실행
        run_model_based_analysis(need_df, supply_df, config, output_path)
        
        print("\n✓✓✓ 전체 프로세스 완료 ✓✓✓")
        
    except Exception as e:
        print(f"\n✗✗✗ 에러 발생 ✗✗✗")
        print(f"에러 타입: {type(e).__name__}")
        print(f"에러 메시지: {e}")
        import traceback
        print(f"\n전체 traceback:")
        traceback.print_exc()