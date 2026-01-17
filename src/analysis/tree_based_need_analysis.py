"""
========================================
자살률-Need 지표 동반성 분석
========================================

목적:
- 자살률이 높은 지역에서
  어떤 Need 지표들이 함께 높게 나타나는지를 탐색
- "원인 규명"이 아니라,
  지역 단위에서 반복적으로 관찰되는
  구조적 동반 패턴(co-occurrence)을 파악하는 것이 목적

방법:
- RandomForest 회귀 모델
- SHAP을 이용한 변수 기여도 해석

주의:
- 인과관계 추론 불가 (n=25, 소표본)
- 예측 성능 경쟁 목적 아님
- 정책 판단을 위한 '설명 가능한 탐색 분석'
========================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

from config import BASE_DIR, DATA_DIR


def run_tree_based_analysis():
    """
    자살률-Need 지표 동반성 분석 메인 함수
    
    RandomForest와 SHAP을 이용하여 자살률과 함께 나타나는
    Need 지표들의 패턴을 분석합니다.
    """
    
    # =====================================================
    # 1. 경로 설정
    # =====================================================
    INPUT_PATH = DATA_DIR / 'need_tidy.csv'
    OUTPUT_DIR = BASE_DIR / 'data' / 'outputs' / 'RandomForestModel'
    
    # 출력 디렉토리가 없으면 자동 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 분석 시작 로그
    print("=" * 60)
    print("자살률-Need 지표 동반성 분석 (RandomForest + SHAP)")
    print("=" * 60)
    print(f"입력 파일: {INPUT_PATH}")
    print(f"출력 경로: {OUTPUT_DIR}")
    print("=" * 60)
    
    # =====================================================
    # 2. 데이터 로드
    # =====================================================
    df = pd.read_csv(INPUT_PATH)
    
    print(f"\n✓ 데이터 로드 완료: {df.shape[0]}개 자치구, {df.shape[1]}개 변수")
    
    # =====================================================
    # 3. 변수 분리
    # =====================================================
    y = df['suicide_rate'].values
    
    feature_cols = [
        col for col in df.columns
        if col not in ['district', 'suicide_rate']
    ]
    X = df[feature_cols].values
    
    print(f"\n✓ Target: suicide_rate")
    print(f"✓ Features ({len(feature_cols)}개):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    # =====================================================
    # 4. RandomForest 학습
    # =====================================================
    RF_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'min_samples_split': 3,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"\n{'=' * 60}")
    print("RandomForest 학습 시작")
    print(f"{'=' * 60}")
    for key, val in RF_PARAMS.items():
        print(f"  {key}: {val}")
    
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X, y)
    
    # =====================================================
    # 5. 예측 및 성능 평가
    # =====================================================
    y_pred = rf_model.predict(X)
    
    train_r2 = r2_score(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    cv_scores = cross_val_score(
        rf_model,
        X,
        y,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    print(f"\n{'=' * 60}")
    print("모델 성능 (참고용)")
    print(f"{'=' * 60}")
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  CV R² (5-fold): {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
    print(f"\n⚠️ 주의: n=25 소표본이므로 성능 지표는 참고용")
    print(f"    → 예측 성능보다 '변수 간 동반성 패턴' 파악이 목적")
    print(f"{'=' * 60}")
    
    # =====================================================
    # 6. Feature Importance 추출
    # =====================================================
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    })
    
    importance_df = importance_df.sort_values(
        'importance',
        ascending=False
    )
    importance_df['importance_rank'] = range(
        1, len(importance_df) + 1
    )
    
    fi_path = OUTPUT_DIR / 'rf_feature_importance.csv'
    importance_df.to_csv(
        fi_path,
        index=False,
        encoding='utf-8-sig'
    )
    
    print(f"\n✓ Feature Importance 저장: {fi_path}")
    print("\n[Feature Importance Top 5]")
    print(importance_df.head().to_string(index=False))
    
    # =====================================================
    # 7. SHAP 분석
    # =====================================================
    print(f"\n{'=' * 60}")
    print("SHAP 분석 시작 (TreeExplainer)")
    print(f"{'=' * 60}")
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    shap_summary = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap_value': np.abs(shap_values).mean(axis=0)
    })
    
    shap_summary = shap_summary.sort_values(
        'mean_abs_shap_value',
        ascending=False
    )
    shap_summary['shap_rank'] = range(
        1, len(shap_summary) + 1
    )
    
    shap_path = OUTPUT_DIR / 'rf_shap_summary.csv'
    shap_summary.to_csv(
        shap_path,
        index=False,
        encoding='utf-8-sig'
    )
    
    print(f"\n✓ SHAP Summary 저장: {shap_path}")
    print("\n[SHAP Importance Top 5]")
    print(shap_summary.head().to_string(index=False))
    
    # =====================================================
    # 8. 예측 결과 저장
    # =====================================================
    predictions_df = pd.DataFrame({
        'district': df['district'],
        'suicide_rate_actual': y,
        'suicide_rate_predicted': y_pred,
        'residual': y - y_pred
    })
    
    pred_path = OUTPUT_DIR / 'rf_predictions.csv'
    predictions_df.to_csv(
        pred_path,
        index=False,
        encoding='utf-8-sig'
    )
    
    print(f"\n✓ 예측 결과 저장: {pred_path}")
    
    # =====================================================
    # 9. 결과 해석 가이드
    # =====================================================
    print(f"\n{'=' * 60}")
    print("📊 결과 해석 가이드 (안전한 표현)")
    print(f"{'=' * 60}")
    print("""
✅ 올바른 해석 (동반성/패턴):
  • "자살률이 높은 자치구에서 {변수명}도 함께 높게 나타나는 경향"
  • "{변수명}은 자살률 변동과 강한 동반성을 보임"
  • "RandomForest 모델이 자살률 패턴을 학습하는 데 {변수명}을 주요 특징으로 활용"
  • "SHAP 분석 결과, {변수명}이 예측에 가장 큰 기여"

❌ 피해야 할 해석 (인과/정책):
  • "{변수명}이 자살률을 증가/감소시킨다" → 인과관계 주장 불가
  • "{변수명}을 개선하면 자살률이 낮아진다" → 정책 효과 추정 불가
  • "이 모델로 미래 자살률 예측 가능" → n=25, 예측 목적 아님

🔍 맥락:
  - n=25 (서울시 자치구) 소표본 → 통계적 일반화 제한적
  - 지역 단위 집계 데이터 → 생태학적 오류(ecological fallacy) 가능성
  - 트리 모델 특성상 비선형/상호작용 패턴 포착
  - 본 분석은 '탐색적(exploratory)' 성격
""")
    
    print(f"\n{'=' * 60}")
    print("✅ 분석 완료")
    print(f"{'=' * 60}")
    print(f"저장된 파일:")
    print(f"  1. {fi_path}")
    print(f"  2. {shap_path}")
    print(f"  3. {pred_path}")
    print(f"{'=' * 60}\n")
    
    # =====================================================
    # 10. 요약 통계 출력
    # =====================================================
    print("📈 요약 통계")
    print(f"{'=' * 60}")
    print(f"최고 중요도 변수 (Feature Importance):")
    print(f"  → {importance_df.iloc[0]['feature']}")
    print(f"     (importance: {importance_df.iloc[0]['importance']:.4f})")
    print(f"\n최고 중요도 변수 (SHAP):")
    print(f"  → {shap_summary.iloc[0]['feature']}")
    print(f"     (mean |SHAP|: {shap_summary.iloc[0]['mean_abs_shap_value']:.4f})")
    print(f"{'=' * 60}\n")


# =====================================================
# 실행 진입점
# =====================================================
def main():
    """직접 실행 시 사용"""
    run_tree_based_analysis()


if __name__ == "__main__":
    main()