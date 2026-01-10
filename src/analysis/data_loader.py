"""
데이터 로드 및 기본 전처리
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, NEED_VARS, SUPPLY_VARS


def load_data():
    """Need와 Supply 데이터 로드 및 병합"""
    df_need = pd.read_csv(DATA_DIR / "need_tidy.csv")
    df_supply = pd.read_csv(DATA_DIR / "supply_tidy.csv")
    
    df = df_need.merge(df_supply, on='district', how='inner')
    
    print("=" * 60)
    print("📊 데이터 로드 완료")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"구 개수: {len(df)}")
    print(f"변수 개수: {len(df.columns) - 1}")
    print("\n변수 목록:")
    print("Need (11개):", df_need.columns.tolist()[1:])
    print("Supply (9개):", df_supply.columns.tolist()[1:])
    print("\n첫 5개 구:")
    print(df.head())
    print("\n기초 통계:")
    print(df.describe())
    print("\n결측치 확인:")
    print(df.isnull().sum().sum())
    print("\n✅ 데이터 준비 완료!")
    
    return df


def normalize_to_100(series, direction='positive'):
    """
    0~100점으로 정규화
    
    Parameters:
    -----------
    series : pd.Series
        정규화할 시리즈
    direction : str
        'positive': 높을수록 나쁨 (자살률, 우울감 등)
        'negative': 낮을수록 나쁨 (인프라 등)
    
    Returns:
    --------
    np.array : 정규화된 값 (0~100)
    """
    scaler = MinMaxScaler(feature_range=(0, 100))
    
    if direction == 'positive':
        normalized = scaler.fit_transform(series.values.reshape(-1, 1))
    else:
        normalized = 100 - scaler.fit_transform(series.values.reshape(-1, 1))
    
    return normalized.flatten()


def normalize_data(df):
    """Need와 Supply 변수 정규화"""
    print("=" * 60)
    print("📏 변수 정규화")
    print("=" * 60)
    
    # Need 정규화
    df_need_norm = df[['district']].copy()
    for var in NEED_VARS:
        df_need_norm[f'{var}_norm'] = normalize_to_100(df[var], direction='positive')
        print(f"  ✅ {var:40s} → 정규화 완료")
    
    # Supply 정규화
    df_supply_norm = df[['district']].copy()
    for var in SUPPLY_VARS:
        df_supply_norm[f'{var}_norm'] = normalize_to_100(df[var], direction='negative')
        print(f"  ✅ {var:40s} → 정규화 완료")
    
    print("\n정규화 결과 샘플:")
    print(df_need_norm.head())
    print("\n✅ 모든 변수 0~100점 변환 완료")
    
    return df_need_norm, df_supply_norm