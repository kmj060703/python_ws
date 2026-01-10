"""
메인 실행 스크립트
"""
from config import OUTPUT_DIR
from data_loader import load_data, normalize_data
from index_calculator import (
    calculate_need_index,
    calculate_supply_index,
    calculate_gap_index,
    save_rankings
)
from visualization import plot_quadrant_chart
from policy_simulation import run_policy_simulation


def main():
    """메인 분석 파이프라인"""
    
    # 1. 데이터 로드
    df = load_data()
    
    # 2. 정규화
    df_need_norm, df_supply_norm = normalize_data(df)
    
    # 3. Need Index 계산
    df_need_norm = calculate_need_index(df_need_norm)
    
    # 4. Supply Index 계산
    df_supply_norm = calculate_supply_index(df_supply_norm)
    
    # 5. Gap Index 계산
    df_final, median_need, median_supply = calculate_gap_index(
        df, df_need_norm, df_supply_norm
    )
    
    # 6. 순위 저장
    save_rankings(df, df_need_norm, df_supply_norm)
    
    # 7. 최종 결과 저장
    df_final.to_csv(
        OUTPUT_DIR / "mhvi_final_result.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    print("\n" + "=" * 60)
    print("💾 결과 저장 완료")
    print("=" * 60)
    print(f"\n저장된 변수:")
    print(f"  - district (구)")
    print(f"  - Need_Index (위험도)")
    print(f"  - Supply_Index (인프라 부족도)")
    print(f"  - Gap_Index (격차)")
    print(f"  - Quadrant (4사분면 분류)")
    print("\n🎉 분석 완료!")
    print("=" * 60)
    
    # 8. 시각화
    plot_quadrant_chart(df_final, median_need, median_supply)
    
    # 9. 정책 시뮬레이션
    print("\n" + "=" * 60)
    print("🤖 정책 시뮬레이션 시작")
    print("=" * 60)
    run_policy_simulation()
    
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()