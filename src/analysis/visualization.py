"""
visualization.py

4사분면 시각화
"""
import matplotlib.pyplot as plt


def plot_quadrant_chart(df_final, median_need, median_supply):
    """4사분면 시각화"""
    plt.figure(figsize=(8, 8))
    
    # 분면별 색상
    color_map = {
    'A': ('(A) 과잉공급형 (공급↑, 필요↓)', '#4CAF50'),
    'B': ('(B) 양호형 (공급↓, 필요↓)', '#2196F3'),
    'C': ('(C) 심각 부족형 (공급↓, 필요↑)', '#F44336'),
    'D': ('(D) 고위험 대응형 (공급↑, 필요↑ / 사각지대)', '#FF9800')
    }

    
    # 산점도
    for quad, (label, color) in color_map.items():
        subset = df_final[df_final['Quadrant'] == quad]
        plt.scatter(
            subset['Supply_Index'],
            subset['Need_Index'],
            label=label,
            color=color,
            s=60,
            alpha=0.75
        )

    
    # 중앙값 기준선
    plt.axhline(median_need, color='black', linestyle='--', linewidth=1)
    plt.axvline(median_supply, color='black', linestyle='--', linewidth=1)
    
    # 라벨링 
    for _, row in df_final.iterrows():
        plt.annotate(
            row['district'],
            (row['Supply_Index'], row['Need_Index']),
            textcoords="offset points",
            xytext=(4, 4),
            ha='left',
            fontsize=8,
            fontweight='bold',
            color='black'
        )

    
    # 라벨 & 제목
    plt.xlabel("Supply Index (인프라 공급도)")
    plt.ylabel("Need Index (위험도)")
    plt.title("Need–Supply 기반 4사분면 분류 (Gap Index 상위 10개 강조)")
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()