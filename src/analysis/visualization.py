"""
visualization.py

4사분면 시각화

역할 요약:
- Need_Index(위험도)와 Supply_Index(공급 수준)를
  하나의 좌표계로 시각화
- 중앙값 기준 4사분면을 통해
  각 지역의 정책적 위치와 유형을 직관적으로 표현

이 시각화는:
"어디가 위험한가?"를 넘어서
"어떤 유형의 개입이 필요한가?"를 한눈에 보여주는 도구다.
"""
import matplotlib.pyplot as plt


def plot_quadrant_chart(df_final, median_need, median_supply):
    """
    4사분면 시각화 함수

    입력:
    - df_final:
        district, Need_Index, Supply_Index, Quadrant 컬럼을 포함한 최종 결과 테이블
    - median_need:
        Need_Index 중앙값 (위/아래 분면 구분 기준)
    - median_supply:
        Supply_Index 중앙값 (좌/우 분면 구분 기준)

    출력:
    - matplotlib 산점도 (화면 출력)
    """

    # -----------------------------------------------------
    # 1. Figure 설정
    # -----------------------------------------------------
    # 정사각형 비율 → 두 축의 상대적 크기를 왜곡 없이 비교
    plt.figure(figsize=(8, 8))

    # -----------------------------------------------------
    # 2. 분면별 라벨 및 색상 정의
    # -----------------------------------------------------
    # 각 Quadrant에 대해:
    # - 사람이 바로 이해할 수 있는 설명 라벨
    # - 정책적 의미를 직관적으로 전달하는 색상 지정
    #
    # 색상 의도:
    # A (과잉공급형): 녹색 → 비교적 안정 / 여유
    # B (양호형):     파랑 → 큰 문제는 없으나 관찰 필요
    # C (심각 부족형): 빨강 → 즉각적 개입 필요
    # D (고위험 대응형): 주황 → 공급은 있으나 위험 지속 (사각지대 가능성)
    color_map = {
        'A': ('(A) 과잉공급형 (공급↑, 필요↓)', '#4CAF50'),
        'B': ('(B) 양호형 (공급↓, 필요↓)', '#2196F3'),
        'C': ('(C) 심각 부족형 (공급↓, 필요↑)', '#F44336'),
        'D': ('(D) 고위험 대응형 (공급↑, 필요↑ / 사각지대)', '#FF9800')
    }

    # -----------------------------------------------------
    # 3. 분면별 산점도 그리기
    # -----------------------------------------------------
    # 각 Quadrant에 속한 자치구만 필터링하여
    # 동일한 색상과 라벨로 묶어서 시각화
    for quad, (label, color) in color_map.items():
        subset = df_final[df_final['Quadrant'] == quad]
        plt.scatter(
            subset['Supply_Index'],   # x축: 공급 수준
            subset['Need_Index'],     # y축: 위험 수준
            label=label,
            color=color,
            s=60,                     # 점 크기
            alpha=0.75                # 투명도 → 겹침 완화
        )

    # -----------------------------------------------------
    # 4. 중앙값 기준선 표시
    # -----------------------------------------------------
    # 중앙값(median)을 기준으로
    # Need / Supply를 High / Low로 나누는 시각적 기준선
    #
    # 이 선을 기준으로
    # - 위/아래: 위험 수준
    # - 좌/우: 공급 수준
    # 을 직관적으로 구분할 수 있음
    plt.axhline(
        median_need,
        color='black',
        linestyle='--',
        linewidth=1
    )
    plt.axvline(
        median_supply,
        color='black',
        linestyle='--',
        linewidth=1
    )

    # -----------------------------------------------------
    # 5. 자치구 이름 라벨링
    # -----------------------------------------------------
    # 각 점 옆에 district 이름을 붙여
    # "이 점이 어느 지역인지" 즉시 식별 가능하게 함
    #
    # offset을 주는 이유:
    # - 점과 텍스트가 겹쳐 가독성이 떨어지는 것을 방지
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

    # -----------------------------------------------------
    # 6. 축 라벨 및 제목
    # -----------------------------------------------------
    plt.xlabel("Supply Index (인프라 공급도)")
    plt.ylabel("Need Index (위험도)")
    plt.title("Need–Supply 기반 4사분면 분류")

    # -----------------------------------------------------
    # 7. 기타 시각적 요소
    # -----------------------------------------------------
    plt.legend()            # 분면 설명 범례
    plt.grid(alpha=0.3)     # 가독성 향상을 위한 연한 그리드
    plt.tight_layout()      # 레이아웃 자동 조정
    plt.show()              # 화면 출력
