#charts_3.py
# 본 파일은 정신건강 정책 분석 대시보드에서 사용되는
# 주요 시각화 컴포넌트들을 함수 단위로 모듈화한 파일이다.
#
# 지도, 산점도, 레이더 차트, AI 기반 사각지대 분석 등
# 정책 의사결정 지원을 위한 핵심 시각화를 담당한다.

import folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import branca.colormap as cm
import json

# 1. MHVI 지도 시각화 (정신건강 취약 지수 / 인프라 분포)

# 서울시 전체 영역이 화면에 균형 있게 표시되도록 지도 경계 설정
def draw_mhvi_map(geo_data, data_df):
    seoul_bounds = [[37.42, 126.75], [37.70, 127.18]]
    
    # 보고서/행정용 시각화를 위해 사용자 인터랙션을 최소화한 지도 설정
    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11,
        zoom_control=False,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        touchZoom=False,
        tiles="cartodbpositron",
        attr=' '
    )
    
    # 지도 하단 Attribution 문구 제거 
    m.get_root().html.add_child(folium.Element("""
        <style>.leaflet-control-attribution { display: none !important; }</style>
    """))
    m.fit_bounds(seoul_bounds)

    # 데이터 컬럼 자동 감지
    # - Need_Index 존재 시: 정신건강 취약도 지도
    # - 없을 경우: 인프라(center_count) 분포 지도
    if 'Need_Index' in data_df.columns:
        col_to_plot = "Need_Index"
        legend_title = "정신건강 취약 지수"
        colors = ['#fffbeb', '#fef3c7', '#fde047', '#fb923c', '#f97316', '#dc2626', '#991b1b']
    else:
        col_to_plot = "center_count"
        legend_title = "정신건강 인프라 수"
        colors = ['#fffbeb', '#fef3c7', '#fde047', '#fb923c', '#f97316', '#dc2626', '#991b1b']

    if 'name' not in data_df.columns and 'district' in data_df.columns:
        data_df = data_df.copy()
        data_df['name'] = data_df['district']

    # 자치구명 기준으로 값 매핑
    data_dict = data_df.set_index('name')[col_to_plot].to_dict()

    # GeoJSON 각 자치구에 대응되는 값 주입
    for feature in geo_data['features']:
        gu_name = feature['properties'].get('SIG_KOR_NM')
        if gu_name in data_dict:
            feature['properties']['value'] = data_dict[gu_name]
        else:
            feature['properties']['value'] = 0

    # 데이터 분포 기반 색상 스케일 자동 보정
    vmin = data_df[col_to_plot].min()
    vmax = data_df[col_to_plot].max()
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=legend_title)
    
    # 기본 지도 스타일 정의
    def style_function(feature):
        value = feature['properties'].get('value', 0)
        return {
            'fillColor': colormap(value),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    # 마우스 오버 시 강조 효과
    def highlight_function(feature):
        return {
            'fillColor': '#ffffff',
            'color': 'black',
            'weight': 3,
            'fillOpacity': 0.9,
        }

    # GeoJSON 레이어 추가
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['SIG_KOR_NM', 'value'],
            aliases=['지역구:', f'{legend_title}:'],
            localize=True,
            sticky=False
        )
    ).add_to(m)

    colormap.add_to(m)
    return m

# 2. 수요-공급 격차 산점도 (4사분면 분석)
def draw_gap_scatter(df):
    if 'Need_Index' in df.columns and 'Supply_Index' in df.columns:
        fig = px.scatter(
            df, 
            x="Supply_Index", 
            y="Need_Index", 
            text="district", 
            hover_data=["Gap_Index"],
            color="Quadrant",
            color_discrete_map={
                'D: 고위험 대응형': '#FF4B4B',
                'C: 심각 부족형 ⚠️': '#FF8C00',
                'B: 양호형': '#4CAF50',
                'A: 과잉공급형': '#2196F3'
            },
            category_orders={
                "Quadrant": [
                    "A: 과잉공급형", 
                    "B: 양호형", 
                    "C: 심각 부족형 ⚠️", 
                    "D: 고위험 대응형"
                ]
            },
            labels={
                "Supply_Index": "공급 수준 (Supply Index)", 
                "Need_Index": "정신건강 위험도 (Need Index)",
                "Quadrant": "유형"
            },
            title="수요(위험도) vs 공급(인프라) 4사분면 분석"
        )
        
        # 중앙값 기준 사분면 구분선
        median_need = df['Need_Index'].median()
        median_supply = df['Supply_Index'].median()
        
        fig.add_vline(x=median_supply, line_width=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=median_need, line_width=1, line_dash="dash", line_color="gray")
        
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(dragmode=False)
        fig.update_traces(marker=dict(size=12), textposition='top center')
        
    else:
        # 데이터 부족 시 기본 산점도 대체 제공
        fig = px.scatter(
            df, x="center_count", y="center_count", 
            text="name", size="center_count", color="center_count",
            labels={"center_count": "인프라 수준"},
            title="데이터 부족: 기본 인프라 산점도"
        )
        fig.update_traces(textposition='top center')
        
    return fig

# 3. 자치구별 레이더 차트 (상대 비교)
def draw_radar_chart(df, selected_gu):
    cols = {
        'welfare_budget_per_capita': '1인당 복지예산',
        'single_households': '1인 가구 수',
        'perceived_stress_rate': '스트레스 인지율',
        'depression_experience_rate': '우울감 경험률',
        'suicide_rate': '자살률'
    }
    
    categories = list(cols.values())
    
    # 지표 간 스케일 차이를 제거하기 위한 정규화 (0~10)
    df_norm = df.copy()
    for col in cols.keys():
        min_val = df[col].min()
        max_val = df[col].max()
        df_norm[col] = (df[col] - min_val) / (max_val - min_val) * 10
        
    target_data = df_norm[df_norm['district'] == selected_gu].iloc[0]
    values = [target_data[col] for col in cols.keys()]
    
    # 레이더 차트 폐곡선 처리
    values += values[:1]
    categories += categories[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, 
        theta=categories, 
        fill='toself', 
        name=selected_gu,
        line_color='#14b8a6',
        fillcolor='rgba(20, 184, 166, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 10],
                tickfont=dict(size=10)
            )
        ),
        dragmode=False,
        showlegend=False,
        title={
            'text': f"{selected_gu} 5대 주요 지표 분석 (상대적 수준 0~10)",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=80, b=50, l=50, r=50)
    )
    return fig

# 4. AI 기반 정책 사각지대 바차트
def draw_ai_blindspot_bar(df_rank):
    fig = px.bar(
        df_rank.head(10), 
        x='Inefficiency', 
        y='district', 
        orientation='h',
        color='Inefficiency', 
        color_continuous_scale='Oranges',
        labels={
            'Inefficiency': '사각지대 의심 지수', 
            'district': '자치구'
        },
        title="🚨 정신건강 정책 사각지대"
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        dragmode=False,
        xaxis_title="사각지대 의심 지수 (높음 = 구조적 우선 점검 필요)",
        yaxis_title="",
        height=450,
        margin=dict(l=100, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    
    return fig

# 5. SHAP 기반 정책 사각지대 기여 요인 분석
def draw_shap_waterfall(df_shap, target_gu):
    filtered = df_shap[df_shap['district'] == target_gu]
    if filtered.empty:
        return None

    # 영문 변수명을 정책 실무용 한글 용어로 매핑
    label_map = {
        "suicide_rate": "자살률",
        "depression_experience_rate": "우울감 경험률",
        "perceived_stress_rate": "스트레스 인지율",
        "high_risk_drinking_rate": "고위험 음주율",
        "unmet_medical_need_rate": "미충족 의료 필요율",
        "elderly_population_rate": "노인 인구 비율",
        "old_dependency_ratio": "노년부양비",
        "single_households": "1인 가구 수",
        "basic_livelihood_recipients": "기초생활수급자 수",
        "unemployment_rate": "실업률",
        "welfare_budget_per_capita": "1인당 복지예산",
        "medical_institutions_count": "의료기관 수",
        "health_promotion_centers_count": "건강증진센터 수",
        "elderly_leisure_welfare_facilities_count": "노인 여가복지시설",
        "in_home_elderly_welfare_facilities_count": "재가노인복지시설",
        "parks_count": "공원 수",
        "libraries_count": "도서관 수",
        "public_sports_facilities_count": "공공 체육시설 수",
        "cultural_satisfaction": "문화생활 만족도"
    }

    # gu_data = filtered.drop(['district', 'Inefficiency'], axis=1).T
    # gu_data.columns = ['Effect']
    # gu_data.index = [label_map.get(col, col) for col in gu_data.index]
    # gu_data = gu_data.sort_values(by='Effect')

    gu_data = filtered.drop(['district', 'Inefficiency'], axis=1).T
    gu_data.columns = ['Effect']
    gu_data.index = [label_map.get(col, col) for col in gu_data.index]

    # 실전 데이터 환경에서 발생 가능한 비수치값 방어
    gu_data['Effect'] = pd.to_numeric(gu_data['Effect'], errors='coerce')
    gu_data = gu_data.dropna(subset=['Effect'])

    gu_data = gu_data.sort_values(by='Effect')

    # 음수(위험 완화) / 양수(위험 증가) 색상 구분
    colors = ['#14b8a6' if x < 0 else '#f97316' for x in gu_data['Effect']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=gu_data.index,
        x=gu_data['Effect'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>영향도: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{target_gu} 사각지대 의심 지수 기여 요인 (SHAP)",
        xaxis_title="영향도 (🟠 지표값 높음=사각지대↑ / 🟢 지표값 낮음=사각지대↓)",
        yaxis_title="",
        dragmode=False,
        height=500,
        margin=dict(l=150, r=50, t=60, b=50),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(color='#0f172a', size=12),  # 모든 텍스트 검정색
        title_font=dict(color='#0f172a', size=16, family='Pretendard')
    )
    
    fig.add_vline(x=0, line_width=2, line_color="gray", line_dash="solid")
    
    fig.update_xaxes(
        fixedrange=True, 
        showgrid=True, 
        gridcolor='rgba(0,0,0,0.1)',
        tickfont=dict(color='#000000', size=13),
        title_font=dict(color='#000000', size=13, family='Pretendard')
    )
    
    fig.update_yaxes(
        fixedrange=True,
        tickfont=dict(color='#000000', size=15, family='Pretendard')
    )
    
    return fig
    return fig

# 6. 인프라 상위 10개 자치구 바차트 (기본 분석용)
def draw_top10_bar(df):
    top10 = df.nlargest(10, 'center_count')
    fig = px.bar(
        top10, x="name", y="center_count", 
        color="center_count", text_auto=True,
        color_continuous_scale="YlOrRd",
        title="서울시 인프라 상위 10개 구"
    )
    fig.update_layout(xaxis_title="지역구", yaxis_title="센터 수")
    return fig

# 7. 클러스터 지도 (탐색적 분석)
def draw_cluster_map(geo_data, df):
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")
    
    # 임시 클러스터 기준 (탐색용)
    df['cluster'] = df['center_count'] % 3 
    folium.Choropleth(
        geo_data=geo_data,
        data=df,
        columns=["name", "cluster"],
        key_on="feature.properties.SIG_KOR_NM",
        fill_color="Set3",
        legend_name="지역 클러스터"
    ).add_to(m)
    return m