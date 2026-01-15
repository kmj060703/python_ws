import folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import branca.colormap as cm
import json

# 1. MHVI 지도 (Need Index 시각화)
def draw_mhvi_map(geo_data, data_df):
    seoul_bounds = [[37.42, 126.75], [37.70, 127.18]]
    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11,
        zoom_control=False,
        dragging=False, # 드래그(이동) 완전히 차단
        scrollWheelZoom=False,
        doubleClickZoom=False,
        touchZoom=False,
        tiles="cartodbpositron",
        attr=' '
    )
    m.get_root().html.add_child(folium.Element("""
        <style>.leaflet-control-attribution { display: none !important; }</style>
    """))
    m.fit_bounds(seoul_bounds) # 서울 영역에 맞게 자동 줌/이동

    # 데이터 컬럼 자동 감지 및 설정
    if 'Need_Index' in data_df.columns:
        col_to_plot = "Need_Index"
        legend_title = "정신건강 취약 지수 (높음=위험)"
        # 색상 스케일 (YlOrRd)
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026']
    else:
        col_to_plot = "center_count"
        legend_title = "정신건강 인프라 수"
        colors = ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'] # PuBu

    # 구 이름 컬럼 확보
    if 'name' not in data_df.columns and 'district' in data_df.columns:
        data_df = data_df.copy()
        data_df['name'] = data_df['district']

    # 데이터 매핑을 위한 딕셔너리 생성
    data_dict = data_df.set_index('name')[col_to_plot].to_dict()

    # GeoJSON properties에 데이터 병합
    for feature in geo_data['features']:
        gu_name = feature['properties'].get('SIG_KOR_NM')
        if gu_name in data_dict:
            feature['properties']['value'] = data_dict[gu_name]
        else:
            feature['properties']['value'] = 0

    # 색상 맵 생성
    vmin = data_df[col_to_plot].min()
    vmax = data_df[col_to_plot].max()
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=legend_title)
    
    # 스타일 함수
    def style_function(feature):
        value = feature['properties'].get('value', 0)
        return {
            'fillColor': colormap(value),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    # 하이라이트 함수 (마우스 오버 시)
    def highlight_function(feature):
        return {
            'fillColor': '#ffffff',
            'color': 'black',
            'weight': 3,
            'fillOpacity': 0.9,
        }

    # GeoJson 레이어 추가 (클릭 이벤트 활성화)
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

    # 범례 추가
    colormap.add_to(m)

    return m

# 2. Gap 분석 산점도 (MHVI 데이터 반영)
def draw_gap_scatter(df):
    if 'Need_Index' in df.columns and 'Supply_Index' in df.columns:
        fig = px.scatter(
            df, 
            x="Supply_Index", 
            y="Need_Index", 
            text="district", 
            # size="Gap_Index",  <-- 에러 원인 제거 (음수 값 포함 가능)
            hover_data=["Gap_Index"], # 툴팁에 Gap 정보 표시
            color="Quadrant",
            color_discrete_map={
                'D: 고위험 대응형': '#FF4B4B', # 고위험
                'C: 심각 부족형 ⚠️': '#FF8C00', # 심각 부족
                'B: 양호형': '#4CAF50', # 양호
                'A: 과잉공급형': '#2196F3'  # 과잉 공급
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
        
        # 4사분면 기준선 (중앙값)
        median_need = df['Need_Index'].median()
        median_supply = df['Supply_Index'].median()
        
        fig.add_vline(x=median_supply, line_width=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=median_need, line_width=1, line_dash="dash", line_color="gray")
        
        # 줌/드래그 차단 설정
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(dragmode=False)

        # 점 크기 고정
        fig.update_traces(marker=dict(size=12), textposition='top center')
        
    else:
        # 데이터 없을 경우 기존 더미 차트
        fig = px.scatter(
            df, x="center_count", y="center_count", 
            text="name", size="center_count", color="center_count",
            labels={"center_count": "인프라 수준"},
            title="데이터 부족: 기본 인프라 산점도"
        )
        fig.update_traces(textposition='top center')
        
    return fig

# 3. 클러스터 지도 (기본 유지)
def draw_cluster_map(geo_data, df):
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")
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

# 4. 레이더 차트 (실제 데이터 반영)
def draw_radar_chart(df, selected_gu):
    # 분석할 5가지 지표 선택
    # 1. 1인당 복지예산 (인프라)
    # 2. 1인 가구 수 (사회적 고립/밀도)
    # 3. 스트레스 인지율
    # 4. 우울감 경험률
    # 5. 자살률
    
    cols = {
        'welfare_budget_per_capita': '1인당 복지예산',
        'single_households': '1인 가구 수',
        'perceived_stress_rate': '스트레스 인지율',
        'depression_experience_rate': '우울감 경험률',
        'suicide_rate': '자살률'
    }
    
    categories = list(cols.values())
    
    # 정규화 (0~10점 척도)
    df_norm = df.copy()
    for col in cols.keys():
        min_val = df[col].min()
        max_val = df[col].max()
        # 0~10점으로 변환
        df_norm[col] = (df[col] - min_val) / (max_val - min_val) * 10
        
    target_data = df_norm[df_norm['district'] == selected_gu].iloc[0]
    values = [target_data[col] for col in cols.keys()]
    
    # 닫힌 도형을 위해 첫 번째 값 추가
    values += values[:1]
    categories += categories[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, 
        theta=categories, 
        fill='toself', 
        name=selected_gu,
        line_color='#FF4B4B'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 10],
                tickfont=dict(size=10)
            )
        ),
        dragmode=False, # 드래그 차단
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

# 5. TOP 10 바차트 (기본 유지)
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

# --- AI 진단용 추가 함수 ---

def draw_ai_blindspot_bar(df_rank):
    fig = px.bar(
        df_rank.head(10), 
        x='Inefficiency', y='district', orientation='h',
        color='Inefficiency', color_continuous_scale='Reds',
        labels={'Inefficiency': '관리 필요 지수', 'district': '자치구'}
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        dragmode=False
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

def draw_shap_waterfall(df_shap, target_gu):
    # 데이터 존재 여부 확인 (에러 방지 핵심)
    filtered = df_shap[df_shap['district'] == target_gu]
    if filtered.empty:
        return None

    # 한글 매핑 정보
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

    gu_data = filtered.drop(['district', 'Inefficiency'], axis=1).T
    gu_data.columns = ['Effect']
    
    # 인덱스(영문 컬럼명)를 한글로 변환
    gu_data.index = [label_map.get(col, col) for col in gu_data.index]
    
    gu_data = gu_data.sort_values(by='Effect')

    fig = px.bar(
        gu_data, x='Effect', y=gu_data.index, orientation='h',
        color='Effect', color_continuous_scale='RdBu_r',
        labels={'Effect': '위험 영향도 (빨간색=위험)', 'y': '지표'}
    )
    fig.update_layout(
        dragmode=False,
        yaxis_title="진단 지표"
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig