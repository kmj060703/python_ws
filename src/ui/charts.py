import folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

    # 데이터 컬럼 자동 감지
    if 'Need_Index' in data_df.columns:
        col_to_plot = "Need_Index"
        legend_title = "정신건강 취약 지수 (높음=위험)"
        fill_color = "YlOrRd"  # 위험할수록 붉은색
        
        # 툴팁용 컬럼 이름 통일 (district -> name 변환 불필요, GeoJSON과 매칭은 key_on으로 해결)
        # 하지만 data_df에 'name'이 없고 'district'만 있을 수 있음.
        if 'name' not in data_df.columns and 'district' in data_df.columns:
            data_df['name'] = data_df['district']
            
    else:
        col_to_plot = "center_count"
        legend_title = "정신건강 인프라 수"
        fill_color = "PuBu" # 인프라 많을수록 파란색 (기존 YlOrRd에서 변경 고려 가능하지만 유지)

    folium.Choropleth(
        geo_data=geo_data,
        data=data_df,
        columns=["name", col_to_plot],
        key_on="feature.properties.SIG_KOR_NM",
        fill_color=fill_color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_title
    ).add_to(m)
    
    folium.GeoJson(
        geo_data,
        style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["SIG_KOR_NM"], aliases=["지역구:"], localize=True)
    ).add_to(m)
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
                'D': '#FF4B4B', # 고위험
                'C': '#FF8C00', # 심각 부족
                'B': '#4CAF50', # 양호
                'A': '#2196F3'  # 과잉 공급
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
        labels={'Inefficiency': '방치 지수', 'district': '자치구'}
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

    gu_data = filtered.drop(['district', 'Inefficiency'], axis=1).T
    gu_data.columns = ['Effect']
    gu_data = gu_data.sort_values(by='Effect')

    fig = px.bar(
        gu_data, x='Effect', y=gu_data.index, orientation='h',
        color='Effect', color_continuous_scale='RdBu_r',
        labels={'Effect': '기여도 (빨간색=위험)', 'index': '지표'}
    )
    fig.update_layout(dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig