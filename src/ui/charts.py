import folium
import plotly.express as px
import plotly.graph_objects as go

# 1. MHVI 지도
def draw_mhvi_map(geo_data, infra_data):
    seoul_bounds = [[37.42, 126.75], [37.70, 127.18]]
    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11.5,
        min_zoom=11.5,
        zoom_control=False,
        dragging=False,
        scrollWheelZoom=False,
        tiles="cartodbpositron",
        attr=' '
    )
    # 지도 하단에 문구 제거
    m.get_root().html.add_child(folium.Element("""
        <style>.leaflet-control-attribution { display: none !important; }</style>
    """))
    m.fit_bounds(seoul_bounds)

    folium.Choropleth(
        geo_data=geo_data,
        data=infra_data,
        columns=["name", "center_count"],
        key_on="feature.properties.SIG_KOR_NM",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="정신건강 인프라 지수"
    ).add_to(m)
    
    folium.GeoJson(
        geo_data,
        style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["SIG_KOR_NM"], aliases=["지역구:"], localize=True)
    ).add_to(m)
    return m

# 2. Gap 분석 산점도
def draw_gap_scatter(df):
    # 임시로 center_count를 x, y축으로 활용
    fig = px.scatter(
        df, x="center_count", y="center_count", 
        text="name", size="center_count", color="center_count",
        labels={"center_count": "인프라 수준"},
        title="수요 대비 인프라 격차 분석"
    )
    fig.update_traces(textposition='top center')
    return fig

# 3. 클러스터 지도
def draw_cluster_map(geo_data, df):
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")
    # 구별로 임시 그룹 할당
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

# 4. 레이더 차트
def draw_radar_chart(df, selected_gu):
    # 샘플 지표
    categories = ['인프라 수준', '인구 밀도', '스트레스 인지율', '우울감 경험률', '자살 생각률']
    
    # 선택된 구의 데이터 가져오기 (샘플)
    target_data = df[df['name'] == selected_gu].iloc[0]
    val = target_data['center_count']
    values = [val * 2, 5, 3, 4, val]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name=selected_gu,
        line_color='#FF4B4B'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title=f"{selected_gu} 정신건강 지표 분석"
    )
    return fig

# 5. TOP 10 바차트
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