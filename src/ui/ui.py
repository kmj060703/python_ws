import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
import pandas as pd
import charts

# 1. 경로 설정
current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
GEO_PATH = os.path.join(ROOT_DIR, "data", "raw", "seoul_municipalities.geojson")
INFRA_PATH = os.path.join(ROOT_DIR, "data", "infra", "centers.csv")

# 2. 페이지 설정
st.set_page_config(
    page_title="서울시 정신건강 인사이트 플랫폼",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 3. 세션 상태 & URL 파라미터
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.current_page = query_params["page"]

# 4. CSS 스타일
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700;800&display=swap');
* { font-family: 'Pretendard', sans-serif; }

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* 배경 */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #f0fdf4 0%,
        #ccfbf1 40%,
        #e0f2fe 100%
    ) !important;
}


/* 홈 */
.home-hero {
    text-align: center;
    padding: 4rem 2rem 2.5rem;
}

.home-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #0f172a;   /* 다크 네이비 */
    letter-spacing: -0.02em;
}

.home-subtitle {
    font-size: 1.15rem;
    margin-top: 1rem;
    color: #475569;
    font-weight: 500;
}

/* 제목 아래 포인트 라인 */
.title-divider {
    width: 64px;
    height: 4px;
    margin: 20px auto 28px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #2dd4bf,
        #38bdf8
    );
}

/* 컬럼 간격 */
[data-testid="column"] {
    padding: 0 1rem;
}

/* 카드 버튼 */
.card-wrapper {
    position: relative;
    height: 300px;
    width: 100%;
    border: none;
    background: none;
    padding: 0;
    cursor: pointer;
    margin-bottom: 2.5rem;   /* 카드 간 세로 간격 */
}

/* 카드 */
.analysis-card {
    height: 100%;
    background: rgba(255,255,255,.95);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,.15);
    transition: all .3s ease;
    text-align: center;
    border: 2px solid transparent;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1rem;
}

.card-wrapper:hover .analysis-card {
    transform: translateY(-12px);
    box-shadow: 0 20px 60px rgba(102,126,234,.4);
    border-color: #667eea;
}

.card-icon { font-size: 3.5rem; }
.card-title { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
.card-desc { font-size: .95rem; color: #64748b; }

/* 서브페이지 */
.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-desc {
    background: linear-gradient(135deg,#e0f2fe,#ddd6fe);
    border-left: 4px solid #667eea;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}

.back-button button {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important;
    border-radius: 12px !important;
}

header, footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# 5. 데이터 로드
@st.cache_data
def load_data():
    geo, df = None, None
    if os.path.exists(GEO_PATH) and os.path.exists(INFRA_PATH):
        with open(GEO_PATH, "r", encoding="utf-8") as f:
            geo = json.load(f)
        df = pd.read_csv(INFRA_PATH)
    return geo, df

geo_data, infra_data = load_data()

# 6. 홈 화면
if st.session_state.current_page == "home":

    st.markdown("""
    <div class="home-hero">
        <div class="home-title">서울시 정신건강 인사이트</div>
        <div class="home-subtitle">
            데이터 기반 정신건강 인프라 분석 및 정책 제언 플랫폼
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='title-divider'></div>", unsafe_allow_html=True)

    cards = [
        ("mhvi", "🗺️", "MHVI 지도", "서울시 25개 자치구의 정신건강 취약 지수를 지도에 시각화합니다"),
        ("gap", "📊", "Gap 분석", "인구 대비 정신건강 인프라의 수요-공급 격차를 분석합니다"),
        ("cluster", "🎯", "클러스터 분석", "유사한 특성을 가진 자치구들을 그룹화하여 시각화합니다"),
        ("radar", "📈", "구별 비교", "선택한 자치구의 다차원 정신건강 지표를 비교합니다"),
        ("top10", "🏆", "TOP 10", "인프라가 우수한 상위 10개 자치구를 확인합니다"),
        ("data", "📋", "데이터 테이블", "전체 자치구의 상세 데이터를 확인합니다")
    ]

    cols = st.columns(3)
    for i, (key, icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"""
            <form method="get">
                <input type="hidden" name="page" value="{key}">
                <button class="card-wrapper" type="submit">
                    <div class="analysis-card">
                        <div class="card-icon">{icon}</div>
                        <div class="card-title">{title}</div>
                        <div class="card-desc">{desc}</div>
                    </div>
                </button>
            </form>
            """, unsafe_allow_html=True)

# 7. 서브 페이지
else:
    col_back, _ = st.columns([1, 8])
    with col_back:
        st.markdown('<div class="back-button">', unsafe_allow_html=True)
        if st.button("← 홈으로"):
            st.query_params.clear()
            st.session_state.current_page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if geo_data is None or infra_data is None:
        st.error("데이터를 불러올 수 없음")
    else:
        page = st.session_state.current_page

        if page == 'mhvi':
            st.markdown("<h1 class='page-title'>🗺️ MHVI 지도</h1>", unsafe_allow_html=True)
            st.markdown("<div class='page-desc'>서울시 25개 자치구의 정신건강 취약 지수를 시각화합니다.</div>", unsafe_allow_html=True)
            m = charts.draw_mhvi_map(geo_data, infra_data)
            st_folium(m, width="100%", height=600, returned_objects=[], key="map_mhvi")

        elif page == 'gap':
            st.markdown("<h1 class='page-title'>📊 수요-공급 격차 분석</h1>", unsafe_allow_html=True)
            fig = charts.draw_gap_scatter(infra_data)
            st.plotly_chart(fig, use_container_width=True)

        elif page == 'cluster':
            st.markdown("<h1 class='page-title'>🎯 클러스터 분석</h1>", unsafe_allow_html=True)
            m = charts.draw_cluster_map(geo_data, infra_data)
            st_folium(m, width="100%", height=600, returned_objects=[], key="map_cluster")

        elif page == 'radar':
            st.markdown("<h1 class='page-title'>📈 자치구별 세부 지표 비교</h1>", unsafe_allow_html=True)
            selected_gu = st.selectbox("자치구를 선택하세요", infra_data['name'].unique())
            fig = charts.draw_radar_chart(infra_data, selected_gu)
            st.plotly_chart(fig, use_container_width=True)

        elif page == 'top10':
            st.markdown("<h1 class='page-title'>🏆 인프라 우수 지역 TOP 10</h1>", unsafe_allow_html=True)
            fig = charts.draw_top10_bar(infra_data)
            st.plotly_chart(fig, use_container_width=True)

        elif page == 'data':
            st.markdown("<h1 class='page-title'>📋 구별 상세 데이터</h1>", unsafe_allow_html=True)
            st.dataframe(
                infra_data.sort_values(by="center_count", ascending=False),
                use_container_width=True,
                hide_index=True
            )