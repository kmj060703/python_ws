import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
import pandas as pd
import charts_3 as charts

DISCLAIMER = """
⚠️ 본 플랫폼의 모든 분석 결과는 **인과관계를 의미하지 않으며**  
정책 검토를 위한 **참고용 분석 결과**입니다.
"""

INDEX_DESC = """
- **Need Index(정신건강 위험도 수준)**: 자살률, 우울감 경험률, 스트레스 인지율 등 주요 정신건강 위험 지표를 가중합하여 산출한 종합 위험 지수  
- **Supply Index(인프라 지수)**: 의료, 복지, 문화, 체육 등 정신건강 관련 인프라 지표를 표준화한 뒤 가중합하여 산출한 공급 수준 지수  
- **Gap Index(격차 지수)** = Need − Supply : 공급 대비 인프라 지원 정도를 파악하기 위한 지수 
  (＋ 값일수록 need 대비 supply이가 부족한 지역 / - 값일수록 상대적 공급 여유 지역 )
"""

# 1. 경로 설정
current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
GEO_PATH = os.path.join(ROOT_DIR, "data", "raw", "seoul_municipalities.geojson")
INFRA_PATH = os.path.join(ROOT_DIR, "data", "infra", "centers.csv")
NEED_PATH = os.path.join(ROOT_DIR, "data", "processed", "need_tidy.csv")
SUPPLY_PATH = os.path.join(ROOT_DIR, "data", "processed", "supply_tidy.csv")
MHVI_PATH = os.path.join(ROOT_DIR, "data", "processed", "mhvi_final_result.csv")

# 데이터 결과물 경로
RANK_PATH = os.path.join(ROOT_DIR, "data", "outputs", "tables", "ai_blindspot_ranking.csv")
SHAP_PATH = os.path.join(ROOT_DIR, "data", "outputs", "tables", "ai_blindspot_shap.csv")
POLICY_PATH = os.path.join(ROOT_DIR, "data", "outputs", "recommend_policy", "need_policy_recommendation_by_district.csv")

# 2. 페이지 설정
st.set_page_config(
    page_title="서울시 정신건강 인사이트 플랫폼",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 3. 세션 상태
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.current_page = query_params["page"]

# 4. CSS 스타일 (색상 개선 + expander 텍스트 수정)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700;800&display=swap');
* { font-family: 'Pretendard', sans-serif; }

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #f0fdf4 0%,
        #ccfbf1 40%,
        #e0f2fe 100%
    ) !important;
}

.home-hero {
    text-align: center;
    padding: 4rem 2rem 2.5rem;
}

.home-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.home-subtitle {
    font-size: 1.15rem;
    margin-top: 1rem;
    color: #475569;
    font-weight: 500;
    line-height: 1.6;
}

.title-divider {
    width: 64px;
    height: 4px;
    margin: 20px auto 28px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #14b8a6,
        #06b6d4
    );
}

[data-testid="column"] {
    padding: 0 1rem;
}

.card-wrapper {
    position: relative;
    height: 300px;
    width: 100%;
    border: none;
    background: none;
    padding: 0;
    cursor: pointer;
    margin-bottom: 2.5rem;
}

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
    box-shadow: 0 20px 60px rgba(20,184,166,.4);
    border-color: #14b8a6;
}

.card-icon { font-size: 3.5rem; }
.card-title { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
.card-desc { font-size: .95rem; color: #64748b; line-height: 1.5; }

.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg,#14b8a6,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

.page-desc {
    background: linear-gradient(135deg,#e0f2fe,#ccfbf1);
    border-left: 4px solid #14b8a6;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.back-button button {
    background: linear-gradient(135deg,#14b8a6,#06b6d4) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* 개선된 info box */
.stAlert {
    border-radius: 12px;
    border-left: 4px solid #14b8a6;
}

/* ===== expander 텍스트 색상 수정 (핵심) ===== */
.streamlit-expanderHeader {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid #e2e8f0 !important;
}

/* expander 헤더 내부 텍스트 강제 검은색 */
.streamlit-expanderHeader p,
.streamlit-expanderHeader span,
.streamlit-expanderHeader div,
.streamlit-expanderHeader summary {
    color: #0f172a !important;
}

/* selectbox 스타일 개선 */
[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 2px solid #14b8a6 !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] > div:hover {
    border-color: #0d9488 !important;
}

/* selectbox 텍스트 */
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #0f172a !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background-color: #ffffff !important;
    padding: 1.5rem !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
}

.streamlit-expanderContent p,
.streamlit-expanderContent div,
.streamlit-expanderContent li,
.streamlit-expanderContent span,
.streamlit-expanderContent {
    color: #0f172a !important;
}

/* 우선순위 번호 강조 */
.streamlit-expanderHeader strong {
    color: #14b8a6 !important;
    font-weight: 800 !important;
}

/* Markdown 내 모든 텍스트 강제 검은색 */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] {
    color: #0f172a !important;
}

/* 정책 제언 텍스트 강제 */
.element-container p,
.element-container div {
    color: #0f172a !important;
}

header, footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# 5. 데이터 로드
@st.cache_data
def load_data():
    geo, df, radar_df, mhvi_df = None, None, None, None
    
    # 기본 인프라 데이터 및 지도
    if os.path.exists(GEO_PATH) and os.path.exists(INFRA_PATH):
        with open(GEO_PATH, "r", encoding="utf-8") as f:
            geo = json.load(f)
        df = pd.read_csv(INFRA_PATH)
        
    # 레이더 차트용 통합 데이터
    if os.path.exists(NEED_PATH) and os.path.exists(SUPPLY_PATH):
        try:
            df_need = pd.read_csv(NEED_PATH)
            df_supply = pd.read_csv(SUPPLY_PATH)
            radar_df = pd.merge(df_need, df_supply, on='district', how='inner')
        except Exception as e:
            st.error(f"레이더 데이터 로드 중 오류: {e}")

    # MHVI 데이터
    if os.path.exists(MHVI_PATH):
        try:
            mhvi_df = pd.read_csv(MHVI_PATH)
        except Exception as e:
            st.error(f"MHVI 데이터 로드 중 오류: {e}")
            
    return geo, df, radar_df, mhvi_df

geo_data, infra_data, radar_df, mhvi_df = load_data()


# 6. 홈 화면
if st.session_state.current_page == "home":
    st.markdown("""
    <div class="home-hero">
        <div class="home-title">서울시 정신건강 인사이트</div>
        <div class="home-subtitle">
            데이터 기반 분석으로 제안하는<br>
            서울시 정신건강 정책 개입 우선순위
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='title-divider'></div>", unsafe_allow_html=True)

    cards = [
        ("mhvi", "🗺️", "지역별 정신건강 현황", "서울시 25개 자치구의 정신건강 지수 시각화"),
        ("gap", "📊", "수요-공급 격차 분석", "지역별 정신적 위험도 대비 인프라 공급의 불균형 진단"),
        ("ai_diagnosis", "🤖", "AI 정책 사각지대 탐색", "데이터 패턴 분석을 통한 잠재 위험 지역 발견"),
        ("policy_sim", "📈", "맞춤형 정책 제안", "자치구별 우선 개입 영역 및 정책 방향 제시"),
        ("radar", "📈", "자치구 세부 비교", "선택한 지역의 다차원 지표 상세 분석"),
        ("data", "📋", "전체 데이터 보기", "모든 자치구의 통합 데이터 테이블")
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
        st.error("데이터를 불러올 수 없습니다. 데이터 파일 경로를 확인해주세요.")
    else:
        page = st.session_state.current_page

        if page == 'mhvi':
            st.markdown("<h1 class='page-title'>🗺️ 지역별 정신건강 현황 지도</h1>", unsafe_allow_html=True)
            
            st.info("""
💡 **이 지도는 무엇을 보여주나요?**  
지도의 색상은 각 자치구의 **정신건강 위험도 수준**을 나타냅니다.  
수치가 높을수록 정신건강 위험 요인(자살률, 우울감, 스트레스 등)이 상대적으로 높은 지역입니다.

🖱️ **지도를 클릭**하면 해당 자치구의 상세 분석 페이지로 이동합니다.
            """)
            
            with st.expander("📊 지수 계산 방법 자세히 보기"):
                st.caption(INDEX_DESC)
            
            target_df = mhvi_df if mhvi_df is not None else infra_data
            
            if mhvi_df is None:
                st.warning("⚠️ MHVI 데이터(mhvi_final_result.csv)가 없어 기본 인프라 지도를 표시합니다.")
                
            m = charts.draw_mhvi_map(geo_data, target_df)
            
            map_output = st_folium(m, width="100%", height=600, returned_objects=["last_object_clicked"], key="map_mhvi")

            def is_point_in_polygon(x, y, poly):
                n = len(poly)
                inside = False
                p1x, p1y = poly[0]
                for i in range(n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside

            def find_gu_by_coord(geo_data, lat, lng):
                point_x, point_y = lng, lat
                
                for feature in geo_data['features']:
                    gu_name = feature['properties'].get('SIG_KOR_NM')
                    geometry = feature['geometry']
                    geom_type = geometry['type']
                    coords = geometry['coordinates']
                    
                    if geom_type == 'Polygon':
                        if is_point_in_polygon(point_x, point_y, coords[0]):
                            return gu_name
                    elif geom_type == 'MultiPolygon':
                        for poly in coords:
                            if is_point_in_polygon(point_x, point_y, poly[0]):
                                return gu_name
                return None

            if map_output['last_object_clicked']:
               clicked_lat = map_output['last_object_clicked'].get('lat')
               clicked_lng = map_output['last_object_clicked'].get('lng')
               
               properties = map_output['last_object_clicked'].get('properties', {})
               clicked_gu = properties.get('SIG_KOR_NM') or properties.get('name') or properties.get('SIG_ENG_NM')
               
               if not clicked_gu and clicked_lat and clicked_lng:
                   clicked_gu = find_gu_by_coord(geo_data, clicked_lat, clicked_lng)

               if clicked_gu:
                   st.success(f"✅ **{clicked_gu}** 선택됨! 상세 페이지로 이동합니다.")
                   st.session_state['selected_gu_from_map'] = clicked_gu
                   st.session_state.current_page = 'radar'
                   st.query_params['page'] = 'radar'
                   st.rerun()
               else:
                   st.warning("⚠️ 선택한 위치에서 지역구 정보를 찾을 수 없습니다. 지도의 구역 내부를 클릭해주세요.")

        elif page == 'gap':
            st.markdown("<h1 class='page-title'>📊 수요-공급 격차(GAP) 분석</h1>", unsafe_allow_html=True)
            
            st.info("""
💡 **4사분면 분석**  
각 자치구를 정신건강 **취약도(Need)**와 **인프라 수준(Supply)**으로 분류합니다.

- 🟢 **A (과잉공급형)**: 인프라는 충분하나 취약도가 낮은 지역
- 🟡 **B (양호형)**: 인프라도 충분하고 취약도도 높은 지역  
- 🔴 **C (심각 부족형)**: 취약도는 높으나 인프라가 부족한 지역 → **우선 개입 필요**
- 🔵 **D (고위험 대응형 / 사각지대)**: 취약도는 매우 높지만 인프라는 갖춰진 지역
            """)
            
            if mhvi_df is not None:
                fig = charts.draw_gap_scatter(mhvi_df)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                with st.expander("📋 상세 데이터 보기"):
                    display_cols = ['district', 'Need_Index', 'Supply_Index', 'Gap_Index', 'Quadrant']
                    display_df = mhvi_df[display_cols].copy()
                    display_df.columns = ['자치구', '취약 지수', '인프라 지수', '격차 지수', '유형']
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("⚠️ MHVI 데이터가 없어 기본 인프라 데이터로 표시합니다.")
                fig = charts.draw_gap_scatter(infra_data)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        elif page == 'ai_diagnosis':
            st.markdown("<h1 class='page-title'>🤖 정책 사각지대 진단</h1>", unsafe_allow_html=True)
            
            st.info("""
💡 **사각지대의 정의 : 인프라(Supply) 공급 수준이 높음에도 불구하고 정신건강 위험도(Need)가 여전히 높은 지역**

이는 **투입된 예산과 자원이 충분함**에도 불구하고, 
정책 효과가 실제 위험 완화로 이어지지 않거나, **자원이 비효율적으로 활용**되고 있을 가능성을 의미합니다. 

따라서 이러한 지역은 단순한 고위험 지역이 아니라
**기존 정책의 효과성 점검과 구조적 개선이 필요한 [비효율적 고위험 지역]**으로 간주해야 합니다.

이는 다음과 같은 **정책 사각지대**를 의미할 수 있습니다:
- 📍 **인프라는 존재하지만, 실제 지역의 수요에 비해 질적 적합성이 낮은 경우**
- 📍 **물리적/제도적 공급은 있으나, 접근성 또는 활용도가 낮아 실효성이 떨어지는 경우** 
- 📍 **단기적 인프라 확충으로는 개선되지 않는, 구조적/누적적 위험 요인이 지속되는 경우** 

👉 이런 지역은 **단순 인프라 확충이 아닌, 맞춤형 정책 개입**이 필요합니다.

---
🚫 **사각지대의 핵심**  
즉, 해당 지역의 문제는 **"지원의 양"**이 아니라 **지원 방식과 구조의 문제**일 수 있습니다. 
이러한 지역은 단순한 공급 확대가 아니라, 지역 특성과 위험 요인에 기반한 **맞춤형 정책 개입과 정책 설계의 재검토**가 요구됩니다. 
            """)
            
            st.markdown("---")
            st.markdown("### 📊 분석 결과")
            
            if os.path.exists(RANK_PATH) and os.path.exists(SHAP_PATH):
                df_rank = pd.read_csv(RANK_PATH)
                df_shap = pd.read_csv(SHAP_PATH)
                
                # 좌우 레이아웃
                c1, c2 = st.columns([1, 1.2])
                
                # with c1:
                #     st.markdown("**1️⃣ 정책 사각지대 의심 지역**")
                #     st.caption("Need와 Supply가 동시에 높아 구조적 점검이 필요한 지역")
                #     st.plotly_chart(charts.draw_ai_blindspot_bar(df_rank), use_container_width=True, config={'displayModeBar': False})
                    
                with c1:
                    st.markdown("**1️⃣ 정책 사각지대 의심 지역 (D유형)**")
                    st.caption("취약도와 인프라가 모두 높아 정책 효과 점검이 필요한 지역")

                    # 🔥 D유형만 필터링
                    df_d_blindspot = df_rank[df_rank["Quadrant"] == "D"]

                    if df_d_blindspot.empty:
                        st.success("✅ 현재 사각지대(D유형)에 해당하는 지역이 없습니다.")
                    else:
                        st.plotly_chart(
                            charts.draw_ai_blindspot_bar(df_d_blindspot),
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )

                with c2:
                    st.markdown("**2️⃣ 지역별 사각지대 원인 분석**")
                    selected_gu = st.selectbox("🔍 분석할 자치구 선택", infra_data['name'].unique(), key="ai_gu_select")
                    
                    fig_shap = charts.draw_shap_waterfall(df_shap, selected_gu)
                    if fig_shap:
                        st.caption("""
                        **📈 그래프 해석 방법:**
                        
                        이 차트는 각 지표가 **사각지대 의심 지수에 얼마나 영향을 주는지** 보여줍니다.
                        
                        - **🟠 주황색 막대 (오른쪽 →)**: 
                          - 이 지표의 **값이 높아서** 사각지대 지수를 **증가**시킴
                          - 예: "1인당 복지예산"이 주황이면 → 복지예산이 **많은데도** 사각지대 의심
                        
                        - **🟢 청록색 막대 (왼쪽 ←)**: 
                          - 이 지표의 **값이 낮아서** 사각지대 지수를 **감소**시킴
                          - 예: "도서관 수"가 청록이면 → 도서관이 **적어서** 사각지대 지수 낮아짐
                        
                        💡 **핵심**: 주황색이 많다 = 해당 지표가 높은데도 Need/Supply 균형이 안 맞음
                        """)
                        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.success(f"""
✅ **{selected_gu}**는 **정상 범주** 지역입니다.  
Need/Supply 균형이 적절하여 별도의 구조적 점검이 필요하지 않습니다.
                        """)
            else:
                st.warning("⚠️ 분석 데이터를 찾을 수 없습니다.")

        elif page == 'policy_sim':
            st.markdown("<h1 class='page-title'>📈 자치구별 맞춤형 정책 제안</h1>", unsafe_allow_html=True)

            st.markdown("""
            <div class="page-desc">
                💡 <strong>데이터 기반 정책 우선순위</strong><br>
                각 자치구의 주요 취약 요인을 분석하여 
                <strong>우선 개입이 필요한 영역</strong>과 
                <strong>구체적인 정책 방향</strong>을 제시합니다.
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists(POLICY_PATH):
                df_poly = pd.read_csv(POLICY_PATH)
                
                # 자치구 선택 UI 개선
                st.markdown("### 📍 자치구 선택")
                selected_gu = st.selectbox(
                    "분석할 자치구를 선택하세요", 
                    df_poly['district'].unique(),
                    label_visibility="collapsed"
                )
                
                res = df_poly[df_poly['district'] == selected_gu].iloc[0]
                
                factor_map = {
                    "suicide_rate": "자살률",
                    "depression_experience_rate": "우울감 경험률",
                    "perceived_stress_rate": "스트레스 인지율",
                    "high_risk_drinking_rate": "고위험 음주율",
                    "unmet_medical_need_rate": "미충족 의료율",
                    "unemployment_rate": "실업률",
                    "elderly_population_rate": "노인 인구 비율",
                    "old_dependency_ratio": "노년 부양비",
                    "single_households": "1인 가구 수",
                    "basic_livelihood_recipients": "기초생활수급자 수"
                }

                # 주요 위험 요인 카드
                st.markdown(f"### 🎯 {selected_gu} 주요 위험 요인 TOP 3")
                
                cols = st.columns(3)
                badge_colors = ["#dc2626", "#f97316", "#fbbf24"]  # 빨강, 주황, 노랑
                emoji_list = ["🔴", "🟠", "🟡"]
                
                for i in range(1, 4):
                    factor_key = f'top{i}_factor'
                    if factor_key in res:
                        factor_raw = res[factor_key]
                        factor_name = factor_map.get(factor_raw, factor_raw)
                        
                        with cols[i-1]:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #ffffff, #f8fafc);
                                        padding: 1.5rem;
                                        border-radius: 12px;
                                        border: 2px solid {badge_colors[i-1]};
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                                        text-align: center;
                                        min-height: 120px;
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: center;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji_list[i-1]}</div>
                                <div style="background: {badge_colors[i-1]};
                                            color: white;
                                            padding: 0.25rem 0.75rem;
                                            border-radius: 999px;
                                            font-size: 0.875rem;
                                            font-weight: 700;
                                            display: inline-block;
                                            margin: 0 auto 0.75rem;">
                                    우선순위 {i}
                                </div>
                                <div style="font-size: 1.1rem;
                                            font-weight: 700;
                                            color: #0f172a;">
                                    {factor_name}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 상세 정책 제안
                st.markdown(f"### 📋 {selected_gu} 맞춤형 정책 제안")
                
                for i in range(1, 4):
                    factor_key = f'top{i}_factor'
                    policy_key = f'policy_direction_{i}'
                    
                    if factor_key in res and policy_key in res:
                        factor_raw = res[factor_key]
                        factor_name = factor_map.get(factor_raw, factor_raw)
                        policy_desc = res[policy_key]
                        
                        with st.expander(f"{emoji_list[i-1]} **우선순위 {i}: {factor_name} 기반 정책**", expanded=(i==1)):
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
                                        padding: 1rem;
                                        border-radius: 8px;
                                        border-left: 4px solid {badge_colors[i-1]};
                                        margin-bottom: 1rem;">
                                <strong style="color: {badge_colors[i-1]};">🎯 주요 타겟 지표:</strong> {factor_name}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**💡 정책 제언:**")
                            policy_lines = policy_desc.split('\n')
                            for line in policy_lines:
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")
            else:
                st.warning("⚠️ 정책 제언 데이터(need_policy_recommendation_by_district.csv)를 찾을 수 없습니다.")

        elif page == 'radar':
            st.markdown("<h1 class='page-title'>📈 자치구별 세부 지표 비교</h1>", unsafe_allow_html=True)
            
            st.info("""
💡 **5개 핵심 지표의 균형 분석**  
선택한 자치구의 정신건강 관련 5가지 주요 지표를 시각화합니다.  
각 축의 값이 클수록 해당 영역의 수치가 높음을 의미합니다.
            """)
            
            if radar_df is not None:
                gu_list = radar_df['district'].unique().tolist()
                default_index = 0
                
                if 'selected_gu_from_map' in st.session_state and st.session_state.selected_gu_from_map in gu_list:
                    default_index = gu_list.index(st.session_state.selected_gu_from_map)
                
                selected_gu = st.selectbox("📍 자치구 선택", gu_list, index=default_index)
                fig = charts.draw_radar_chart(radar_df, selected_gu)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                selected_data = radar_df[radar_df['district'] == selected_gu].iloc[0]
                
                # 서울시 평균 계산
                seoul_avg = {
                    'welfare_budget_per_capita': radar_df['welfare_budget_per_capita'].mean(),
                    'medical_institutions_count': radar_df['medical_institutions_count'].mean(),
                    'suicide_rate': radar_df['suicide_rate'].mean(),
                    'single_households': radar_df['single_households'].mean(),
                    'perceived_stress_rate': radar_df['perceived_stress_rate'].mean()
                }
                
                st.markdown("### 📌 주요 특징 (서울시 평균 대비)")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    val = selected_data.get('welfare_budget_per_capita', 0)
                    avg = seoul_avg['welfare_budget_per_capita']
                    delta = val - avg
                    st.metric(
                        "1인당 복지예산 (천원)", 
                        f"{val:,.1f}",
                        f"{delta:+,.1f} (평균 {avg:,.1f})",
                        delta_color="normal"
                    )
                with col2:
                    val = selected_data.get('medical_institutions_count', 0)
                    avg = seoul_avg['medical_institutions_count']
                    delta = val - avg
                    st.metric(
                        "의료기관 수", 
                        f"{val:.0f}개",
                        f"{delta:+.0f}개 (평균 {avg:.0f}개)",
                        delta_color="normal"
                    )
                with col3:
                    val = selected_data.get('suicide_rate', 0)
                    avg = seoul_avg['suicide_rate']
                    delta = val - avg
                    st.metric(
                        "자살률", 
                        f"{val:.1f}",
                        f"{delta:+.1f} (평균 {avg:.1f})",
                        delta_color="inverse"  # 자살률은 낮을수록 좋음
                    )
                
                # 추가 지표
                st.markdown("### 📊 추가 지표")
                col4, col5 = st.columns(2)
                
                with col4:
                    val = selected_data.get('single_households', 0)
                    avg = seoul_avg['single_households']
                    delta = val - avg
                    st.metric(
                        "1인 가구 수", 
                        f"{val:.0f}가구",
                        f"{delta:+.0f} (평균 {avg:.0f})",
                        delta_color="off"
                    )
                with col5:
                    val = selected_data.get('perceived_stress_rate', 0)
                    avg = seoul_avg['perceived_stress_rate']
                    delta = val - avg
                    st.metric(
                        "스트레스 인지율", 
                        f"{val:.1f}%",
                        f"{delta:+.1f}%p (평균 {avg:.1f}%)",
                        delta_color="inverse"
                    )
            else:
                st.error("⚠️ 세부 지표 데이터를 불러올 수 없습니다.")

        elif page == 'data':
            st.markdown("<h1 class='page-title'>📋 자치구별 상세 데이터</h1>", unsafe_allow_html=True)
            
            st.info("""
💡 **전체 데이터 통합 뷰**  
MHVI 지수(Need, Supply, Gap)와 원본 세부 지표를 통합한 전체 데이터입니다.  
우측 상단 아이콘으로 컬럼 필터 및 정렬이 가능합니다.
            """)
            
            if radar_df is not None and mhvi_df is not None:
                master_df = pd.merge(mhvi_df, radar_df, on='district', how='outer')
                
                main_cols = ['district', 'Quadrant', 'Need_Index', 'Supply_Index', 'Gap_Index']
                other_cols = [c for c in master_df.columns if c not in main_cols]
                master_df = master_df[main_cols + other_cols]
                
                col_rename = {
                    'district': '자치구',
                    'Quadrant': '유형',
                    'Need_Index': '취약지수',
                    'Supply_Index': '인프라지수',
                    'Gap_Index': '격차지수',
                    'suicide_rate': '자살률',
                    'depression_experience_rate': '우울감경험률',
                    'perceived_stress_rate': '스트레스인지율',
                    'single_households': '1인가구수',
                    'welfare_budget_per_capita': '1인당복지예산',
                    'libraries_count': '도서관수',
                    'parks_count': '공원수',
                    'medical_institutions_count': '의료기관수'
                }
                display_df = master_df.rename(columns=col_rename)

                st.dataframe(display_df, use_container_width=True, height=600)
                
                csv = display_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 전체 데이터 다운로드 (CSV)",
                    data=csv,
                    file_name="seoul_mental_health_full_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ 상세 데이터를 불러올 수 없어 기본 데이터만 표시합니다.")
                st.dataframe(infra_data, use_container_width=True)