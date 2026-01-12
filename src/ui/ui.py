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
NEED_PATH = os.path.join(ROOT_DIR, "data", "processed", "need_tidy.csv")
SUPPLY_PATH = os.path.join(ROOT_DIR, "data", "processed", "supply_tidy.csv")
MHVI_PATH = os.path.join(ROOT_DIR, "data", "processed", "mhvi_final_result.csv")

# 데이터 결과물 경로
RANK_PATH = os.path.join(ROOT_DIR, "data", "outputs", "tables", "ai_blindspot_ranking.csv")
SHAP_PATH = os.path.join(ROOT_DIR, "data", "outputs", "tables", "ai_blindspot_shap.csv")
POLICY_PATH = os.path.join(ROOT_DIR, "data", "outputs", "tables", "policy_recommendations_rf.csv")

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

# 4. CSS 스타일 (원본 그대로 유지)
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
}

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
    box-shadow: 0 20px 60px rgba(102,126,234,.4);
    border-color: #667eea;
}

.card-icon { font-size: 3.5rem; }
.card-title { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
.card-desc { font-size: .95rem; color: #64748b; }

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

# 6. 홈 화면 (카드 HTML 구조 보존)
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
        ("mhvi", "🗺️", "MHVI 지도", "서울시 25개 자치구의 정신건강 취약 지수 시각화"),
        ("gap", "📊", "Gap 분석", "수요-공급 격차 및 인프라 부족 지역 분석"),
        ("ai_diagnosis", "🤖", "AI 사각지대", "공급 대비 과도 위험 지역 및 원인 진단"),
        ("policy_sim", "📈", "정책 시나리오", "구별 맞춤형 정책 처방 및 개선 효과 예측"),
        ("radar", "📈", "구별 비교", "선택한 자치구의 다차원 지표 비교"),
        ("data", "📋", "데이터 테이블", "전체 자치구의 상세 데이터 확인")
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
            if mhvi_df is not None:
                m = charts.draw_mhvi_map(geo_data, mhvi_df)
                st_folium(m, width="100%", height=600, returned_objects=[], key="map_mhvi")
            else:
                st.warning("MHVI 데이터(mhvi_final_result.csv)가 없어 기본 인프라 지도를 표시합니다.")
                m = charts.draw_mhvi_map(geo_data, infra_data)
                st_folium(m, width="100%", height=600, returned_objects=[], key="map_mhvi_infra")

        elif page == 'gap':
            st.markdown("<h1 class='page-title'>📊 수요-공급 격차 분석</h1>", unsafe_allow_html=True)
            if mhvi_df is not None:
                fig = charts.draw_gap_scatter(mhvi_df)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # 데이터 테이블 표시 (선택사항)
                with st.expander("상세 데이터 보기"):
                    st.dataframe(mhvi_df[['district', 'Need_Index', 'Supply_Index', 'Gap_Index', 'Quadrant']])
            else:
                st.warning("MHVI 데이터가 없어 기본 인프라 데이터로 표시합니다.")
                fig = charts.draw_gap_scatter(infra_data)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        elif page == 'ai_diagnosis':
            st.markdown("<h1 class='page-title'>🤖 AI 구조적 사각지대 진단</h1>", unsafe_allow_html=True)
            if os.path.exists(RANK_PATH) and os.path.exists(SHAP_PATH):
                df_rank = pd.read_csv(RANK_PATH)
                df_shap = pd.read_csv(SHAP_PATH)
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.plotly_chart(charts.draw_ai_blindspot_bar(df_rank), use_container_width=True, config={'displayModeBar': False})
                with c2:
                    selected_gu = st.selectbox("진단 구 선택", infra_data['name'].unique())
                    fig_shap = charts.draw_shap_waterfall(df_shap, selected_gu)
                    if fig_shap:
                        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info(f"**{selected_gu}**는 AI 진단 결과 **구조적 사각지대(이상 징후)가 발견되지 않은 정상 범주** 지역입니다. 특이 사항이 없어 별도의 세부 원인 분석 데이터를 제공하지 않습니다.")
            else:
                st.warning("AI 데이터 없음")

        elif page == 'policy_sim':
            st.markdown("<h1 class='page-title'>📈 정책 시뮬레이션</h1>", unsafe_allow_html=True)
            if os.path.exists(POLICY_PATH):
                df_poly = pd.read_csv(POLICY_PATH)
                selected_gu = st.selectbox("구 선택", df_poly['district'].unique())
                res = df_poly[df_poly['district'] == selected_gu].iloc[0]
                
                # 정책 변수명 한글 매핑
                policy_map = {
                    "welfare_budget_per_capita": "1인당 복지 예산 증액",
                    "cultural_satisfaction": "문화 환경 만족도 개선",
                    "parks_count": "공원 인프라 확충",
                    "libraries_count": "도서관 시설 확충",
                    "public_sports_facilities_count": "공공 체육 시설 확충",
                    "medical_institutions_count": "의료 기관 접근성 개선",
                    "health_promotion_centers_count": "건강 증진 센터 확충",
                    "elderly_leisure_welfare_facilities_count": "노인 여가 복지 시설 확충",
                    "in_home_elderly_welfare_facilities_count": "재가 노인 복지 시설 확충"
                }
                
                # 1순위 추천 정책 가져오기 (없으면 원본 유지)
                rec_var = res.get('rec1_lever', '')
                policy_name = policy_map.get(rec_var, rec_var if rec_var else "인프라 보완")

                st.success(f"### {selected_gu} 처방")
                st.metric("추천 정책", policy_name)
                
                # 상세 설명 (선택적)
                st.info(f"💡 **{policy_name}**을(를) 우선적으로 고려하면 자살률 감소 효과가 가장 클 것으로 예측됩니다.")
            else:
                st.warning("시나리오 데이터 없음")

        elif page == 'radar':
            st.markdown("<h1 class='page-title'>📈 자치구별 세부 지표 비교</h1>", unsafe_allow_html=True)
            if radar_df is not None:
                selected_gu = st.selectbox("자치구 선택", radar_df['district'].unique())
                fig = charts.draw_radar_chart(radar_df, selected_gu)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.error("세부 지표 데이터를 불러올 수 없습니다.")

        elif page == 'data':
            st.markdown("<h1 class='page-title'>📋 구별 상세 데이터</h1>", unsafe_allow_html=True)
            
            if radar_df is not None and mhvi_df is not None:
                # 데이터 병합 (지수 + 원본 데이터)
                master_df = pd.merge(mhvi_df, radar_df, on='district', how='outer')
                
                # 컬럼 순서 재배치 (중요 지표 먼저)
                main_cols = ['district', 'Quadrant', 'Need_Index', 'Supply_Index', 'Gap_Index']
                other_cols = [c for c in master_df.columns if c not in main_cols]
                master_df = master_df[main_cols + other_cols]
                
                # 컬럼명 한글화 (가독성 향상)
                col_rename = {
                    'district': '자치구',
                    'Quadrant': '유형(4사분면)',
                    'Need_Index': '취약 지수(Need)',
                    'Supply_Index': '인프라 지수(Supply)',
                    'Gap_Index': '격차 지수(Gap)',
                    'suicide_rate': '자살률',
                    'depression_experience_rate': '우울감 경험률',
                    'perceived_stress_rate': '스트레스 인지율',
                    'single_households': '1인 가구 수',
                    'welfare_budget_per_capita': '1인당 복지예산',
                    'libraries_count': '도서관 수',
                    'parks_count': '공원 수',
                    'medical_institutions_count': '의료기관 수'
                }
                display_df = master_df.rename(columns=col_rename)

                st.info("💡 **MHVI 지수**와 **세부 원본 데이터**를 통합한 전체 데이터입니다.")
                st.dataframe(display_df, use_container_width=True, height=600)
                
                # CSV 다운로드 버튼
                csv = display_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 전체 데이터 다운로드 (CSV)",
                    data=csv,
                    file_name="seoul_mental_health_full_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("상세 데이터를 불러올 수 없어 기본 데이터만 표시합니다.")
                st.dataframe(infra_data, use_container_width=True)