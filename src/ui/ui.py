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
        ("mhvi", "🗺️", "정신건강 취약 지도", "서울시 25개 자치구의 정신건강 취약 지수 시각화"),
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
            st.markdown("<h1 class='page-title'>🗺️ 정신건강 취약 지도</h1>", unsafe_allow_html=True)
            st.info("💡 **지도의 각 구를 클릭**하면 해당 지역의 상세 분석 페이지로 이동합니다.")
            
            target_df = mhvi_df if mhvi_df is not None else infra_data
            
            if mhvi_df is None:
                st.warning("MHVI 데이터(mhvi_final_result.csv)가 없어 기본 인프라 지도를 표시합니다.")
                
            m = charts.draw_mhvi_map(geo_data, target_df)
            
            # 클릭 이벤트 감지를 위해 returned_objects 설정
            map_output = st_folium(m, width="100%", height=600, returned_objects=["last_object_clicked"], key="map_mhvi")

            # 디버깅: 데이터 로드 상태 및 클릭 정보 확인
            if radar_df is not None:
                st.sidebar.success("✅ 상세 데이터 로드 완료")
            else:
                st.sidebar.error("❌ 상세 데이터 로드 실패")

            # --- 좌표 기반 구 찾기 함수 (GeoJSON 파싱) ---
            def is_point_in_polygon(x, y, poly):
                """Ray-casting algorithm to check if point (x,y) is in polygon"""
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
                point_x, point_y = lng, lat # GeoJSON uses (lng, lat)
                
                for feature in geo_data['features']:
                    gu_name = feature['properties'].get('SIG_KOR_NM')
                    geometry = feature['geometry']
                    geom_type = geometry['type']
                    coords = geometry['coordinates']
                    
                    if geom_type == 'Polygon':
                        # Polygon: [ [ring1], [ring2], ... ] - 첫 번째 링이 외곽선
                        if is_point_in_polygon(point_x, point_y, coords[0]):
                            return gu_name
                    elif geom_type == 'MultiPolygon':
                        # MultiPolygon: [ [[ring]], [[ring]], ... ]
                        for poly in coords:
                            if is_point_in_polygon(point_x, point_y, poly[0]):
                                return gu_name
                return None
            
            # ---------------------------------------------

            if map_output['last_object_clicked']:
               clicked_lat = map_output['last_object_clicked'].get('lat')
               clicked_lng = map_output['last_object_clicked'].get('lng')
               
               # 1. 속성 정보로 시도
               properties = map_output['last_object_clicked'].get('properties', {})
               clicked_gu = properties.get('SIG_KOR_NM') or properties.get('name') or properties.get('SIG_ENG_NM')
               
               # 2. 좌표로 시도 (속성 정보 없을 경우)
               if not clicked_gu and clicked_lat and clicked_lng:
                   clicked_gu = find_gu_by_coord(geo_data, clicked_lat, clicked_lng)

               if clicked_gu:
                   st.success(f"'{clicked_gu}' 선택됨! 상세 페이지로 이동합니다.")
                   st.session_state['selected_gu_from_map'] = clicked_gu
                   st.session_state.current_page = 'radar'
                   st.query_params['page'] = 'radar'
                   st.rerun()
               else:
                   st.warning("선택한 위치에서 지역구 정보를 찾을 수 없습니다. 정확한 구역을 클릭해주세요.")

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
                
                # 변수명 한글 매핑 (원인 지표용)
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

                st.success(f"### {selected_gu} 맞춤형 정책 제언")
                
                # 3가지 정책 방향 표시
                for i in range(1, 4):
                    factor_key = f'top{i}_factor'
                    policy_key = f'policy_direction_{i}'
                    
                    if factor_key in res and policy_key in res:
                        factor_raw = res[factor_key]
                        factor_name = factor_map.get(factor_raw, factor_raw)
                        policy_desc = res[policy_key]
                        
                        with st.expander(f"**순위 {i}: {factor_name} 기반 정책**", expanded=(i==1)):
                            st.write(f"🎯 **주요 타겟 지표:** {factor_name}")
                            st.info(f"💡 **정책 제언:**\n\n{policy_desc}")
            else:
                st.warning("정책 제언 데이터(need_policy_recommendation_by_district.csv)를 찾을 수 없습니다.")

        elif page == 'radar':
            st.markdown("<h1 class='page-title'>📈 자치구별 세부 지표 비교</h1>", unsafe_allow_html=True)
            if radar_df is not None:
                gu_list = radar_df['district'].unique().tolist()
                default_index = 0
                
                # 지도에서 클릭해서 넘어온 경우 해당 구 선택
                if 'selected_gu_from_map' in st.session_state and st.session_state.selected_gu_from_map in gu_list:
                    default_index = gu_list.index(st.session_state.selected_gu_from_map)
                    # 한 번 사용 후 초기화 (선택사항, 여기선 유지)
                    # del st.session_state['selected_gu_from_map'] 
                
                selected_gu = st.selectbox("자치구 선택", gu_list, index=default_index)
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