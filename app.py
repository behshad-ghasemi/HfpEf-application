import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
import io

# ================== Load private model from GitHub using secret token ==================

url = "https://raw.githubusercontent.com/behshad-ghasemi/Hfpef-application-secrets/main/inference_objects.pkl"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {st.secrets['github']['token']}"
}

response = requests.get(url, headers=headers)
model = pickle.load(io.BytesIO(response.content))

# ================== Streamlit Page Config ==================
st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== FIXED: Load secrets correctly ==================
config = st.secrets    # <---- این مهم ترین اصلاح است

# ================== Authentication ==================
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    expiry_days=config['cookie']['expiry_days']
)

try:
    authenticator.login()
except TypeError:
    authenticator.login('main')

if st.session_state.get("authentication_status") == False:
    st.error('username or password is not correct!')
    st.stop()

if st.session_state.get("authentication_status") == None:
    st.warning('Insert your user-name and password, please.')
    st.stop()


name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

# ================== Main App ==================
if authentication_status:

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        try:
            logo = Image.open("logo.png")
            st.image(logo, width=150)
        except:
            st.write("🏥")

    with col2:
        st.title("🏥 HFpEF Prediction System")
        st.markdown("**Heart Failure with Preserved Ejection Fraction Risk Assessment**")
        st.markdown("*This application is developed by Behshad Ghaseminezhadabdolmaleki (Beth Gasemin)*")

    with col3:
        st.write(f"👤 **{name}**")
        authenticator.logout('Logout', 'main')

    st.markdown("---")

    # ---------- Load models ----------
    @st.cache_resource
    def load_models():
        try:
            with open('inference_objects.pkl', 'rb') as f:
                objects = pickle.load(f)
            return objects
        except FileNotFoundError:
            st.error('Not founded')
            st.stop()

    inference_objects = load_models()

    preprocessor = inference_objects['preprocessor']
    multi_reg = inference_objects['multi_reg']
    model_hf = inference_objects['model_hf']
    bio_features_scaled = inference_objects['bio_features_scaled']
    mediator_features_scaled = inference_objects['mediator_features_scaled']
    NUM_FEATURES = inference_objects['NUM_FEATURES']
    causal_ranges = inference_objects['causal_ranges']

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Stage 1 Model", inference_objects['best_stage1_model_name'])
        st.metric("Stage 1 R²", f"{inference_objects['best_stage1_r2']:.3f}")
        st.metric("Stage 2 Model", inference_objects['best_stage2_model_name'])
        st.metric("Stage 2 AUC", f"{inference_objects['best_stage2_auc']:.3f}")
        st.markdown("---")
        st.info('This model finds the probability of HFpEF based on Biomarkers')

    # ---------- Biomarker Form ----------
    st.header('"🔬Biomarker information:"')

    bio_features_original = [f.replace('num__', '') for f in bio_features_scaled]

    with st.form("biomarker_form"):
        st.markdown('Insert biomarker values of your patient, please.')

        cols = st.columns(3)
        user_data = {}

        for idx, biomarker in enumerate(bio_features_original):
            col_idx = idx % 3
            with cols[col_idx]:
                value = st.number_input(
                    f"**{biomarker}**",
                    value=None,
                    format="%.2f",
                    help=f"Insert {biomarker} value",
                    key=f"bio_{idx}"
                )
                user_data[biomarker] = value if value is not None else np.nan

        submitted = st.form_submit_button("🔍 Forcasting HFpEF ", use_container_width=True)

    # ---------- Prediction ----------
    if submitted:
        with st.spinner("Calculating..."):

            bio_df_input = pd.DataFrame([user_data], columns=bio_features_original)

            full_input = pd.DataFrame(columns=NUM_FEATURES)
            for b_orig in bio_features_original:
                if b_orig in NUM_FEATURES:
                    full_input[b_orig] = [user_data[b_orig]]

            for col in NUM_FEATURES:
                if col not in full_input.columns:
                    full_input[col] = np.nan

            bio_scaled = preprocessor.transform(full_input)
            bio_scaled_df = pd.DataFrame(bio_scaled, columns=preprocessor.get_feature_names_out())
            bio_input_scaled = bio_scaled_df[bio_features_scaled]

            scaled_mediators = multi_reg.predict(bio_input_scaled)
            scaled_mediators_df = pd.DataFrame(scaled_mediators, columns=mediator_features_scaled)

            num_scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
            full_scaled = np.zeros((1, len(NUM_FEATURES)))
            mediator_raw_names = [m.replace('num__', '') for m in mediator_features_scaled]

            for i, feature in enumerate(NUM_FEATURES):
                if feature in mediator_raw_names:
                    idx = mediator_raw_names.index(feature)
                    full_scaled[0, i] = scaled_mediators[0, idx]

            actual_values = num_scaler.inverse_transform(full_scaled)
            mediator_actual = {}

            for i, feature in enumerate(NUM_FEATURES):
                if feature in mediator_raw_names:
                    mediator_actual[feature] = actual_values[0, i]

            actual_mediators_df = pd.DataFrame([mediator_actual])

            hf_proba = model_hf.predict_proba(scaled_mediators_df.values)
            p_hf = float(hf_proba[0, 1])

            st.markdown("---")
            st.header("📊  Result: ")

            col1, col2, col3 = st.columns([2, 1, 2])

            with col2:
                if p_hf >= 0.7:
                    st.error(f"### 🔴 {p_hf:.1%}")
                    st.error("**High Risk**")
                elif p_hf >= 0.5:
                    st.warning(f"### 🟠 {p_hf:.1%}")
                    st.warning("**Medium Risk**")
                else:
                    st.success(f"### 🟢 {p_hf:.1%}")
                    st.success("**Low Risk**")

            st.markdown("### 📈 LV mass / E'e avg / LAD forcasting based on biomarkers")

            mediator_df_display = actual_mediators_df.T
            mediator_df_display.columns = ['Forcasting quantity:']
            mediator_df_display.index.name = 'Mediator:'

            st.dataframe(
                mediator_df_display.style.format("{:.2f}"),
                use_container_width=True
            )

            st.markdown("### 💊  recommendation")

            if p_hf >= 0.5:
                st.warning("""
                **⚠️ PLEASE :**
               - Complete assessment 
               - Cardiologist review needed!
                """)
            else:
                st.success("""
                **✅ Normal situation**
                """)

            st.markdown("---")

            report = f"""
              HFpEF Probability Report
            =====================
            
            Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            User: {name}
            
            Probability HFpEF Risk: {p_hf:.1%}
            Situation: {'High Risk' if p_hf >= 0.7 else 'Medium Risk' if p_hf >= 0.5 else 'Low Risk'}
            
            Biomarkers input
            {bio_df_input.to_string()}
            
            Mediator Forcasting values:
            {actual_mediators_df.to_string()}
            """

            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"HFpEF_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# ================== تنظیمات صفحه ==================
st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== استایل CSS پیشرفته ==================
st.markdown("""
<style>
    /* پس‌زمینه گرادیانت حرفه‌ای */
    .stApp {
        background: #1e293b;
    }
    
    /* استایل اصلی محتوا */
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
    }
    
    /* هدرها */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    h2 {
        color: #2563eb;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #1e40af;
        font-weight: 600;
        font-size: 1.4rem !important;
    }
    
    /* دکمه‌ها */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* کادرهای متریک */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #64748b;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    /* فرم ورودی */
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    
    /* لیبل‌های ورودی */
    .stNumberInput label {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }
    
    /* جدول */
    .dataframe {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        font-weight: 700;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e0f2fe;
        transition: all 0.2s ease;
    }
    
    .dataframe tbody td {
        padding: 0.75rem;
        font-size: 0.95rem;
    }
    
    /* سایدبار */
    [data-testid="stSidebar"] {
        background: #764ba2;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white !important;
        border-bottom-color: rgba(255, 255, 255, 0.3);
    }
    
    /* کادرهای پیام */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        padding: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
    }
    
    /* جداکننده */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    }
    
    /* فرم */
    [data-testid="stForm"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    /* دکمه دانلود */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
    
    /* کارت‌های نتیجه */
    div[data-testid="column"] > div {
        background: white;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* انیمیشن لودینگ */
    .stSpinner > div {
        border-top-color: #3b82f6!important;
    }
    
    /* هدر لوگو */
    [data-testid="column"]:first-child img {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="column"]:first-child img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)
