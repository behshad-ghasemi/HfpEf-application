import streamlit_authenticator as stauth
import yaml
import copy
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
import io
import streamlit as st


url = "https://raw.githubusercontent.com/behshad-ghasemi/Hfpef-application-secrets/main/inference_objects.pkl"
headers = {"Authorization": f"Bearer {st.secrets['github']['token']}"}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    st.error("❌ Could not download model file from private GitHub repository.")
    st.stop()
import streamlit_authenticator as stauth
import yaml
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
import io
import streamlit as st


GITHUB_TOKEN = st.secrets["github"]["token"]


model_objects = pickle.load(io.BytesIO(response.content))
preprocessor = model_objects['preprocessor']
multi_reg = model_objects['multi_reg']
model_hf = model_objects['model_hf']
bio_features_scaled = model_objects['bio_features_scaled']
mediator_features_scaled = model_objects['mediator_features_scaled']
NUM_FEATURES = model_objects['NUM_FEATURES']
causal_ranges = model_objects['causal_ranges']


st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


secrets_url = "https://raw.githubusercontent.com/behshad-ghasemi/Hfpef-application-secrets/main/secrets.toml"
secrets_headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

secrets_response = requests.get(secrets_url, headers=secrets_headers)

if secrets_response.status_code != 200:
    st.error("❌ Could not download secrets.toml from private GitHub repository.")
    st.stop()

secrets_text = secrets_response.text
config = yaml.safe_load(secrets_text)

credentials = {
    "usernames": dict(config["credentials"]["usernames"])
}

cookie = dict(config["cookie"])

authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"]
)

try:
    authenticator.login()
except:
    authenticator.login("main")

if st.session_state.get("authentication_status") == False:
    st.error("❌ Username or password incorrect.")
    st.stop()

if st.session_state.get("authentication_status") is None:
    st.warning("🔐 Please enter your username and password.")
    st.stop()

name = st.session_state["name"]


col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    try:
        st.image("logo.png", width=150)
    except:
        st.write("🏥")

with col2:
    st.title("🏥 HFpEF Prediction System")
    st.markdown("""
        <div style='text-align: center;'>
            Heart Failure with Preserved Ejection Fraction Risk Assessment
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; font-style: italic;'>
            Developed by Behshad Ghaseminezhadabdolmaleki (Beth Gasemin)
        </div>
    """, unsafe_allow_html=True)


with col3:
    st.write(f"👤 **{name}**")
    authenticator.logout("Logout", "main")

st.markdown("---")




with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Stage 1 Model", model_objects['best_stage1_model_name'])
    st.metric("Stage 1 R²", f"{model_objects['best_stage1_r2']:.3f}")
    st.metric("Stage 2 Model", model_objects['best_stage2_model_name'])
    st.metric("Stage 2 AUC", f"{model_objects['best_stage2_auc']:.3f}")

    st.markdown("---")
    st.info("The model estimates HFpEF risk using biomarkers.")



st.header("🔬 Biomarker Information")

bio_features_original = [f.replace("num__", "") for f in bio_features_scaled]



bio_units = {
    "PINK1": "ng/mL",
    "Galectin3": "ng/mL",
    "mir-7110": "unitless",
    "DHEAs": "ng/mL",
    "SHBG": "nmol/L",
    "mir125": "unitless"
}

bio_stats = {
    "PINK1": {"mean": 5.2, "std": 1.3},
    "Galectin3": {"mean": 12.5, "std": 3.2},
    "mir-7110": {"mean": 0.8, "std": 0.2},
    "DHEAs": {"mean": 3.5, "std": 1.1},
    "SHBG": {"mean": 50, "std": 12},
    "mir125": {"mean": 1.2, "std": 0.4}
}

with st.form("biomarker_form"):
    st.markdown("Enter patient biomarker values (only zero or positive numbers).")

    cols = st.columns(3)
    user_data = {}
    alerts = []

    for i, biomarker in enumerate(bio_features_original):
        col = cols[i % 3]
        with col:
            value = st.number_input(
                f"{biomarker} ({bio_units.get(biomarker, '')})",
                value=0.0,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key=f"bio_{i}"
            )
            user_data[biomarker] = value

            mean = bio_stats[biomarker]["mean"]
            std = bio_stats[biomarker]["std"]
            if value < mean - 2*std or value > mean + 2*std:
                alerts.append(f"⚠️ {biomarker} value ({value}) is outside the range , Insert the correct value please!")

    submitted = st.form_submit_button("🔍 Predict HFpEF", use_container_width=True)

if submitted:
    if alerts:
        for alert in alerts:
            st.error(alert)
        st.stop() 
    else:
        st.success("All inputs within normal ranges ✅")
    with st.spinner("Calculating..."):

        
        df_bio = pd.DataFrame([user_data], columns=bio_features_original)

        
        full_input = pd.DataFrame(columns=NUM_FEATURES)
        for col in NUM_FEATURES:
            if col in df_bio.columns:
                full_input[col] = df_bio[col]
            else:
                full_input[col] = np.nan

        
        scaled = preprocessor.transform(full_input)
        df_scaled = pd.DataFrame(scaled, columns=preprocessor.get_feature_names_out())
        bio_scaled_input = df_scaled[bio_features_scaled]

        # Stage 1 prediction (mediators)
        predicted_scaled_mediators = multi_reg.predict(bio_scaled_input)
        df_pred_scaled = pd.DataFrame(predicted_scaled_mediators, columns=mediator_features_scaled)

        # Convert back to actual values
        num_scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        zeros_full = np.zeros((1, len(NUM_FEATURES)))

        med_raw = [m.replace("num__", "") for m in mediator_features_scaled]

        for i, f in enumerate(NUM_FEATURES):
            if f in med_raw:
                idx = med_raw.index(f)
                zeros_full[0, i] = predicted_scaled_mediators[0, idx]

        actual_vals = num_scaler.inverse_transform(zeros_full)

        mediator_actual = {
            f: actual_vals[0, i] for i, f in enumerate(NUM_FEATURES) if f in med_raw
        }
        df_mediators_actual = pd.DataFrame([mediator_actual]).T
        df_mediators_actual.columns = ["Predicted Value"]

        # Stage 2: HF probability
        hf_proba = model_hf.predict_proba(df_pred_scaled.values)[0, 1]

        st.markdown("---")
        st.header("📊 Prediction Result")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            if hf_proba >= 0.7:
                st.error(f"### 🔴 {hf_proba:.1%}\n**High Risk**")
            elif hf_proba >= 0.5:
                st.warning(f"### 🟠 {hf_proba:.1%}\n**Medium Risk**")
            else:
                st.success(f"### 🟢 {hf_proba:.1%}\n**Low Risk**")

        st.markdown("### 🧪 Predicted Mediator Values")
        st.dataframe(df_mediators_actual, use_container_width=True)

        st.markdown("### 💊 Recommendation")

        if hf_proba >= 0.5:
            st.warning("⚠️ Additional cardiology review recommended.")
        else:
            st.success("Normal condition.")

        # Download report
        st.markdown("---")
        report = f"""
HFpEF Probability Report
=========================

Date: {pd.Timestamp.now()}

Probability: {hf_proba:.1%}
Risk Level: {'High' if hf_proba>=0.7 else 'Medium' if hf_proba>=0.5 else 'Low'}

Biomarker input:
{df_bio.to_string()}

Predicted mediators:
{df_mediators_actual.to_string()}
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="HFpEF_Report.txt",
            mime="text/plain"
        )



st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, .stApp {
        background: #2f3f57 !important; 
        color: #ECF0F1 !important;      
    }

    /* استایل اصلی محتوا */
    .main .block-container {
        color: #1e3a8a !important; /* متن داخل کادرها */
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
        color: #1e3a8a; /* متن متریک‌ها روی پس‌زمینه سفید */
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #64748b;
    }
    .stMetric {
        background: #FFFFFF; /* کادر متریک سفید */
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
        background: #FFFFFF; /* کادر فرم سفید */
        color: #1e3a8a;      /* متن داخل فرم */
    }
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    .stNumberInput label {
        font-weight: 600;
        color: #1e3a8a; /* متن لیبل‌ها روی کادر سفید */
        font-size: 0.95rem;
    }

    /* جدول */
    .dataframe {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background: #FFFFFF; /* پس‌زمینه جدول سفید */
        color: #1e3a8a;      /* متن جدول */
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
        background: #001F3F; /* سایدبار لوکس تیره */
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
        border-left: 5px solid #3d2687;
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
        background: #FFFFFF; /* فرم سفید */
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        color: #1e3a8a; /* متن داخل فرم */
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
        background: #FFFFFF; /* کارت‌های وسط سفید */
        border-radius: 12px;
        padding: 1rem;
        color: #1e3a8a;
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


