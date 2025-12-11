import streamlit_authenticator as stauth
import yaml
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
import io
import streamlit as st

# ================== Model Loading ==================
url = "https://raw.githubusercontent.com/behshad-ghasemi/Hfpef-application-secrets/main/inference_objects.pkl"
headers = {"Authorization": f"Bearer {st.secrets['github']['token']}"}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    st.error("❌ Could not download model file from private GitHub repository.")
    st.stop()

GITHUB_TOKEN = st.secrets["github"]["token"]

model_objects = pickle.load(io.BytesIO(response.content))
preprocessor = model_objects['preprocessor']
multi_reg = model_objects['multi_reg']
model_hf = model_objects['model_hf']
bio_features_scaled = model_objects['bio_features_scaled']
mediator_features_scaled = model_objects['mediator_features_scaled']
NUM_FEATURES = model_objects['NUM_FEATURES']
CAT_FEATURES = model_objects['CAT_FEATURES']
causal_ranges = model_objects['causal_ranges']

# ================== Page Config ==================
st.set_page_config(
    page_title="HFpEF Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Authentication ==================
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

# ================== Header ==================
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

# ================== Sidebar ==================
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Stage 1 Model", model_objects['best_stage1_model_name'])
    st.metric("Stage 1 R²", f"{model_objects['best_stage1_r2']:.3f}")
    st.metric("Stage 2 Model", model_objects['best_stage2_model_name'])
    st.metric("Stage 2 AUC", f"{model_objects['best_stage2_auc']:.3f}")
    st.markdown("---")
    st.info("The model estimates HFpEF risk using biomarkers.")

# ================== Biomarker Configuration ==================
st.header("🔬 Biomarker Information")

# Extract original biomarker names from scaled names
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
    "PINK1": {"mean": 4.147694837, "std": 5.366805867},
    "Galectin3": {"mean": 6.642251141, "std": 5.376785606},
    "mir-7110": {"mean": 87.56797769, "std": 496.8851913},
    "DHEAs": {"mean": 2049.97, "std": 2553.86406},
    "SHBG": {"mean": 2144.46, "std": 2403.79653},
    "mir125": {"mean": 1.861721004, "std": 2.610673131}
}

# ================== Input Form ==================
with st.form("biomarker_form"):
    st.markdown("Enter patient biomarker values (only zero or positive numbers).")

    cols = st.columns(3)
    user_data = {}

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

    submitted = st.form_submit_button("🔍 Predict HFpEF", use_container_width=True)

# ================== Prediction Logic ==================
if submitted:
    # Validation
    std_values = np.array([bio_stats[b]["std"] for b in bio_features_original])
    mean_values = np.array([bio_stats[b]["mean"] for b in bio_features_original])
    
    cv_values = np.where(mean_values > 0, std_values / mean_values, 0)
    median_cv = np.median(cv_values[np.isfinite(cv_values)])
    
    if median_cv <= 0 or not np.isfinite(median_cv):
        median_cv = 0.5
    
    alerts = []
    widened_biomarkers = []

    for biomarker in bio_features_original:
        mean = bio_stats[biomarker]["mean"]
        std = bio_stats[biomarker]["std"]
        
        cv = std / mean if mean > 0 else 0
        
        if cv > median_cv * 1.5:
            adaptive_factor = min(cv / median_cv, 4.0)
            widened_biomarkers.append(biomarker)
        else:
            adaptive_factor = 1.0
        
        lower = mean - 3.0 * std * adaptive_factor
        upper = mean + 3.0 * std * adaptive_factor
        lower = max(0, lower)

        val = user_data.get(biomarker, np.nan)
        if np.isnan(val):
            alerts.append(f"⚠️ {biomarker} value is missing.")
            continue

        if val < lower or val > upper:
            alerts.append(
                f"⚠️ {biomarker} value ({val:.2f}) is outside the acceptable range "
                f"({lower:.2f} - {upper:.2f}). "
                f"{'[Widened range due to high variability]' if adaptive_factor > 1 else ''}"
            )

    vals = list(user_data.values())
    vals_for_uniqueness = [v if not pd.isna(v) else "__NA__" for v in vals]
    
    if len(set(vals_for_uniqueness)) == 1:
        st.error("⚠️ All biomarker values are identical. Please enter realistic patient data.")
        st.stop()

    numeric_vals = np.array([v for v in vals if (v is not None and not pd.isna(v))], dtype=float)
    if numeric_vals.size == 0:
        st.error("⚠️ No numeric biomarker values provided.")
        st.stop()

    if np.std(numeric_vals) < 1e-3:
        st.error("⚠️ Biomarker values have too little variation. Please enter realistic patient data.")
        st.stop()

    if alerts:
        for alert in alerts:
            st.error(alert)
        st.stop()

    if widened_biomarkers:
        st.info(f"ℹ️ **Adaptive Validation Applied**: The following biomarkers have high population variability (CV > {median_cv*1.5:.2f}), so their acceptable ranges were widened accordingly:\n\n" + 
                "\n".join([f"• **{b}** (std={bio_stats[b]['std']:.1f})" for b in widened_biomarkers]))

    st.success("✅ All inputs within acceptable ranges")

    # ============ PYTHON-MATCHING PREDICTION PIPELINE ============
    
    with st.spinner("Calculating..."):
    
        # -------------------------------------------
        # 1. Stage 1 INPUT: biomarker-only dataframe
        # -------------------------------------------
        # Python uses ONLY biomarkers for Stage 1
        bio_df_input = pd.DataFrame([user_data], columns=bio_features_original)
    
        # We MUST create a dataframe with all numeric features
        # but FILL MEDIATORS WITH 0 (Python used standardized values ≈ 0 at prediction time)
        full_input = pd.DataFrame(columns=NUM_FEATURES)
        for col in NUM_FEATURES:
            if col in bio_features_original:
                full_input[col] = [user_data[col]]  # original biomarker
            else:
                full_input[col] = [0.0]            # mediators = zero (standard normal mean)
    
        # -------------------------------------------
        # 2. Scale full numeric feature row
        # -------------------------------------------
        scaled_full = preprocessor.transform(full_input)
        scaled_full_df = pd.DataFrame(scaled_full, columns=preprocessor.get_feature_names_out())
    
        # -------------------------------------------
        # 3. Extract ONLY biomarker-scaled columns
        # -------------------------------------------
        X_stage1 = scaled_full_df[bio_features_scaled]
    
        # -------------------------------------------
        # 4. Stage 1 prediction (scaled mediators)
        # -------------------------------------------
        scaled_mediators = multi_reg.predict(X_stage1)
        scaled_mediators_df = pd.DataFrame(scaled_mediators, columns=mediator_features_scaled)
    
        # -------------------------------------------
        # 5. Convert scaled mediators → actual mediators
        # -------------------------------------------
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    
        # create a dummy zero vector
        scaled_vector = np.zeros((1, len(NUM_FEATURES)))
    
        # fill mediator positions ONLY
        for m_scaled in mediator_features_scaled:
            raw_name = m_scaled.replace("num__", "")
            idx = NUM_FEATURES.index(raw_name)
            value_scaled = scaled_mediators_df[m_scaled].values[0]
            scaled_vector[0, idx] = value_scaled
    
        # inverse-transform
        actual_vector = scaler.inverse_transform(scaled_vector)
    
        # extract mediator actual values into df
        mediator_actual = {}
        for m_scaled in mediator_features_scaled:
            raw_name = m_scaled.replace("num__", "")
            idx = NUM_FEATURES.index(raw_name)
            mediator_actual[raw_name] = actual_vector[0, idx]
    
        actual_mediators_df = pd.DataFrame([mediator_actual])
    
        # -------------------------------------------
        # 6. Stage 2 prediction (USES SCALED MEDIATORS)
        # -------------------------------------------
        hf_proba = model_hf.predict_proba(scaled_mediators_df.values)[0, 1]
        
        # ============ END OF CORRECTED LOGIC ============

        # Display results
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

        st.markdown("### 🧪 Predicted Mediator Values (ACTUAL)")
        display_df = actual_mediators_df.T
        display_df.columns = ["Predicted Value"]
        st.dataframe(display_df, use_container_width=True)

        # Recommendation
        st.markdown("### 💊 Recommendation")
        if hf_proba >= 0.5:
            st.warning("⚠️ Additional cardiology review recommended.")
        else:
            st.success("✅ Normal condition - Continue regular monitoring.")

        # Download report
        st.markdown("---")
        report = f"""
HFpEF Probability Report
=========================

Date: {pd.Timestamp.now()}
Patient: {name}

Probability: {hf_proba:.1%}
Risk Level: {'High' if hf_proba>=0.7 else 'Medium' if hf_proba>=0.5 else 'Low'}

Biomarker Input (Original Values):
"""
        for bio in bio_features_original:
            report += f"   • {bio:50s}: {user_data[bio]:10.2f}\n"
        
        report += f"""
Predicted Mediators (ACTUAL VALUES):
"""
        for col in actual_mediators_df.columns:
            val = actual_mediators_df[col].values[0]
            report += f"   • {col:50s}: {val:10.2f}\n"
        
        report += f"""
Model Information:
- Stage 1: {model_objects['best_stage1_model_name']} (R² = {model_objects['best_stage1_r2']:.3f})
- Stage 2: {model_objects['best_stage2_model_name']} (AUC = {model_objects['best_stage2_auc']:.3f})

Validation Notes:
- Adaptive thresholds were applied for biomarkers with high variability
- Biomarkers with widened ranges: {', '.join(widened_biomarkers) if widened_biomarkers else 'None'}
"""
        st.download_button(
            "📥 Download Report",
            report,
            file_name=f"HFpEF_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


# ================== Styling ==================
st.markdown("""
<style>
    html, body, .stApp {
        background: #2f3f57 !important; 
        color: #ECF0F1 !important;      
    }

    .main .block-container {
        color: #1e3a8a !important;
    }

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
        background: #FFFFFF;
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

    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: #FFFFFF;
        color: #1e3a8a;
    }
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    .stNumberInput label {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.95rem;
    }

    .dataframe {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background: #FFFFFF;
        color: #1e3a8a;
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

    [data-testid="stSidebar"] {
        background: #001F3F;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    [data-testid="stSidebar"] h2 {
        color: white !important;
        border-bottom-color: rgba(255, 255, 255, 0.3);
    }

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

    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    }

    [data-testid="stForm"] {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        color: #1e3a8a;
    }

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

    div[data-testid="column"] > div {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        color: #1e3a8a;
    }

    .stSpinner > div {
        border-top-color: #3b82f6!important;
    }

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
