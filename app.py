import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Traffic Accident Severity Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS / AESTHETICS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19;
        color: #ffffff;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Main Hero Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 20px;
        padding: 2.25rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        line-height: 1.25;
    }

    .main-desc {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0;
        max-width: 780px;
    }

    /* Container Card Styling (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(17, 24, 39, 0.75) !important;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.35);
        padding: 1.6rem 1.75rem !important;
        margin-bottom: 1.5rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 10px 30px -2px rgba(99, 102, 241, 0.15);
    }

    /* Crisp White Typography for Section Titles & Labels */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 1.35rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        letter-spacing: -0.01em;
    }

    .feature-group-label {
        font-size: 0.98rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-top: 1.35rem;
        margin-bottom: 0.75rem;
        letter-spacing: 0.01em;
    }

    /* All Input Field Labels & Form Text in High-Contrast White */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        color: #ffffff !important;
    }

    /* Radio and Checkbox Label Text */
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] [role="radiogroup"] p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] label span {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Help tooltips & icons */
    [data-testid="stTooltipHoverTarget"] svg {
        fill: #94a3b8 !important;
    }

    /* Model Info Pill */
    .model-info-box {
        background: rgba(99, 102, 241, 0.15);
        border-left: 3px solid #818cf8;
        padding: 0.85rem 1.15rem;
        border-radius: 0 12px 12px 0;
        font-size: 0.95rem;
        color: #f8fafc !important;
        line-height: 1.5;
        height: 100%;
        display: flex;
        align-items: center;
    }

    .model-info-box strong {
        color: #ffffff !important;
    }

    /* Primary Action Button */
    .stButton {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .stButton>button {
        width: 100%;
        height: 3.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #4f46e5 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.015em;
        border: none;
        border-radius: 14px;
        box-shadow: 0 6px 20px -2px rgba(79, 70, 229, 0.45);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #4338ca 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -2px rgba(79, 70, 229, 0.6);
        color: #ffffff !important;
    }

    .stButton>button:active {
        transform: translateY(0);
    }

    /* Severity Result Display */
    .severity-result-card {
        border-radius: 18px;
        padding: 2rem 2.25rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-width: 1.5px;
        border-style: solid;
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .sev-1 {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.08) 100%);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.2);
    }
    .sev-2 {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.08) 100%);
        border-color: rgba(245, 158, 11, 0.5);
        box-shadow: 0 10px 25px -5px rgba(245, 158, 11, 0.2);
    }
    .sev-3 {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.2) 0%, rgba(234, 88, 12, 0.08) 100%);
        border-color: rgba(249, 115, 22, 0.5);
        box-shadow: 0 10px 25px -5px rgba(249, 115, 22, 0.2);
    }
    .sev-4 {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.08) 100%);
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.2);
    }

    .severity-heading {
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        line-height: 1.2;
    }
    .sev-1 .severity-heading { color: #34d399 !important; }
    .sev-2 .severity-heading { color: #fbbf24 !important; }
    .sev-3 .severity-heading { color: #fb923c !important; }
    .sev-4 .severity-heading { color: #f87171 !important; }

    .severity-explanation {
        font-size: 1.15rem;
        color: #ffffff !important;
        font-weight: 500;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# RESOURCE CACHING AND MODEL LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource(show_spinner="Loading trained machine learning models and scaler...")
def load_artifacts():
    errors = []
    rf_model, xgb_model, scaler = None, None, None

    if os.path.exists(RF_PATH):
        try:
            rf_model = joblib.load(RF_PATH)
        except Exception as e:
            errors.append(f"Failed to load Random Forest model (rf_model.pkl): {str(e)}")
    else:
        errors.append("rf_model.pkl not found in current directory.")

    if os.path.exists(XGB_PATH):
        try:
            xgb_model = joblib.load(XGB_PATH)
        except Exception as e:
            errors.append(f"Failed to load XGBoost model (xgb_model.pkl): {str(e)}")
    else:
        errors.append("xgb_model.pkl not found in current directory.")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            errors.append(f"Failed to load Scaler (scaler.pkl): {str(e)}")
    else:
        errors.append("scaler.pkl not found in current directory.")

    return rf_model, xgb_model, scaler, errors

rf_model, xgb_model, scaler, load_errors = load_artifacts()

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="main-title">Traffic Accident Severity Prediction</div>
    <p class="main-desc">Predict the severity of traffic accidents using machine learning models trained on environmental and road feature data.</p>
</div>
""", unsafe_allow_html=True)

if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    st.stop()

# ==========================================
# MODEL SELECTION
# ==========================================
with st.container(border=True):
    col_model, col_info = st.columns([1, 2], gap="large")
    with col_model:
        model_choice = st.radio(
            "Choose Prediction Model:",
            options=["Random Forest", "XGBoost"],
            horizontal=True,
            help="Select which trained ensemble model to use for evaluating severity."
        )
    with col_info:
        if model_choice == "Random Forest":
            st.markdown('<div class="model-info-box">🌲 <span><strong>Random Forest Classifier</strong>: Robust ensemble of decision trees trained with balanced class weights for stability.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="model-info-box">⚡ <span><strong>XGBoost Classifier</strong>: High-performance gradient boosted decision trees optimized for multi-class classification.</span></div>', unsafe_allow_html=True)

# ==========================================
# INPUT FIELDS (TWO-COLUMN LAYOUT)
# ==========================================
DAY_NAME_TO_INT = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

WEATHER_OPTIONS = [
    "Clear", "Cloudy", "Overcast", "Rain", "Light Rain",
    "Heavy Rain", "Snow", "Light Snow", "Fog", "Thunderstorm", "Other"
]

SEVERITY_DESCRIPTIONS = {
    1: "Minor incident. Little to no impact on traffic flow",
    2: "Moderate incident. Some disruption to traffic expected",
    3: "Serious incident. Significant traffic disruption likely",
    4: "Major incident. Severe impact on traffic, expect long delays"
}

col_left, col_right = st.columns(2, gap="large")

# --- LEFT COLUMN: Weather & Environmental Conditions ---
with col_left:
    with st.container(border=True):
        st.markdown('<div class="section-title">🌤️ Environmental & Weather Conditions</div>', unsafe_allow_html=True)
        
        temperature = st.number_input(
            "Temperature in Fahrenheit",
            min_value=-20.0,
            max_value=120.0,
            value=65.0,
            step=1.0,
            help="Ambient temperature recorded at the accident scene (-20°F to 120°F)."
        )
        
        visibility = st.number_input(
            "Visibility in miles",
            min_value=0.0,
            max_value=10.0,
            value=10.0,
            step=0.5,
            help="Atmospheric visibility in miles (0 to 10 miles)."
        )
        
        wind_speed = st.number_input(
            "Wind Speed in mph",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Wind speed recorded in miles per hour (0 to 100 mph)."
        )
        
        weather_condition = st.selectbox(
            "Weather condition",
            options=WEATHER_OPTIONS,
            index=0,
            help="Primary weather phenomenon observed at the time of the collision."
        )
        
        time_of_day = st.selectbox(
            "Time of day",
            options=["Day", "Night"],
            index=0,
            help="Ambient light condition (Day vs. Night based on sunrise/sunset)."
        )

# --- RIGHT COLUMN: Incident & Roadway Features ---
with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-title">🛣️ Incident & Roadway Infrastructure</div>', unsafe_allow_html=True)
        
        distance_affected = st.number_input(
            "Distance affected in miles",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.1,
            help="Length of the roadway segment impacted by the incident (0 to 50 miles)."
        )
        
        hour_of_day = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=12,
            help="Time the accident occurred in 24-hour military format (0 = midnight, 23 = 11 PM)."
        )
        
        day_of_week = st.selectbox(
            "Day of week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
            help="Day of the week the incident occurred."
        )
        
        st.markdown('<div class="feature-group-label">Nearby Roadway Features:</div>', unsafe_allow_html=True)
        
        col_cb1, col_cb2 = st.columns(2, gap="medium")
        with col_cb1:
            junction = st.checkbox("Junction nearby", value=False)
            traffic_signal = st.checkbox("Traffic Signal nearby", value=False)
            crossing = st.checkbox("Crossing nearby", value=False)
        with col_cb2:
            amenity = st.checkbox("Amenity nearby", value=False)
            stop_sign = st.checkbox("Stop sign nearby", value=False)
            railway = st.checkbox("Railway nearby", value=False)

# ==========================================
# PREDICTION LOGIC & PIPELINE
# ==========================================
predict_clicked = st.button("🚨 Predict Accident Severity", use_container_width=True)

if predict_clicked:
    try:
        # Select active model
        active_model = rf_model if model_choice == "Random Forest" else xgb_model
        
        # Determine target feature names from the model
        if hasattr(active_model, "feature_names_in_"):
            model_feature_names = list(active_model.feature_names_in_)
        elif hasattr(rf_model, "feature_names_in_"):
            model_feature_names = list(rf_model.feature_names_in_)
        else:
            raise ValueError("Unable to retrieve feature schema from model.")

        # Determine features that the scaler expects
        scaler_feature_names = getattr(scaler, "feature_names_in_", None)

        # 1. Initialize complete feature dictionary with 0.0
        feature_dict = {col: 0.0 for col in model_feature_names}

        # 2. Assign default values for unprompted features
        if "Month" in feature_dict:
            feature_dict["Month"] = 6.0
        if "Humidity(%)" in feature_dict:
            feature_dict["Humidity(%)"] = 65.0
        if "Pressure(in)" in feature_dict:
            feature_dict["Pressure(in)"] = 29.9
        if "Accident_Duration" in feature_dict:
            feature_dict["Accident_Duration"] = 30.0
        if "Accident_Duration_Min" in feature_dict:
            feature_dict["Accident_Duration_Min"] = 30.0
        if "Start_Lat" in feature_dict:
            feature_dict["Start_Lat"] = 0.0
        if "Start_Lng" in feature_dict:
            feature_dict["Start_Lng"] = 0.0

        # 3. Assign user-provided numerical inputs
        feature_dict["Temperature(F)"] = float(temperature)
        feature_dict["Visibility(mi)"] = float(visibility)
        feature_dict["Wind_Speed(mph)"] = float(wind_speed)
        feature_dict["Distance(mi)"] = float(distance_affected)
        feature_dict["Hour"] = float(hour_of_day)
        feature_dict["DayOfWeek"] = float(DAY_NAME_TO_INT[day_of_week])

        # 4. Convert Time of day: Day=1, Night=0 for Sunrise_Sunset_Day
        is_day = 1.0 if time_of_day == "Day" else 0.0
        if "Sunrise_Sunset_Day" in feature_dict:
            feature_dict["Sunrise_Sunset_Day"] = is_day
        if "Sunrise_Sunset_Night" in feature_dict:
            feature_dict["Sunrise_Sunset_Night"] = 1.0 if time_of_day == "Night" else 0.0

        # 5. Convert Weather condition using notebook one-hot encoding scheme
        # Check if column matching exact selected weather exists; if not, fall back to Other
        weather_col = f"Weather_Condition_{weather_condition}"
        if weather_col in feature_dict:
            feature_dict[weather_col] = 1.0
        elif "Weather_Condition_Other" in feature_dict:
            feature_dict["Weather_Condition_Other"] = 1.0

        # 6. Convert booleans to integers (1 for checked, 0 for unchecked)
        if "Junction" in feature_dict:
            feature_dict["Junction"] = 1.0 if junction else 0.0
        if "Traffic_Signal" in feature_dict:
            feature_dict["Traffic_Signal"] = 1.0 if traffic_signal else 0.0
        if "Crossing" in feature_dict:
            feature_dict["Crossing"] = 1.0 if crossing else 0.0
        if "Amenity" in feature_dict:
            feature_dict["Amenity"] = 1.0 if amenity else 0.0
        if "Stop" in feature_dict:
            feature_dict["Stop"] = 1.0 if stop_sign else 0.0
        if "Railway" in feature_dict:
            feature_dict["Railway"] = 1.0 if railway else 0.0

        # Build feature DataFrame
        df_input = pd.DataFrame([feature_dict])

        # 7. Apply scaler to numerical columns only
        if scaler is not None:
            if scaler_feature_names is not None:
                # Ensure all scaler columns are present in df_input
                for sc_col in scaler_feature_names:
                    if sc_col not in df_input.columns:
                        if sc_col == "Month":
                            df_input[sc_col] = 6.0
                        elif sc_col == "Humidity(%)":
                            df_input[sc_col] = 65.0
                        elif sc_col == "Pressure(in)":
                            df_input[sc_col] = 29.9
                        elif sc_col in ("Accident_Duration", "Accident_Duration_Min"):
                            df_input[sc_col] = 30.0
                        else:
                            df_input[sc_col] = 0.0
                df_input[scaler_feature_names] = scaler.transform(df_input[scaler_feature_names])
            else:
                # Fallback to specified numerical columns
                candidate_num_cols = [
                    "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Distance(mi)",
                    "Humidity(%)", "Pressure(in)", "Hour", "DayOfWeek", "Month",
                    "Accident_Duration" if "Accident_Duration" in df_input.columns else "Accident_Duration_Min"
                ]
                cols_to_scale = [c for c in candidate_num_cols if c in df_input.columns]
                df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])

        # 8. Align exact columns in order expected by model
        df_features = df_input[model_feature_names]

        # 9. Run prediction using the selected model
        raw_pred = active_model.predict(df_features)[0]

        # In the training pipeline, Severity was remapped from 1-4 to 0-3 (classes: 0, 1, 2, 3)
        # Remap back to 1-4
        if int(raw_pred) in [0, 1, 2, 3]:
            pred_severity = int(raw_pred) + 1
        else:
            pred_severity = int(raw_pred)

        explanation = SEVERITY_DESCRIPTIONS.get(
            pred_severity,
            "Unknown severity level."
        )

        # 10. Display result
        sev_class = f"sev-{pred_severity}"
        st.markdown(f"""
        <div class="severity-result-card {sev_class}">
            <div class="severity-heading">Predicted Severity: {pred_severity}</div>
            <div class="severity-explanation">Severity {pred_severity}: {explanation.split(': ', 1)[-1] if ': ' in explanation else explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        # Additional probability breakdown if model supports predict_proba
        if hasattr(active_model, "predict_proba"):
            try:
                probs = active_model.predict_proba(df_features)[0]
                classes = getattr(active_model, "classes_", [0, 1, 2, 3])
                
                with st.expander("📊 View Class Confidence Probabilities", expanded=False):
                    prob_cols = st.columns(len(classes))
                    for idx, (cls_idx, prob) in enumerate(zip(classes, probs)):
                        sev_num = int(cls_idx) + 1 if int(cls_idx) in [0, 1, 2, 3] else int(cls_idx)
                        with prob_cols[idx]:
                            st.metric(f"Severity {sev_num}", f"{prob * 100:.1f}%")
                            st.progress(float(prob))
            except Exception:
                pass

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please verify that all inputs are valid and model files are accessible.")
