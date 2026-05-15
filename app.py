import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="HeartGuard AI | Stroke Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
    }

    .stButton>button:hover {
        background-color: #ff3333;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.3);
    }

    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }

    .result-container {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .header-style {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Responsive adjustment */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    /* Remove sidebar scrollbar */
    [data-testid="stSidebar"] > div:first-child {
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_assets():
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, expected_columns = load_assets()

# --- Sidebar ---
with st.sidebar:
    if os.path.exists("header.png"):
        st.image("header.png", use_container_width=True)
    st.title("HeartGuard AI")
    st.markdown("---")
    st.info("""
    **Project Overview**
    This AI system predicts the risk of heart disease based on clinical data points. 
    It uses a K-Nearest Neighbors (KNN) model trained on medical records.
    """)
    st.write("**Developed by:** Harsh")
    st.write("**Course:** Machine Learning")
    st.markdown("---")
    st.caption("Disclaimer: This tool is for educational purposes only and does not replace professional medical advice.")

# --- Main UI ---
st.markdown("""
    <div class="header-style">
        <h1>Heart Health Analysis Dashboard</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Predicting cardiovascular risk with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.warning("Please ensure model files (KNN_heart.pkl, scaler.pkl, columns.pkl) are in the project directory.")
else:
    # Use tabs for a cleaner UX
    tab1, tab2 = st.tabs(["🚀 Prediction Tool", "📊 About the Model"])

    with tab1:
        st.subheader("Patient Information Input")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 👤 Demographics")
                age = st.slider("Age", 18, 100, 40, help="Patient's age in years")
                sex = st.selectbox("Sex", ["M", "F"], help="Biological sex of the patient")
                
            with col2:
                st.markdown("### 💓 Vitals")
                resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120, help="Resting blood pressure")
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level")
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

            with col3:
                st.markdown("### 🏥 Clinical Data")
                chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], 
                                         help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
                resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
                
            st.markdown("---")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
            with col5:
                exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
            with col6:
                st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
                
            oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest")

            submit_button = st.form_submit_button(label="Analyze Heart Health")

        if submit_button:
            # Processing Input
            raw_input = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_' + sex: 1,
                'ChestPainType_' + chest_pain: 1,
                'RestingECG_' + resting_ecg: 1,
                'ExerciseAngina_' + exercise_angina: 1,
                'ST_Slope_' + st_slope: 1
            }

            input_df = pd.DataFrame([raw_input])
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[expected_columns]
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

            # Results Section
            st.markdown("---")
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-container" style="background-color: #ffe5e5; border-left: 10px solid #ff4b4b;">
                        <h2 style="color: #ff4b4b;">⚠️ High Risk Detected</h2>
                        <p style="font-size: 1.1rem; color: #333;">Our analysis suggests a significant risk of heart disease. 
                        Please consult a medical professional immediately.</p>
                        {f'<h4 style="color: black;">Confidence Score: {prediction_proba:.2%}</h4>' if prediction_proba else ''}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-container" style="background-color: #e5f9e5; border-left: 10px solid #28a745;">
                        <h2 style="color: #28a745;">✅ Low Risk Detected</h2>
                        <p style="font-size: 1.1rem; color: #333;">Our analysis suggests a low risk of heart disease. 
                        Maintain a healthy lifestyle and regular checkups!</p>
                        {f'<h4 style="color: black;">Confidence Score: {1-prediction_proba:.2%}</h4>' if prediction_proba else ''}
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.header("Model & Feature Analysis")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Key Features Explained")
            st.write("""
            - **Chest Pain Type**: Classification of chest pain symptoms.
            - **Oldpeak**: Exercise-induced ST depression relative to rest.
            - **ST Slope**: The slope of the peak exercise ST segment.
            - **Max HR**: Peak heart rate achieved during exercise.
            """)
            
        with col_b:
            st.subheader("System Performance")
            st.metric(label="Model Accuracy", value="85%", delta="Verified")
            st.write("The model uses K-Nearest Neighbors (KNN) to classify heart health based on patterns in historical data.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Created for Academic Project • 2026</p>", unsafe_allow_html=True)