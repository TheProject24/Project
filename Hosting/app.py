import streamlit as st
import joblib
import pandas as pd

# --- 1. CONFIGURATION ---

TOP_10_FEATURES = [
    'failures', 'Mjob_services', 'goout', 'sex_M', 'famsup_yes',
    'Walc', 'guardian_mother', 'nursery_yes', 'age', 'Mjob_health'
]


# --- 2. MODEL LOADING ---

@st.cache_resource
def load_model():
    try:
        model = joblib.load('reduced_student_logistic_model.pkl')
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'reduced_student_logistic_model.pkl' not found. Please train and save the model before running the app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model = load_model()

# --- 3. STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Student Risk Predictor", layout="centered")

st.title("🎓 Student Risk Prediction App")
st.markdown("Use the top 10 influential features to assess a student's risk of failure (G3 $\\le 10$).")
st.markdown("---")

inputs = {}

# --- A. Academic & Behavioral Factors (Most Important) ---
st.subheader("1. Academic & Social Factors (Highest Impact)")

col1, col2 = st.columns(2)

with col1:
    # 1. failures (Numeric, most important feature)
    inputs['failures'] = st.slider(
        'Past School Failures',
        min_value=0, max_value=4, value=0, step=1,
        help="Number of times the student failed a class in the past."
    )
with col2:
    # 2. goout (Ordinal, high risk)
    inputs['goout'] = st.slider(
        'Time Spent Going Out',
        min_value=1, max_value=5, value=3, step=1,
        help="Scale from 1 (Very low) to 5 (Very high)."
    )

# 3. Walc (Ordinal, Alcohol Consumption)
inputs['Walc'] = st.slider(
    'Weekend Alcohol Consumption',
    min_value=1, max_value=5, value=1, step=1,
    help="Scale from 1 (Very low) to 5 (Very high)."
)

# --- B. Demographic & Binary Factors ---
st.subheader("2. Family & Demographic Status")

col3, col4, col5 = st.columns(3)

# 4. sex_M (Binary)
with col3:
    sex_M_val = st.radio("Student Gender:", ('Female', 'Male'), index=0)
    inputs['sex_M'] = 1 if sex_M_val == 'Male' else 0

# 5. famsup_yes (Binary)
with col4:
    famsup_yes_val = st.radio("Family Educational Support:", ('No', 'Yes'), index=0)
    inputs['famsup_yes'] = 1 if famsup_yes_val == 'Yes' else 0

# 6. nursery_yes (Binary)
with col5:
    nursery_yes_val = st.radio("Attended Nursery School:", ('No', 'Yes'), index=1)
    inputs['nursery_yes'] = 1 if nursery_yes_val == 'Yes' else 0

# 7. age (Numeric)
inputs['age'] = st.slider(
    'Student Age',
    min_value=15, max_value=22, value=16, step=1
)

# --- C. Parent Job/Guardian Factors (Encoded) ---
st.subheader("3. Parent Characteristics")

col6, col7, col8 = st.columns(3)

# 8. Mjob_services (Binary, mother's job)
with col6:
    Mjob_services_val = st.selectbox(
        'Mother\'s Job is Services:',
        ('No (Other Job)', 'Yes (Services)'),
        index=0,
        help="Trained on one-hot encoded features."
    )
    inputs['Mjob_services'] = 1 if Mjob_services_val == 'Yes (Services)' else 0

# 9. Mjob_health (Binary, mother's job)
with col7:
    Mjob_health_val = st.selectbox(
        'Mother\'s Job is Health:',
        ('No (Other Job)', 'Yes (Health)'),
        index=0,
        help="Trained on one-hot encoded features."
    )
    inputs['Mjob_health'] = 1 if Mjob_health_val == 'Yes (Health)' else 0

# 10. guardian_mother (Binary)
with col8:
    guardian_mother_val = st.radio("Guardian is Mother:", ('No (Father/Other)', 'Yes (Mother)'), index=1)
    inputs['guardian_mother'] = 1 if guardian_mother_val == 'Yes (Mother)' else 0

st.markdown("---")

# --- 4. PREDICTION LOGIC ---

if st.button('🎯 Predict Student Outcome'):

    # 1. Create a DataFrame from the inputs dictionary
    # Note: We must ensure all 10 features are explicitly handled or set to 0.

    # Create the input row, setting all 10 features explicitly from inputs dict

    # Handle the fact that only one Mjob (services or health) can be 1
    # If the user selects both, we respect the last input, but ideally, Mjob should be a single select box.

    input_row = {feature: 0 for feature in TOP_10_FEATURES}

    # Transfer user inputs to the row
    input_row['failures'] = inputs['failures']
    input_row['goout'] = inputs['goout']
    input_row['Walc'] = inputs['Walc']
    input_row['age'] = inputs['age']

    # Transfer binary inputs (they are already 0 or 1)
    input_row['sex_M'] = inputs['sex_M']
    input_row['famsup_yes'] = inputs['famsup_yes']
    input_row['nursery_yes'] = inputs['nursery_yes']
    input_row['guardian_mother'] = inputs['guardian_mother']

    # Handle Mjob features (they are exclusive in the real dataset, but we treat them as independent binary columns here)
    input_row['Mjob_services'] = inputs['Mjob_services']
    input_row['Mjob_health'] = inputs['Mjob_health']

    # Convert to DataFrame
    input_df = pd.DataFrame([input_row])

    # Ensure the columns are in the exact order the model was trained on
    input_df = input_df[TOP_10_FEATURES]

    # Make Prediction
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
    except ValueError as e:
        st.error(
            f"Prediction Error: The input features may not match the model's required features. Ensure the model was trained on the 10 selected features.")
        st.exception(e)
        st.stop()

    # 5. Display Results
    st.markdown("### Prediction Result")

    if prediction == 1:
        st.success("✅ Prediction: PASS - LOW RISK")
        st.markdown(
            f"The model predicts the student **WILL PASS** (G3 > 10) with a probability of **{prediction_proba[1] * 100:.1f}%**.")
        st.balloons()
    else:
        st.error("🚨 Prediction: FAIL - HIGH RISK")
        st.markdown(
            f"The model predicts the student **WILL FAIL** (G3 $\\le 10$) with a probability of **{prediction_proba[0] * 100:.1f}%**.")
        st.warning("Intervention is strongly recommended for this student.")

    st.markdown("---")
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
        }
    </style>""", unsafe_allow_html=True)