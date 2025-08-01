import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(page_title="Coral Reef Species Prediction", layout="wide")

# Title and description
st.title("Coral Species Multi-Label Predictor")
st.markdown("Input environmental conditions to predict the presence of coral species.")

# Load dataset for reference
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\Tejas\OneDrive\Desktop\CORAL\data\coral_reef_sites.csv')

try:
    data = load_data()
except FileNotFoundError:
    st.error("Dataset not found at 'data/coral_reef_sites.csv'. Please ensure it exists.")
    st.stop()

# Define features and species
ENV_FEATURES = [
    "region", "depth", "substrate_type", "structural_complexity",
    "water_temperature", "salinity", "light_availability", "coral_cover",
    "algal_cover", "marine_protection_status", "proximity_to_human_activity"
]
SPECIES = [
    "species_Porites", "species_Favia", "species_Acropora", "species_Montipora",
    "species_Pocillopora", "species_Fungia", "species_Turbinaria", "species_Euphyllia",
    "species_Goniopora", "species_Diploria", "species_Millepora"
]
numeric_cols = [
    "depth", "structural_complexity", "water_temperature", "salinity",
    "coral_cover", "algal_cover", "proximity_to_human_activity"
]
categorical_cols = ["region", "substrate_type", "light_availability", "marine_protection_status"]

# Initialize encoders and scaler
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder().fit(data[col])
scaler = StandardScaler().fit(data[numeric_cols])

# Load pre-trained model
MODEL_PATH = r"C:\Users\Tejas\OneDrive\Desktop\CORAL\data\classifier_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    st.success("Pre-trained model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure classifier_model.pkl exists.")
    st.stop()

# Input form
st.header("Enter Environmental Features")
with st.form("coral_input_form"):
    region = st.selectbox("Region", options=sorted(data['region'].unique()))
    depth = st.slider("Depth (meters)", 0.0, 100.0, 10.0, step=0.1)
    substrate_type = st.selectbox("Substrate Type", options=sorted(data['substrate_type'].unique()))
    structural_complexity = st.slider("Structural Complexity (0-1)", 0.0, 1.0, 0.5, step=0.01)
    water_temperature = st.slider("Water Temperature (°C)", 10.0, 35.0, 26.0, step=0.1)
    salinity = st.slider("Salinity (ppt)", 20.0, 45.0, 35.0, step=0.1)
    light_availability = st.selectbox("Light Availability", options=sorted(data['light_availability'].unique()))
    coral_cover = st.slider("Coral Cover (%)", 0.0, 100.0, 50.0, step=0.1)
    algal_cover = st.slider("Algal Cover (%)", 0.0, 100.0, 25.0, step=0.1)
    marine_protection_status = st.selectbox("Marine Protection Status", options=sorted(data['marine_protection_status'].unique()))
    proximity_to_human_activity = st.slider("Proximity to Human Activity (km)", 0.0, 50.0, 5.0, step=0.1)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'region': [region],
        'depth': [depth],
        'substrate_type': [substrate_type],
        'structural_complexity': [structural_complexity],
        'water_temperature': [water_temperature],
        'salinity': [salinity],
        'light_availability': [light_availability],
        'coral_cover': [coral_cover],
        'algal_cover': [algal_cover],
        'marine_protection_status': [marine_protection_status],
        'proximity_to_human_activity': [proximity_to_human_activity]
    })
    
    # Preprocess input
    for col in categorical_cols:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except ValueError as e:
            st.error(f"Invalid value for {col}. Valid options are: {list(encoders[col].classes_)}")
            st.stop()
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # Ensure input_data has the same column order as ENV_FEATURES
    input_data = input_data[ENV_FEATURES]
    
    # Prediction
    try:
        y_pred = model.predict(input_data)
        pred_labels = y_pred.toarray()[0]  # Convert sparse matrix to dense array
        st.subheader("Prediction Results:")
        results = {species: "Present" if pred_labels[i] == 1 else "Absent" for i, species in enumerate(SPECIES)}
        results_df = pd.DataFrame([results])
        st.table(results_df)
        st.markdown("---")
        st.markdown("*Note:* The results depend on the pre-trained model. Ensure feature preprocessing matches model expectations.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    st.write("App is running...")