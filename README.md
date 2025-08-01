# Coral Reef Species Multi-Label Predictor

## Overview

The **Coral Reef Species Multi-Label Predictor** is a Streamlit web application that predicts the presence of 11 coral species based on environmental conditions. The app leverages a pre-trained multi-label classifier (`classifier_model.pkl`), trained on the synthetic dataset `coral_reef_sites.csv`, which contains environmental feature and species presence data for 10,000 coral reef sites. Users input environmental features (e.g., depth, region, light availability) into a web form, and the app displays predictions in a results table.

The app is built with Python, utilizing:
- `pandas`
- `scikit-learn`
- `scikit-multilearn`
- `streamlit`
- `joblib`

All preprocessing logic is embedded in `app.py`; no additional modules are required.

## File Structure

```plaintext
CORAL/
├── data/
│   ├── coral_reef_sites.csv    # Dataset: features and species labels
│   └── classifier_model.pkl    # Pre-trained model
├── app.py                     # Streamlit app
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
```

### Files Explained

- **data/coral_reef_sites.csv**  
  - 10,000 rows detailing coral reef conditions and species.
  - **Features:**  
    - Numerical: `depth`, `structural_complexity`, `water_temperature`, `salinity`, `coral_cover`, `algal_cover`, `proximity_to_human_activity`
    - Categorical: `region`, `substrate_type`, `light_availability`, `marine_protection_status`
  - **Labels:**  
    - 11 species (e.g., `species_Porites`, `species_Acropora`) as 1 (present) or 0 (absent).

- **data/classifier_model.pkl**  
  - Pre-trained `BinaryRelevance` model (with `RandomForestClassifier` backend), saved using joblib.

- **app.py**
  - Streamlit app that loads data/model, processes user input, and displays predictions.

- **requirements.txt**
  - Lists all required Python libraries with versions.

## Requirements

- **Python:** 3.8–3.10
- **Virtual Environment:** Strongly recommended
- **OS:** Windows 10/11 (tested)

## Setup Instructions

### 1. Ensure Data Files

Place both `coral_reef_sites.csv` and `classifier_model.pkl` in:
```
C:\Users\Tejas\OneDrive\Desktop\CORAL\data\
```

### 2. Create and Activate a Virtual Environment

```bash
cd C:\Users\Tejas\OneDrive\Desktop\CORAL
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Ensure `requirements.txt` contains:
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
scikit-multilearn==0.2.0
streamlit==1.36.0
joblib==1.2.0
```
Install with:
```bash
pip install -r requirements.txt
```

### 4. Verify app.py

- Ensure `app.py` resides in `C:\Users\Tejas\OneDrive\Desktop\CORAL\`.
- Check that it references the model at the correct path:  
  `C:\Users\Tejas\OneDrive\Desktop\CORAL\data\classifier_model.pkl`.

### 5. Run the Application

```bash
streamlit run app.py
```
- Open [http://localhost:8501](http://localhost:8501) in your browser.

## How to Use

### 1. Open the App

- Open [http://localhost:8501](http://localhost:8501) after running the Streamlit server.

### 2. Enter Features in Form

- **Numerical:**  
  - Depth (meters), structural complexity (0–1), water temperature (°C), salinity (ppt), coral cover (%), algal cover (%), proximity to human activity (km)
- **Categorical:**  
  - Region, substrate type, light availability, marine protection status (choose from dropdowns)

#### Example Input

- Region: Caribbean
- Depth: 10.0
- Light Availability: High
- Marine Protection Status: Protected

### 3. Run Prediction

- Click **"Predict"** to generate results.

### 4. View Results

- Predictions are shown in a table format:

| species_Porites | species_Favia | ... | species_Millepora |
|-----------------|---------------|-----|-------------------|
| Present         | Absent        | ... | Present           |

## Troubleshooting

- **Dataset Not Found:**  
  - Ensure `C:\Users\Tejas\OneDrive\Desktop\CORAL\data\coral_reef_sites.csv` exists.
  - Check that the path in `app.py` matches.

- **Model Not Found:**  
  - Ensure `classifier_model.pkl` is at `C:\Users\Tejas\OneDrive\Desktop\CORAL\data\classifier_model.pkl`.

- **Invalid Value for Categorical Features:**  
  Run:
  ```python
  python -c "import pandas as pd; print(pd.read_csv('data/coral_reef_sites.csv')['light_availability'].unique())"
  ```
  - Ensure your dropdown selections match these (e.g., ['High', 'Medium', 'Low']).

- **Prediction Errors (e.g., `ValueError: X has different features`):**
  - Check that features in `app.py` exactly match those used for training.
  - To check model type:
    ```python
    import joblib
    print(type(joblib.load('C:/Users/Tejas/OneDrive/Desktop/CORAL/data/classifier_model.pkl')))
    ```
    Expected: ``

- **Streamlit Not Running:**  
  - Confirm the virtual environment is active.
  - Reinstall dependencies if necessary: `pip install -r requirements.txt`.

## Notes

- **Preprocessing:**  
  - The app encodes categorical features (e.g., `light_availability`) with `LabelEncoder` and scales numericals with `StandardScaler`, matching training-time transformations.
- **Model:**  
  - Assumes a `BinaryRelevance` classifier trained on 11 features and 11 species.
- **Dataset:**  
  - The app utilizes `coral_reef_sites.csv` to initialize encoders and scalers, ensuring user inputs are fully compatible with the classifier.
