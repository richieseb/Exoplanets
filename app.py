import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Chai to Code | Transit Finder", page_icon="☕", layout="wide")

st.title("☕ Team Chai to Code: Exoplanet Transit Detection Pipeline")
st.markdown("---")

# --- CUSTOM FUNCTIONS ---
def apply_astronomical_detrending(X_matrix, window_length=101, polyorder=2):
    detrended_matrix = np.zeros_like(X_matrix)
    for i in range(X_matrix.shape[0]):
        trend = savgol_filter(X_matrix[i], window_length=window_length, polyorder=polyorder)
        detrended_matrix[i] = X_matrix[i] - trend + 1.0
    return detrended_matrix

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("Pipeline Controls")
confidence_threshold = st.sidebar.slider("Classification Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

# --- 2. DATASET UPLOAD ---
uploaded_file = st.file_index = st.file_uploader("Upload Light Curve CSV File (e.g., exoTest.csv)", type=["csv"])

if uploaded_file is not None:
    # Load data
    with st.spinner("Parsing astronomical datasets..."):
        df = pd.read_csv(uploaded_file)
        
    st.success(f"Successfully loaded {df.shape[0]} stellar observations!")
    
    # Process Labels and Features
    if 'LABEL' in df.columns:
        y_true = df['LABEL'].values - 1
        X_raw = df.drop(columns=['LABEL']).values
    else:
        y_true = None
        X_raw = df.values

    # Apply the Signal Processing step
    X_clean = apply_astronomical_detrending(X_raw)
    X_input = np.expand_dims(X_clean, axis=-1)

    # --- 3. MODEL INFERENCE ---
    # Quick mock/load of your model (Make sure you save your model in Colab first using cnn_model.save('best_model.h5'))
    try:
        model = tf.keras.models.load_model('best_model.h5')
        y_prob = model.predict(X_input).flatten()
        y_pred = (y_prob > confidence_threshold).astype(int)
    except Exception as e:
        st.error("Model file 'best_model.h5' not found. Displaying simulation mode.")
        # Fallback simulation if model isn't uploaded locally yet
        np.random.seed(42)
        y_prob = np.random.uniform(0.0, 1.0, size=df.shape[0])
        y_pred = (y_prob > confidence_threshold).astype(int)

    # --- 4. SUMMARY DASHBOARD METRICS ---
    detected_planets = np.where(y_pred == 1)[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stars Scanned", f"{df.shape[0]}")
    col2.metric("Exoplanet Candidates Flagged", f"{len(detected_planets)}")
    col3.metric("Operational Precision Limit", f"{confidence_threshold * 100}%")

    st.markdown("### 🎯 Flagged Pipeline Candidates")
    
    if len(detected_planets) > 0:
        # Let user select which detected planet they want to inspect
        selected_star = st.selectbox("Select a Candidate Star Index to Inspect:", detected_planets)
        
        # Plotting the visual verification dashboard
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(X_clean[selected_star], color='crimson', lw=1.5, label="Filtered Signal")
        ax.axhline(np.mean(X_clean[selected_star]), color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_title(f"Visual Verification Profile — Star Index {selected_star}")
        ax.set_xlabel("Time (Cadence Intervals)")
        ax.set_ylabel("Normalized Flux")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=":")
        
        st.pyplot(fig)
        
        st.info(f"💡 **Pipeline Diagnostics:** Confidence score for this candidate is **{y_prob[selected_star]*100:.2f}%**.")
    else:
        st.write("No stars crossed the current confidence threshold setting.")
