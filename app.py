import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt

st.title("🧠 EEG Seizure Detection System (Using Machine Learning and Signal Processing)")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model("eeg_model_new.h5")

# ---------------------------
# FILTER FUNCTION
# ---------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=256):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ---------------------------
# INPUT
# ---------------------------
st.header("1. EEG Input")

uploaded_file = st.file_uploader("Upload EEG (.edf file)", type=["edf"])

if uploaded_file is not None:
    try:
        import mne

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()

        st.success("EDF file loaded successfully")

        # ---------------------------
        # SLIDING WINDOW
        # ---------------------------
        window_size = 2000
        max_std = 0
        best_segment = data[0][:2000]

        for i in range(0, len(data[0]) - window_size, window_size):
            segment = data[0][i:i+window_size] * 1e6
            if np.std(segment) > max_std:
                max_std = np.std(segment)
                best_segment = segment

        # APPLY FILTER
        signal = bandpass_filter(best_segment)

    except Exception as e:
        st.error(f"Error reading EDF file: {e}")
        signal = np.random.normal(0, 1, 2000)

else:
    st.info("No file uploaded → using sample signal")
    signal = np.random.normal(0, 1, 2000)

# ---------------------------
# VISUALIZATION
# ---------------------------
st.header("2. Signal Visualization")

fig, ax = plt.subplots()
ax.plot(signal)
ax.set_title("Filtered EEG Signal")
st.pyplot(fig)

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
st.header("3. Feature Extraction")

mean_val = np.mean(signal)
std_val = np.std(signal)
peak_val = np.max(np.abs(signal))

st.write("Mean:", mean_val)
st.write("Standard Deviation:", std_val)
st.write("Peak Amplitude:", peak_val)

# ---------------------------
# ML PREDICTION (DISPLAY ONLY)
# ---------------------------
st.header("4. ML Model Output")

features = np.array([[mean_val/50, std_val/200, peak_val/500]])

prediction = model.predict(features)[0][0]

st.write("Model Prediction Value:", prediction)

# ---------------------------
# FINAL DECISION (RELIABLE)
# ---------------------------
# FINAL DETECTION (HYBRID)
# ---------------------------
st.header("5. Final Detection")

# combine ML + signal logic
if prediction > 0.6 or std_val > 200 or peak_val > 800:
    st.error("⚠️ Seizure Detected")
    st.write("Detected using ML prediction + signal abnormality")
else:
    st.success("✅ No Seizure")
    st.write("Normal EEG pattern")
# ---------------------------
# CLINICAL RECOMMENDATION
# ---------------------------
st.header("6. Clinical Recommendation")

if prediction > 0.6 or std_val > 200 or peak_val > 800:
    result = "Seizure"
    st.warning("""
⚠️ The EEG signal shows abnormal neural activity consistent with seizure-like patterns.

Recommendation:
- Immediate neurological evaluation is advised  
- Correlate with patient clinical symptoms  
- Continuous EEG monitoring recommended  
- MRI/CT scan may be required  

Risk Level: HIGH
""")
else:
    result = "Normal"
    st.success("""
✅ The EEG signal appears within normal physiological limits.

Recommendation:
- No immediate clinical concern  
- Routine monitoring is sufficient  
- Follow-up if symptoms persist  

Risk Level: LOW
""")

# ---------------------------
# DOWNLOAD REPORT
# ---------------------------
st.header("7. Download Report")

report = f"""
EEG REPORT

Mean: {mean_val}
Standard Deviation: {std_val}
Peak Amplitude: {peak_val}

Model Prediction: {prediction}

Final Result: {result}

Clinical Recommendation:
{"Neurological evaluation required" if result == "Seizure" else "Routine monitoring sufficient"}
"""

st.download_button(
    label="Download Report",
    data=report,
    file_name="EEG_Report.txt",
    mime="text/plain"
)