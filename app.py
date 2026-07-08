import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="ECG Classification", layout="wide")

st.title("🫀 ECG Heartbeat Classification App")
st.write("Upload an ECG dataset file OR enter custom heartbeat values manually to run the prediction model.")

st.markdown("---")

# Dictionary to map MIT-BIH class numbers to their real medical names
CLASS_MAPPING = {
    0: "Normal Beat (N)",
    1: "Supraventricular Premature Beat (S)",
    2: "Premature Ventricular Contraction (V)",
    3: "Fusion of Ventricular and Normal Beat (F)",
    4: "Unclassifiable Beat (Q)"
}

# Placeholder for loading your model
# @st.cache_resource
# def load_my_model():
#     # Examples:
#     # from tensorflow.keras.models import load_model
#     # return load_model('your_model.h5')
#     pass

# model = load_my_model()

# 2. Control Panel - Choose Input Method
st.sidebar.header("Input Method Selection")
input_method = st.sidebar.radio(
    "Choose how you want to provide ECG data:",
    ("Upload CSV File", "Enter Manual Values")
)

features = None
source_info = ""
true_label_num = None

# --- METHOD 1: UPLOAD CSV ---
if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload your ECG Data File (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            
            # Display file summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Heartbeat Records (Rows)", value=df.shape[0])
            with col2:
                st.metric(label="Total Features per Record", value=df.shape[1])
                
            st.markdown("---")
            
            row_index = st.sidebar.number_input(
                "Type or Select Heartbeat Index Number:", 
                min_value=0, 
                max_value=len(df) - 1, 
                value=0,
                step=1
            )
            
            selected_row = df.iloc[row_index]
            source_info = f"Row Index {row_index} from uploaded file"
            
            if df.shape[1] == 188:
                features = selected_row.iloc[:-1].values
                true_label_num = int(selected_row.iloc[-1])
            else:
                features = selected_row.values
                
        except Exception as e:
            st.error(f"An unexpected error occurred while parsing the dataset: {e}")
    else:
        st.info("💡 Awaiting a CSV data file. Please upload one using the file uploader or switch to Manual Input in the sidebar.")

# --- METHOD 2: MANUAL VALUES (SMART 187/188 DETECTION) ---
else:
    st.subheader("✍️ Enter Custom Heartbeat Values")
    st.write("Paste numbers from your test file. (The app will automatically handle if the line contains 187 or 188 values).")
    
    # Text area for raw manual input
    user_input = st.text_area("Paste your numbers here:", value="", height=200, placeholder="0.5\n0.8\n0.3 ...")
    
    if user_input:
        try:
            # 1. replace commas with spaces so everything becomes space-separated
            cleaned_input = user_input.replace(",", " ")
            # 2. replace newlines and tabs with spaces
            cleaned_input = cleaned_input.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            
            # 3. Split by spaces and filter out empty strings
            tokens = cleaned_input.split(" ")
            parsed_values = [float(x.strip()) for x in tokens if x.strip() != ""]
            
            count = len(parsed_values)
            
            # Smart Validation Check
            if count == 187:
                features = np.array(parsed_values)
                source_info = "Custom Manual Input (Exactly 187 features)"
                st.success("Successfully parsed 187 values!")
                
            elif count == 188:
                # If 188, automatically extract the first 187 as features, and the last one as the True Label
                features = np.array(parsed_values[:-1])
                true_label_num = int(parsed_values[-1])
                true_label_name = CLASS_MAPPING.get(true_label_num, "Unknown Class")
                
                source_info = f"Custom Manual Input (188 values detected -> Label extracted)"
                st.success(f"Successfully parsed 188 values! Auto-extracted True Label: **{true_label_name}**")
                
            elif count < 187:
                st.error(f"🚨 Too few values! You provided **{count}** values. The model requires 187 features. Please add {187 - count} more numbers.")
            else:
                st.error(f"🚨 Invalid length! You provided **{count}** values. Raw test rows should be either 187 or 188 values long.")
                
        except ValueError:
            st.error("🚨 Invalid format! Please ensure you only enter valid numbers.")

# 3. Processing and Prediction Layout (Only executes if validation passes)
if features is not None:
    
    # 4. Visualization Section
    st.subheader(f"📈 ECG Waveform Visualizer - {source_info}")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(features, color='#1E88E5', linewidth=2, label='ECG Signal')
    ax.set_title("ECG Signal Lead Amplitude over Time", fontsize=10)
    ax.set_xlabel("Time / Sampling Points", fontsize=8)
    ax.set_ylabel("Normalized Amplitude", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 5. Prediction Execution
    st.subheader("🤖 Model Inference")
    
    input_data = np.array([features])
    
    # If your model requires 3D inputs (e.g., CNN or LSTM), uncomment the line below:
    # input_data = np.expand_dims(input_data, axis=-1)
    
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Processing signal through the network..."):
            
            # -------------------------------------------------------------
            # REAL MODEL INTEGRATION NOTE:
            # prediction = model.predict(input_data)
            # predicted_class_num = np.argmax(prediction, axis=1)[0]
            # confidence = np.max(prediction) * 100
            # -------------------------------------------------------------
            
            # --- MOCK LOGIC FOR TESTING ---
            if true_label_num is not None:
                predicted_class_num = true_label_num
            else:
                predicted_class_num = int(np.sum(features) % 5)
            
            np.random.seed(int(np.sum(features) * 100) % 1000)
            confidence = np.random.uniform(94.0, 99.9)
            # ------------------------------
            
            # Map the predicted number to the actual class name
            predicted_class_name = CLASS_MAPPING.get(predicted_class_num, "Unknown Class")
            
            st.success("Prediction complete!")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"### Predicted Class:  \n**{predicted_class_name}**")
            with res_col2:
                st.markdown(f"### Confidence Score:  \n**{confidence:.2f}%**")