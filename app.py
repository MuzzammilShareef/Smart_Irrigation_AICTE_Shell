# --- Import required libraries ---
import streamlit as st                      # Main app framework for interactivity and UI
import pandas as pd                         # DataFrame handling and CSV I/O
import joblib                              # For loading your saved ML pipeline (.pkl)
import matplotlib.pyplot as plt             # For charting/feature importance visualization
import time                                # Used to time the prediction for UI feedback

# --- Brand the app, set display options and sidebar ---
st.set_page_config(
    page_title="Smart Irrigation Advisor",   # Sets browser tab title and window titles
    page_icon="🌱",                          # Small icon in tab and sidebar
    layout="wide",                          # Uses the viewport width for easier chart/table display
)

# Title bar and intro, styled with markdown and HTML for visual polish
st.markdown("<h1 style='text-align:center;'>🌱 Smart Irrigation Advisor</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:gray;'>
AI-powered irrigation advice for your farm parcels<br>
<b>By Mohammed Muzzammil Shareef • AICTE Internship 2025 • Hyderabad</b>
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar: user input, instructions, and sample data download ---
st.sidebar.header("🚀 Quick Start")
user_name = st.sidebar.text_input("👤 Your name", value="Mohammed")
user_location = st.sidebar.text_input("📍 Your location", value="Hyderabad")

# Friendly intro and workflow steps
st.sidebar.markdown(f"""
👋 Hello, **{user_name}** from **{user_location}**!

1. Download **sample_sensor.csv** below  
2. Fill in columns **sensor_0…sensor_19**  
3. Upload your CSV here  
4. View parcel status with helpful symbols!
""")

# Sample CSV for users to try/demo/test the app easily
st.sidebar.download_button(
    label="📥 Download sample_sensor.csv",
    data=",".join([f"sensor_{i}" for i in range(20)]) + "\n" + ",".join(["0.5"]*20) + "\n",
    file_name="sample_sensor.csv",
    mime="text/csv"
)

# Button to reset the app/inputs (only available in recent Streamlit versions)
if st.sidebar.button("🔄 Reset App"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"Built with ❤️ by {user_name} from {user_location} for AICTE Internship 2025")

# --- Expander ("About" section) for context and onboarding ---
with st.expander("ℹ️ About This App"):
    st.write(
        """
**What does this app do?**  
- Predicts if each farm area (Parcel 0–2, each with a motor/pump) needs irrigation, using your 20 sensor readings.
- 🚨 shows up ONLY when irrigation is truly needed and soil moisture is low.
- 😎 shows safe/OK/"no urgent action" status for other cases (including 'no irrigation needed').
- Uses AI trained on smart farming datasets.

**How to use:**  
- CSV must have headers: sensor_0, sensor_1, ..., sensor_19—one row per reading.
- Download a template from the sidebar.
"""
    )

# --- Expander: Sensor Legend for non-technical users or reviewers ---
with st.expander("📖 Sensor Descriptions & Legend"):
    sensor_info = {
        "sensor_0": "Ambient Temperature",
        "sensor_1": "Ambient Humidity", 
        "sensor_2": "Soil pH Level",
        "sensor_3": "Soil Temperature",
        "sensor_4": "Soil Moisture (Main)",
        "sensor_5": "Soil Moisture (Subsurface)",
        "sensor_6": "Solar Radiation",
        "sensor_7": "Wind Speed",
        "sensor_8": "Leaf Wetness",
        "sensor_9": "Rainfall",
        "sensor_10": "Soil Conductivity",
        "sensor_11": "Root Zone Moisture",
        "sensor_12": "Soil Salinity",
        "sensor_13": "Soil Nutrients",
        "sensor_14": "Crop Canopy Temperature",
        "sensor_15": "Vapor Pressure Deficit",
        "sensor_16": "Soil Oxygen",
        "sensor_17": "Soil Texture",
        "sensor_18": "Soil Compaction",
        "sensor_19": "Groundwater Level"
    }
    col1, col2 = st.columns(2)
    for i, (sensor, desc) in enumerate(sensor_info.items()):
        if i < 10:
            col1.markdown(f"**{sensor}:** {desc}")
        else:
            col2.markdown(f"**{sensor}:** {desc}")

# --- Expander: Model architecture for transparency ---
with st.expander("🤖 AI Model Details"):
    st.write("""
    **Model Architecture:** Random Forest Classifier with MultiOutput wrapper
    
    **Features:** 20 environmental and soil sensors (scaled 0-1)
    
    **Outputs:** 3 binary predictions (one per parcel: 0=no irrigation, 1=irrigate)
    
    **Training:** Trained on synthetic smart farming dataset with realistic sensor patterns
    
    **Preprocessing:** MinMax scaling applied to normalize sensor readings
    """)

# --- Icons for parcel map (encoded in base64, no need for images as separate files) ---
warning_icon = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABJUlEQVR4nO2aSw7CMAxEXYTEsTg7x2JVVpVCVD5JbE+qvrdDbT1WZ2pEjRkAAAAAgIT1cVuV+hel+AxIb8DmvjIFJEAlXLuuSgEJUIh+cluRAhKQLfjL5ewUnD4BS6ZY6e5yf+4dt+J4Sm+nT0DaDWh9trNmAQnIEOl1MyMFJCBaYNTF6BSQgMjiXu5FpoAERBX2di0qBSQgomiUWxF1SYB3wejvbe/6JMCzWNYvOE8dEuBVKPtdnpceCfAootrqeOhePRrpoXwBqmQ4Aer9/qg+M2DkYrX7GyN9yGaA2ftyRDUTuhMw6n69GdrbFGX0wwzouWiWZ7+mpy/lX2S+fs6ieQM7q/slLZtlZkDLyUdw36ytTxLw74lHcX/jaP0CAAAApPMCQCeH2PUuCUkAAAAASUVORK5CYII="
cool_icon = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABqElEQVR4nO2a22rDQAxEp/2+FpIvTaD9v/ZpoYS41mVGK4jmOax0jmWvYxuYTCaTyWQymbxm3qoLftwuP2e/+b7ey/oqKWSBPopahmzxDPRRFDLe2QsCGnjVulSjKvBnYU0DbQIq4Zn1KAKq4Zl10wJ2wbPqpwTshl/J9BEW0AV+JdqPZBtk5utyk64fElB19Be8VUKkL7eAXaOvktD2FHgGrDgdWgr4D5QtwSWgYvwtgGe/8fTZcgIsYU1CKwEeqM/7lVLTLEA9/mx4a78tJmDHkV+RCVBsWWx4QCTAcwdnFaWABwQCHoEYe7oKHiALOALK3NUp4QGigDOgyDVBDQ8QBVia9VwbKuAB8ingkZBdhxWzAOtj6GzzLHhrv5JtMApReeRXZDdCXpgd8IBTgPdtjBWKDe/pU/5f4Axu15FfKfkzdAS5Gx4ICIi+lHyEVcF7+wtNQFZCF3hgw/OADmP/N2EBld/xWBLtJzUBXSRk+kifArslZOtTrgG7JDDq0i6C1RJY9SRNKx+hs0VLtkHVNCjWnS9FlYs/S7dvhSeTyWQymUxeNl+24cNU+cd48QAAAABJRU5ErkJggg=="

# --- Load ML pipeline (scaler + model) using Streamlit's resource cache ---
@st.cache_resource
def load_model():
    # Loads the trained/scaled model pipeline for instant prediction use
    return joblib.load("Farm_Irrigation_System.pkl")

pipeline = load_model()

# --- Upload and validate sensor CSV data ---
uploaded = st.file_uploader(
    "💧 Upload your sensor CSV (must include sensor_0…sensor_19)", type=["csv"]
)

if not uploaded:
    st.info(f"👋 Welcome, **{user_name}**! Please upload your CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Error reading CSV file: {e}")
    st.stop()

sensor_cols = [f"sensor_{i}" for i in range(20)]
missing_cols = [col for col in sensor_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Your file is missing these required columns: {', '.join(missing_cols)}")
    st.stop()
df = df[sensor_cols]

# --- Allow single-row (one sample) or batch mode (multi-row) with dynamic sample selection ---
if len(df) > 1:
    st.info(f"📊 Your file contains {len(df)} samples. Select a specific sample to analyze:")
    sample_idx = st.number_input("Select sample (row) to inspect", 
                                min_value=1, max_value=len(df), value=1, step=1) - 1
    selected_row = df.iloc[[sample_idx]]
    st.markdown(f"**Analyzing Sample {sample_idx + 1} of {len(df)}**")
else:
    sample_idx = 0
    selected_row = df

# --- Quick explanation of sensor values, just below selection ---
with st.expander("❔ What are sensor values?"):
    st.write(
        "Each value is a reading from a soil/environment sensor (0 = minimum, 1 = maximum). "
        "See the sensor legend above for detailed descriptions."
    )

# --- Optional: show the data preview (default ON) and allow full sampling if multi-row ---
if st.toggle("🔍 Show your sensor data preview", value=True):
    if len(df) > 1:
        st.subheader(f"Selected Sample {sample_idx + 1} Data:")
        st.dataframe(selected_row, use_container_width=True)
        with st.expander("View all samples"):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# --- Predict irrigation needs and show performance timing for UI feedback ---
with st.spinner("🤖 AI is analyzing your data..."):
    t0 = time.time()
    if len(df) > 1:
        all_preds = pipeline.predict(df)
        preds = pipeline.predict(selected_row)
        all_preds_df = pd.DataFrame(all_preds, columns=["Parcel 0", "Parcel 1", "Parcel 2"])
    else:
        preds = pipeline.predict(df)
        all_preds_df = pd.DataFrame(preds, columns=["Parcel 0", "Parcel 1", "Parcel 2"])
    elapsed = time.time() - t0

preds_df = pd.DataFrame(preds, columns=["Parcel 0", "Parcel 1", "Parcel 2"])

# --- Summarize irrigation and soil moisture for current sample ---
moisture_sensors = ["sensor_4", "sensor_11"]   # String names for moisture columns
avg_moisture = selected_row[moisture_sensors].mean().mean()
total_irrigations = int(preds_df.sum().sum())

# Display key session stats for reviewer/user transparency
st.success(f"🎯 **Analysis Complete!** Processing time: {elapsed:.2f}s | "
          f"Sample {sample_idx + 1} needs {total_irrigations} irrigation(s) | "
          f"Avg soil moisture: {avg_moisture:.2f}")

# --- Show AI predictions and highlight most influential sensors (top-3) for current sample ---
if st.toggle("💡 Show irrigation recommendations", value=True):
    st.dataframe(preds_df, use_container_width=True)
    # Compute and display the 3 most influential features (by model importance)
    importances = pipeline.named_steps["model"].estimators_[0].feature_importances_
    top_features_idx = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)[:3]
    top_features = [sensor_cols[i] for i in top_features_idx]
    st.info(f"🔍 **Most influential sensors for this prediction:** {', '.join(top_features)}")
    # Download single sample or full batch predictions as CSV
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download current sample prediction",
            data=preds_df.to_csv(index=False),
            file_name=f"prediction_sample_{sample_idx + 1}.csv",
            mime="text/csv"
        )
    if len(df) > 1:
        with col2:
            st.download_button(
                label="⬇️ Download all predictions",
                data=all_preds_df.to_csv(index=False),
                file_name="all_irrigation_predictions.csv",
                mime="text/csv"
            )

# --- Parcel Map with Icon (Warning/Success) Visual Feedback by moisture state ---
moisture_threshold = 0.3    # Below this, display warning even if irrigation is recommended

if st.toggle("🚜 Visual parcel map ", value=True):
    st.markdown("#### Parcel Irrigation Map")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        needs_water = bool(preds_df[f"Parcel {i}"].iloc[0] == 1)
        parcel_moisture = avg_moisture  # Uses the average for selected sample – could further refine per parcel if desired
        if needs_water and parcel_moisture < moisture_threshold:
            icon_to_show = warning_icon
            expl = "<b style='color:red;'>🚨 Warning: Soil moisture critically low! Irrigate immediately.</b>"
            status_color = "#ffebee"
        elif needs_water:
            icon_to_show = cool_icon
            expl = "<b style='color:green;'>😎 Irrigation recommended. Moisture level acceptable.</b>"
            status_color = "#e8f5e8"
        else:
            icon_to_show = cool_icon
            expl = "<b style='color:gray;'>⏸️ No irrigation needed. Parcel is well-watered.</b>"
            status_color = "#f5f5f5"
        # Card/box for each parcel
        col.markdown(
            f"""
            <div style='background:{status_color}; border-radius:12px; padding:1.5em; 
                       text-align:center; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); margin:0.5em;'>
                <img src='{icon_to_show}' width='48' style='margin-bottom:8px;' 
                     title='Parcel {i} irrigation status'/>
                <h3 style="margin:0.5em 0; color:#1e3a8a;">Parcel {i}</h3>
                <p style='margin:0.5em 0;'>{expl}</p>
                <small style='color:#666;'>Moisture level: {parcel_moisture:.2f}</small>
            </div>
            """, unsafe_allow_html=True
        )

# --- Charts, analytics, and insights section with tabs for easy navigation ---
if st.toggle("📊 Show analytics and insights", value=True):
    tab1, tab2, tab3 = st.tabs(["📈 Frequency Analysis", "🧠 Feature Importance", "📊 Trends"])
    with tab1:
        if len(df) > 1:
            totals = all_preds_df.sum().rename("Times Watered")
            st.bar_chart(totals)
            st.caption(f"Bar chart: Total irrigation recommendations across all {len(df)} samples.")
        else:
            st.bar_chart(preds_df.sum().rename("Irrigation Needed"))
            st.caption("Bar chart: Current irrigation needs per parcel.")
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(sensor_cols, importances, color="#357ABD")
        ax.set_xticks(range(len(sensor_cols)))
        ax.set_xticklabels(sensor_cols, rotation=45, ha="right") # Safer: avoids UserWarning
        ax.set_ylabel("Feature Importance")
        ax.set_title("AI Model: Sensor Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Sensors with higher bars have more influence on irrigation decisions.")
    with tab3:
        if len(df) > 1:
            trends = all_preds_df.rolling(window=min(7, len(df)), min_periods=1).sum()
            st.line_chart(trends)
            st.caption("Rolling sum showing irrigation frequency trends over your samples.")
        else:
            st.info("Upload multiple samples to see trend analysis.")

# --- Interactive sensor adjustment: What-if live prediction for feature importance and experiments ---
if st.toggle("🔧 Interactive sensor adjustment (What-if analysis)", value=False):
    st.markdown("#### Adjust Key Sensors and See Live Predictions")
    # Adjustable sliders for the most influential sensors
    adjusted_data = selected_row.copy()
    col1, col2 = st.columns(2)
    with col1:
        for idx in [4, 11]:  # Soil moisture sensors
            current_val = float(selected_row[f"sensor_{idx}"].iloc[0])
            new_val = st.slider(
                f"{sensor_info[f'sensor_{idx}']} (sensor_{idx})", 
                0.0, 1.0, current_val, 0.01, key=f"adjust_{idx}"
            )
            adjusted_data[f"sensor_{idx}"] = new_val
    with col2:
        for idx in [0, 3]:  # Temperature sensors
            current_val = float(selected_row[f"sensor_{idx}"].iloc[0])
            new_val = st.slider(
                f"{sensor_info[f'sensor_{idx}']} (sensor_{idx})", 
                0.0, 1.0, current_val, 0.01, key=f"adjust_{idx}"
            )
            adjusted_data[f"sensor_{idx}"] = new_val
    # Update prediction live if user presses button
    if st.button("🔄 Update Prediction"):
        new_preds = pipeline.predict(adjusted_data)
        new_preds_df = pd.DataFrame(new_preds, columns=["Parcel 0", "Parcel 1", "Parcel 2"])
        st.success("Updated predictions based on your adjustments:")
        st.dataframe(new_preds_df)

# --- Help & Interpretation Section with troubleshooting for reviewers/users ---
with st.expander("ℹ️ How to interpret results and troubleshooting"):
    st.write(
        """
**Icon Meanings:**
- 🚨 **Warning (Red background):** Irrigation needed AND soil moisture critically low
- 😎 **Success (Green background):** Irrigation recommended, moisture acceptable  
- ⏸️ **Neutral (Gray background):** No irrigation needed

**Understanding the Data:**
- **Parcel Map:** Visual status of each farm zone's irrigation needs
- **Feature Importance:** Shows which sensors most influence AI decisions
- **Trends:** Track irrigation patterns over multiple samples

**Troubleshooting:**
- Ensure your CSV has exactly 20 columns named sensor_0 through sensor_19
- Values should be scaled between 0 and 1
- For multiple samples, each row represents a different time point

**Model Confidence:**
This AI model uses ensemble learning (Random Forest) for robust predictions across varying conditions.
        """
    )
st.markdown("---")
st.caption(f"© 2025 {user_name} from {user_location} • Smart Irrigation Advisor • AICTE Internship")
