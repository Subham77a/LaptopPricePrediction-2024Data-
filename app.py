import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ---------------- LOAD MODEL & DATA ----------------
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ---------------- TITLE ----------------
st.title("💻 Laptop Price Predictor")
st.caption("Predict laptop price based on hardware & physical specifications")

st.divider()

# ===================== 🏷 BRAND SECTION =====================
st.subheader("🏷 Brand & Identity")

col1, col2 = st.columns(2)
with col1:
    Brand = st.selectbox("Laptop Brand", df['Brand'].unique())
with col2:
    Processor_Brand = st.selectbox("Processor Brand", df['Processor_Brand'].unique())

st.divider()

# ===================== ⚙ CPU & MEMORY =====================
st.subheader("⚙ CPU & Memory")

col_cpu, col_mem = st.columns(2, gap="large")

# -------- CPU BLOCK --------
with col_cpu:
    st.markdown("### 🧠 Processor")

    Processor_power = st.selectbox(
        "ProcessorTier (e.g. 5 = Intel i5 / Ryzen 5)",
        sorted(df['Processor_power'].unique())
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    Intel_Gen = st.selectbox(
        "Intel Generation (0 for AMD / Other)",
        sorted(df['Intel_Gen'].unique())
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    Ghz = st.number_input(
        "Processor Speed (Base GHz)",
        min_value=0.0,
        step=0.1
    )

# -------- MEMORY BLOCK --------
with col_mem:
    st.markdown("### 💾 Memory")

    RAM = st.selectbox(
        "Installed RAM (GB)",
        sorted(df['RAM'].unique())
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    RAM_Expandable = st.selectbox(
        "Expandable RAM (GB)",
        sorted(df['RAM_Expandable'].unique())
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    RAM_TYPE = st.selectbox(
        "RAM Type",
        df['RAM_TYPE'].unique()
    )

st.divider()


# ===================== 🎮 GPU SECTION =====================
st.subheader("🎮 Graphics")
GPU = st.selectbox("Graphics Processor", df['GPU'].unique())

st.divider()

# ===================== 🖥 PHYSICAL OVERVIEW =====================
st.subheader("🖥 Physical Overview")

col1, col2, col3 = st.columns(3)

with col1:
    Display = st.number_input(
        "Display Size (inches)",
        min_value=10.0,
        max_value=18.0,
        step=0.1
    )

with col2:
    Display_type = st.selectbox(
        "Display Type",
        df['Display_type'].unique()
    )

with col3:
    Adapter = st.selectbox(
        "Adapter Power (W)",
        sorted(df['Adapter'].unique())
    )

st.divider()

# ===================== 💾 STORAGE =====================
st.subheader("💾 Storage")

col1, col2 = st.columns(2)
with col1:
    SSD = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))
with col2:
    HDD = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))

st.divider()

# ===================== 📊 PREDICTION =====================
st.subheader("📊 Price Estimation")

if st.button("🔍 Predict Laptop Price", use_container_width=True):

    input_df = pd.DataFrame([{
        'Brand': Brand,
        'Processor_Brand': Processor_Brand,
        'RAM_Expandable': int(RAM_Expandable),
        'RAM': int(RAM),
        'RAM_TYPE': RAM_TYPE,
        'Ghz': float(Ghz),
        'Display_type': Display_type,
        'Display': int(Display),
        'GPU': GPU,
        'SSD': float(SSD),
        'HDD': float(HDD),
        'Adapter': int(Adapter),
        'Intel_Gen': int(Intel_Gen),
        'Processor_power': int(Processor_power)
    }])

    # ---- FIX: convert log-price to actual price ----
    predicted_log_price = pipe.predict(input_df)[0]
    predicted_price = np.exp(predicted_log_price)

    st.success(f"💰 **Estimated Laptop Price:** ₹{int(predicted_price):,}")
    st.caption("🔎 Estimated using historical pricing patterns (±10–15%)")
