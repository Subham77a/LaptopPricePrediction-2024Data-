import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests

BACKEND_VALIDATE_URL = "http://localhost:3000/api/auth/validate-token"

st.set_page_config(page_title="Secure Streamlit App")

# 🔐 Get token from URL (NEW API)
token = st.query_params.get("token")

if not token:
    st.error("❌ No authentication token")
    st.stop()

# 🔍 Validate token with backend
try:
    res = requests.get(
        BACKEND_VALIDATE_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=5
    )

    if res.status_code != 200:
        st.error("❌ Invalid or expired token")
        st.stop()

    user = res.json().get("user")

except Exception:
    st.error("❌ Authentication failed")
    st.stop()

# ✅ Authenticated UI
st.success("✅ Authenticated")
st.write(f"Welcome **{user['username']}**")

st.header("Secure Streamlit Dashboard")
st.write("Only logged-in users can see this.")


# CONFIG
st.set_page_config(
    page_title="Laptop Performance and Buying Guide",
    page_icon="💻",
    layout="centered"
)

#LOAD MODEL & DATA
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# SESSION STATE INIT 
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None

#  BRAND INTELLIGENCE 
brand_intelligence = {
    "HP": {"reliability": 85, "service": 90},
    "Dell": {"reliability": 90, "service": 95},
    "Lenovo": {"reliability": 88, "service": 92},
    "ASUS": {"reliability": 80, "service": 85},
    "Acer": {"reliability": 78, "service": 80},
    "MSI": {"reliability": 75, "service": 70}
}

#  HELPER FUNCTIONS 
def normalize(val, min_val, max_val):
    return max(0, min(100, (val - min_val) / (max_val - min_val) * 100))

def performance_score(cpu_tier, ram, gpu):
    cpu_score = normalize(cpu_tier, 1, 9)
    ram_score = normalize(ram, 4, 64)
    gpu_score = 85 if "RTX" in gpu else 70 if "GTX" in gpu else 55
    return round(0.4 * cpu_score + 0.3 * ram_score + 0.3 * gpu_score, 2)

def display_score(display_size, display_type):
    panel = display_type.lower()

    if "oled" in panel:
        panel_score = 95
    elif "ips" in panel:
        panel_score = 80
    elif "va" in panel:
        panel_score = 70
    else:
        panel_score = 60

    if display_size == 14:
        size_score = 95
    elif 13 <= display_size < 14 or 14 < display_size <= 15.6:
        size_score = 85
    elif display_size < 13:
        size_score = 75
    else:
        size_score = 70

    return round(0.6 * panel_score + 0.4 * size_score, 2)

def price_fairness(actual, predicted):
    diff = actual - predicted
    if diff <= 0:
        return 100
    elif diff <= 5000:
        return 75
    elif diff <= 10000:
        return 50
    else:
        return 25

def laptop_value_index(perf, price, brand, service, display):
    return round(
        0.25 * perf +
        0.25 * price +
        0.15 * brand +
        0.15 * service +
        0.20 * display, 2
    )

#  TITLE 
st.title("💻Benchmark Surgeon")
st.caption("Predict laptop price & evaluate real-world value")

st.divider()

# BRAND
st.subheader("🏷 Brand & Identity")
col1, col2 = st.columns(2)

with col1:
    Brand = st.selectbox("Laptop Brand", df['Brand'].unique())
with col2:
    Processor_Brand = st.selectbox("Processor Brand", df['Processor_Brand'].unique())

st.divider()

# ⚙ CPU & MEMORY
st.subheader("⚙ CPU & Memory")
col_cpu, col_mem = st.columns(2)

with col_cpu:
    Processor_power = st.selectbox(
        "Processor Tier(eg:intel i5/ryzen 5 = 5)",
        sorted(df['Processor_power'].unique())
    )
    Intel_Gen = st.selectbox(
        "Intel Generation (0 for AMD / Other)",
        sorted(df['Intel_Gen'].unique())
    )
    Ghz = st.number_input("Processor Speed (base speed in GHz)", min_value=0.0, step=0.1)

with col_mem:
    RAM = st.selectbox("Installed RAM (GB)", sorted(df['RAM'].unique()))
    RAM_Expandable = st.selectbox("Expandable RAM (GB)", sorted(df['RAM_Expandable'].unique()))
    RAM_TYPE = st.selectbox("RAM Type", df['RAM_TYPE'].unique())

st.divider()

# 🎮 GPU
st.subheader("🎮 Graphics")
GPU = st.selectbox("Graphics Processor", df['GPU'].unique())

st.divider()

# 🖥 DISPLAY
st.subheader("🖥 Display")
col1, col2, col3 = st.columns(3)

with col1:
    Display = st.number_input("Display Size (inches)", 10.0, 18.0, step=0.1)
with col2:
    Display_type = st.selectbox("Display Type", df['Display_type'].unique())
with col3:
    Adapter = st.selectbox("Adapter Power (W)", sorted(df['Adapter'].unique()))

st.divider()

# 💾 STORAGE 
st.subheader("💾 Storage")
col1, col2 = st.columns(2)

with col1:
    SSD = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))
with col2:
    HDD = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))

st.divider()

#  📊 PRICE INPUT 
actual_price = st.number_input(
    "Enter Actual Market Price (₹)",
    min_value=10000,
    step=1000
)

#  🔍 PREDICT
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

    predicted_log_price = pipe.predict(input_df)[0]
    st.session_state.predicted_price = np.exp(predicted_log_price)

#  📋 SCORECARD 
if st.session_state.predicted_price is not None:

    predicted_price = st.session_state.predicted_price
    st.success(f"💰 Estimated Laptop Price: ₹{int(predicted_price):,}")

    brand_info = brand_intelligence.get(Brand, {"reliability": 70, "service": 70})

    perf_score = performance_score(Processor_power, RAM, GPU)
    disp_score = display_score(Display, Display_type)
    price_score = price_fairness(actual_price, predicted_price)
    brand_score = brand_info["reliability"]
    service_score = brand_info["service"]

    final_score = laptop_value_index(
        perf_score,
        price_score,
        brand_score,
        service_score,
        disp_score
    )

    price_diff = actual_price - predicted_price

    st.divider()
    st.subheader("📋 Laptop Value Scorecard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance", f"{perf_score} / 100")
    c2.metric("Display", f"{disp_score} / 100")
    c3.metric("Price Fairness", f"{price_score} / 100")
    c4.metric("Brand Trust", f"{brand_score} / 100")

    st.metric("🔢 Final Laptop Value Score", f"{final_score} / 100")
    st.progress(final_score / 100)
    st.caption(f"💡 Price difference: ₹{int(price_diff):,}")

    st.divider()

    if price_diff <= 0:
        st.success("Strong Buy — Priced below or equal to predicted value")
    elif price_diff <= 3000:
        st.success(" Very close to fair price - Buy now")
    elif price_diff <= 7000:
        if final_score >= 70:
            st.info(" Recommended — Slightly overpriced but strong value")
        else:
            st.warning("⚠ Consider Waiting — Price high for value")
    elif price_diff <= 12000:
        if final_score >= 75:
            st.warning("⚠ Buy Only if Needed — Good laptop, overpriced")
        else:
            st.error(" Not Recommended — Poor value")
    else:
        st.error(" Avoid for Now — Significantly overpriced")
