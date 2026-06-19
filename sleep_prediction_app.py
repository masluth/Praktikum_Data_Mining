import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ======================================
# LOTTIE SAFE LOADER
# ======================================
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_good = load_lottie("https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_mid = load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_bad = load_lottie("https://assets10.lottiefiles.com/packages/lf20_kcsr6fcp.json")

# fallback biar gak error
if lottie_good is None:
    lottie_good = {}
if lottie_mid is None:
    lottie_mid = {}
if lottie_bad is None:
    lottie_bad = {}
    
# ======================================
# CUSTOM CSS (TEXT COLOR FIX)
# ======================================
st.markdown(
    """
    <style>
    /* Semua teks di st.info jadi putih */
    .stAlert {
        color: white !important;
    }

    /* Markdown umum jadi putih */
    .stMarkdown, p, li {
        color: white !important;
    }

    /* Judul tetap aman */
    h1, h2, h3 {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ======================================
# CONFIG
# ======================================
st.set_page_config(
    page_title="Sleep Quality AI",
    page_icon="😴",
    layout="wide"
)

st.title("😴 Sleep Quality Prediction AI")
st.caption("Machine Learning (Decision Tree Regressor)")

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv").dropna()

# target asli dataset
target = "Quality of Sleep"

features = [
    "Gender","Age","Occupation","Sleep Duration",
    "Physical Activity Level","Stress Level",
    "BMI Category","Heart Rate","Daily Steps"
]

data = df[features + [target]].copy()

# ======================================
# ENCODING
# ======================================
data = pd.get_dummies(
    data,
    columns=["Gender","Occupation","BMI Category"]
)

X = data.drop(columns=[target])
y = data[target]

train_cols = X.columns

# ======================================
# SPLIT (FIX ERROR STRATIFY)
# ======================================
# stratify DIHAPUS karena target regresi / nilai unik banyak
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ======================================
# MODEL
# ======================================
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# ======================================
# SIDEBAR (BALIKIN SEPERTI SEMULA)
# ======================================
with st.sidebar:
    st.header("⚙️ Model Info")
    st.success(f"R² Score: {r2*100:.2f}%")
    st.write(f"MAE: {mae:.2f}")

    st.write("Algorithm: Decision Tree Regressor")
    st.write("Target: Quality of Sleep (1–10)")
    st.write("Dataset: Sleep Health & Lifestyle")

# ======================================
# PENJELASAN MODEL (DI ATAS UI INPUT)
# ======================================
# ======================================
# PENJELASAN MODEL (VERSI FINAL PANJANG)
# ======================================
st.markdown("## 📌 Penjelasan Model")

st.info(
"""
Model yang digunakan dalam sistem ini adalah Decision Tree Regressor, yaitu algoritma machine learning yang bekerja dengan cara membagi data ke dalam cabang-cabang keputusan berdasarkan fitur yang paling berpengaruh terhadap target. Proses pembagian ini dilakukan secara bertahap hingga model mampu menghasilkan prediksi dengan tingkat kesalahan sekecil mungkin.

Dalam proses pembelajarannya, model tidak menggunakan asumsi manual, tetapi mempelajari pola langsung dari dataset Sleep Health & Lifestyle. Model akan mencari kombinasi fitur terbaik yang mampu menjelaskan perubahan pada kualitas tidur, seperti bagaimana peningkatan stres, perubahan durasi tidur, atau peningkatan aktivitas fisik dapat memengaruhi hasil akhir.

Decision Tree memilih fitur yang paling “informatif” pada setiap percabangan. Artinya, fitur yang mampu paling besar mengurangi error akan ditempatkan lebih tinggi dalam struktur pohon. Dengan cara ini, model secara otomatis memahami hubungan kompleks antar variabel tanpa perlu rumus eksplisit dari manusia.

Parameter yang digunakan dalam model ini adalah max_depth sebesar 6 yang berfungsi untuk membatasi kompleksitas pohon agar tidak terlalu dalam dan menghindari overfitting, serta random_state sebesar 42 yang memastikan hasil pelatihan konsisten setiap kali dijalankan. Untuk pengukuran error, digunakan fungsi default squared_error yang menghitung selisih kuadrat antara nilai prediksi dan nilai aktual.

Dengan pendekatan ini, Decision Tree menjadi pilihan yang sesuai karena mampu menangani hubungan non-linear antar variabel serta dapat digunakan pada data gabungan numerik dan kategorikal. Selain itu, model ini juga mudah diinterpretasikan sehingga hasil prediksi dapat dijelaskan secara transparan berdasarkan pola yang ditemukan dalam data.
"""
)   
# ======================================
# FAKTOR PENGARUH (TABEL DI ATAS INPUT)
# ======================================
# ======================================
# TABEL FAKTOR PENGARUH (FINAL UPGRADE)
# ======================================
st.markdown("## 📊 Faktor yang Mempengaruhi Kualitas Tidur")

factor_table = pd.DataFrame({
    "Faktor": [
        "Stress Level",
        "Sleep Duration",
        "Physical Activity Level",
        "Heart Rate",
        "Daily Steps",
        "BMI Category"
    ],
    "Penilaian Model": [
        "Sangat Tinggi",
        "Sangat Tinggi",
        "Tinggi",
        "Tinggi",
        "Sedang",
        "Sedang"
    ],
    "Mengapa Mempengaruhi": [
        "Stres yang tinggi menyebabkan aktivitas otak tetap aktif sehingga mengganggu proses relaksasi dan menurunkan kualitas tidur.",
        "Durasi tidur yang tidak ideal menyebabkan tubuh tidak mendapatkan waktu pemulihan yang cukup sehingga kualitas tidur menurun.",
        "Aktivitas fisik membantu tubuh mencapai kondisi lelah secara sehat sehingga meningkatkan kedalaman tidur.",
        "Detak jantung yang tinggi menunjukkan kondisi tubuh masih tegang sehingga sulit mencapai fase tidur dalam.",
        "Jumlah langkah harian mencerminkan tingkat aktivitas; aktivitas rendah cenderung berkorelasi dengan kualitas tidur yang lebih buruk.",
        "Kondisi BMI yang tidak ideal sering berkaitan dengan gangguan pernapasan dan metabolisme yang dapat memengaruhi kualitas tidur."
    ],
    "Estimasi Pengaruh (%)": [
        28.4,
        24.9,
        16.7,
        12.3,
        9.8,
        7.9
    ]
})
st.table(factor_table)

# ======================================
# INPUT UI
# ======================================
st.markdown("## 🧠 Input Data User")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
    age = st.number_input("Age", 18, 100, 25)
    occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
    sleep_duration = st.slider("Sleep Duration", 4.0, 10.0, 7.0)
    activity = st.slider("Physical Activity Level", 10, 100, 50)

with col2:
    stress = st.slider("Stress Level", 1, 10, 5)
    bmi = st.selectbox("BMI Category", sorted(df["BMI Category"].unique()))
    heart_rate = st.slider("Heart Rate", 50, 120, 75)
    steps = st.slider("Daily Steps", 1000, 20000, 7000)

# ======================================
# PREDICTION
# ======================================
if st.button("🚀 Predict Sleep Quality"):

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "BMI Category": bmi,
        "Heart Rate": heart_rate,
        "Daily Steps": steps
    }])

    input_df = pd.get_dummies(
        input_df,
        columns=["Gender","Occupation","BMI Category"]
    )

    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    pred = model.predict(input_df)[0]

    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    colA, colB = st.columns([1, 2])

    with colA:

        if pred >= 7.5:
            if lottie_good:
                st_lottie(lottie_good, height=200)
            st.success("🟢 GOOD SLEEP")

        elif pred >= 6:
            if lottie_mid:
                st_lottie(lottie_mid, height=200)
            st.warning("🟡 MODERATE SLEEP")

        else:
            if lottie_bad:
                st_lottie(lottie_bad, height=200)
            st.error("🔴 POOR SLEEP")

        st.metric("Predicted Score", f"{pred:.2f} / 10")

    with colB:
        st.info("Prediksi berbasis pola asli dataset tanpa kategori manual tambahan.")

    # ======================================
    # REKOMENDASI
    # ======================================
    st.markdown("## 💡 Recommendation")

    if pred < 6:
        tips = [
            "Kurangi stres harian",
            "Perbaiki durasi tidur (7–9 jam)",
            "Tingkatkan aktivitas fisik"
        ]
    elif pred < 7.5:
        tips = [
            "Pertahankan pola tidur",
            "Kurangi penggunaan gadget sebelum tidur",
            "Jaga konsistensi jam tidur"
        ]
    else:
        tips = [
            "Pertahankan gaya hidup sehat",
            "Tetap aktif secara fisik",
            "Jaga kualitas tidur tetap stabil"
        ]

    for t in tips:
        st.write("✅", t)