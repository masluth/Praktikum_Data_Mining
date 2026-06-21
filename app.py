import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score
)

st.markdown("""
<style>

[data-testid="stMetric"]{
    background-color:#111827;
    border:1px solid #374151;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
}

[data-testid="stMetricValue"]{
    font-size:42px;
    font-weight:bold;
}

[data-testid="stMetricLabel"]{
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

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
    page_title="TidurFit",
    page_icon="logo_python.png",
    layout="wide"
)

# ======================================
# SPLASH SCREEN
# ======================================

if "splash_done" not in st.session_state:

    splash = st.empty()

    with splash.container():

        st.markdown("<br><br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2,1,2])

        with col2:
            st.image(
                "logo_app.png",
                width=180
            )

        st.markdown(
            """
            <h1 style='text-align:center;'>
            TidurFit
            </h1>

            <h4 style='text-align:center;'>
            Integrated Health AI System
            </h4>
            """,
            unsafe_allow_html=True
        )

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

    time.sleep(2)

    st.session_state["splash_done"] = True

    st.rerun()

col1, col2, col3 = st.columns([1,2,1])

with col2:
    
    st.markdown(
        """
        <h1 style='text-align:center; margin-bottom:0px;'>
        TidurFit
        </h1>

        <h4 style='text-align:center; margin-top:0px;'>
        Integrated Health AI System
        </h4>
        """,
        unsafe_allow_html=True
    )
    
if "bmi_category" not in st.session_state:
    st.session_state["bmi_category"] = "Normal"

if "gender" not in st.session_state:
    st.session_state["gender"] = "Female"

if "age" not in st.session_state:
    st.session_state["age"] = 21
    
tab1, tab2, tab3 = st.tabs(
    [
        "Analisis Kondisi Berat Badan",
        "Analisis Kualitas Tidur",
        "Tentang Sistem"
    ]
)
# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("Sleep_Data_Sampled.csv").dropna()

# target asli dataset
target = "Quality of Sleep"

features = [
    "Gender","Age","Occupation","Sleep Duration",
    "Physical Activity Level","Stress Level",
    "BMI Category","Heart Rate","Daily Steps"
]

data = df[features + [target]].copy()

obesity_df = pd.read_csv(
    "ObesityDataSet_raw_and_data_sinthetic.csv"
)

# ======================================
# MODEL OBESITY
# ======================================

obesity_features = [
    "Gender",
    "Age",
    "Height",
    "Weight"
]

X_obesity = obesity_df[obesity_features]

y_obesity = obesity_df["NObeyesdad"]

# encoding gender
X_obesity = pd.get_dummies(
    X_obesity,
    columns=["Gender"]
)

# simpan nama kolom
obesity_train_cols = X_obesity.columns

# split data
Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    X_obesity,
    y_obesity,
    test_size=0.2,
    random_state=42
)

# model classifier
obesity_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

obesity_model.fit(
    Xo_train,
    yo_train
)

# ======================================
# EVALUASI MODEL OBESITY
# ======================================

obesity_pred = obesity_model.predict(
    Xo_test
)

obesity_acc = accuracy_score(
    yo_test,
    obesity_pred
)
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
train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)

# ======================================
# FEATURE IMPORTANCE
# ======================================

importance_df = pd.DataFrame({
    "Faktor-Faktor": X.columns,
    "Persentase Nilai": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Persentase Nilai",
    ascending=False
)

# ======================================
# SIDEBAR (BALIKIN SEPERTI SEMULA)
# ======================================
with st.sidebar:
    st.header("Informasi Mengenai Model Yang Digunakan")

    st.success(f"Dataset Size: {len(df)} Rows")
    st.success(f"Train R²: {train_r2*100:.2f}%")
    st.success(f"Test R²: {r2*100:.2f}%")

    st.success(f"MAE: {mae:.2f}")
    
    st.success(
    f"Obesity Accuracy: {obesity_acc*100:.2f}%"
)
with tab2:


    st.info(
    """
    Hasil Analisis Kondisi Berat Badan akan digunakan secara otomatis sebagai salah satu faktor dalam prediksi kualitas tidur untuk menghasilkan analisis yang lebih akurat dan komprehensif.
    """    )  
    # ======================================
    # INPUT USER
    # ======================================
    st.markdown("Input Dulu Ya")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.session_state.get(
     "gender",
     "Female"
     )

        age = st.session_state.get(
         "age",
          21
         )

        st.text_input(
        "Gender",
        value=gender,
        disabled=True
        )

        st.text_input(
            "Age",
            value=str(age),
            disabled=True
            )
    
        occupation = st.selectbox(
            "Occupation",
            sorted(df["Occupation"].unique())
        )

        sleep_duration = st.slider(
            "Sleep Duration",
            4.0,
            10.0,
            7.0
        )

        activity = st.slider(
            "Physical Activity Level",
            10,
            100,
            50
        )

    with col2:

        stress = st.slider(
            "Stress Level",
            1,
            10,
            5
        )

        bmi_category = st.session_state.get(
        "bmi_category",
        "Normal"
        )

        st.text_input(
        "BMI Category",
        value=bmi_category,
        disabled=True
        )

        heart_rate = st.slider(
            "Heart Rate",
            50,
            120,
            75
        )

        steps = st.slider(
            "Daily Steps",
            1000,
            20000,
            7000
        )

    # ======================================
    # PREDIKSI
    # ======================================
    if st.button("Ayo Liat Score Kalian Gais :v"):

        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Physical Activity Level": activity,
            "Stress Level": stress,
            "BMI Category": bmi_category,
            "Heart Rate": heart_rate,
            "Daily Steps": steps
        }])

        input_df = pd.get_dummies(
        input_df,
        columns=[
        "Gender",
        "Occupation",
        "BMI Category"
        ]
    
        )
        input_df = input_df.reindex(
        columns=train_cols,
        fill_value=0
        )
        
        pred = model.predict(input_df)[0]

        st.markdown("---")
        st.markdown("Ini Score Kalian Yaww :)")

        colA, colB = st.columns([1,2])

        with colA:

            if pred >= 7.5:

                if lottie_good:
                    st_lottie(
                        lottie_good,
                        height=200
                    )

                st.success(
                    "Bagus Di Pertahanin Ya Gais :>"
                )

            elif pred >= 6:

                if lottie_mid:
                    st_lottie(
                        lottie_mid,
                        height=200
                    )

                st.warning(
                    "Masih Bagus Si, Cuman Diperbaiki Lagi Yaw :v"
                )

            else:

                if lottie_bad:
                    st_lottie(
                        lottie_bad,
                        height=200
                    )

                st.error(
                    "Tidurnya Diatur Lagi Yaw :("
                )

            st.metric(
                "Predicted Score",
                f"{pred:.2f} / 10"
            )

        with colB:

            st.info(
                "Prediksi ini dihitung berdasarkan dataset yang kami gunakan."
            )
            
            if pred >= 8:
                st.success(
                 "Kualitas tidur Anda sangat baik."
            )

            elif pred >= 7:
                st.info(
                "Kualitas tidur Anda tergolong baik."
            )

            elif pred >= 6:
                st.warning(
                "Kualitas tidur Anda cukup baik namun masih dapat ditingkatkan."
            )

            else:
                st.error(
                "Kualitas tidur Anda tergolong rendah dan memerlukan perhatian lebih."
            )

        st.subheader("Ini Rekomendasi Dari Akuu yaaa")

        tips = []

            # Analisis Durasi Tidur
        if sleep_duration < 7:
            tips.append(
        "Durasi tidur Anda masih kurang. Disarankan tidur 7–9 jam per malam agar tubuh mendapatkan waktu pemulihan yang optimal."
    )

# Analisis Stress
        if stress >= 7:
            tips.append(
        "Tingkat stres Anda cukup tinggi. Cobalah melakukan relaksasi, olahraga ringan, atau mengurangi aktivitas yang memicu stres sebelum tidur."
    )

# Analisis Aktivitas Fisik
        if activity < 40:
            tips.append(
        "Aktivitas fisik harian masih rendah. Menambah aktivitas fisik dapat membantu meningkatkan kualitas tidur dan kesehatan tubuh."
    )

# Analisis Daily Steps
        if steps < 5000:
            tips.append(
        "Jumlah langkah harian masih rendah. Usahakan berjalan lebih aktif dan mencapai minimal 6.000–8.000 langkah per hari."
    )

# Analisis Heart Rate
        if heart_rate > 90:
            tips.append(
        "Detak jantung relatif tinggi. Pastikan tubuh mendapatkan istirahat yang cukup dan hindari konsumsi kafein berlebihan sebelum tidur."
    )

# Analisis BMI
        if bmi_category == "Obese":
            tips.append(
        "Kondisi berat badan berlebih dapat meningkatkan risiko gangguan tidur. Menjaga pola makan dan aktivitas fisik dapat membantu meningkatkan kualitas tidur."
    )

        elif bmi_category == "Underweight":
            tips.append(
        "Pastikan kebutuhan nutrisi harian terpenuhi agar tubuh memiliki energi yang cukup untuk mendukung pola tidur yang sehat."
    )

# Jika semua bagus
        if len(tips) == 0:

            if pred >= 8:

                tips.append(
                "Kualitas tidur Anda sangat baik. Pertahankan pola hidup sehat yang saat ini sudah berjalan dengan baik."
                )

            elif pred >= 7:

                tips.append(
                "Kualitas tidur Anda tergolong baik. Tetap pertahankan kebiasaan positif agar kualitas tidur tetap stabil."
                )

            else:

                tips.append(
                "Meskipun tidak ditemukan faktor risiko yang menonjol, kualitas tidur Anda masih dapat ditingkatkan dengan menjaga konsistensi jam tidur dan pola hidup sehat."
                )

    # Tampilkan hasil
        for t in tips:
            st.write("✅", t)
            
    st.markdown("""
    <h3 style='margin-top:20px;'>
    Hasil Evaluasi Model
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    col1.metric(
    "Dataset Size",
    "15,000"
    )

    col2.metric(
    "Train R²",
    f"{train_r2*100:.2f}%"
    )

    col3.metric(
    "Test R²",
    f"{r2*100:.2f}%"
    )

    # ======================================
    # TABEL FAKTOR
    # ======================================
    st.markdown("### Faktor-Faktor Yang Mempengaruhi Kualitas Tidur")

    st.dataframe(importance_df)
      
with tab1:

    st.header("Alat untuk mengukur BMI")
    
    st.info(
    """
    Modul ini menggunakan algoritma Decision Tree Classifier yang dilatih menggunakan Obesity Dataset untuk memprediksi kategori kondisi berat badan pengguna berdasarkan data Gender, Age, Height, dan Weight.

    Hasil prediksi akan dikonversi menjadi BMI Category dan digunakan secara otomatis sebagai salah satu faktor dalam analisis kualitas tidur pada modul berikutnya.
    """
    )

    gender_bmi = st.selectbox(
    "Gender",
    ["Male", "Female"]
    )

    age_bmi = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=21
    )
    
    height_cm = st.number_input(
    "Height (cm)",
    100,
    250,
    170
    )

    weight = st.number_input(
        "Weight (kg)",
        20.0,
        300.0,
        70.0
    )

    if st.button("Analisis dulu yaww"):

   # ======================================
# PREDIKSI MODEL OBESITY
# ======================================

        input_obesity = pd.DataFrame([{
            "Gender": gender_bmi,
            "Age": age_bmi,
            "Height": height_cm / 100,
            "Weight": weight
            }])

        input_obesity = pd.get_dummies(
            input_obesity,
            columns=["Gender"]
            )

        input_obesity = input_obesity.reindex(
            columns=obesity_train_cols,
            fill_value=0
            )

        obesity_result = obesity_model.predict(
            input_obesity
            )[0]
        
        mapping = {
        "Insufficient_Weight": "Underweight",
        "Normal_Weight": "Normal",
        "Overweight_Level_I": "Overweight",
        "Overweight_Level_II": "Overweight",
        "Obesity_Type_I": "Obese",
        "Obesity_Type_II": "Obese",
        "Obesity_Type_III": "Obese"
        }

        bmi_category = mapping[obesity_result]
        
    # tampilkan
        st.markdown("### Hasil Analisis Berat Badan")

        st.metric(
        "Predicted Category",
        obesity_result.replace("_", " ")
        )

        st.metric(
        "BMI Category",
        bmi_category
        )

        st.metric(
        "Model Accuracy",
        f"{obesity_acc*100:.2f}%"
        )

        st.success(
        f"Model memprediksi kondisi berat badan Anda sebagai {obesity_result.replace('_', ' ')}."
        )
        st.info(
        "Hasil kategori BMI ini akan digunakan secara otomatis sebagai salah satu faktor dalam analisis kualitas tidur pada tab berikutnya."
)
        st.session_state["gender"] = gender_bmi
        st.session_state["age"] = age_bmi
        st.session_state["bmi_category"] = bmi_category
        st.session_state["bmi_ready"] = True
        
        
with tab3:

    st.header("Tentang Sistem")

    st.info(
    """
    Integrated Health AI System merupakan aplikasi berbasis Machine Learning yang dirancang untuk membantu pengguna memahami kondisi kesehatan berdasarkan kondisi fisik dan pola hidup sehari-hari.

    Sistem ini terdiri dari dua modul utama, yaitu Analisis Kondisi Berat Badan dan Analisis Kualitas Tidur. Kedua modul tersebut menggunakan dataset dan model Machine Learning yang berbeda, namun saling terhubung dalam satu alur kerja yang terintegrasi.

    Pada modul pertama, sistem menggunakan algoritma Decision Tree Classifier untuk menganalisis kondisi berat badan pengguna berdasarkan data Gender, Age, Height, dan Weight. Hasil analisis tersebut menghasilkan kategori kondisi berat badan yang kemudian dikonversi menjadi BMI Category sebagai representasi kondisi fisik pengguna.

    Selanjutnya, BMI Category digunakan secara otomatis sebagai salah satu faktor pada modul Analisis Kualitas Tidur. Modul kedua menggunakan algoritma Decision Tree Regressor untuk memprediksi Quality of Sleep berdasarkan berbagai faktor, seperti Sleep Duration, Stress Level, Physical Activity Level, Heart Rate, Daily Steps, Occupation, serta BMI Category.

    Dengan memanfaatkan berbagai faktor yang berkaitan dengan kondisi fisik dan pola hidup pengguna, sistem dapat memberikan analisis kualitas tidur yang lebih komprehensif serta membantu pengguna memahami faktor-faktor yang mempengaruhi kualitas tidur mereka.    """
    )
    
    st.subheader("Latar Belakang Sistem")

    st.write(
    """
    Kualitas tidur merupakan salah satu faktor penting yang mempengaruhi kesehatan, produktivitas, dan kualitas hidup seseorang. Namun, kualitas tidur dipengaruhi oleh berbagai faktor, seperti durasi tidur, tingkat aktivitas fisik, tingkat stres, kondisi fisik, serta kebiasaan hidup sehari-hari.

    Perkembangan teknologi Machine Learning memungkinkan proses analisis berbagai faktor tersebut dilakukan secara lebih cepat dan efektif. Dengan memanfaatkan data yang relevan, sistem dapat membantu pengguna memahami kondisi kesehatan serta memperoleh gambaran mengenai kualitas tidur yang dimiliki.

    Oleh karena itu, Integrated Health AI System dikembangkan untuk mengintegrasikan analisis kondisi berat badan dan analisis kualitas tidur dalam satu platform. Sistem ini memanfaatkan model Machine Learning untuk memberikan informasi yang lebih komprehensif berdasarkan kondisi fisik dan pola hidup pengguna.
    """
    )
    
    st.subheader("Tujuan Sistem")

    st.markdown("""
    - Membantu pengguna mengetahui kondisi berat badan berdasarkan data fisik yang dimasukkan.
    - Membantu pengguna memprediksi kualitas tidur berdasarkan berbagai faktor seperti durasi tidur, tingkat stres, aktivitas fisik, detak jantung, jumlah langkah harian, serta kondisi fisik pengguna.
    - Mengintegrasikan analisis kondisi berat badan dan kualitas tidur dalam satu sistem yang saling terhubung.
    - Memberikan informasi yang mudah dipahami sebagai bahan evaluasi kesehatan pengguna.
    """)

    st.subheader("Dataset yang Digunakan")

    st.write("• Obesity Dataset")
    st.write(
    "Digunakan untuk melatih model klasifikasi kondisi berat badan menggunakan algoritma Decision Tree Classifier."
)
    st.write("• Sleep Data Sampled Dataset")
    st.write(
    """
    Digunakan untuk melatih model prediksi kualitas tidur menggunakan algoritma Decision Tree Regressor. Dataset ini berisi berbagai faktor yang mempengaruhi kualitas tidur, seperti Occupation, Sleep Duration, Physical Activity Level, Stress Level, Heart Rate, Daily Steps, serta BMI Category.
    """
)

    st.subheader("Algoritma yang Digunakan")

    st.write("• Decision Tree Classifier")
    st.write(
    "Digunakan untuk memprediksi kategori kondisi berat badan pengguna."
)
    st.write("• Decision Tree Regressor")
    st.write(
    "Digunakan untuk memprediksi nilai Quality of Sleep dengan skala 1 sampai 10."
)

    st.subheader("Alur Sistem")

    st.code(
    """
Input Data Fisik
↓
Analisis Berat Badan
↓
BMI Category
↓
Analisis Kualitas Tidur
↓
Prediksi Quality of Sleep
    """
    )

    st.markdown("---")

    st.subheader("Profil Developer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:

        st.image(
        "logo_python.png",
        width=180
    )

        st.markdown("### Nama Developer 1")

        st.write("NIM : XXXXX")
        st.write("Kelas : I231D")
    
    with col2:

        st.image(
        "logo_python.png",
        width=180
    )

        st.markdown("### Nama Developer 2")

        st.write("NIM : XXXXX")
        st.write("Kelas : I231D")
    
    with col3:

        st.image(
        "logo_python.png",
        width=180
    )

        st.markdown("### Nama Developer 3")

        st.write("NIM : XXXXX")
        st.write("Kelas : I231D")
