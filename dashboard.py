import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ==============================
# KONFIGURASI DASAR
# ==============================
st.set_page_config(page_title="🌌 SpaceVision AI", layout="wide")

# ==============================
# GAYA LUAR ANGKASA (BIRU-HITAM-PUTIH)
# ==============================
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0a0f24 20%, #001233 100%);
            color: white;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            text-align: center;
        }
        .desc {
            text-align: center;
            color: #dbeafe;
            font-size: 18px;
        }
        .button-box {
            display: flex;
            justify-content: center;
            gap: 50px;
            margin-top: 40px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            color: white;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
            padding: 15px 40px;
            border: none;
            transition: 0.3s;
            box-shadow: 0 0 20px rgba(37,99,235,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.08);
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            box-shadow: 0 0 30px rgba(96,165,250,0.7);
        }
        .back-button>button {
            background: none;
            color: #93c5fd;
            border: 1px solid #2563eb;
            padding: 8px 20px;
            border-radius: 10px;
        }
        .back-button>button:hover {
            background-color: #1e3a8a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# MUAT MODEL
# ==============================
@st.cache_resource
def load_yolo():
    return YOLO("model/FauzanAkbar_Laporan4.pt")

@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model("model/Fauzan Akbar_Laporan 2.h5")

yolo_model = load_yolo()
clf_model = load_classifier()

# ==============================
# SISTEM HALAMAN
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(page):
    st.session_state.page = page

# ==============================
# HALAMAN UTAMA
# ==============================
if st.session_state.page == "home":
    st.markdown("<h1>🌌 SpaceVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='desc'>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa.<br>Pilih misi eksplorasimu di bawah ini 🚀</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🪐 Klasifikasi Gambar"):
            goto("classify")
    with col2:
        if st.button("🛸 Deteksi Objek"):
            goto("detect")

# ==============================
# HALAMAN KLASIFIKASI GAMBAR
# ==============================
elif st.session_state.page == "classify":
    st.markdown("<h1>🪐 Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    st.markdown("<p class='desc'>Unggah gambar dan biarkan AI menentukan kelasnya di antara bintang-bintang ✨</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar Asli", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🚀 Mengklasifikasikan..."):
            pred = clf_model.predict(img_array)
            class_idx = np.argmax(pred)
            prob = np.max(pred)

        st.success(f"🌌 Hasil Prediksi: **Kelas {class_idx}** (Probabilitas: {prob:.2f})")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Halaman Utama", key="back1"):
        goto("home")

# ==============================
# HALAMAN DETEKSI OBJEK
# ==============================
elif st.session_state.page == "detect":
    st.markdown("<h1>🛸 Deteksi Objek</h1>", unsafe_allow_html=True)
    st.markdown("<p class='desc'>Unggah gambar dan biarkan AI menemukan objek di dalamnya 👁️‍🗨️</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar Asli", use_container_width=True)

        with st.spinner("🛰️ Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Halaman Utama", key="back2"):
        goto("home")
