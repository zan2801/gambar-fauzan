import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ==============================
# KONFIGURASI DASAR
# ==============================
st.set_page_config(page_title="🌌 Klasifikasi & Deteksi Objek", layout="wide")

# ==============================
# GAYA DAN TEMA LUAR ANGKASA
# ==============================
st.markdown("""
    <style>
        body {
            background: radial-gradient(circle at top left, #0b0c2a, #1c2350);
            color: white;
        }
        .title {
            text-align: center;
            color: #ffffff;
            font-size: 40px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #dddddd;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 12px;
            height: 60px;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #60a5fa;
            transform: scale(1.05);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
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
# NAVIGASI HALAMAN
# ==============================
menu = st.sidebar.radio(
    "🧭 Navigasi",
    ["🏠 Halaman Utama", "🛸 Deteksi Objek", "🪐 Klasifikasi Gambar"]
)

# ==============================
# HALAMAN UTAMA
# ==============================
if menu == "🏠 Halaman Utama":
    st.markdown("<h1 class='title'>🌠 Gambar & Deteksi Objek</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Pilih misi eksplorasi AI di galaksi luar angkasa 🚀</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.ibb.co/hLhL5Q7/astronaut.png", use_container_width=True)

    with col2:
        st.markdown("### Pilih Mode:")
        if st.button("🪐 Klasifikasi Gambar"):
            st.session_state["page"] = "klasifikasi"
        if st.button("🛸 Deteksi Objek"):
            st.session_state["page"] = "deteksi"

    if "page" in st.session_state:
        if st.session_state["page"] == "klasifikasi":
            menu = "🪐 Klasifikasi Gambar"
        elif st.session_state["page"] == "deteksi":
            menu = "🛸 Deteksi Objek"

# ==============================
# HALAMAN DETEKSI OBJEK
# ==============================
if menu == "🛸 Deteksi Objek":
    st.markdown("<h1 class='title'>🛸 Deteksi Objek</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Unggah gambar dan biarkan AI mendeteksi objek seperti penjelajah kosmik!</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("🚀 Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="🪞 Gambar Asli", use_container_width=True)

        with st.spinner("Mendeteksi objek... 🪐"):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="🔍 Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

# ==============================
# HALAMAN KLASIFIKASI GAMBAR
# ==============================
if menu == "🪐 Klasifikasi Gambar":
    st.markdown("<h1 class='title'>🪐 Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Unggah gambar dan biarkan AI menentukan kelasnya di antara bintang-bintang ✨</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("🌌 Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="🪞 Gambar Asli", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("Mengklasifikasikan... 🌠"):
            pred = clf_model.predict(img_array)
            class_idx = np.argmax(pred)
            prob = np.max(pred)

        st.success(f"🌌 Hasil Prediksi: **Kelas {class_idx}** (Probabilitas: {prob:.2f})")
