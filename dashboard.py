import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

BG_COLOR = "#05091a"
TEXT_COLOR = "#ffffff"
TEXT_SUB = "#708090"
TITLE_MODE_COLOR = "#00BFFF"

# Styling dasar + animasi bintang
st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_SUB};
            overflow: hidden;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
        }}
        @keyframes twinkle {{
            0% {{ opacity: 0.3; transform: scale(1); }}
            50% {{ opacity: 1; transform: scale(1.3); }}
            100% {{ opacity: 0.3; transform: scale(1); }}
        }}
    </style>
""", unsafe_allow_html=True)


# ==========================
# BINTANG LATAR
# ==========================
def draw_stars(num_stars=400):
    star_colors = ["#FFD700", "#FFF8DC", "#B0E0E6", "#F0E68C", "#FFFFFF"]
    stars_html = "<div style='position:fixed; inset:0; z-index:0; pointer-events:none;'>"
    for _ in range(num_stars):
        left = random.uniform(0, 100)
        top = random.uniform(0, 100)
        size = random.randint(4, 14)
        opacity = round(random.uniform(0.25, 0.95), 2)
        duration = round(random.uniform(1.8, 4.0), 2)
        color = random.choice(star_colors)
        stars_html += (
            f"<div style='position:absolute; left:{left}%; top:{top}%; "
            f"font-size:{size}px; color:{color}; opacity:{opacity}; "
            f"animation:twinkle {duration}s infinite ease-in-out;'>⭐</div>"
        )
    stars_html += "</div>"
    st.markdown(stars_html, unsafe_allow_html=True)


# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/FauzanAkbar_Laporan4.pt")
    classifier = tf.keras.models.load_model("model/Fauzan Akbar_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()


# ==========================
# NAVIGASI
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "main"


# ==========================
# HEADER
# ==========================
def header(title, subtitle=""):
    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR};'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR}; font-size:18px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.write("")


# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    draw_stars()
    header("🪐 SpaceVision AI", "Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f"<h3 style='text-align:center; color:{TITLE_MODE_COLOR};'>Pilih Misi Kamu:</h3>", unsafe_allow_html=True)
        st.write("")
        if st.button("🧠 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
        if st.button("🛰️ Deteksi Objek", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()


# ==========================
# HALAMAN KLASIFIKASI
# ==========================
elif st.session_state.page == "classify":
    draw_stars()
    header("🧠 Klasifikasi Gambar", "Unggah gambar untuk diidentifikasi oleh model AI kamu")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        try:
            # otomatis sesuaikan ukuran input model
            input_shape = classifier.input_shape[1:3]
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            st.success(f"Hasil Prediksi: {class_index}")
            st.write("Probabilitas:", float(np.max(prediction)))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()


# ==========================
# HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    draw_stars()
    header("🛰️ Deteksi Objek", "Unggah gambar untuk mendeteksi objek menggunakan YOLO")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        try:
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal melakukan deteksi: {e}")

    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
