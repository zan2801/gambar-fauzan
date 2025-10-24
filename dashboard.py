import streamlit as st
import random
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Klasifikasi & Deteksi Objek", page_icon="🚀", layout="wide")

BG_COLOR = "#05091a"     
TEXT_COLOR = "#ffffff"
TEXT_JUGA = "#708090"

# Gaya dasar halaman
st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_JUGA};
            overflow: hidden;
        }}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
        [data-testid="stToolbar"] {{right: 2rem;}}
        @keyframes twinkle {{
            0% {{opacity: 0.3; transform: scale(1);}}
            50% {{opacity: 1; transform: scale(1.4);}}
            100% {{opacity: 0.3; transform: scale(1);}}
        }}
        @keyframes satelliteMove {{
            0%   {{ transform: translateX(0) translateY(0) rotate(0deg); opacity: 0.6; }}
            50%  {{ transform: translateX(30px) translateY(-15px) rotate(20deg); opacity: 1; }}
            100% {{ transform: translateX(60px) translateY(0) rotate(-10deg); opacity: 0.6; }}
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================
# SIMPAN HALAMAN
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
# BINTANG & SATELIT
# ==========================
def draw_stars_and_satellites(num_stars=370, num_satellites=6):
    """Bintang penuh layar + animasi satelit di belakang gambar"""
    star_colors = ["#FFD700", "#FFF8DC", "#B0E0E6", "#F0E68C", "#FFFFFF"]
    html = ""

    # Tambah bintang
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(4, 14)
        opacity = random.uniform(0.3, 1)
        duration = random.uniform(1.5, 4)
        color = random.choice(star_colors)

        html += f"""
            <div style="
                position: fixed;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: {color};
                opacity: {opacity};
                z-index: -1;
                pointer-events: none;
                animation: twinkle {duration}s infinite ease-in-out;
            ">⭐</div>
        """

    # Tambah beberapa satelit kecil di belakang gambar
    for _ in range(num_satellites):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(40, 70)
        duration = random.uniform(6, 12)
        html += f"""
            <div style="
                position: fixed;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: #cfcfcf;
                opacity: 0.7;
                z-index: 0;
                pointer-events: none;
                animation: satelliteMove {duration}s infinite ease-in-out alternate;
            ">🛰️</div>
        """

    st.markdown(html, unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/FauzanAkbar_Laporan4.pt")
    classifier = tf.keras.models.load_model("model/Fauzan Akbar_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    draw_stars_and_satellites()
    header("Klasifikasi & Deteksi Objek", "Klasifikasikan dan deteksi objek dengan mudah, cepat, dan akurat💫")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<h3 style='text-align:center; color:#FFD700;'>🚀Pilih Misi Kamu:</h3>", unsafe_allow_html=True)
        st.write("")
        if st.button("🌑 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
        if st.button("🪐 Deteksi Objek", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()

# ==========================
# HALAMAN KLASIFIKASI
# ==========================
elif st.session_state.page == "classify":
    draw_stars_and_satellites()
    header("🌑 Klasifikasi Gambar", "Unggah gambar untuk melakukan klasifikasi")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        try:
            input_shape = classifier.input_shape[1:3]
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)

            label_map = {0: "Tumor Otak", 1: "Otak Sehat"}
            class_label = label_map.get(class_index, "Tidak Dikenal")

            st.success(f"Hasil Prediksi: {class_label}")
            st.write("Probabilitas:", f"{np.max(prediction):.4f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ==========================
# HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    draw_stars_and_satellites()
    header("🪐 Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        results = yolo_model(img)
        result_img = results[0].plot()

        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
