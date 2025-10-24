import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import random

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

BG_COLOR = "#05091a"      # warna background
TEXT_COLOR = "#ffffff"    # warna teks umum
SIDEBAR_TITLE_COLOR = "#00BFFF"  # warna teks "Pilih Mode"

# Styling background & bintang
st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_COLOR};
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
    """,
    unsafe_allow_html=True,
)

# Fungsi bintang
def draw_stars(num_stars=300):
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
    yolo_model = YOLO("model/FauzanAkbar_Laporan4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Fauzan Akbar_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
draw_stars()
st.title("🪐 SpaceVision AI")

st.sidebar.markdown(
    f"<h3 style='color:{SIDEBAR_TITLE_COLOR};'>Pilih Mode:</h3>",
    unsafe_allow_html=True
)
menu = st.sidebar.selectbox("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))
