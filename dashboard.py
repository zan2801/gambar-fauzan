import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Image Intelligence App",
    page_icon="🧠",
    layout="centered"
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/FauzanAkbar_Laporan4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Fauzan Akbar_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header
# ==========================
st.markdown("""
<h1 style='text-align:center;'>🧠 Image Intelligence App</h1>
<p style='text-align:center; color:gray;'>
Dibuat oleh <b>Fauzan Akbar</b> — Deteksi Objek & Klasifikasi Gambar berbasis AI
</p>
<hr style='border: 1px solid #e0e0e0;'>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar dan pilih mode analisis di atas 👆")

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)
    st.write("---")

    if menu == "Deteksi Objek (YOLO)":
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="🧾 Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")

    elif menu == "Klasifikasi Gambar":
        with st.spinner("🧠 Menganalisis gambar..."):
            img_resized = img.resize((224, 224))  # sesuaikan dengan input model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

        st.subheader("🔎 Hasil Prediksi")
        st.metric(label="Kelas Prediksi", value=f"{class_index}")
        st.progress(confidence)
        st.caption(f"Probabilitas: **{confidence:.2%}**")

        st.success("🎉 Analisis selesai!")

else:
    st.info("👆 Unggah gambar untuk memulai analisis.")

# ==========================
# Footer
# ==========================
st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>✨ Powered by TensorFlow & YOLO — Streamlit Edition</p>",
    unsafe_allow_html=True
)
