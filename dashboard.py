import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

# ===============================
# Fungsi untuk Membuat Langit Berbintang
# ===============================
def draw_starry_sky(num_stars=150):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#020b26")  # Warna langit malam (gelap kebiruan)
    x = np.random.rand(num_stars)
    y = np.random.rand(num_stars)
    sizes = np.random.randint(10, 80, size=num_stars)
    colors = np.random.choice(["white", "#A9DFFF", "#FFDAB9", "#B0E0E6"], size=num_stars)
    ax.scatter(x, y, s=sizes, color=colors, alpha=0.8)
    ax.axis("off")
    st.pyplot(fig)

# ===============================
# Navigasi Halaman
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# Tombol navigasi balik
def go_home():
    st.session_state.page = "home"

# ===============================
# Halaman Utama
# ===============================
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center; color: white;'>🪐 SpaceVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #B0E0E6;'>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa.<br>Pilih misimu di bawah ini 🚀</p>", unsafe_allow_html=True)
    
    draw_starry_sky(200)

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        pass
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🌌 Klasifikasi Gambar"):
                st.session_state.page = "klasifikasi"
        with b2:
            if st.button("🛰️ Deteksi Objek"):
                st.session_state.page = "deteksi"
    with col3:
        pass

# ===============================
# Halaman Klasifikasi
# ===============================
elif st.session_state.page == "klasifikasi":
    st.markdown("<h1 style='text-align: center; color: white;'>🌌 Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    draw_starry_sky(120)
    uploaded_file = st.file_uploader("🪐 Pilih gambar dari komputermu", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang dipilih", use_column_width=True)
        st.success("Gambar berhasil diunggah! 🚀")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        go_home()

# ===============================
# Halaman Deteksi
# ===============================
elif st.session_state.page == "deteksi":
    st.markdown("<h1 style='text-align: center; color: white;'>🛰️ Deteksi Objek</h1>", unsafe_allow_html=True)
    draw_starry_sky(120)
    uploaded_file = st.file_uploader("🚀 Pilih gambar dari komputermu", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang dipilih", use_column_width=True)
        st.info("Deteksi sedang diproses... ✨ (tapi ini simulasi dulu 😄)")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        go_home()
