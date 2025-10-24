import streamlit as st
import random

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

# Warna tema
PRIMARY_COLOR = "lightskyblue"
BG_COLOR = "black"
TEXT_COLOR = "white"

# Simpan state halaman
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# FUNGSI HEADER
# ==========================
def header(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.write(subtitle)
    st.write("")

# ==========================
# FUNGSI MENAMPILKAN BINTANG
# ==========================
def draw_stars(n=80):
    """Menampilkan simbol bintang di posisi acak menggunakan kolom"""
    rows = []
    for _ in range(n):
        row = ["⭐" if random.random() < 0.03 else "" for _ in range(50)]
        rows.append("".join(row))
    st.text("\n".join(rows))

# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    # Background sederhana
    st.markdown(f"<div style='background-color:{BG_COLOR};height:100vh'></div>", unsafe_allow_html=True)

    # Bintang di latar
    draw_stars(60)

    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR};'>🪐 SpaceVision AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR};'>Jelajahi kecerdasan buatan di galaksi luar angkasa 🚀</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.empty()
    with col2:
        if st.button("🧠 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
        if st.button("🛰️ Deteksi Objek", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()
    with col3:
        st.empty()

# ==========================
# HALAMAN KLASIFIKASI
# ==========================
elif st.session_state.page == "classify":
    draw_stars(40)
    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR};'>🧠 Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR};'>Unggah gambar untuk diidentifikasi model AI kamu</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ==========================
# HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    draw_stars(40)
    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR};'>🛰️ Deteksi Objek</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR};'>Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
