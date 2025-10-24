import streamlit as st
import random

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="SpaceVision AI", layout="centered")

# ===============================
# FUNGSI LATAR BELAKANG BINTANG
# ===============================
def draw_stars(num_stars=100):
    """Buat latar belakang bintang di layer belakang"""
    stars_html = """
    <div style="
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        overflow: hidden;
        z-index: 0;
    ">
    """
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(6, 14)
        opacity = random.uniform(0.3, 0.9)
        stars_html += f"""
        <span style="
            position: absolute;
            left: {left}%;
            top: {top}%;
            font-size: {size}px;
            color: gold;
            opacity: {opacity};
            pointer-events: none;
        ">⭐</span>
        """
    stars_html += "</div>"
    st.markdown(stars_html, unsafe_allow_html=True)


# ===============================
# FUNGSI GAYA UTAMA
# ===============================
def set_background():
    st.markdown(
        """
        <style>
        body {
            background-color: #03091E;
            color: white;
        }
        .main {
            background-color: transparent;
        }
        h1, h2, h3 {
            text-align: center;
            color: white;
        }
        .space-button {
            display: block;
            width: 300px;
            margin: 20px auto;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 15px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            background-color: #0B1E42;
            color: white;
        }
        .space-button:hover {
            background-color: #1A3B73;
            transform: scale(1.05);
        }
        @keyframes twinkle {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        span {
            animation: twinkle 2s infinite ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===============================
# FUNGSI HALAMAN
# ===============================
def main_page():
    st.markdown("<h1>🪐 SpaceVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-top:40px;'>Pilih Misi Kamu:</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Klasifikasi Gambar", key="classify", use_container_width=True):
            st.session_state.page = "klasifikasi"
    with col2:
        if st.button("🛰️ Deteksi Objek", key="detect", use_container_width=True):
            st.session_state.page = "deteksi"


def klasifikasi_page():
    st.markdown("<h1>🧠 Misi Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    st.write("Unggah gambar untuk mengenali objek di dalamnya.")
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "utama"


def deteksi_page():
    st.markdown("<h1>🛰️ Misi Deteksi Objek</h1>", unsafe_allow_html=True)
    st.write("Unggah gambar untuk mendeteksi objek di dalamnya.")
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "utama"


# ===============================
# SISTEM NAVIGASI
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "utama"

set_background()
draw_stars(120)

if st.session_state.page == "utama":
    main_page()
elif st.session_state.page == "klasifikasi":
    klasifikasi_page()
elif st.session_state.page == "deteksi":
    deteksi_page()
