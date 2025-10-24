import streamlit as st
import random

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="SpaceVision AI", layout="centered")

# ===============================
# FUNGSI LATAR BELAKANG BINTANG
# ===============================
def draw_stars(num_stars=120):
    """Buat layer bintang di latar belakang (tidak menutupi elemen utama)."""
    stars_html = """
    <div style="
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    ">
    """
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(6, 14)
        opacity = random.uniform(0.3, 0.8)
        stars_html += f"""
        <span style="
            position: absolute;
            left: {left}%;
            top: {top}%;
            font-size: {size}px;
            color: gold;
            opacity: {opacity};
            animation: twinkle {random.uniform(1.5, 3)}s infinite ease-in-out;
        ">⭐</span>
        """
    stars_html += "</div>"

    st.markdown(stars_html, unsafe_allow_html=True)


# ===============================
# FUNGSI GAYA GLOBAL
# ===============================
def set_style():
    st.markdown("""
        <style>
        body {
            background-color: #03091E;
            color: white;
        }
        .main {
            background-color: transparent;
        }
        h1, h2, h3, p {
            text-align: center;
            color: white;
        }
        @keyframes twinkle {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        .stButton button {
            display: block;
            margin: 10px auto;
            width: 250px;
            background-color: #0B1E42;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1A3B73;
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)


# ===============================
# HALAMAN
# ===============================
def main_page():
    st.markdown("<h1>🪐 SpaceVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀</h3>", unsafe_allow_html=True)
    st.markdown("<br><h3>Pilih Misi Kamu:</h3>", unsafe_allow_html=True)

    if st.button("🧠 Klasifikasi Gambar"):
        st.session_state.page = "klasifikasi"
        st.rerun()
    if st.button("🛰️ Deteksi Objek"):
        st.session_state.page = "deteksi"
        st.rerun()


def klasifikasi_page():
    st.markdown("<h1>🧠 Misi Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    st.write("Unggah gambar untuk mengenali objek di dalamnya.")
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "utama"
        st.rerun()


def deteksi_page():
    st.markdown("<h1>🛰️ Misi Deteksi Objek</h1>", unsafe_allow_html=True)
    st.write("Unggah gambar untuk mendeteksi objek di dalamnya.")
    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "utama"
        st.rerun()


# ===============================
# SISTEM NAVIGASI
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "utama"

set_style()
draw_stars(120)

if st.session_state.page == "utama":
    main_page()
elif st.session_state.page == "klasifikasi":
    klasifikasi_page()
elif st.session_state.page == "deteksi":
    deteksi_page()
