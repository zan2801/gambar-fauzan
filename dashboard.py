import streamlit as st
import random

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

PRIMARY_COLOR = "#3b82f6"
BG_COLOR = "#040b24"
TEXT_COLOR = "#ffffff"

# ==========================
# SETUP STATE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# GAYA GLOBAL
# ==========================
st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
            overflow: hidden;
        }}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
        [data-testid="stToolbar"] {{visibility: hidden;}}
        button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            padding: 0.6em 1.2em;
            cursor: pointer;
            transition: 0.3s;
        }}
        button:hover {{
            background-color: #2563eb;
            transform: scale(1.05);
        }}
    </style>
""", unsafe_allow_html=True)


# ==========================
# FUNGSI BINTANG
# ==========================
def draw_stars(num_stars=100):
    stars_html = ""
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(6, 16)
        opacity = random.uniform(0.3, 0.9)
        stars_html += f"""
            <div style="
                position: fixed;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: #ffffff;
                opacity: {opacity};
                pointer-events: none;
                z-index: 1;
            ">⭐</div>
        """
    st.markdown(stars_html, unsafe_allow_html=True)


# ==========================
# FUNGSI HEADER
# ==========================
def header(title, subtitle=""):
    st.markdown(f"""
        <div style="text-align:center; z-index:2; position:relative;">
            <h1 style="color:{TEXT_COLOR};">{title}</h1>
            <p style="color:{TEXT_COLOR}; font-size:18px;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("")


# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    draw_stars(130)
    header("🪐 SpaceVision AI", "Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
        if st.button("🧠 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
        if st.button("🛰️ Deteksi Objek", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()


# ==========================
# HALAMAN KLASIFIKASI
# ==========================
elif st.session_state.page == "classify":
    draw_stars(120)
    header("🧠 Klasifikasi Gambar", "Unggah gambar untuk diidentifikasi model AI kamu")

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
    draw_stars(120)
    header("🛰️ Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
