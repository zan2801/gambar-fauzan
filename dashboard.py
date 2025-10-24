import streamlit as st
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

# Warna tema
PRIMARY_COLOR = "#3b82f6"
BG_COLOR = "#0b0f2e"
TEXT_COLOR = "#ffffff"

# ==========================
# GAYA GLOBAL
# ==========================
st.markdown(f"""
    <style>
        /* Latar belakang utama */
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_COLOR};
        }}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
        [data-testid="stToolbar"] {{right: 2rem;}}
        
        /* Pastikan konten di atas layer bintang */
        [data-testid="stAppViewContainer"] > div:first-child {{
            position: relative;
            z-index: 2;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================
# FUNGSI BINTANG
# ==========================
def draw_stars(num_stars=120):
    """Buat latar belakang bintang di layer paling belakang."""
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
        size = random.randint(6, 16)
        opacity = random.uniform(0.3, 0.9)
        stars_html += f"""
            <div style="
                position: absolute;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: gold;
                opacity: {opacity};
                pointer-events: none;
            ">⭐</div>
        """
    stars_html += "</div>"
    st.markdown(stars_html, unsafe_allow_html=True)

# ==========================
# SIMPAN STATUS HALAMAN
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# FUNGSI HEADER
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
        st.write("### Pilih Misi Kamu:")
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
    header("🧠 Klasifikasi Gambar", "Unggah gambar untuk diidentifikasi model AI kamu")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ==========================
# HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    draw_stars()
    header("🛰️ Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
