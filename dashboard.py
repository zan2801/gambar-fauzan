import streamlit as st
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

# Tema warna
PRIMARY_COLOR = "#3b82f6"  # biru
BG_COLOR = "#0b0f2e"       # biru tua kehitaman
TEXT_COLOR = "#ffffff"     # putih

# Simpan state halaman
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# FUNGSI: HEADER
# ==========================
def header(title, subtitle=""):
    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR};'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR}; font-size:18px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.write("")

# ==========================
# FUNGSI: LATAR BELAKANG BINTANG
# ==========================
def draw_stars(num_stars=80):
    stars_html = ""
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(10, 22)  # ukuran font kecil agar tidak terlalu mencolok
        opacity = random.uniform(0.4, 1)
        stars_html += f"""
            <div style="
                position: absolute;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: {TEXT_COLOR};
                opacity: {opacity};
                z-index: 0;
            ">⭐</div>
        """
    st.markdown(f"""
        <div style="
            position: fixed;
            width: 100%;
            height: 100%;
            background-color: {BG_COLOR};
            overflow: hidden;
            z-index: -1;
        ">
            {stars_html}
        </div>
    """, unsafe_allow_html=True)

# ==========================
# TAMPILKAN BINTANG DI SEMUA HALAMAN
# ==========================
draw_stars(100)

# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    header("🪐 SpaceVision AI", "Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.empty()
    with col2:
        st.write("### Pilih Misi Kamu:")
        st.write("")
        if st.button("🧠 Klasifikasi Gambar", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
        if st.button("🛰️ Deteksi Objek", use_container_width=True):
            st.session_state.page = "detect"
            st.rerun()
    with col3:
        st.empty()

# =======
