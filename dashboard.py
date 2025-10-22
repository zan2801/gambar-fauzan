import streamlit as st
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

PRIMARY_COLOR = "#3b82f6"
BG_COLOR = "#0b0f2e"
TEXT_COLOR = "#ffffff"

# Gaya dasar halaman
st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_COLOR};
        }}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
        [data-testid="stToolbar"] {{right: 2rem;}}
    </style>
""", unsafe_allow_html=True)

# Simpan halaman
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
# BINTANG (dengan zona aman)
# ==========================
def draw_stars(num_stars=120, safe_zone=(30, 70, 35, 65)):
    """safe_zone = (left_min, left_max, top_min, top_max)"""
    stars_html = ""
    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)

        # Hindari zona aman
        if safe_zone[0] < left < safe_zone[1] and safe_zone[2] < top < safe_zone[3]:
            continue

        size = random.randint(8, 18)
        opacity = random.uniform(0.3, 0.9)

        stars_html += f"""
            <div style="
                position: fixed;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: gold;
                opacity: {opacity};
                z-index: 1;
                pointer-events: none;
            ">⭐</div>
        """
    st.markdown(stars_html, unsafe_allow_html=True)


# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    draw_stars(num_stars=150, safe_zone=(28, 72, 30, 70))
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
    draw_stars(num_stars=150, safe_zone=(20, 80, 20, 80))
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
    draw_stars(num_stars=150, safe_zone=(20, 80, 20, 80))
    header("🛰️ Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
