import streamlit as st
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

PRIMARY_COLOR = "#3b82f6"
BG_COLOR = "#05091a"
TEXT_COLOR = "#ffffff"
TEXT_JUGA = "#708090"

# Gaya dasar halaman (CSS inline)
st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_JUGA};
            overflow: hidden;
        }}
        [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
        [data-testid="stToolbar"] {{right: 2rem;}}
        @keyframes twinkle {{
            0% {{opacity: 0.3; transform: scale(1);}}
            50% {{opacity: 1; transform: scale(1.4);}}
            100% {{opacity: 0.3; transform: scale(1);}}
        }}
        /* Custom style untuk tombol agar terlihat lebih konsisten */
        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            height: 38px;
            padding: 6px 12px;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background-color: #5aa0ff;
        }}
    </style>
""", unsafe_allow_html=True)

# Simpan halaman awal di session state
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# HEADER
# ==========================
def header(title, subtitle=""):
    st.markdown(f"<h1 style='text-align:center; color:{TEXT_COLOR}; margin:0.2rem 0;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='text-align:center; color:{TEXT_COLOR}; margin:0.1rem 0 1rem 0; font-size:16px'>{subtitle}</p>", unsafe_allow_html=True)

# ==========================
# BINTANG FULL LAYAR
# ==========================
def draw_stars(num_stars=400):
    """Bintang penuh layar dengan warna & ukuran bervariasi"""
    star_colors = ["#FFD700", "#FFF8DC", "#B0E0E6", "#F0E68C", "#FFFFFF"]
    stars_html = ""

    for _ in range(num_stars):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(4, 14)
        opacity = round(random.uniform(0.3, 1), 2)
        duration = round(random.uniform(1.5, 4), 2)
        color = random.choice(star_colors)

        stars_html += f"""
            <div style="
                position: fixed;
                left: {left}%;
                top: {top}%;
                font-size: {size}px;
                color: {color};
                opacity: {opacity};
                z-index: 0;
                pointer-events: none;
                animation: twinkle {duration}s infinite ease-in-out;
            ">⭐</div>
        """

    st.markdown(stars_html, unsafe_allow_html=True)

# ==========================
# HALAMAN UTAMA
# ==========================
if st.session_state.page == "main":
    draw_stars(num_stars=400)
    header("🪐 SpaceVision AI", "Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀")

    # Tiga kolom — kita gunakan kol tengah untuk konten
    col1, col2, col3 = st.columns([1, 1, 1])
    # pastikan setiap 'with' memiliki blok terindentasi
    with col1:
        # bisa kosong, tapi harus ada blok; gunakan empty() agar aman
        st.empty()

    with col2:
        # <-- pastikan ini terindentasi (4 spasi)
        # Ubah warna "Pilih Misi Kamu:" saja dengan markdown inline
        st.markdown(f"<h3 style='text-align:center; color:#87CEEB; margin-bottom:6px;'>Pilih Misi Kamu:</h3>", unsafe_allow_html=True)
        st.write("")  # spacing
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
    draw_stars(num_stars=400)
    header("🧠 Klasifikasi Gambar", "Unggah gambar untuk diidentifikasi model AI kamu")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    st.write("")  # spacing
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ==========================
# HALAMAN DETEKSI
# ==========================
elif st.session_state.page == "detect":
    draw_stars(num_stars=400)
    header("🛰️ Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    st.write("")  # spacing
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
