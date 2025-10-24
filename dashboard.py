import streamlit as st
import random

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

BG_COLOR = "#05091a"      # Latar belakang gelap
TEXT_COLOR = "#ffffff"    # Warna teks utama
TEXT_JUGA = "#708090"     # Warna teks biasa (default)
MISSION_COLOR = "#3b82f6" # <-- Ubah warna "Pilih Misi Kamu:" di sini (hex atau nama warna)

# ==========================
# Gaya global (boleh pakai HTML/CSS via Python)
# hanya untuk background & animasi bintang (tombol tetap default)
# ==========================
st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {BG_COLOR} !important;
            color: {TEXT_JUGA};
            overflow: hidden;
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        [data-testid="stToolbar"] {{ right: 2rem; }}

        @keyframes twinkle {{
            0% {{ opacity: 0.3; transform: scale(1); }}
            50% {{ opacity: 1; transform: scale(1.3); }}
            100% {{ opacity: 0.3; transform: scale(1); }}
        }}
        /* pastikan konten Streamlit tetap di atas layer bintang */
        [data-testid="stAppViewContainer"] > div:first-child {{
            position: relative;
            z-index: 2;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Inisialisasi halaman (session state)
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================
# Fungsi header
# ==========================
def header(title: str, subtitle: str = ""):
    st.markdown(
        f"<h1 style='text-align:center; color:{TEXT_COLOR}; margin:6px 0;'>{title}</h1>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='text-align:center; color:{TEXT_COLOR}; margin:0 0 12px 0; font-size:16px'>{subtitle}</p>",
            unsafe_allow_html=True,
        )

# ==========================
# Fungsi draw_stars
# ==========================
def draw_stars(num_stars: int = 300):
    """
    Inject HTML spans/divs untuk bintang.
    Bintang dibuat di z-index:0 (di bawah konten).
    """
    star_colors = ["#FFD700", "#FFF8DC", "#B0E0E6", "#F0E68C", "#FFFFFF"]
    stars_html = "<div style='position:fixed; inset:0; z-index:0; pointer-events:none;'>"
    for _ in range(num_stars):
        left = random.uniform(0, 100)
        top = random.uniform(0, 100)
        size = random.randint(4, 14)
        opacity = round(random.uniform(0.25, 0.95), 2)
        duration = round(random.uniform(1.8, 4.0), 2)
        color = random.choice(star_colors)
        stars_html += (
            f"<div style='position:absolute; left:{left}%; top:{top}%; "
            f"font-size:{size}px; color:{color}; opacity:{opacity}; "
            f"animation:twinkle {duration}s infinite ease-in-out;'>⭐</div>"
        )
    stars_html += "</div>"
    # render HTML layer for stars; safe because we allow html here
    st.markdown(stars_html, unsafe_allow_html=True)

# ==========================
# HALAMAN UTAMA
# ==========================
def page_main():
    draw_stars(num_stars=350)
    header("🪐 SpaceVision AI", "Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🚀")

    # tiga kolom, konten di tengah
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.empty()
    with col2:
        # hanya warna "Pilih Misi Kamu:" yang diubah via MISSION_COLOR
        st.markdown(
            f"<h3 style='text-align:center; color:{MISSION_COLOR}; margin-bottom:8px;'>Pilih Misi Kamu:</h3>",
            unsafe_allow_html=True,
        )
        st.write("")  # spacer
        # tombol default Streamlit (warna bawaan)
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
def page_classify():
    draw_stars(num_stars=300)
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
def page_detect():
    draw_stars(num_stars=300)
    header("🛰️ Deteksi Objek", "Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    st.write("")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ==========================
# Router sederhana
# ==========================
if st.session_state.page == "main":
    page_main()
elif st.session_state.page == "classify":
    page_classify()
elif st.session_state.page == "detect":
    page_detect()
else:
    # fallback ke main kalau state aneh
    st.session_state.page = "main"
    page_main()
