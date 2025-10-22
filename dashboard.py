import streamlit as st
from streamlit.components.v1 import html

# ========== Konfigurasi Halaman ==========
st.set_page_config(page_title="SpaceVision AI", page_icon="🪐", layout="wide")

# ========== CSS Tema + Animasi ==========
st.markdown("""
    <style>
    body {
        background: radial-gradient(ellipse at bottom, #0d1b2a 0%, #000 100%);
        overflow: hidden;
        height: 100vh;
        color: white;
    }

    /* Animasi bintang */
    .stars {
        width: 2px;
        height: 2px;
        background: white;
        position: absolute;
        animation: twinkle 2s infinite ease-in-out alternate;
        border-radius: 50%;
    }

    @keyframes twinkle {
        from {opacity: 0.2;}
        to {opacity: 1;}
    }

    .starfield {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
        overflow: hidden;
    }

    /* Pusat konten */
    .main-container {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        text-align: center;
    }

    /* Tombol */
    .space-btn {
        background: #1e90ff;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        margin: 1rem;
        font-size: 1.2rem;
        box-shadow: 0 0 20px rgba(30,144,255,0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .space-btn:hover {
        background: #63a4ff;
        box-shadow: 0 0 30px rgba(99,164,255,0.8);
    }
    </style>
""", unsafe_allow_html=True)


# ========== Fungsi Buat Bintang ==========
def draw_stars(n=100):
    import random
    stars_html = ""
    for _ in range(n):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        delay = random.random() * 2
        stars_html += f'<div class="stars" style="top:{y}%;left:{x}%;animation-delay:{delay}s;"></div>'
    st.markdown(f'<div class="starfield">{stars_html}</div>', unsafe_allow_html=True)


# ========== Navigasi Halaman ==========
if "page" not in st.session_state:
    st.session_state.page = "main"

# ========== Halaman Utama ==========
if st.session_state.page == "main":
    draw_stars(120)
    st.markdown("""
        <div class="main-container">
            <h1>🪐 <b>SpaceVision AI</b></h1>
            <p>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa.<br>
            Pilih misi eksplorasimu di bawah ini 🚀</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🧠 Klasifikasi Gambar", key="classify_btn"):
            st.session_state.page = "classify"
            st.rerun()
    with col2:
        if st.button("🛰️ Deteksi Objek", key="detect_btn"):
            st.session_state.page = "detect"
            st.rerun()

# ========== Halaman Klasifikasi ==========
elif st.session_state.page == "classify":
    draw_stars(100)
    st.title("🧠 Klasifikasi Gambar")
    st.write("Unggah gambar untuk mengidentifikasi objek berdasarkan model AI kamu.")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()

# ========== Halaman Deteksi ==========
elif st.session_state.page == "detect":
    draw_stars(100)
    st.title("🛰️ Deteksi Objek")
    st.write("Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO.")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = "main"
        st.rerun()
