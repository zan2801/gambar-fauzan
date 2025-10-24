import streamlit as st
import random

# =========================
# Background dan animasi bintang
# =========================
st.markdown("""
<style>
/* Latar belakang luar angkasa */
body {
    background-color: #050A30;
    overflow: hidden;
}

/* Efek bintang */
.star {
    position: absolute;
    background-color: white;
    border-radius: 50%;
    animation: twinkle 2s infinite ease-in-out;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
}

/* Container utama */
.main-container {
    position: relative;
    z-index: 2;
    text-align: center;
    padding: 80px 20px;
    color: white;
}

/* Judul utama */
h1 {
    font-size: 3em;
    color: #FFFFFF;
    margin-bottom: 10px;
}

/* Subjudul */
p {
    font-size: 1.2em;
    color: #A8C7FA;
}

/* Warna khusus untuk teks "Pilih Misi Kamu:" */
#judul-misi {
    color: #FFD700; /* 🎨 ubah warna di sini */
    font-weight: bold;
    font-size: 28px;
    margin-top: 40px;
}

/* Tombol misi */
button {
    background-color: #0B3D91;
    border: none;
    border-radius: 10px;
    color: white;
    padding: 15px 25px;
    font-size: 18px;
    margin: 10px;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
button:hover {
    background-color: #1E90FF;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Render bintang acak di seluruh halaman
# =========================
num_stars = 200  # jumlah bintang
stars_html = ""
for _ in range(num_stars):
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    size = random.uniform(0.5, 2.5)
    duration = random.uniform(1.5, 3)
    stars_html += f"""
    <div class="star" style="
        top:{y}%;
        left:{x}%;
        width:{size}px;
        height:{size}px;
        animation-duration:{duration}s;">
    </div>
    """

# Tampilkan bintang
st.markdown(f"<div style='position:fixed;top:0;left:0;width:100%;height:100%;z-index:1;'>{stars_html}</div>", unsafe_allow_html=True)

# =========================
# Konten utama
# =========================
st.markdown("""
<div class="main-container">
    <h1>🚀 SpaceVision AI</h1>
    <p>Jelajahi dunia kecerdasan buatan di galaksi luar angkasa 🪐</p>
    <h2 id="judul-misi">Pilih Misi Kamu:</h2>
</div>
""", unsafe_allow_html=True)

# =========================
# Tombol Streamlit
# =========================
col1, col2 = st.columns(2)
with col1:
    if st.button("🌌 Eksplorasi AI"):
        st.success("Menjalankan misi Eksplorasi AI...")

with col2:
    if st.button("🛰️ Analisis Data Luar Angkasa"):
        st.success("Menjalankan misi Analisis Data Luar Angkasa...")
