import streamlit as st
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi & Deteksi Objek", layout="wide")

# Inisialisasi session state untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi navigasi
def go_to(page_name):
    st.session_state.page = page_name

# ==============================
# HALAMAN DEPAN
# ==============================
if st.session_state.page == "home":
    st.image("halaman depan.png", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Klasifikasi Gambar", use_container_width=True):
            go_to("klasifikasi")
    with col2:
        if st.button("🧠 Deteksi Objek", use_container_width=True):
            go_to("deteksi")

# ==============================
# HALAMAN KLASIFIKASI GAMBAR
# ==============================
elif st.session_state.page == "klasifikasi":
    st.image("halaman klasifikasi gambar.png", use_container_width=True)

    # Tombol kembali
    if st.button("⬅️ Kembali ke Beranda"):
        go_to("home")

    # Upload file
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.success("Gambar berhasil dimuat! (tempatkan model klasifikasi di sini)")

# ==============================
# HALAMAN DETEKSI OBJEK
# ==============================
elif st.session_state.page == "deteksi":
    st.image("halaman deteksi objek.png", use_container_width=True)

    # Tombol kembali
    if st.button("⬅️ Kembali ke Beranda"):
        go_to("home")

    # Upload file
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.success("Gambar berhasil dimuat! (tempatkan model YOLO di sini)")
