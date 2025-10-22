import streamlit as st

st.set_page_config(page_title="Gambar & Deteksi Objek", layout="wide")

# Judul utama
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Gambar & Deteksi Objek</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Silahkan pilih sesuai kebutuhan anda</p>", unsafe_allow_html=True)

# Spacer agar elemen agak turun
st.write("")
st.write("")

# Buat layout tengah
col_empty_left, col_content, col_empty_right = st.columns([1, 2, 1])

with col_content:
    # Dua tombol sejajar di tengah
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Klasifikasi Gambar")
        st.write("Klik untuk melakukan klasifikasi gambar")
        if st.button("Buka Klasifikasi Gambar"):
            st.switch_page("pages/klasifikasi.py")  # sesuaikan nama file halaman kamu

    with col2:
        st.subheader("📦 Deteksi Objek")
        st.write("Klik untuk melakukan deteksi objek")
        if st.button("Buka Deteksi Objek"):
            st.switch_page("pages/deteksi.py")  # sesuaikan nama file halaman kamu
