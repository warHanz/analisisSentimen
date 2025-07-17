import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Crypto Exchange Analytics",
    page_icon="📊",
    layout="wide"
)

# --- PERBAIKAN ImportError ---
# Pastikan nama folder adalah 'pages' (dengan 's').
# Pastikan nama modul (tentang, preprocessing, analisis) sesuai dengan nama file Python Anda.
from page import tentang, scraping, preprocessing, analisis # Menggunakan 'analisis' sesuai kode Anda sebelumnya

# --- CSS Kustom untuk Tampilan Lebih Baik dan Card ---
st.markdown("""
    <style>
    /* Mengatur lebar konten utama */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Mempercantik judul */
    h1 {
        color: #4CAF50; /* Warna hijau cerah */
        text-align: center;
        font-size: 2.8em;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    /* Gaya untuk deskripsi dan poin-poin */
    .stMarkdown, .st-emotion-cache-nahz7x p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #333;
    }
    /* Kontainer teks perkenalan */
    .welcome-text {
        background-color: #f0f2f6; /* Latar belakang abu-abu muda */
        padding: 20px 30px;
        border-radius: 10px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.05);
        margin-top: 25px;
        text-align: justify;
    }
    .welcome-text ul {
        list-style-type: disc; /* Menggunakan bullet point standar */
        padding-left: 25px;
    }
    .welcome-text li {
        margin-bottom: 8px;
    }
    /* Styling untuk sidebar secara keseluruhan */
    .st-emotion-cache-kmq1ps {
        background-color: #262730; /* Latar belakang sidebar yang lebih gelap */
        color: #f0f2f6;
    }
    /* Untuk item menu di sidebar */
    .st-emotion-cache-1r6dm7x {
        background-color: #262730 !important;
    }
    /* Styling untuk expander */
    .streamlit-expanderHeader {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }

    /* --- CSS BARU UNTUK CARD --- */
    .card-container {
        display: flex; /* Menggunakan flexbox untuk menata kartu bersebelahan */
        justify-content: space-around; /* Memberi ruang di antara kartu */
        flex-wrap: wrap; /* Memungkinkan kartu untuk wrap ke baris berikutnya jika layar kecil */
        gap: 30px; /* Jarak antara kartu */
        margin-top: 40px;
        margin-bottom: 40px;
    }
    .feature-card {
        background-color: #ffffff; /* Latar belakang kartu putih */
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1); /* Bayangan yang lebih lembut */
        padding: 30px;
        flex: 1; /* Agar kartu membesar mengisi ruang yang tersedia */
        min-width: 280px; /* Lebar minimum kartu */
        max-width: 45%; /* Batas lebar agar tidak terlalu besar */
        text-align: center;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out; /* Transisi untuk efek hover */
        border: 1px solid #eee; /* Garis tepi tipis */
    }
    .feature-card:hover {
        transform: translateY(-8px); /* Menggeser kartu sedikit ke atas saat di-hover */
        box-shadow: 0 12px 25px rgba(0,0,0,0.2); /* Bayangan lebih kuat saat di-hover */
    }
    .feature-card h3 {
        color: #2C3E50; /* Warna judul kartu yang gelap */
        font-size: 1.8em;
        margin-bottom: 15px;
    }
    .feature-card p {
        color: #555;
        font-size: 1em;
    }
    .card-icon {
        font-size: 3em; /* Ukuran ikon */
        color: #4CAF50; /* Warna ikon hijau cerah */
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Fungsi untuk Memuat Konten Halaman ---
def load_page_content(page_name):
    if page_name == "Home":

        # --- Bagian Hero Image ---
        try:
            # PERBAIKAN DEPRECATION WARNING: Menggunakan use_container_width=True
            st.image("assets/hero/hero.jpg", use_container_width=True, caption="")
        except FileNotFoundError:
            st.warning("File gambar 'assets/hero/hero.jpg' tidak ditemukan. Pastikan path sudah benar.")

        st.markdown("---") # Garis pemisah setelah hero image

        # Konten teks perkenalan utama
        # st.markdown(
        #     """
        #     <div class='welcome-text'>
        #         <p> <b>analisis sentimen</b> dan <b>preprocessing data</b> pada pasar <b>cryptocurrency</b> yang dinamis.</p>
        #         <p>Di sini, Anda dapat mengunggah data, membersihkannya, dan melakukan analisis mendalam untuk mendapatkan wawasan berharga tentang opini publik.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        
        st.markdown("---") # Garis pemisah

        
        st.markdown(
            """
            <div class="card-container">
                <div class="feature-card">
                    <div class="card-icon">📊</div>
                    <h3>Visualisasi Tabel Preprocessing Data</h3>
                    <p>Lihat dan pahami struktur data setelah prosesing data.</p>
                </div>
                <div class="feature-card">
                    <div class="card-icon">📈</div>
                    <h3>Visualisasi Hasil Analisis</h3>
                    <p>Hasil Sentimen Positif, netral dan negatif serta akurasi model Supervised Learning dan Naive Bayes Classifier.</p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.markdown("---") # Garis pemisah

    elif page_name == "Tentang":
        tentang.app()
    elif page_name == "Scraping":
        scraping.app()
    elif page_name == "Preprocessing":
        preprocessing.app()
    elif page_name == "Analisis":
        analisis.app()

# --- Sidebar Navigasi Utama ---
with st.sidebar:

    st.markdown("##") # Spasi
    selected_option = option_menu(
        menu_title="Analisis Exchange",
        options=["Home", "Tentang", "Scraping", "Preprocessing", "Analisis"],
        icons=["house", "info-circle", "gear", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )
# --- Panggil Fungsi Konten Halaman ---
load_page_content(selected_option)