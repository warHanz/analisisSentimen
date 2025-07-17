import streamlit as st
from google_play_scraper import reviews, Sort
import pandas as pd
import time

# --- Konfigurasi Aplikasi ---
APP_TITLE = "Scraping Data"
APP_DESCRIPTION = "Aplikasi ini memungkinkan Anda mengambil ulasan terbaru dari aplikasi exchange crypto  di Google Play Store."
APP_ID_INDODAX = "id.co.bitcoin"
APP_ID_PINTU = "com.valar.pintu"
APP_ID_TOKOCRYPTO = "com.binance.cloud.tokocrypto"
ALLOWED_APP_IDS = {
    "Indodax": APP_ID_INDODAX,
    "Pintu": APP_ID_PINTU,
    "TokoCrypto": APP_ID_TOKOCRYPTO 
}

# --- Fungsi Utama Halaman Scraping ---
def app():

    # Custom CSS untuk mempercantik tampilan
    st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stSelectbox, .stNumberInput {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
        }
        .stProgress .st-bo {
            background-color: #007bff;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #343a40;
            text-align: center;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 18px;
            color: #6c757d;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Aplikasi
    st.markdown(f'<div class="header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subheader">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)

    # Layout dengan container
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Pilih Aplikasi dan Jumlah Ulasan")
        
        # Membagi layout menjadi dua kolom untuk input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dropdown untuk memilih aplikasi
            selected_app_name = st.selectbox(
                "📱 Pilih aplikasi:",
                options=list(ALLOWED_APP_IDS.keys()),
                index=0,
                help="Pilih antara Indodax atau Pintu untuk mengambil ulasan."
            )
        
        with col2:
            # Input jumlah ulasan
            count = st.number_input(
                "🔢 Jumlah ulasan (0 untuk semua):",
                min_value=0,
                value=100,
                step=10,
                help="Masukkan jumlah ulasan yang ingin diambil. Proses mungkin memakan waktu lebih lama untuk jumlah besar."
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Tombol untuk memulai scraping
    if st.button("🚀 Ambil Ulasan", use_container_width=True):
        all_reviews = []
        continuation_token = None
        batch_size = 200

        # Info dan progress bar
        st.info(f"Mengambil ulasan untuk **{selected_app_name}**...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            with st.spinner("⏳ Sedang mengambil ulasan, mohon tunggu..."):
                reviews_fetched_count = 0
                
                while True:
                    current_batch_count = batch_size
                    if count > 0:
                        remaining_to_fetch = count - reviews_fetched_count
                        if remaining_to_fetch <= 0:
                            break
                        current_batch_count = min(batch_size, remaining_to_fetch)

                    result, continuation_token = reviews(
                        ALLOWED_APP_IDS[selected_app_name],
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        count=current_batch_count,
                        continuation_token=continuation_token
                    )

                    if not result:
                        break

                    all_reviews.extend(result)
                    reviews_fetched_count = len(all_reviews)

                    # Update progress bar dan status
                    if count > 0:
                        progress = min(1.0, reviews_fetched_count / count)
                        progress_bar.progress(progress)
                        status_text.text(f"📥 Mengambil ulasan: {reviews_fetched_count}/{count}")
                    else:
                        status_text.text(f"📥 Mengambil ulasan: {reviews_fetched_count} (tanpa batas)")

                    if not continuation_token or len(result) < current_batch_count:
                        break

                    time.sleep(0.5)

            st.success(f"✅ Berhasil mengambil {len(all_reviews)} ulasan untuk {selected_app_name}!")

            # Format data ulasan
            formatted_reviews = []
            for review in all_reviews:
                formatted_reviews.append({
                    'Tanggal': review['at'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Nama Pengguna': review['userName'],
                    'Rating': review['score'],
                    'Reviews Text': review['content'],
                })

            # Konversi ke DataFrame
            reviews_df = pd.DataFrame(formatted_reviews)

            # Tampilkan ulasan dalam tabel interaktif
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### 📊 Ulasan {selected_app_name}")
            st.dataframe(
                reviews_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Rating": st.column_config.NumberColumn(
                        format="%d ⭐",
                        help="Rating dari 1 hingga 5 bintang"
                    ),
                    "Ulasan": st.column_config.TextColumn(
                        width="large",
                        help="Teks ulasan dari pengguna"
                    )
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Tombol untuk mengunduh CSV
            csv = reviews_df.to_csv(index=False).encode('utf-8')
            file_name = f"ulasan_{selected_app_name.lower().replace(' ', '_')}.csv"
            st.download_button(
                label="📥 Unduh Ulasan (CSV)",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            st.warning("Pastikan koneksi internet Anda stabil dan coba lagi.")

if __name__ == "__main__":
    app()