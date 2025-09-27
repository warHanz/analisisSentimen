# Analisis Sentimen Ulasan Exchange Crypto

## Deskripsi

Aplikasi ini adalah dashboard interaktif berbasis Streamlit untuk analisis sentimen pada ulasan aplikasi exchange cryptocurrency (Indodax, Pintu, Tokocrypto) dari Google Play Store. Proses meliputi scraping data, preprocessing, analisis sentimen, dan visualisasi hasil.

## Fitur Utama

- **Scraping Data**: Mengambil ulasan terbaru dari Google Play Store secara otomatis.
- **Preprocessing Data**: Membersihkan, normalisasi, tokenisasi, stopword removal, dan stemming pada data ulasan.
- **Analisis Sentimen**: Melabeli data dengan model lexicon-based dan supervised learning (SVM & Naive Bayes Classifier).
- **Visualisasi**: Menampilkan hasil analisis dalam bentuk tabel, grafik, word cloud, dan confusion matrix.
- **Download Hasil**: Unduh hasil analisis dan visualisasi dalam format ZIP, CSV, atau JSON.

## Struktur Folder

- `home.py` : Halaman utama dashboard.
- `page/` : Berisi modul Streamlit untuk scraping, preprocessing, analisis, dan tentang aplikasi.
- `models/` : Implementasi model SVM dan NBC serta utilitas pelatihan dan evaluasi.
- `utils/` : Fungsi preprocessing, analisis, dan visualisasi.
- `assets/` : Dataset, kamus sentimen, gambar, dan hasil analisis.

## Cara Menjalankan

1. **Instalasi Dependensi**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi**
   ```powershell
   streamlit run home.py
   ```
3. **Navigasi**
   - Home: Informasi dan fitur utama.
   - Tentang: Penjelasan aplikasi dan alur kerja.
   - Scraping: Ambil ulasan dari Google Play Store.
   - Preprocessing: Proses dan bersihkan data ulasan.
   - Analisis: Analisis sentimen dan evaluasi model.

## Alur Kerja

1. **Scraping**: Pilih aplikasi dan jumlah ulasan, unduh data CSV.
2. **Preprocessing**: Unggah data, lakukan pembersihan, normalisasi, tokenisasi, stopword removal, dan stemming.
3. **Analisis Sentimen**: Unggah data hasil preprocessing, lakukan pelabelan sentimen dan evaluasi model.
4. **Visualisasi & Download**: Lihat hasil analisis dan unduh visualisasi/grafik.

## Dataset & Kamus

- Dataset ulasan: `assets/dataset/ulasan_*.csv`
- Kamus sentimen: `assets/datasentimen/kamus_positif.xlsx`, `kamus_negatif.xlsx`
- Kamus kata baku: `assets/kamus/`

## Dependensi

Lihat `requirements.txt` untuk daftar lengkap package Python yang digunakan.

## Kontribusi

Pull request dan saran pengembangan sangat diterima.

## Lisensi

Proyek ini untuk keperluan akademik dan riset. Silakan gunakan dan modifikasi sesuai kebutuhan.
