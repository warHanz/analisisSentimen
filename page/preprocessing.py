import streamlit as st
import pandas as pd
from utils.preprocessing_utils import *
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import os
from PIL import Image

# Cache the loading of previous data to improve performance
@st.cache_data
def load_previous_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file CSV: {str(e)}")
        return None

def save_plot_as_image(fig, filename="preprocessing_chart.png"):
    """Save Plotly figure as an image file."""
    fig.write_image(filename, format="png")
    return filename

def app():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader>label {
            color: #2c3e50;
            font-weight: bold;
        }
        .stSubheader {
            color: #2c3e50;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .success-message {
            color: #27ae60;
            font-weight: bold;
        }
        .error-message {
            color: #c0392b;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("📊 Preprocessing Data Dashboard")
    st.markdown("""
        Unggah file CSV atau Excel untuk membersihkan dan menyiapkan data Anda sebelum analisis lebih lanjut. Anda juga dapat melihat dataset yang telah diproses sebelumnya dengan tampilan yang interaktif.
    """)

    # Tabs for better organization
    tab1, tab2 = st.tabs(["🔄 Proses Data Baru", "📂 Lihat Dataset Lama"])

    with tab1:
        st.subheader("Unggah dan Proses Data Baru")
        uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'], key="new_data_uploader", help="Unggah file data Anda untuk diproses.")

        if uploaded_file is not None:
            # Load data
            with st.spinner("Memuat data..."):
                data = load_data(uploaded_file)
                if data is None:
                    st.error("Gagal memuat data. Pastikan format file sesuai.")
                    return
            
            # Initialize counts dictionary
            counts_dict = {
                "Data Awal": len(data),
                "Setelah Deduplikasi": 0,
                "Setelah Cleaning": 0,
                "Setelah Normalisasi": 0,
                "Setelah Tokenisasi": 0,
                "Setelah Stopword Removal": 0,
                "Setelah Stemming": 0
            }

            # Preprocess data (deduplication)
            with st.spinner("Melakukan deduplikasi..."):
                processed_data, deduplicated_count = preprocess_data(data)
                if processed_data is None:
                    st.error("Gagal melakukan deduplikasi.")
                    return
                counts_dict["Setelah Deduplikasi"] = deduplicated_count

            # Initialize lists for processed texts
            cleaned_texts = []
            normalized_texts = []
            tokenized_texts = []
            stopwords_removed = []
            stemmed_texts = []

            # Process each review
            with st.spinner("Memproses teks..."):
                for text in processed_data['Reviews Text']:
                    cleaned = clean_text(text)
                    cleaned_texts.append(cleaned)
                    if cleaned:
                        counts_dict["Setelah Cleaning"] += 1

                    normalized = normalize_text(cleaned)
                    normalized_texts.append(normalized)
                    if normalized:
                        counts_dict["Setelah Normalisasi"] += 1

                    tokens = tokenize_text(normalized)
                    tokenized_texts.append(tokens)
                    if tokens:
                        counts_dict["Setelah Tokenisasi"] += 1

                    no_stopwords = remove_stopwords(tokens)
                    stopwords_removed.append(no_stopwords)
                    if no_stopwords:
                        counts_dict["Setelah Stopword Removal"] += 1

                    stemmed = stem_tokens(no_stopwords)
                    stemmed_texts.append(stemmed)
                    if stemmed:
                        counts_dict["Setelah Stemming"] += 1

            # Update processed_data
            processed_data['Cleaned Text'] = cleaned_texts
            processed_data['Normalized Text'] = normalized_texts
            processed_data['Tokenized Text'] = tokenized_texts
            processed_data['Stopwords Removed'] = stopwords_removed
            processed_data['Stemmed Text'] = stemmed_texts

            # Display processed data
            st.subheader("Tabel Hasil Preprocessing")
            st.dataframe(
                processed_data[['Reviews Text', 'Cleaned Text', 'Normalized Text', 'Tokenized Text', 'Stopwords Removed', 'Stemmed Text']],
                use_container_width=True,
                height=400
            )

            # Save preprocessed data as CSV
            with st.spinner("Menyimpan data..."):
                if save_preprocessed_data(processed_data):
                    st.markdown('<p class="success-message">Data berhasil disimpan ke temp_preprocessed_data.csv</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="error-message">Gagal menyimpan data.</p>', unsafe_allow_html=True)

            # Save and download preprocessed data as JSON
            st.subheader("Unduh Hasil Preprocessing sebagai JSON")
            processed_data.to_json('preprocessed_data.json', orient='records', lines=False, force_ascii=False)
            with open('preprocessed_data.json', 'rb') as f:
                st.download_button(
                    label="Unduh Data sebagai JSON",
                    data=f,
                    file_name="preprocessed_data.json",
                    mime="application/json",
                    key="download_json_new"
                )

            # Plot data counts
            st.subheader("Visualisasi Jumlah Data per Tahap Preprocessing")
            fig = go.Figure(data=[
                go.Bar(
                    x=list(counts_dict.keys()),
                    y=list(counts_dict.values()),
                    marker_color='#4CAF50',
                    text=list(counts_dict.values()),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Jumlah Data pada Setiap Tahap Preprocessing",
                xaxis_title="Tahap Preprocessing",
                yaxis_title="Jumlah Data",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True, key="counts_chart_new")

            # Save plot as image and provide download button
            st.subheader("Unduh Grafik Preprocessing")
            chart_filename = save_plot_as_image(fig)
            with open(chart_filename, 'rb') as f:
                st.download_button(
                    label="Unduh Grafik sebagai PNG",
                    data=f,
                    file_name=chart_filename,
                    mime="image/png",
                    key="download_chart_new"
                )

    with tab2:
        st.subheader("Lihat Dataset Lama")
        # File uploader for both CSV and PNG
        uploaded_files = st.file_uploader(
            "Pilih file CSV dan PNG hasil preprocessing sebelumnya", 
            type=['csv', 'png'], 
            accept_multiple_files=True, 
            key="previous_files_uploader", 
            help="Unggah file CSV (misalnya, temp_preprocessed_data.csv) dan file PNG (misalnya, preprocessing_chart.png)."
        )

        csv_file = None
        png_file = None
        if uploaded_files:
            for file in uploaded_files:
                if file.name.endswith('.csv'):
                    csv_file = file
                elif file.name.endswith('.png'):
                    png_file = file

        if csv_file is not None:
            with st.spinner("Memuat dataset lama..."):
                previous_data = load_previous_data(csv_file)
                if previous_data is None:
                    return

                # Check required columns
                required_columns = ['Reviews Text', 'Cleaned Text', 'Normalized Text', 'Tokenized Text', 'Stopwords Removed', 'Stemmed Text']
                missing_cols = [col for col in required_columns if col not in previous_data.columns]
                if missing_cols:
                    st.markdown(f'<p class="error-message">Kolom berikut tidak ditemukan: {", ".join(missing_cols)}</p>', unsafe_allow_html=True)
                    return

                # Convert string representations of lists if necessary
                for col in ['Tokenized Text', 'Stopwords Removed', 'Stemmed Text']:
                    if previous_data[col].apply(lambda x: isinstance(x, str)).any():
                        try:
                            previous_data[col] = previous_data[col].apply(lambda x: eval(x) if isinstance(x, str) and pd.notna(x) else x)
                        except Exception as e:
                            st.markdown(f'<p class="error-message">Gagal mengonversi kolom {col}: {str(e)}</p>', unsafe_allow_html=True)
                            return

                # Display previous data
                st.subheader("Tabel Dataset Lama")
                st.dataframe(
                    previous_data[['Reviews Text', 'Cleaned Text', 'Normalized Text', 'Tokenized Text', 'Stopwords Removed', 'Stemmed Text']],
                    use_container_width=True,
                    height=400
                )

                # Display uploaded chart if available
                if png_file is not None:
                    st.subheader("Grafik Preprocessing Sebelumnya")
                    try:
                        chart_image = Image.open(png_file)
                        st.image(chart_image, caption="Grafik Jumlah Data per Tahap Preprocessing (Sebelumnya)", use_container_width=True)
                    except Exception as e:
                        st.markdown(f'<p class="error-message">Gagal memuat gambar grafik: {str(e)}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="error-message">File PNG grafik tidak diunggah. Harap unggah file PNG untuk menampilkan visualisasi.</p>', unsafe_allow_html=True)

        else:
            st.markdown('<p class="error-message">Harap unggah file CSV untuk melihat dataset lama.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()