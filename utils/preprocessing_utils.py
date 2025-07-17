import os
import re
import ast
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.graph_objects as go
import streamlit as st

nltk.download(['punkt', 'stopwords'], quiet=True)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

STOPWORDS_TO_REMOVE = {
    'tidak', 'bukan', 'belum', 'jangan', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami',
    'itu', 'ini', 'ada', 'perlu', 'sama', 'saat', 'seperti', 'dari', 'ke', 'untuk',
    'pada', 'dan', 'atau', 'yang', 'dengan', 'di', 'ya', 'asu', 'sih', 'lah', 'gak',
    'nggak', 'kok', 'dong', 'nih', 'aja', 'bisa', 'cuma', 'woi', 'indodax','aplikasi', 'mudah', 'mula',
    'pemula', 'belajar', 'fitur', 'crypto', 'kripto', 'trading', 'investasi', 'skali', 'sekali', 
    'kali', 'mengerti'
}

CUSTOM_STOPWORDS = {
    'adalah', 'akan', 'antara', 'apa', 'apabila', 'atas', 'bagaimana', 'bagi', 'bahwa',
    'berada', 'berapa', 'berikut', 'bersama', 'boleh', 'banyak', 'baru', 'bila', 'bilang',
    'cara', 'cukup', 'dapat', 'demi', 'demikian', 'depan', 'dulu', 'harus', 'hanya',
    'hingga', 'ia', 'ialah', 'jika', 'justru', 'kapan', 'karena', 'karenanya', 'kemudian',
    'kini', 'lagi', 'lalu', 'lebih', 'maupun', 'melainkan', 'menjadi', 'menuju', 'meski',
    'mungkin', 'oleh', 'paling', 'para', 'per', 'pernah', 'saja', 'sambil', 'sampai',
    'sangat', 'sebelum', 'sekarang', 'secara', 'sedang', 'selain', 'selama', 'seluruh',
    'semua', 'sesudah', 'setelah', 'setiap', 'sudah', 'supaya', 'tapi', 'tentang',
    'terhadap', 'termasuk', 'ternyata', 'tetap', 'tetapi', 'tiap', 'tuju', 'turut', 'umum',
    'yaitu' 
}

STOPWORDS_ALL = set(stopwords.words('indonesian')).union(STOPWORDS_TO_REMOVE)


# Load normalization dictionary from assets/kamus/kamuskatabaku.xlsx
def load_normalization_dict():
    try:
        norm_file = "assets/kamus/kamuskatabaku.xlsx"
        if os.path.exists(norm_file):
            norm_df = pd.read_excel(norm_file)
            # Assuming columns 'non_baku' and 'baku' for non-standard and standard words
            norm_dict = dict(zip(norm_df['tidak_baku'].str.lower(), norm_df['kata_baku'].str.lower()))
            return norm_dict
        else:
            st.error("File kamuskatabaku.xlsx tidak ditemukan di assets/kamus/")
            return {}
    except Exception as e:
        st.error(f"Gagal memuat kamus normalisasi: {e}")
        return {}


def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception:
        st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
        return None


def preprocess_data(data):
    required_columns = ['Tanggal', 'Nama Pengguna', 'Rating', 'Reviews Text']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return None, 0
    processed = data[required_columns].drop_duplicates(subset='Reviews Text', keep='first')
    return processed, len(processed)

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z\s]', '', text).strip()
    return text

def normalize_text(text):
    if not text:
        return ''
    norm_dict = load_normalization_dict()
    return ' '.join(norm_dict.get(word, word) for word in text.split())

def tokenize_text(text):
    if text:
        return word_tokenize(text)
    return []

def remove_stopwords(tokens):
    if not tokens:
        return []
    filtered_tokens = [token for token in tokens if token not in STOPWORDS_ALL]
    return filtered_tokens

def stem_tokens(tokens):
    if not tokens:
        return []
    stemmed = [stemmer.stem(token) for token in tokens]
    return [token for token in stemmed if len(token) >= 3]

def plot_processing_counts(counts_dict):
    steps = list(counts_dict.keys())
    counts = list(counts_dict.values())
    fig = go.Figure([go.Bar(x=steps, y=counts, text=counts, textposition='auto',
                            marker_color='#66B3FF', width=0.8)])
    fig.update_layout(
        title="Jumlah Data Setelah Setiap Tahap Pemrosesan",
        xaxis_title="Tahap Pemrosesan",
        yaxis_title="Jumlah Data",
        title_x=0.5,
        height=400
    )
    return fig

def load_preprocessed_data(filepath="temp_preprocessed_data.csv"):
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'stemmed' in df.columns:
                df['stemmed'] = df['stemmed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            return df
        else:
            st.error("File data yang telah diproses tidak ditemukan.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat data yang telah diproses: {e}")
        return None

def save_preprocessed_data(data, filepath="temp_preprocessed_data.csv"):
    try:
        data.to_csv(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan data yang telah diproses: {e}")
        return False