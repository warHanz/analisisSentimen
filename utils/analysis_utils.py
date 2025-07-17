import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from collections import Counter
import zipfile
from io import BytesIO
import streamlit as st

def load_preprocessed_data(file=None, filepath="temp_preprocessed_data.csv"):
    try:
        if file is not None:
            # Baca langsung dari UploadedFile
            df = pd.read_csv(file)
        elif os.path.exists(filepath):
            # Baca dari filepath jika tidak ada file diunggah
            df = pd.read_csv(filepath)
        else:
            st.error(f"File {filepath} tidak ditemukan.")
            return None

        # Konversi kolom 'Stemmed Text' dari string ke list jika perlu
        if 'Stemmed Text' in df.columns:
            df['Stemmed Text'] = df['Stemmed Text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

def load_json_data(file):
    try:
        df = pd.read_json(file, orient='records', lines=False)
        # Konversi 'Stemmed Text' ke list jika diperlukan
        if 'Stemmed Text' in df.columns:
            df['Stemmed Text'] = df['Stemmed Text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file JSON: {str(e)}")
        return None

def load_sentiment_lexicon():
    positive_file = "assets/datasentimen/kamus_positif.xlsx"
    negative_file = "assets/datasentimen/kamus_negatif.xlsx"

    try:
        if os.path.exists(positive_file):
            positive_lexicon = set(pd.read_excel(positive_file)['word'].str.lower())
        else:
            st.error(f"File {positive_file} tidak ditemukan.")
            positive_lexicon = set()
    except Exception as e:
        st.error(f"Gagal memuat {positive_file}: {str(e)}")
        positive_lexicon = set()

    try:
        if os.path.exists(negative_file):
            negative_lexicon = set(pd.read_excel(negative_file)['word'].str.lower())
        else:
            st.error(f"File {negative_file} tidak ditemukan.")
            negative_lexicon = set()
    except Exception as e:
        st.error(f"Gagal memuat {negative_file}: {str(e)}")
        negative_lexicon = set()

    return positive_lexicon, negative_lexicon

def label_sentiment(text, positive_words, negative_words):
    """
    Memberikan label sentimen berdasarkan teks (string, bukan list token).
    """
    tokens = text.split()  # Asumsi teks sudah dalam bentuk string
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    sentiment_score = positive_count - negative_count

    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment_score, sentiment

def preprocess_for_model(tokens):
    return ' '.join(tokens)

def generate_wordcloud(tokens_list, title, color):
    all_words = ' '.join(word for tokens in tokens_list for word in tokens)
    return WordCloud(width=800, height=400, background_color='white',
                     colormap=color, max_words=100).generate(all_words)

def generate_frequency_chart(tokens_list, top_n=10, color='viridis'):
    all_words = [word for tokens in tokens_list for word in tokens]
    word_freq = Counter(all_words).most_common(top_n)
    words, freqs = zip(*word_freq)
    
    # Map colormap ke warna spesifik untuk konsistensi
    color_dict = {
        'viridis': '#4ECDC4',  # Warna utama untuk Semua Data
        'Greens': '#28a745',   # Warna untuk Positif
        'Greys': '#6c757d',    # Warna untuk Netral
        'Reds': '#dc3545'      # Warna untuk Negatif
    }
    
    bar_color = color_dict.get(color, '#4ECDC4')  # Default ke viridis jika color tidak dikenal
    
    fig = go.Figure([go.Bar(
        x=words,
        y=freqs,
        text=freqs,
        textposition='auto',
        marker_color=bar_color,
        width=0.8
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Kata Paling Sering Muncul",
        xaxis_title="Kata",
        yaxis_title="Frekuensi",
        title_x=0.5,
        height=400,  # Konsisten dengan word cloud
        plot_bgcolor='white',  # Latar belakang putih seperti word cloud
        paper_bgcolor='white',
        font=dict(size=12, color='#2C3E50'),  # Font konsisten dengan CSS
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(tickangle=45),  # Memutar label x untuk keterbacaan
        showlegend=False
    )
    return fig

def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions, labels=['Negative', 'Neutral', 'Positive'])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Neutral', 'Positive'],
                    y=['Negative', 'Neutral', 'Positive'])
    fig.update_layout(title=f"Confusion Matrix - {model_name}", title_x=0.5)
    return fig

def plot_data_split(train_size, test_size):
    fig = go.Figure([go.Bar(x=['Data Training', 'Data Uji'], y=[train_size, test_size],
                            text=[train_size, test_size], textposition='auto',
                            marker_color=['#66B3FF', '#FF9999'], width=0.8)])
    fig.update_layout(title="Perbandingan Data Training dan Uji",
                      yaxis_title="Jumlah Data", title_x=0.5, height=400)
    return fig

def plot_accuracy_comparison(results):
    fig = go.Figure([go.Bar(x=list(results.keys()), y=[results[name]['acc'] for name in results],
                            text=[f"{results[name]['acc']:.2f}" for name in results],
                            textposition='auto', marker_color=['#FF6B6B', '#4ECDC4'], width=0.8)])
    fig.update_layout(title="Akurasi Model", yaxis_title="Akurasi",
                      yaxis_range=[0, 1], title_x=0.5, height=400)
    return fig

def display_metric_box(label, value, emoji):
    """Menampilkan metrik dengan emoticon dan styling minimalis."""
    st.markdown(
        f"<div class='metric-box'>"
        f"{emoji} <span class='metric-label'>{label}:</span> "
        f"<span class='metric-value'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

def display_metric_box(label, value, emoji):
    """Menampilkan metrik dengan emoticon dan styling minimalis."""
    st.markdown(
        f"<div class='metric-box'>"
        f"{emoji} <span class='metric-label'>{label}:</span> "
        f"<span class='metric-value'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True
    )