import streamlit as st
import os
import base64
from pathlib import Path

def app():

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div style="text-align: center;">
            <h1 style="font-size: 3rem; color: #333; margin-bottom: 1rem; font-weight: 700;">
                📊 Tentang Aplikasi
            </h1>
            <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
                Platform Analisis Sentimen untuk Ulasan Aplikasi Exchange Crypto
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tentang Kami Section
    st.markdown("""
    <div class="content-section">
        <div class="section-text">
            <p>
                Aplikasi ini dirancang untuk memberikan analisis sentimen dengan melakukan procesisng data seperti Pembersihan data meliputi penghapusan noise, normalisasi teks, penanganan emoji, stopwords removal,  
                tokenisasi, stemming dan preprocessing untuk bahasa Indonesia.
            <br>
                Dengan menggunakan metode Support Vector Mechine dan Naive Bayes Classifier, aplikasi ini dapat memberikan label sentimen pada ulasan aplikasi exchange crypto. 
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cara Kerja Section
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">Bagaimana Cara Kerjanya?</h2>
        <div class="section-text">
            <p>Aplikasi ini menggunakan pendekatan sistematis untuk memberikan analisis sentimen sebagai berikut:</p>
        </div>
        <ol style="margin-left: 2rem; text-align: left;">
            <li>
                <b>Data Scraping:</b>
                Menggunakan teknik web scraping untuk mengumpulkan review dari Google Play Store menggunakan google play scraper.
            </li>
            <li>
                <b>Data Preprocessing & Cleaning:</b>
                Pembersihan data meliputi penghapusan noise, normalisasi teks, penanganan emoji, stopwords removal,  
                tokenisasi, stemming dan preprocessing untuk bahasa Indonesia.
            </li>
            <li>
                <b>Sentiment Analysis:</b>
                Implementasi model Support Vector Mechine dan Naive Bayes Classifier dan penerapan pelabelan data dengan model lexicon based untuk sentimen positif, netral dan negatif.
            </li>
            <li>
                <b>Visualisasi:</b>
                Presentasi hasil dalam dashboard interaktif dengan berbagai tabel, chart, grafik, 
                dan visualisasi yang dapat dipahami.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    

    app()