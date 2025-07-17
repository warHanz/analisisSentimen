import streamlit as st
import zipfile
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils.analysis_utils import *
from models.models import prepare_data, train_and_evaluate_multiple_splits, plot_data_split, plot_accuracy_comparison, plot_confusion_matrix

def display_metric_box(label, value, emoji):
    """Menampilkan metrik dengan emoticon dan styling minimalis."""
    st.markdown(
        f"<div class='metric-box'>"
        f"{emoji} <span class='metric-label'>{label}:</span> "
        f"<span class='metric-value'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

def app():
    # CSS Kustom dari kode awal
    st.markdown("""
        <style>
        h1, h2, h3, h4 {
            color: #2C3E50;
            text-align: center;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }
        h1 { font-size: 2.5em; color: #4CAF50; }
        h2 { font-size: 2.0em; }
        h3 { font-size: 1.6em; }
        .stMarkdown p {
            font-size: 1.1em;
            line-height: 1.6;
            color: #555;
            text-align: center;
        }
        .metric-card {
            background-color: #f8f8f8;
            border-left: 5px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            text-align: center;
        }
        .metric-card h3 {
            color: #4CAF50;
            font-size: 1.5em;
            margin-bottom: 5px;
        }
        .metric-card p {
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin: 0;
        }
        .metric-box {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 6px;
            margin: 2px 0;
            font-size: 0.9em;
            color: #000;
            font-weight: 400;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .metric-box .metric-label, .metric-box .metric-value {
            font-weight: bold;
            color: #000;
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
        .stSelectbox label {
            font-size: 1.1em;
            color: #2C3E50;
            text-align: center;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

    # Fungsi untuk mengonversi tabel ke gambar
    def table_to_image(df, title):
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.3))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1)
        ax.set_title(title, fontsize=9, pad=10)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    # Fungsi untuk mengonversi figure Plotly ke PNG dengan penanganan error
    def plotly_to_png(fig):
        buf = BytesIO()
        try:
            pio.write_image(fig, file=buf, format='png', scale=2)
            buf.seek(0)
            return buf
        except Exception as e:
            st.error(f"Gagal mengekspor gambar Plotly: {str(e)}")
            return None

    # Fungsi untuk mengonversi word cloud ke PNG
    def wordcloud_to_png(wc, title):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=9, pad=10)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    # Fungsi untuk memformat nilai metrik
    def format_value(val):
        return '{:.2f}%'.format(val) if isinstance(val, (int, float)) else str(val)

    st.title("📊 Analisis Sentimen")
    st.markdown("Unggah file CSV untuk analisis data.")

    sentiment_map = {
        'Sentimen Positif': 'Positive',
        'Sentimen Netral': 'Neutral',
        'Sentimen Negatif': 'Negative'
    }
    color_map = {
        'Positive': 'Greens',
        'Neutral': 'Greys',
        'Negative': 'Reds'
    }

    uploaded_file = st.file_uploader("Unggah CSV", type=["csv"], key="preprocessed_data_uploader")
    if uploaded_file is None:
        st.info("Unggah file CSV untuk analisis.")
    else:
        with st.spinner("Memproses..."):
            processed_data = load_preprocessed_data(file=uploaded_file)
            if processed_data is None:
                st.error("Gagal memuat data. Periksa format CSV.")
                return

            expected_stemmed_col = 'Stemmed Text'
            if expected_stemmed_col not in processed_data.columns:
                st.error(f"Kolom '{expected_stemmed_col}' tidak ditemukan.")
                return

            processed_data['stemmed_text'] = processed_data[expected_stemmed_col].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else x)

            positive_words, negative_words = load_sentiment_lexicon()
            processed_data[['Sentiment Score', 'Sentiment']] = processed_data['stemmed_text'].apply(
                lambda x: pd.Series(label_sentiment(x, positive_words, negative_words)))

            st.markdown("<h2>Hasil Visualisasi</h2>", unsafe_allow_html=True)
            total_data = len(processed_data)
            st.markdown(f"<div class='metric-card'><h3>Total Data</h3><p>{total_data}</p></div>", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("<h3>Distribusi Sentimen</h3>", unsafe_allow_html=True)
            col_data_table, col_sentiment_dist = st.columns([3, 2])
            with col_data_table:
                st.write("#### Contoh Data")
                st.dataframe(processed_data[['stemmed_text', 'Sentiment Score', 'Sentiment']].head(10))
            with col_sentiment_dist:
                st.write("#### Proporsi Sentimen")
                sentiment_counts = processed_data['Sentiment'].value_counts()
                unique_classes = processed_data['Sentiment'].nunique()
                if unique_classes <= 1:
                    st.error(f"Hanya {unique_classes} kelas sentimen: {processed_data['Sentiment'].unique()}.")
                else:
                    colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
                    color_list = [colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Proporsi Sentimen",
                        hole=0.4,
                        color_discrete_sequence=color_list,
                        height=300
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart_new")

            st.markdown("---")
            st.markdown("<h3>Word Cloud</h3>", unsafe_allow_html=True)
            tabs_wc = st.tabs(["Semua", "Positif", "Netral", "Negatif"])
            with tabs_wc[0]:
                fig, ax = plt.subplots(figsize=(6, 3))
                wc_all = generate_wordcloud(processed_data[expected_stemmed_col], "Semua Data", 'viridis')
                ax.imshow(wc_all, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            for idx, (tab_title, sentiment_label) in enumerate(sentiment_map.items()):
                with tabs_wc[idx + 1]:
                    sentiment_data = processed_data[processed_data['Sentiment'] == sentiment_label][expected_stemmed_col]
                    if len(sentiment_data) > 0:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        wc_sentiment = generate_wordcloud(sentiment_data, sentiment_label, color_map[sentiment_label])
                        ax.imshow(wc_sentiment, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info(f"Tidak ada data {tab_title}.")

            st.markdown("---")
            st.markdown("<h3>Top 10 Kata</h3>", unsafe_allow_html=True)
            tabs_freq = st.tabs(["Semua", "Positif", "Netral", "Negatif"])
            with tabs_freq[0]:
                st.plotly_chart(generate_frequency_chart(processed_data[expected_stemmed_col]), use_container_width=True, key="freq_chart_new_all")
            for idx, (tab_title, sentiment_label) in enumerate(sentiment_map.items()):
                with tabs_freq[idx + 1]:
                    sentiment_data = processed_data[processed_data['Sentiment'] == sentiment_label][expected_stemmed_col]
                    if len(sentiment_data) > 0:
                        st.plotly_chart(generate_frequency_chart(sentiment_data), use_container_width=True, key=f"freq_chart_new_{sentiment_label.lower()}")
                    else:
                        st.info(f"Tidak ada data {tab_title}.")

            st.markdown("---")
            st.markdown("<h2>Evaluasi Model</h2>", unsafe_allow_html=True)
            if unique_classes <= 1:
                st.error("Minimal 2 kelas diperlukan.")
            else:
                processed_data['Text_for_Model'] = processed_data[expected_stemmed_col].apply(preprocess_for_model)
                test_sizes = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3}
                all_results = {}
                for split_ratio, test_size in test_sizes.items():
                    with st.spinner(f"Melatih model ({split_ratio})..."):
                        results = train_and_evaluate_multiple_splits(
                            processed_data['Text_for_Model'],
                            processed_data['Sentiment'],
                            test_sizes=[test_size]
                        )
                        all_results.update(results)
                st.success("Analisis selesai untuk semua rasio!")

                for split_ratio in test_sizes.keys():
                    split_key = f"Split {split_ratio}"
                    results = all_results[split_key]['results']
                    train_size = all_results[split_key]['train_size']
                    test_size_count = all_results[split_key]['test_size']
                    y_test = all_results[split_key]['y_test']

                    st.markdown(f"<h3>Evaluasi Rasio {split_ratio}</h3>", unsafe_allow_html=True)
                    col_model_overview1, col_model_overview2 = st.columns(2)
                    with col_model_overview1:
                        st.write("#### Data Latih & Uji")
                        st.plotly_chart(plot_data_split(train_size, test_size_count), use_container_width=True, key=f"split_chart_new_{split_ratio}")
                    with col_model_overview2:
                        st.write("#### Akurasi")
                        st.plotly_chart(plot_accuracy_comparison(results, split_ratio), use_container_width=True, key=f"acc_chart_new_{split_ratio}")

                    st.markdown("---")
                    st.markdown("<h3>Detail Evaluasi</h3>", unsafe_allow_html=True)
                    for name in results:
                        with st.expander(f"{name} Model ({split_ratio})"):
                            col_eval1, col_eval2 = st.columns(2)
                            with col_eval1:
                                st.write("##### Matriks Konfusi")
                                st.plotly_chart(
                                    plot_confusion_matrix(y_test, results[name]['pred'], name, split_ratio),
                                    use_container_width=True,
                                    key=f"cm_{name}_new_{split_ratio}"
                                )
                            with col_eval2:
                                st.write("##### Laporan Klasifikasi")
                                report_df = pd.DataFrame(results[name]['report']).T.round(2)
                                st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))

                                st.write("##### Ringkasan Hasil Evaluasi")
                                st.markdown("<h4 style='margin: 0;'>Ringkasan Hasil Evaluasi</h4>", unsafe_allow_html=True)

                                report = results[name]['report']
                                # Akurasi
                                display_metric_box("Akurasi Keseluruhan", format_value(report['accuracy'] * 100), "🔍")
                                st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
                                # Presisi
                                st.markdown("<strong>Presisi</strong>", unsafe_allow_html=True)
                                for label in ['Positive', 'Neutral', 'Negative']:
                                    if label in report:
                                        display_metric_box(f"Presisi {label}", format_value(report[label]['precision'] * 100), "✅")
                                display_metric_box("Macro Avg Presisi", format_value(report['macro avg']['precision'] * 100), "📊")
                                display_metric_box("Weighted Avg Presisi", format_value(report['weighted avg']['precision'] * 100), "⚖️")
                                st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
                                # Recall
                                st.markdown("<strong>Recall</strong>", unsafe_allow_html=True)
                                for label in ['Positive', 'Neutral', 'Negative']:
                                    if label in report:
                                        display_metric_box(f"Recall {label}", format_value(report[label]['recall'] * 100), "🔄")
                                display_metric_box("Macro Avg Recall", format_value(report['macro avg']['recall'] * 100), "📊")
                                display_metric_box("Weighted Avg Recall", format_value(report['weighted avg']['recall'] * 100), "⚖️")
                                st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
                                # F1-Score
                                st.markdown("<strong>F1-Score</strong>", unsafe_allow_html=True)
                                for label in ['Positive', 'Neutral', 'Negative']:
                                    if label in report:
                                        display_metric_box(f"F1-Score {label}", format_value(report[label]['f1-score'] * 100), "⚖️")
                                display_metric_box("Macro Avg F1-Score", format_value(report['macro avg']['f1-score'] * 100), "📊")
                                display_metric_box("Weighted Avg F1-Score", format_value(report['weighted avg']['f1-score'] * 100), "⚖️")

                st.markdown("---")
                st.subheader("Unduh Hasil")
                if st.button("Unduh ZIP"):
                    with st.spinner("Menyusun ZIP..."):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            pie_fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Proporsi Sentimen",
                                hole=0.4,
                                color_discrete_sequence=color_list,
                                height=300
                            )
                            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
                            pie_fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
                            pie_buf = plotly_to_png(pie_fig)
                            if pie_buf:
                                zip_file.writestr("distribusi.png", pie_buf.read())
                            else:
                                st.warning("Gagal menambahkan distribusi.png ke ZIP.")

                            wc_all = generate_wordcloud(processed_data[expected_stemmed_col], "Semua Data", 'viridis')
                            wc_buf = wordcloud_to_png(wc_all, "Word Cloud - Semua")
                            if wc_buf:
                                zip_file.writestr("wordcloud_semua.png", wc_buf.read())
                            for sentiment_label in sentiment_map.values():
                                sentiment_data = processed_data[processed_data['Sentiment'] == sentiment_label][expected_stemmed_col]
                                if len(sentiment_data) > 0:
                                    wc_sentiment = generate_wordcloud(sentiment_data, sentiment_label, color_map[sentiment_label])
                                    wc_buf = wordcloud_to_png(wc_sentiment, f"Word Cloud - {sentiment_label}")
                                    if wc_buf:
                                        zip_file.writestr(f"wordcloud_{sentiment_label.lower()}.png", wc_buf.read())

                            freq_fig_all = generate_frequency_chart(processed_data[expected_stemmed_col])
                            freq_buf = plotly_to_png(freq_fig_all)
                            if freq_buf:
                                zip_file.writestr("frekuensi_semua.png", freq_buf.read())
                            for sentiment_label in sentiment_map.values():
                                sentiment_data = processed_data[processed_data['Sentiment'] == sentiment_label][expected_stemmed_col]
                                if len(sentiment_data) > 0:
                                    freq_fig = generate_frequency_chart(sentiment_data)
                                    freq_buf = plotly_to_png(freq_fig)
                                    if freq_buf:
                                        zip_file.writestr(f"frekuensi_{sentiment_label.lower()}.png", freq_buf.read())

                            sample_table = processed_data[['stemmed_text', 'Sentiment Score', 'Sentiment']].head(10)
                            table_buf = table_to_image(sample_table, "Contoh Data")
                            if table_buf:
                                zip_file.writestr("contoh_data.png", table_buf.read())

                            for split_ratio in test_sizes.keys():
                                split_key = f"Split {split_ratio}"
                                results = all_results[split_key]['results']
                                train_size = all_results[split_key]['train_size']
                                test_size_count = all_results[split_key]['test_size']
                                y_test = all_results[split_key]['y_test']

                                split_fig = plot_data_split(train_size, test_size_count)
                                split_buf = plotly_to_png(split_fig)
                                if split_buf:
                                    zip_file.writestr(f"{split_ratio.replace(':', '_')}_pembagian_data.png", split_buf.read())
                                else:
                                    st.warning(f"Gagal menambahkan {split_ratio}_pembagian_data.png ke ZIP.")

                                acc_fig = plot_accuracy_comparison(results, split_ratio)
                                acc_buf = plotly_to_png(acc_fig)
                                if acc_buf:
                                    zip_file.writestr(f"{split_ratio.replace(':', '_')}_akurasi.png", acc_buf.read())
                                else:
                                    st.warning(f"Gagal menambahkan {split_ratio}_akurasi.png ke ZIP.")

                                for name in results:
                                    cm_fig = plot_confusion_matrix(y_test, results[name]['pred'], name, split_ratio)
                                    cm_buf = plotly_to_png(cm_fig)
                                    if cm_buf:
                                        zip_file.writestr(f"{split_ratio.replace(':', '_')}_cm_{name}.png", cm_buf.read())
                                    else:
                                        st.warning(f"Gagal menambahkan {split_ratio}_cm_{name}.png ke ZIP.")

                                    report_df = pd.DataFrame(results[name]['report']).T.round(2)
                                    table_buf = table_to_image(report_df, f"Laporan - {name} ({split_ratio})")
                                    if table_buf:
                                        zip_file.writestr(f"{split_ratio.replace(':', '_')}_laporan_{name}.png", table_buf.read())

                        zip_buffer.seek(0)
                        st.download_button(
                            label="Unduh ZIP",
                            data=zip_buffer,
                            file_name="hasil_analisis.zip",
                            mime="application/zip"
                        )
                        st.success("ZIP siap diunduh!")

if __name__ == "__main__":
    app()