from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from functools import lru_cache
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def create_vectorizer(max_features=10000, ngram_range=(1, 3), min_df=2):
    """Create and cache a TF-IDF vectorizer with optimized parameters"""
    logger.info(f"Creating TF-IDF vectorizer with max_features={max_features}, ngram_range={ngram_range}, min_df={min_df}")
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True
    )

def prepare_data(texts, labels, test_size=0.2, random_state=42, max_features=10000, ngram_range=(1, 3), min_df=2):
    """
    Split data into training and testing sets and convert texts to TF-IDF features.
    """
    if texts.empty or labels.empty or len(texts) != len(labels):
        logger.error("Invalid input: texts or labels are empty or mismatched")
        raise ValueError("Texts and labels must be non-empty and have the same length")

    unique_labels = labels.nunique()
    if unique_labels < 2:
        logger.error(f"Hanya ditemukan {unique_labels} kelas unik. Tidak dapat melakukan pembagian stratified atau melatih klasifier.")
        raise ValueError(f"Tidak cukup kelas unik ({unique_labels}) untuk pembagian stratified atau pelatihan model.")

    logger.info(f"Preparing data: splitting with test_size={test_size} and transforming to TF-IDF")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    vectorizer = create_vectorizer(max_features, ngram_range, min_df)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    logger.info(f"Data prepared: {X_train_tfidf.shape[0]} training samples, {X_test_tfidf.shape[0]} test samples")
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Train and evaluate SVM and NBC models with optimized hyperparameter tuning.
    """
    if X_train_tfidf.shape[0] == 0 or X_test_tfidf.shape[0] == 0:
        logger.error("Empty training or test data")
        raise ValueError("Training and test data must not be empty")

    results = {}
    class_names = sorted(y_train.unique().tolist())

    # Optimized SVM with GridSearchCV
    logger.info("Tuning and training SVM model")
    try:
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear'],
            'class_weight': [None, 'balanced']
        }

        svm_grid = GridSearchCV(
            SVC(probability=True),
            svm_param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy',
            verbose=0
        )
        svm_grid.fit(X_train_tfidf, y_train)

        svm_pred = svm_grid.predict(X_test_tfidf)
        results['SVM'] = {
            'pred': svm_pred,
            'acc': accuracy_score(y_test, svm_pred),
            'report': classification_report(y_test, svm_pred, output_dict=True, zero_division=0, target_names=class_names),
            'best_params': svm_grid.best_params_
        }
        logger.info(f"SVM best params: {svm_grid.best_params_} Accuracy: {results['SVM']['acc']:.2f}")
    except Exception as e:
        logger.error(f"Error training SVM model: {e}")
        results['SVM'] = {'error': str(e)}

    # Optimized Naive Bayes with GridSearchCV
    logger.info("Tuning and training NBC model")
    try:
        nb_param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0],
            'fit_prior': [True, False]
        }

        nb_grid = GridSearchCV(
            MultinomialNB(),
            nb_param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy',
            verbose=0
        )
        nb_grid.fit(X_train_tfidf, y_train)

        nb_pred = nb_grid.predict(X_test_tfidf)
        results['NBC'] = {
            'pred': nb_pred,
            'acc': accuracy_score(y_test, nb_pred),
            'report': classification_report(y_test, nb_pred, output_dict=True, zero_division=0, target_names=class_names),
            'best_params': nb_grid.best_params_
        }
        logger.info(f"NBC best params: {nb_grid.best_params_} Accuracy: {results['NBC']['acc']:.2f}")
    except Exception as e:
        logger.error(f"Error training NBC model: {e}")
        results['NBC'] = {'error': str(e)}

    return results

def train_and_evaluate_multiple_splits(texts, labels, test_sizes=[0.1, 0.2, 0.3], random_state=42, max_features=10000, ngram_range=(1, 3), min_df=2):
    """
    Train and evaluate models for multiple data splits.
    """
    all_results = {}
    for test_size in test_sizes:
        split_ratio = f"{int((1-test_size)*100)}:{int(test_size*100)}"
        logger.info(f"Evaluating models with test_size={test_size} ({split_ratio})")
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_data(
            texts, labels, test_size=test_size, random_state=random_state, max_features=max_features, ngram_range=ngram_range, min_df=min_df
        )
        results = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
        all_results[f"Split {split_ratio}"] = {
            'results': results,
            'train_size': X_train_tfidf.shape[0],
            'test_size': X_test_tfidf.shape[0],
            'y_test': y_test,
            'vectorizer': vectorizer
        }
    return all_results

def plot_data_split(train_size, test_size):
    """
    Menghasilkan diagram pie yang menunjukkan pembagian data latih dan uji.
    """
    labels = ['Data Latih', 'Data Uji']
    sizes = [train_size, test_size]
    colors = ['#4CAF50', '#FFC107']

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3,
                                 marker_colors=colors,
                                 hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>")])
    fig.update_layout(title_text='Pembagian Data Latih dan Uji', title_x=0.5, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_accuracy_comparison(results, split_name):
    """
    Menghasilkan diagram batang yang membandingkan akurasi model untuk split tertentu.
    """
    model_names = []
    accuracies = []
    for name, data in results.items():
        if 'acc' in data and 'error' not in data:
            model_names.append(name)
            accuracies.append(data['acc'])

    if not model_names:
        fig = go.Figure()
        fig.update_layout(title_text=f'Tidak ada data akurasi model untuk split {split_name}', height=350, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    df_acc = pd.DataFrame({'Model': model_names, 'Akurasi': accuracies})
    fig = px.bar(df_acc, x='Model', y='Akurasi',
                 title=f'Akurasi Model ({split_name})',
                 color='Model',
                 color_discrete_map={'SVM': '#17A2B8', 'NBC': '#6C757D'},
                 text_auto='.2%',
                 height=350)
    fig.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(range=[0, 1])
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name, split_name):
    """
    Menghasilkan heatmap untuk matriks konfusi.
    """
    all_labels = sorted(np.unique(np.concatenate((y_true, y_pred))).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig = px.imshow(cm,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    x=all_labels,
                    y=all_labels,
                    text_auto=True,
                    color_continuous_scale="Viridis",
                    title=f'Matriks Konfusi: {model_name} ({split_name})',
                    height=400)
    fig.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
    return fig