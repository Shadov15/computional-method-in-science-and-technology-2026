import pandas as pd
import numpy as np
import scipy.sparse
from collections import defaultdict
from scipy.sparse import coo_matrix
from .text_operations import vectorize_text


def l2_normalize_document_vectors(matrix):
    """Normalizuje kolumny macierzy dokumentów do normy L2 = 1."""
    doc_norms = np.sqrt(matrix.power(2).sum(axis=0)).A1
    doc_norms[doc_norms == 0] = 1.0

    normalization = scipy.sparse.diags(1.0 / doc_norms)

    return (matrix @ normalization).tocsr()


def tf_idf_transform(N, data, document_frequency, global_vocab, rows):
    """Transformuje surowe zliczenia TF w wagi TF-IDF."""
    id_to_word = {v: k for k, v in global_vocab.items()}

    for i in range(len(data)):
        word_id = rows[i]
        word = id_to_word[word_id]
        nw = document_frequency[word]
        idf = np.log(N / nw)
        data[i] = data[i] * idf


def bm25_transform(N, data, document_frequency, global_vocab, rows, cols, doc_lengths, k1=1.5, b=0.75):
    """Zwektoryzowana wersja BM25."""
    data_arr = np.array(data, dtype=float)
    rows_arr = np.array(rows)
    cols_arr = np.array(cols)

    # Tworzymy tablicę IDF o rozmiarze słownika, indeksowaną word_id
    df_arr = np.zeros(len(global_vocab))
    for word, word_id in global_vocab.items():
        df_arr[word_id] = document_frequency[word]

    # Obliczamy IDF dla wszystkich słów na raz
    idf_arr = np.log(((N - df_arr + 0.5) / (df_arr + 0.5)) + 1.0)

    avg_doc_len = np.mean(doc_lengths) if N > 0 else 1.0

    idf_per_element = idf_arr[rows_arr]         # IDF dla każdego elementu w `data`
    D_per_element = doc_lengths[cols_arr]       # Długość dokumentu dla każdego elementu w `data`

    tf = data_arr
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * ((1 - b) + (b * (D_per_element / avg_doc_len)))

    transformed_data = idf_per_element * (numerator / denominator)
    for i in range(len(data)):
        data[i] = transformed_data[i]


def create_db(filename, max_k=None, method='custom_tfidf'):
    """
    Buduje bazę danych wyszukiwarki z dokumentów w formacie parquet.
    
    Args:
        filename: Nazwa pliku parquet w data/
        max_k: Liczba komponentów dla SVD (opcjonalnie)
        method: 'custom_tfidf', 'sklearn_tfidf' lub 'bm25'
    
    Returns:
        (csr_matrix, svd_components, vocab, titles, urls)
    """
    df = pd.read_parquet(f"{filename}")
    N = len(df)

    titles = df['title'].tolist()
    urls = df['url'].tolist()
    texts = df['text'].tolist()

    if method == 'sklearn_tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        csr_matrix_sklearn = vectorizer.fit_transform(df['text'])

        global_vocab = vectorizer.vocabulary_
        csr_matrix = csr_matrix_sklearn.T.tocsr()

    else:
        # Logika dla custom_tfidf oraz bm25
        global_vocab = {}
        document_frequency = defaultdict(int)

        rows = []
        cols = []
        data = []

        doc_lengths = np.zeros(N)

        for i, text in enumerate(df['text']):
            word_counts = vectorize_text(text)
            doc_lengths[i] = sum(word_counts.values())

            for word, count in word_counts.items():
                if word not in global_vocab:
                    global_vocab[word] = len(global_vocab)

                word_id = global_vocab[word]
                document_frequency[word] += 1

                rows.append(word_id)
                cols.append(i)
                data.append(count)

        if method == 'custom_tfidf':
            tf_idf_transform(N, data, document_frequency, global_vocab, rows)
        elif method == 'bm25':
            bm25_transform(N, data, document_frequency, global_vocab, rows, cols, doc_lengths)
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(global_vocab), N))
        csr_matrix = l2_normalize_document_vectors(sparse_matrix.tocsr())

    svd_components = None

    if max_k is not None:
        k_to_compute = min(max_k, csr_matrix.shape[0] - 1, csr_matrix.shape[1] - 1)

        U, S, Vt = scipy.sparse.linalg.svds(csr_matrix, k=k_to_compute)

        sort_indices = np.argsort(S)[::-1]
        S = S[sort_indices]
        U = U[:, sort_indices]
        Vt = Vt[sort_indices, :]

        svd_components = (U, S, Vt)

    return csr_matrix, svd_components, global_vocab, titles, urls, texts
