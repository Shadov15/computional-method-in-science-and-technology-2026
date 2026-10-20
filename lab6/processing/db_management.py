import os
import json
import pandas as pd
import numpy as np
import scipy.sparse
from pathlib import Path
import faiss

# ======================================================
# Funkcje do zapisu i odczytu silnika wyszukiwarki
# ======================================================

def _data_path(filename):
    return Path(__file__).resolve().parent.parent / "data" / filename


def _load_texts_fallback(expected_len):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for parquet_path in data_dir.glob("*.parquet"):
        try:
            texts = pd.read_parquet(parquet_path, columns=["text"])['text'].tolist()
            if len(texts) == expected_len:
                return texts
        except Exception:
            continue
    return [""] * expected_len


def save_engine(A, svd_components, vocab, titles, urls, texts, filename="search_db"):
    """Zapisuje macierz rzadką, słownik i metadane dokumentów na dysk."""
    scipy.sparse.save_npz(_data_path(f"{filename}_matrix.npz"), A)

    if svd_components is not None:
        U, S, Vt = svd_components
        np.save(_data_path(f"{filename}_U_max.npy"), U)
        np.save(_data_path(f"{filename}_S_max.npy"), S)
        np.save(_data_path(f"{filename}_Vt_max.npy"), Vt)

        del U, S, Vt        # zwolnienie pamięci

    # Zapis słownika, tytułów i linków do JSON
    metadata = {
        "vocab": vocab,
        "titles": titles,
        "urls": urls,
        "texts": texts,
    }
    with open(_data_path(f"{filename}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("Silnik wyszukiwarki został zapisany na dysku.")


def load_engine(filename="search_db", approx_rows=None):
    """Ładuje i docina gotowy silnik do pamięci w ułamku sekundy."""
    # wczytanie metadanych
    with open(_data_path(f"{filename}_meta.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = metadata.get("texts")
    if texts is None:
        texts = _load_texts_fallback(len(metadata["titles"]))

    # wczytywanie SVD - zabezpieczenie przed crash-em, jeśli plików nie ma
    u_path = _data_path(f"{filename}_U_max.npy")
    if approx_rows is not None and os.path.exists(u_path):
        U = np.load(u_path)
        S = np.load(_data_path(f"{filename}_S_max.npy"))
        Vt = np.load(_data_path(f"{filename}_Vt_max.npy"))

        if approx_rows > len(S):
            raise ValueError(f"Żądane k={approx_rows} jest większe niż maksymalne zapisane k={len(S)}.")

        U_k = U[:, :approx_rows]
        S_k = S[:approx_rows]
        Vt_k = Vt[:approx_rows, :]

        doc_lat = np.ascontiguousarray((S_k[:, np.newaxis] * Vt_k).T, dtype=np.float32)

        faiss.normalize_L2(doc_lat)

        # Inicjalizacja i budowa grafu HNSW
        hnsw_index = faiss.IndexHNSWFlat(approx_rows, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efConstruction = 200  # dokładność grafu
        hnsw_index.add(doc_lat)

        # zwracamy indeks HNSW zamiast macierzy SVD
        return (U_k, S_k, hnsw_index), metadata["vocab"], metadata["titles"], metadata["urls"], texts

    # Jeśli nie chcemy aproksymacji (lub SVD nie istnieje na dysku), ładujemy macierz główną
    A = scipy.sparse.load_npz(_data_path(f"{filename}_matrix.npz"))
    return A, metadata["vocab"], metadata["titles"], metadata["urls"], texts


# ======================================================
# Przykładowe wywołania (Frontend-Mockup)
# ======================================================

if __name__ == "__main__":
    from db_queries import query_db_full, query_db_svd
    
    # Budowa bazy algorytmem BM25
    # A, svd_components, vocab, titles, urls = create_db('dane.parquet', max_k=1000, method='bm25')
    # save_engine(A, svd_components, vocab, titles, urls)

    # Symulacja zapytania z frontend'u
    try:
        engine, vocab, titles, urls, texts = load_engine('search_db', approx_rows=500)

        # Przykładowe zapytanie
        query = "What are some algorithms for traversing a graph?"

        print(f"Szukam dla: '{query}'")

        if isinstance(engine, tuple):
            results = query_db_svd(query, engine, vocab, titles, urls, texts, result_len=10)
        else:
            results = query_db_full(query, engine, vocab, titles, urls, texts, result_len=10)

        print(json.dumps(results, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        print("Nie znaleziono plików bazy. Najpierw odkomentuj logikę create_db().")