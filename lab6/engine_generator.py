import gc
from processing.db_builder import create_db
from processing.db_management import save_engine

# --- KONFIGURACJA ---
DATA_FILENAME = "data/train-00000-of-00041.parquet"
MAX_SVD_K = 1000
METHODS = ["sklearn_tfidf", "custom_tfidf", "bm25"]


def main():
    print("Rozpoczynam zoptymalizowane generowanie silników...\n")

    for method in METHODS:
        print(f"Budowanie dla metody: {method}...")

        # obliczenia
        A, svd_components, vocab, titles, urls, texts = create_db(
            filename=DATA_FILENAME,
            max_k=MAX_SVD_K,
            method=method
        )


        del texts
        texts = None
        gc.collect()  # Wymuszenie czyszczenia zwolnionej pamięci

        target_filename = f"search_db_{method}"
        save_engine(A, svd_components, vocab, titles, urls, texts, filename=target_filename)

        del A, svd_components, vocab, titles, urls
        gc.collect()

        print(f"Zakończono {method} i zwolniono pamięć.\n")

    print("Wszystkie silniki zostały wygenerowane")


if __name__ == "__main__":
    main()