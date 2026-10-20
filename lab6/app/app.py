import flet as ft
from lab6.processing.db_management import load_engine
from lab6.app.settings_page import create_settings_page
from lab6.app.search_page import create_search_page


def main(page: ft.Page):
    page.title = "Sloople"

    def show_settings():
        page.views.clear()
        page.views.append(create_settings_page(page, on_accept=load_and_show_search))
        page.update()

    def load_and_show_search(settings):
        engine_type = settings["engine_type"]
        matrix_type = settings["matrix_type"]
        svd_k = settings["svd_k"] if matrix_type == "svd" else None

        filename_target = f"search_db_{engine_type}"

        try:
            engine, vocab, titles, urls, texts = load_engine(filename_target, approx_rows=svd_k)
            page.views.append(create_search_page(page, engine, vocab, titles, urls, texts, settings, on_back=show_settings))
            page.update()
        except FileNotFoundError:
            # Fallback dla starej bazy, jeśli specyficzna dla silnika nie istnieje
            try:
                engine, vocab, titles, urls, texts = load_engine("search_db", approx_rows=svd_k)
                page.snack_bar = ft.SnackBar(ft.Text(f"Brak pliku '{filename_target}', ładuję domyślne 'search_db'!"))
                page.snack_bar.open = True
                page.views.append(create_search_page(page, engine, vocab, titles, urls, texts, settings, on_back=show_settings))
                page.update()
            except FileNotFoundError:
                page.snack_bar = ft.SnackBar(ft.Text("Brak danych bazy wyszukiwania. Najpierw wygeneruj ją używając builder'a."))
                page.snack_bar.open = True
                page.update()

    show_settings()


def run():
    ft.app(target=main)
