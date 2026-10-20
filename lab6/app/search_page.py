import flet as ft
from pathlib import Path
from lab6.processing.db_queries import query_db_full, query_db_svd

def create_search_page(page: ft.Page, engine, vocab, titles, urls, texts, settings, on_back):
    img_path = str(Path(__file__).resolve().parent / 'sloople_img.png')
    logo = ft.Image(src=img_path, width=250)

    query_field = ft.TextField(hint_text='Enter your query here', expand=True)

    search_layout = ft.ListView(expand=True, spacing=25, padding=20)

    result_len = settings.get("result_len", 10)
    is_svd = settings.get("matrix_type") == "svd"

    def button_clicked(event):
        query = query_field.value

        if query and query.strip():
            query_field.error_text = None

            main_layout.alignment = ft.MainAxisAlignment.START
            logo.width = 120

            search_layout.controls.clear()

            if is_svd and isinstance(engine, tuple):
                results = query_db_svd(query, engine, vocab, titles, urls, texts, result_len=result_len)
            else:
                results = query_db_full(query, engine, vocab, titles, urls, texts, result_len=result_len)

            for res in results:
                url = res["url"]
                result_item = ft.Column(
                    controls=[
                        ft.Text(url, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.GestureDetector(
                            content=ft.Text(
                                res["title"],
                                size=20,
                                color=ft.Colors.BLUE_700,
                                weight=ft.FontWeight.W_500,
                            ),
                            on_tap=lambda e, u=url: page.launch_url(u),
                            mouse_cursor=ft.MouseCursor.CLICK,
                        ),
                        ft.Text(res["snippet"], size=14, color=ft.Colors.ON_SURFACE),
                    ],
                    spacing=2
                )

                search_layout.controls.append(result_item)

            page.update()
        else:
            query_field.error_text = "Query cannot be empty"
            page.update()

    search_button = ft.ElevatedButton(content=ft.Text('Search'), on_click=button_clicked)
    query_field.on_submit = button_clicked

    main_layout = ft.Column(
        controls=[
            ft.Row([
                ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: on_back(), tooltip="Wróć do ustawień")
            ]),
            logo,
            ft.Row([query_field, search_button], alignment=ft.MainAxisAlignment.CENTER),
            search_layout
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True,
    )

    return ft.View(route="/search", controls=[main_layout])

