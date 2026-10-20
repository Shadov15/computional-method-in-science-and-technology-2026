import flet as ft

def create_settings_page(page: ft.Page, on_accept):
    engine_radio = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="sklearn_tfidf", label="tfIdf (sklearn)"),
            ft.Radio(value="custom_tfidf", label="tfIdf custom"),
            ft.Radio(value="bm25", label="BM25")
        ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=40
        ),
        value="bm25"
    )

    svd_slider = ft.Slider(min=200, max=1000, divisions=16, value=500, label="rząd k={value}")

    def on_matrix_change(e):
        svd_slider.disabled = (matrix_radio.value == "full")
        page.update()

    matrix_radio = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="full", label="Full matrix"),
            ft.Radio(value="svd", label="SVD approximation"),
        ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=40
        ),
        value="svd",
        on_change=on_matrix_change
    )

    results_slider = ft.Slider(min=10, max=50, divisions=8, value=10, label="{value} wyników")

    def submit(e):
        settings = {
            "engine_type": engine_radio.value,
            "matrix_type": matrix_radio.value,
            "result_len": int(results_slider.value),
            "svd_k": int(svd_slider.value)
        }
        on_accept(settings)

    on_matrix_change(None) # inicjalizacja stanu SVD suwaka

    return ft.View(
        route="/settings",
        controls=[
            ft.Column(
                controls=[
                    ft.Text("Sloople - Start Configuration", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

                    ft.Text("Choose engine and metric:", size=18, weight=ft.FontWeight.W_500),
                    engine_radio,
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

                    ft.Text("Matrix reprezentation:", size=18, weight=ft.FontWeight.W_500),
                    matrix_radio,
                    ft.Text("SVD approximation row:"),
                    svd_slider,
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

                    ft.Text("Number of results:", size=18, weight=ft.FontWeight.W_500),
                    results_slider,
                    ft.Divider(height=30, color=ft.Colors.TRANSPARENT),

                    ft.ElevatedButton("Accept and load engine", on_click=submit, width=300, height=50, bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE)
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True
            )
        ],
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        padding=50
    )

