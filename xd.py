# lead_card_window.py
# Ventana estilo Zoho CRM para mostrar información del prospecto

import tkinter as tk
from tkinter import ttk

APP_BG = "#015666"      # verde petróleo de fondo
CARD_BG = "#FFFFFF"     # tarjeta blanca
LABEL_FG = "#666666"    # texto gris para etiquetas
VALUE_FG = "#222222"    # texto oscuro para valores
EMAIL_FG = "#d32f2f"    # rojo para email
TITLE_FG = "#333333"    # título

DEFAULT_FONT = ("Segoe UI", 10)
DEFAULT_FONT_BOLD = ("Segoe UI", 10, "bold")
TITLE_FONT = ("Segoe UI", 13, "bold")


class LeadCardWindow(tk.Tk):
    def __init__(self, lead_data: dict):
        super().__init__()
        self.title("Información del prospecto")
        self.configure(bg=APP_BG)
        self.minsize(520, 420)
        # la ventana se puede redimensionar libremente
        self.resizable(True, True)

        # grid principal para que todo se escale
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # contenedor centrado
        outer = tk.Frame(self, bg=APP_BG)
        outer.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        # tarjeta
        card = tk.Frame(outer, bg=CARD_BG, bd=0, relief="flat")
        card.grid(row=0, column=0, sticky="nsew")
        card.rowconfigure(0, weight=1)
        card.columnconfigure(0, weight=0)  # barra lateral
        card.columnconfigure(1, weight=1)  # contenido

        # barra vertical de color a la izquierda
        accent = tk.Frame(card, bg=APP_BG, width=8)
        accent.grid(row=0, column=0, sticky="nsw")
        accent.grid_propagate(False)

        # contenido principal de la tarjeta
        content = tk.Frame(card, bg=CARD_BG)
        content.grid(row=0, column=1, sticky="nsew", padx=(20, 24), pady=(18, 18))
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)

        # --- encabezado ---
        header = tk.Frame(content, bg=CARD_BG)
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 18))

        icon_bg = tk.Frame(header, bg=APP_BG, width=30, height=30)
        icon_bg.pack(side="left", padx=(0, 10))
        icon_bg.pack_propagate(False)

        icon_lbl = tk.Label(
            icon_bg,
            text="👤",
            bg=APP_BG,
            fg="white",
            font=("Segoe UI Emoji", 14)
        )
        icon_lbl.pack(expand=True, fill="both")

        title_lbl = tk.Label(
            header,
            text="Información del prospecto",
            bg=CARD_BG,
            fg=TITLE_FG,
            font=TITLE_FONT
        )
        title_lbl.pack(side="left")

        # ---- Definición de campos y estilos ----
        # kind: normal | bold | email | muted (para "Add Tags")
        self.value_labels = {}

        field_specs = [
            ("Tag", "tag", "muted"),
            ("First Name", "first_name", "bold"),
            ("Email", "email", "email"),
            ("Phone", "phone", "normal"),
            ("Mobile", "mobile", "bold"),
            ("Red Social.", "social", "normal"),
            ("Lead Source", "lead_source", "bold"),
            ("Campaña", "campaign", "normal"),
            ("Lead Owner", "lead_owner", "bold"),
            ("Last Activity Time", "last_activity", "bold"),
            ("Como se debe contactar", "contact_how", "normal"),
            ("Campaña Whats", "campaign_whats", "normal"),
            ("Intento", "attempt", "bold"),
            ("Quien recomienda", "referrer", "normal"),
        ]

        row = 1
        for label_text, key, kind in field_specs:
            label = tk.Label(
                content,
                text=label_text,
                bg=CARD_BG,
                fg=LABEL_FG,
                font=DEFAULT_FONT,
                anchor="w",
            )
            label.grid(row=row, column=0, sticky="w", pady=(0, 6))

            value = lead_data.get(key, "-")
            if value in (None, ""):
                value = "-"

            if kind == "email":
                fg = EMAIL_FG
                font = DEFAULT_FONT_BOLD
            elif kind == "bold":
                fg = VALUE_FG
                font = DEFAULT_FONT_BOLD
            elif kind == "muted":
                fg = LABEL_FG
                font = DEFAULT_FONT
            else:
                fg = VALUE_FG
                font = DEFAULT_FONT

            value_label = tk.Label(
                content,
                text=value,
                bg=CARD_BG,
                fg=fg,
                font=font,
                anchor="w",
            )
            value_label.grid(row=row, column=1, sticky="w", pady=(0, 6))
            self.value_labels[key] = value_label
            row += 1

    def update_lead(self, new_data: dict):
        """
        Permite actualizar los valores desde código:
        window.update_lead(dict_con_datos)
        """
        for key, lbl in self.value_labels.items():
            value = new_data.get(key, "-")
            if value in (None, ""):
                value = "-"
            lbl.config(text=value)


if __name__ == "__main__":
    # Ejemplo de uso con los datos de tu captura
    sample_lead = {
        "tag": "Add Tags",
        "first_name": "Sonia",
        "email": "somleot@hotmail.com",
        "phone": "-",
        "mobile": "+573202977552",
        "social": "-",
        "lead_source": "Chat",
        "campaign": "-",
        "lead_owner": "José Ignacio Salcedo",
        "last_activity": "Nov 27, 2025 08:55 AM",
        "contact_how": "-",
        "campaign_whats": "-",
        "attempt": "None",
        "referrer": "-",
    }

    app = LeadCardWindow(sample_lead)
    app.mainloop()
