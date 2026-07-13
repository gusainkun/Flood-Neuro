import os
import json
import torch
import pickle
import math
import pandas as pd
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkintermapview import TkinterMapView

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.utils.config_loader import load_config, get_project_root
from src.gis.boundary_loader import BoundaryLoader
from src.core.preprocessor import DataPreprocessor
from src.core.feature_engineer import FeatureEngineer
from src.models.deep_learning.cnn_lstm_model import FloodPredictor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.project_config = load_config()
        self.root_path = get_project_root()

        self.boundary_loader = BoundaryLoader()
        self.wards = self.boundary_loader.load_boundaries()

        # Pre-defined topographic flood basins flanking the Bhagirathi River
        self.basins = [
            {
                "name": "Inner Basin",
                "threshold_m": 1.0,
                "coordinates": [
                    (30.728, 78.432), (30.726, 78.438), (30.725, 78.441), (30.727, 78.446), (30.729, 78.451),
                    (30.728, 78.451), (30.726, 78.446), (30.724, 78.441), (30.725, 78.438), (30.727, 78.432)
                ]
            },
            {
                "name": "Mid Basin",
                "threshold_m": 3.0,
                "coordinates": [
                    (30.730, 78.432), (30.728, 78.438), (30.727, 78.441), (30.729, 78.446), (30.731, 78.451),
                    (30.726, 78.451), (30.724, 78.446), (30.722, 78.441), (30.723, 78.438), (30.725, 78.432)
                ]
            },
            {
                "name": "Outer Basin",
                "threshold_m": 5.0,
                "coordinates": [
                    (30.732, 78.432), (30.730, 78.438), (30.729, 78.441), (30.731, 78.446), (30.733, 78.451),
                    (30.724, 78.451), (30.722, 78.446), (30.720, 78.441), (30.721, 78.438), (30.723, 78.432)
                ]
            }
        ]

        # Interactive zoning state variables
        self.drawing_mode = False
        self.drawn_coords = []
        self.temp_markers = []
        self.temp_draw_polygon = None
        self.cursor_text_id = None

        self.title("Local Flood Forecasting and Evacuation System")
        self.geometry("1250x720")

        self._build_menu()

        self.bind("<Control-z>", lambda event: self._undo_last_point())
        self.bind("<Control-Z>", lambda event: self._undo_last_point())
        self.bind("<Escape>", lambda event: self._clear_drawing())

        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bd=0, bg="#1E1E1E", sashwidth=5, sashpad=2)
        self.main_pane.pack(fill="both", expand=True)

        self._build_sidebar()
        self.main_pane.add(self.sidebar, minsize=200)

        self._build_dashboard()
        self.main_pane.add(self.dashboard, minsize=600)

        self._draw_boundaries(predicted_level=None)
        self._restore_layout()

    def _build_menu(self):
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Upload Data", command=self._handle_upload)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo Last Point", command=self._undo_last_point, accelerator="Ctrl+Z")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Active Zone", command=self._clear_drawing, accelerator="Esc")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Start Zoning Ward", command=self._start_drawing_ward)
        tools_menu.add_command(label="Undo Last Point", command=self._undo_last_point)
        tools_menu.add_command(label="Save Zoned Ward", command=self._save_drawn_ward)
        tools_menu.add_command(label="Clear Zoning Tools", command=self._clear_drawing)
        tools_menu.add_separator()
        tools_menu.add_command(label="Manage Wards...", command=self._open_manage_wards_window)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About System", command=self._show_about_dialog)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menu_bar)

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack_propagate(False)

        self.title_label = ctk.CTkLabel(
            self.sidebar,
            text="Flood Risk: --",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.pack(padx=15, pady=20)

        self.loc_label = ctk.CTkLabel(self.sidebar, text="Location Focus:", font=ctk.CTkFont(size=12))
        self.loc_label.pack(padx=15, pady=(5, 2))

        active_loc_name = self.project_config.get("location_details", {}).get("name", "Uttarkashi")
        self.location_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=[active_loc_name],
            state="disabled"
        )
        self.location_menu.pack(padx=15, pady=(0, 15))

        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload Data",
            height=32,
            command=self._handle_upload
        )
        self.upload_btn.pack(padx=15, pady=10)

        self.predict_btn = ctk.CTkButton(
            self.sidebar,
            text="Run Prediction",
            height=32,
            state="disabled",
            command=self._handle_prediction
        )
        self.predict_btn.pack(padx=15, pady=10)

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Status: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(padx=15, pady=15)

        self.result_label = ctk.CTkLabel(
            self.sidebar,
            text="Level: --",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.result_label.pack(padx=15, pady=(15, 2))

        self.date_label = ctk.CTkLabel(
            self.sidebar,
            text="Date: --",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.date_label.pack(padx=15, pady=(0, 15))

        self._build_sidebar_metadata_panel()
        self._build_sidebar_outlook_panel()

    def _build_sidebar_outlook_panel(self):
        self.outlook_frame = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#2B2B2B")
        self.outlook_frame.pack(side="bottom", fill="x", padx=15, pady=(15, 0))

        outlook_title = ctk.CTkLabel(
            self.outlook_frame,
            text="Hydrological Outlook",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        outlook_title.pack(padx=10, pady=(10, 5), anchor="w")

        self.period_lbl = ctk.CTkLabel(
            self.outlook_frame,
            text="Period: --",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.period_lbl.pack(padx=10, pady=2, fill="x")

        self.horizon_lbl = ctk.CTkLabel(
            self.outlook_frame,
            text="Horizon: 7 Days",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.horizon_lbl.pack(padx=10, pady=2, fill="x")

        self.trend_lbl = ctk.CTkLabel(
            self.outlook_frame,
            text="Trend: --",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.trend_lbl.pack(padx=10, pady=2, fill="x")

        self.peak_lbl = ctk.CTkLabel(
            self.outlook_frame,
            text="Est. Peak: --",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.peak_lbl.pack(padx=10, pady=2, fill="x")

        self.lead_lbl = ctk.CTkLabel(
            self.outlook_frame,
            text="Lead Time: --",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.lead_lbl.pack(padx=10, pady=(2, 10), fill="x")

    def _build_sidebar_metadata_panel(self):
        self.metadata_frame = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#2B2B2B")
        self.metadata_frame.pack(side="bottom", fill="x", padx=15, pady=15)

        metadata_title = ctk.CTkLabel(
            self.metadata_frame,
            text="Station Metadata",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        metadata_title.pack(padx=10, pady=(10, 5), anchor="w")

        loc_details = self.project_config.get("location_details", {})
        river = loc_details.get("river_name", "Unknown River")
        center = loc_details.get("map_center", [0.0, 0.0])

        self.river_lbl = ctk.CTkLabel(
            self.metadata_frame,
            text=f"System: {river}",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.river_lbl.pack(padx=10, pady=2, fill="x")

        self.coords_lbl = ctk.CTkLabel(
            self.metadata_frame,
            text=f"Lat: {center[0]:.4f}\nLon: {center[1]:.4f}",
            font=ctk.CTkFont(size=11),
            justify="left",
            anchor="w"
        )
        self.coords_lbl.pack(padx=10, pady=(2, 10), fill="x")

    def _build_dashboard(self):
        self.dashboard = ctk.CTkFrame(self, corner_radius=0)

        self.dashboard_pane = tk.PanedWindow(self.dashboard, orient=tk.VERTICAL, bd=0, bg="#1E1E1E", sashwidth=5,
                                             sashpad=2)
        self.dashboard_pane.pack(fill="both", expand=True)

        self.map_frame = ctk.CTkFrame(self.dashboard_pane, corner_radius=0)

        loc_details = self.project_config.get("location_details", {})
        center = loc_details.get("map_center", [30.726, 78.441])
        zoom = loc_details.get("zoom_level", 14)

        db_path = self.root_path / "data" / "geospatial" / "maps" / "uttarkashi_map.db"

        if db_path.exists():
            print("Offline Mode: Loading physical map from local database.")
            self.map_widget = TkinterMapView(
                self.map_frame,
                corner_radius=10,
                database_path=str(db_path),
                use_database_only=True,
                bg_color="#2B2B2B"
            )
            self.map_widget.set_tile_server("https://a.tile.opentopomap.org/{z}/{x}/{y}.png")
        else:
            print("Development Mode: Loading physical map from online fallback server.")
            self.map_widget = TkinterMapView(self.map_frame, corner_radius=10, bg_color="#2B2B2B")
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}", max_zoom=22)

        self.map_widget.pack(fill="both", expand=True, padx=10, pady=10)
        self.map_widget.set_position(center[0], center[1])
        self.map_widget.set_zoom(zoom)

        self.dashboard_pane.add(self.map_frame, minsize=350)

        self.analytics_frame = ctk.CTkFrame(self.dashboard_pane, corner_radius=0, fg_color="transparent")
        self.analytics_frame.grid_rowconfigure(0, weight=1)
        self.analytics_frame.grid_columnconfigure(0, weight=2)  # Ward Table (wider)
        self.analytics_frame.grid_columnconfigure(1, weight=1)  # Depth Legend Panel
        self.analytics_frame.grid_columnconfigure(2, weight=2)  # Matplotlib Graph (wider)

        # Left side: Current Situation (Ward Status Table)
        self.table_frame = ctk.CTkFrame(self.analytics_frame, corner_radius=10)
        self.table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        self.table_title = ctk.CTkLabel(
            self.table_frame,
            text="Neighborhood Evacuation Status",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.table_title.pack(padx=10, pady=(10, 5), anchor="w")

        self.table_scroll = ctk.CTkScrollableFrame(self.table_frame, fg_color="transparent", height=100)
        self.table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_ward_status_rows()

        # Middle side: Flood Depth Legend Panel (Point 1 & 2 integration)
        self.legend_frame = ctk.CTkFrame(self.analytics_frame, corner_radius=10)
        self.legend_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.legend_title = ctk.CTkLabel(
            self.legend_frame,
            text="Flood Depth Legend",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.legend_title.pack(padx=10, pady=(10, 5), anchor="w")

        legend_items = [
            ("Dry / No Inundation", "transparent"),
            ("< 1.0 m (Shallow)", "#27AE60"),
            ("1.0 to 3.0 m (Low)", "#F1C40F"),
            ("3.0 to 6.0 m (Medium)", "#E67E22"),
            ("> 6.0 m (Extreme)", "#C0392B")
        ]
        for label, bg_color in legend_items:
            item_frame = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
            item_frame.pack(fill="x", padx=10, pady=2)

            color_badge = ctk.CTkLabel(
                item_frame,
                text="  " if bg_color == "transparent" else "  ",
                fg_color=bg_color if bg_color != "transparent" else "transparent",
                corner_radius=4,
                width=24,
                height=16
            )
            color_badge.pack(side="left", padx=5)

            label_widget = ctk.CTkLabel(item_frame, text=label, font=ctk.CTkFont(size=11))
            label_widget.pack(side="left", padx=5)

        # Right side: Live Trend Graph (Matplotlib)
        self.graph_frame = ctk.CTkFrame(self.analytics_frame, corner_radius=10)
        self.graph_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=5)

        self.figure = Figure(figsize=(5, 2), dpi=100)
        self.figure.patch.set_facecolor('#2B2B2B')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=5, pady=5)

        self.plot_line = None
        self.hover_id = self.figure.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._draw_empty_graph()

        self.dashboard_pane.add(self.analytics_frame, minsize=180)

    def _restore_layout(self):
        settings_path = self.root_path / "config" / "layout_settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)

                self.geometry(f"{settings['win_width']}x{settings['win_height']}")
                self.after(150, lambda: self._apply_sashes(
                    settings["sidebar_width"],
                    settings["bottom_y"]
                ))
            except Exception as e:
                print(f"Layout restoration failure: {e}")
                self.after(150, self._apply_default_sashes)
        else:
            self.after(150, self._apply_default_sashes)

    def _apply_sashes(self, x_coord: int, y_coord: int):
        try:
            self.main_pane.sash_place(0, x_coord, 0)
            self.dashboard_pane.sash_place(0, 0, y_coord)
        except Exception:
            pass

    def _apply_default_sashes(self):
        try:
            self.main_pane.sash_place(0, 220, 0)
            self.dashboard_pane.sash_place(0, 0, 450)
        except Exception:
            pass

    def _on_closing(self):
        try:
            sidebar_width = self.main_pane.sash_coord(0)[0]
            bottom_y = self.dashboard_pane.sash_coord(0)[1]

            settings = {
                "win_width": self.winfo_width(),
                "win_height": self.winfo_height(),
                "sidebar_width": sidebar_width,
                "bottom_y": bottom_y
            }

            settings_path = self.root_path / "config" / "layout_settings.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save active layout settings: {e}")

        self.destroy()

    def _draw_boundaries(self, predicted_level=None):
        self.map_widget.delete_all_polygon()
        self.map_polygons = {}

        # 1. First, draw the continuous physical inundation layers under the administrative boundaries
        if predicted_level is not None:
            # We iterate backwards (from outer to inner) so deeper contours draw on top of shallower ones
            for basin in reversed(self.basins):
                threshold = basin["threshold_m"]
                depth = predicted_level - threshold

                if depth <= 0:
                    continue  # Basin is dry

                # Determine inundation hazard color based on calculated water depth
                if depth < 1.0:
                    fill = "#27AE60"  # Green (Shallow)
                    outline = "#1E8449"
                elif 1.0 <= depth < 3.0:
                    fill = "#F1C40F"  # Yellow (Low)
                    outline = "#D4AC0D"
                elif 3.0 <= depth < 6.0:
                    fill = "#E67E22"  # Orange (Medium)
                    outline = "#CA6F1E"
                else:
                    fill = "#C0392B"  # Red (Extreme)
                    outline = "#922B21"

                self.map_widget.set_polygon(
                    basin["coordinates"],
                    fill_color=fill,
                    outline_color=outline,
                    border_width=1,
                    name=f"{basin['name']} (Depth: {depth:.2f}m)"
                )

        # 2. Draw the administrative wards (user-zoned or config-loaded) on top
        for ward in self.wards:
            threshold = ward["threshold_m"]

            if predicted_level is not None and predicted_level >= threshold:
                fill = "#C0392B"
                outline = "#922B21"
            elif predicted_level is not None:
                fill = "#27AE60"
                outline = "#1E8449"
            else:
                fill = "#2E86C1"
                outline = "#1F618D"

            poly = self.map_widget.set_polygon(
                ward["coordinates"],
                fill_color=fill,
                outline_color=outline,
                border_width=2,
                name=ward["name"]
            )
            self.map_polygons[ward["name"]] = poly

    def _build_ward_status_rows(self, predicted_level=None):
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        for i, ward in enumerate(self.wards):
            row_frame = ctk.CTkFrame(self.table_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            name_lbl = ctk.CTkLabel(row_frame, text=ward["name"], width=180, anchor="w", font=ctk.CTkFont(size=12))
            name_lbl.pack(side="left", padx=5)

            if predicted_level is None:
                comparison_text = f"Water: -- (Limit: {ward['threshold_m']}m)"
            else:
                comparison_text = f"Water: {predicted_level:.2f}m (Limit: {ward['threshold_m']}m)"

            comparison_lbl = ctk.CTkLabel(row_frame, text=comparison_text, width=180, anchor="w",
                                          font=ctk.CTkFont(size=11))
            comparison_lbl.pack(side="left", padx=5)

            if predicted_level is None:
                status_text = "Idle"
                status_color = "#A6ACAF"
            elif predicted_level >= ward["threshold_m"]:
                status_text = "EVACUATE"
                status_color = "#E74C3C"
            else:
                status_text = "Safe"
                status_color = "#2ECC71"

            status_lbl = ctk.CTkLabel(
                row_frame,
                text=status_text,
                text_color=status_color,
                font=ctk.CTkFont(size=11, weight="bold")
            )
            status_lbl.pack(side="right", padx=10)

    def _draw_empty_graph(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2B2B2B')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_title("7-Day River Level Trend", fontsize=10, color='white')
        ax.grid(True, color='#444444', linestyle=':')
        self.canvas.draw()

    def _update_trend_graph(self, dates, predictions, threshold):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2B2B2B')

        self.ax = ax
        self.trend_dates = dates
        self.trend_preds = predictions

        self.plot_line, = ax.plot(dates, predictions, marker='o', color='#3498DB', label='Prediction', picker=5)
        ax.axhline(y=threshold, color='#E74C3C', linestyle='--', label='Flood Threshold')

        self.annotation = ax.annotate(
            "", xy=(0, 0), xytext=(-20, 15), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#1E1E1E", ec="#444444", lw=1),
            color="white", fontsize=8, weight="bold"
        )
        self.annotation.set_visible(False)

        ax.set_title("7-Day River Level Trend (Meters)", fontsize=10, color='white', pad=5)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, color='#444444', linestyle=':')
        ax.legend(facecolor='#2B2B2B', edgecolor='#444444', labelcolor='white', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_hover(self, event):
        if self.plot_line is None or not hasattr(self, 'annotation') or not hasattr(self, 'ax'):
            return

        vis = self.annotation.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.plot_line.contains(event)
            if cont:
                pos = self.plot_line.get_xydata()[ind["ind"][0]]
                self.annotation.xy = pos

                date_val = self.trend_dates[ind["ind"][0]]
                level_val = self.trend_preds[ind["ind"][0]]
                text = f"Date: {date_val}\nLevel: {level_val:.2f} m"

                self.annotation.set_text(text)
                self.annotation.set_visible(True)
                self.canvas.draw_idle()
            else:
                if vis:
                    self.annotation.set_visible(False)
                    self.canvas.draw_idle()

    def _start_drawing_ward(self):
        self.drawing_mode = True
        self.drawn_coords = []

        self._clear_temporary_graphics()

        self.status_label.configure(text="Status: Click to Zone", text_color="#F39C12")
        self.map_widget.configure(cursor="crosshair")
        self.map_widget.canvas.bind("<Motion>", self._on_mouse_move)

        self.cursor_text_id = self.map_widget.canvas.create_text(
            0, 0, text="", fill="#A6ACAF", font=("Arial", 8, "bold"), anchor="w"
        )
        self.map_widget.add_left_click_map_command(self._add_drawing_vertex)

        messagebox.showinfo(
            "Zoning Mode",
            "Zoning mode started.\n\nLeft-click directly on the map to place your boundary points, then select Tools -> Save Zoned Ward from the menu bar when finished."
        )

    def _on_mouse_move(self, event):
        if not self.drawing_mode or not self.cursor_text_id:
            return

        try:
            lat, lon = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
            self.map_widget.canvas.itemconfig(
                self.cursor_text_id,
                text=f"Lat: {lat:.4f}\nLon: {lon:.4f}",
                state="normal"
            )
            self.map_widget.canvas.coords(self.cursor_text_id, event.x + 12, event.y - 12)
        except Exception:
            pass

    def _add_drawing_vertex(self, coords):
        if not self.drawing_mode:
            return

        self.drawn_coords.append(coords)

        marker = self.map_widget.set_marker(
            coords[0],
            coords[1],
            text=f"P{len(self.drawn_coords)}\n({coords[0]:.4f}, {coords[1]:.4f})",
            command=self._on_marker_click
        )
        self.temp_markers.append(marker)

        if len(self.drawn_coords) >= 2:
            if self.temp_draw_polygon:
                try:
                    self.temp_draw_polygon.delete()
                except Exception:
                    pass

            self.temp_draw_polygon = self.map_widget.set_polygon(
                self.drawn_coords,
                fill_color="#F39C12",
                outline_color="#D35400",
                border_width=2,
                name="Zoning Preview"
            )

            acres = self._calculate_acreage(self.drawn_coords)
            self.status_label.configure(text=f"Area: {acres:.2f} Acres", text_color="#F39C12")

    def _calculate_acreage(self, coords: list) -> float:
        if len(coords) < 3:
            return 0.0

        lat_to_meters = 111132.0
        lon_to_meters = 111132.0 * math.cos(math.radians(30.726))

        pts = []
        for lat, lon in coords:
            pts.append((lon * lon_to_meters, lat * lat_to_meters))

        area = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        area = abs(area) / 2.0

        return area / 4046.856

    def _on_marker_click(self, marker):
        if not self.drawing_mode:
            return

        confirm = messagebox.askyesno(
            "Delete Vertex",
            f"Do you want to delete vertex '{marker.text.split()[0]}'?"
        )
        if not confirm:
            return

        if marker in self.temp_markers:
            idx = self.temp_markers.index(marker)
            self.temp_markers.pop(idx)
            self.drawn_coords.pop(idx)

            try:
                marker.delete()
            except Exception:
                pass

        for i, m in enumerate(self.temp_markers):
            try:
                pos = m.position
                m.set_text(f"P{i + 1}\n({pos[0]:.4f}, {pos[1]:.4f})")
            except Exception:
                pass

        if len(self.drawn_coords) >= 2:
            if self.temp_draw_polygon:
                try:
                    self.temp_draw_polygon.delete()
                except Exception:
                    pass

            self.temp_draw_polygon = self.map_widget.set_polygon(
                self.drawn_coords,
                fill_color="#F39C12",
                outline_color="#D35400",
                border_width=2,
                name="Zoning Preview"
            )
            acres = self._calculate_acreage(self.drawn_coords)
            self.status_label.configure(text=f"Area: {acres:.2f} Acres", text_color="#F39C12")
        else:
            if self.temp_draw_polygon:
                try:
                    self.temp_draw_polygon.delete()
                except Exception:
                    pass
                self.temp_draw_polygon = None
            self.status_label.configure(text=f"Zoning: P{len(self.drawn_coords)} (0.0 Acres)")

    def _save_drawn_ward(self):
        if not self.drawing_mode or len(self.drawn_coords) < 3:
            messagebox.showerror(
                "Error",
                "You must click at least 3 points on the map to define a valid closed boundary."
            )
            return

        name_dialog = ctk.CTkInputDialog(text="Enter Name of the Ward:", title="New Ward Name")
        ward_name = name_dialog.get_input()
        if not ward_name:
            return

        threshold_dialog = ctk.CTkInputDialog(text="Enter Flood Threshold (meters):", title="New Ward Threshold")
        threshold_input = threshold_dialog.get_input()
        try:
            threshold_val = float(threshold_input)
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid numeric threshold. Saving canceled.")
            return

        self._write_new_ward_to_geojson(ward_name, threshold_val, self.drawn_coords)
        self._clear_drawing()

        self.wards = self.boundary_loader.load_boundaries()
        self._draw_boundaries()
        self._build_ward_status_rows()

        messagebox.showinfo("Success", f"Ward '{ward_name}' saved and loaded successfully.")

    def _write_new_ward_to_geojson(self, name: str, threshold: float, coords: list):
        config = load_config()
        root = get_project_root()
        relative_boundary_path = config["location_details"]["boundary_file"]
        absolute_boundary_path = root / "data" / "geospatial" / relative_boundary_path

        with open(absolute_boundary_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        geojson_coords = [[coord[1], coord[0]] for coord in coords]
        geojson_coords.append(geojson_coords[0])

        new_feature = {
            "type": "Feature",
            "properties": {
                "ward_name": name,
                "flood_threshold_m": threshold,
                "description": "User-defined administrative boundary."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [geojson_coords]
            }
        }

        geojson_data["features"].append(new_feature)

        with open(absolute_boundary_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=4)

    def _undo_last_point(self):
        if not self.drawing_mode or not self.drawn_coords:
            return

        self.drawn_coords.pop()
        if self.temp_markers:
            last_marker = self.temp_markers.pop()
            try:
                last_marker.delete()
            except Exception:
                pass

        if len(self.drawn_coords) >= 2:
            if self.temp_draw_polygon:
                try:
                    self.temp_draw_polygon.delete()
                except Exception:
                    pass

            self.temp_draw_polygon = self.map_widget.set_polygon(
                self.drawn_coords,
                fill_color="#F39C12",
                outline_color="#D35400",
                border_width=2,
                name="Zoning Preview"
            )
            acres = self._calculate_acreage(self.drawn_coords)
            self.status_label.configure(text=f"Area: {acres:.2f} Acres", text_color="#F39C12")
        else:
            if self.temp_draw_polygon:
                try:
                    self.temp_draw_polygon.delete()
                except Exception:
                    pass
                self.temp_draw_polygon = None
            self.status_label.configure(text=f"Zoning: P{len(self.drawn_coords)} (0.0 Acres)")

    def _clear_temporary_graphics(self):
        if self.temp_draw_polygon:
            try:
                self.temp_draw_polygon.delete()
            except Exception:
                pass
            self.temp_draw_polygon = None

        for marker in self.temp_markers:
            try:
                marker.delete()
            except Exception:
                pass
        self.temp_markers = []

        try:
            for marker in list(self.map_widget.canvas_marker_list):
                if marker.text and (marker.text.startswith("P") or "Lat:" in marker.text):
                    marker.delete()
        except Exception:
            pass

    def _clear_drawing(self):
        self.drawing_mode = False
        self.drawn_coords = []
        self._clear_temporary_graphics()
        self.map_widget.left_click_map_command = None

        self.map_widget.configure(cursor="")
        if self.cursor_text_id:
            try:
                self.map_widget.canvas.delete(self.cursor_text_id)
            except Exception:
                pass
            self.cursor_text_id = None
        try:
            self.map_widget.canvas.unbind("<Motion>")
        except Exception:
            pass

        self.status_label.configure(text="Status: Idle", text_color="#FFFFFF")

    def _open_manage_wards_window(self):
        self.manage_window = ctk.CTkToplevel(self)
        self.manage_window.title("Manage Administrative Wards")
        self.manage_window.geometry("520x400")
        self.manage_window.grab_set()

        title_lbl = ctk.CTkLabel(
            self.manage_window,
            text="Edit or Delete Saved Wards",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_lbl.pack(padx=20, pady=15)

        self.manage_scroll = ctk.CTkScrollableFrame(self.manage_window, corner_radius=10)
        self.manage_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._populate_manage_wards_list()

    def _populate_manage_wards_list(self):
        for widget in self.manage_scroll.winfo_children():
            widget.destroy()

        for ward in self.wards:
            row_frame = ctk.CTkFrame(self.manage_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=4)

            name_entry = ctk.CTkEntry(row_frame, width=160)
            name_entry.insert(0, ward["name"])
            name_entry.pack(side="left", padx=5)

            limit_entry = ctk.CTkEntry(row_frame, width=80)
            limit_entry.insert(0, str(ward["threshold_m"]))
            limit_entry.pack(side="left", padx=5)

            old_name = ward["name"]

            save_btn = ctk.CTkButton(
                row_frame,
                text="Save",
                width=60,
                command=lambda n=name_entry, l=limit_entry, o=old_name: self._save_managed_ward(o, n.get(), l.get())
            )
            save_btn.pack(side="left", padx=5)

            delete_btn = ctk.CTkButton(
                row_frame,
                text="Delete",
                width=60,
                fg_color="#C0392B",
                hover_color="#922B21",
                command=lambda o=old_name: self._delete_managed_ward(o)
            )
            delete_btn.pack(side="left", padx=5)

    def _save_managed_ward(self, old_name: str, new_name: str, new_limit_str: str):
        try:
            new_limit = float(new_limit_str)
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid numeric limit threshold.")
            return

        config = load_config()
        root = get_project_root()
        relative_boundary_path = config["location_details"]["boundary_file"]
        absolute_boundary_path = root / "data" / "geospatial" / relative_boundary_path

        with open(absolute_boundary_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        updated = False
        for feature in geojson_data.get("features", []):
            if feature.get("properties", {}).get("ward_name") == old_name:
                feature["properties"]["ward_name"] = new_name
                feature["properties"]["flood_threshold_m"] = new_limit
                updated = True
                break

        if updated:
            with open(absolute_boundary_path, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=4)

            self.wards = self.boundary_loader.load_boundaries()
            self._draw_boundaries()
            self._build_ward_status_rows()
            self._populate_manage_wards_list()
            messagebox.showinfo("Success", f"Ward '{new_name}' updated successfully.")
        else:
            messagebox.showerror("Error", "Failed to update ward details.")

    def _delete_managed_ward(self, name: str):
        confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete ward '{name}'?")
        if not confirm:
            return

        config = load_config()
        root = get_project_root()
        relative_boundary_path = config["location_details"]["boundary_file"]
        absolute_boundary_path = root / "data" / "geospatial" / relative_boundary_path

        with open(absolute_boundary_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        initial_count = len(features)

        filtered_features = [feat for feat in features if feat.get("properties", {}).get("ward_name") != name]

        if len(filtered_features) < initial_count:
            geojson_data["features"] = filtered_features

            with open(absolute_boundary_path, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=4)

            self.wards = self.boundary_loader.load_boundaries()
            self._draw_boundaries()
            self._build_ward_status_rows()
            self._populate_manage_wards_list()
            messagebox.showinfo("Success", f"Ward '{name}' deleted successfully.")
        else:
            messagebox.showerror("Error", "Failed to locate ward for deletion.")

    def _show_about_dialog(self):
        messagebox.showinfo(
            "About System",
            "Local Flood Forecasting and Evacuation System\nVersion: 1.1.0\nConfigured for: Official Indian Administrative Boundaries."
        )

    def _handle_upload(self):
        file_path = filedialog.askopenfilename(
            title="Select Weather Data File",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.uploaded_file_path = file_path
            self.status_label.configure(text="Status: File Loaded")
            self.predict_btn.configure(state="normal")

    def _handle_prediction(self):
        try:
            df = pd.read_csv(self.uploaded_file_path)

            preprocessor = DataPreprocessor()
            cleaned_df = preprocessor.clean_data(df)

            engineer = FeatureEngineer()
            features_df = engineer.create_features(cleaned_df)

            model_dir = self.root_path / "outputs" / "models"

            with open(model_dir / "feature_scaler.pkl", "rb") as f:
                feature_scaler = pickle.load(f)

            with open(model_dir / "target_scaler.pkl", "rb") as f:
                target_scaler = pickle.load(f)

            feature_cols = [col for col in features_df.columns if col not in ["Date", "River_Level_m"]]

            all_features = features_df[feature_cols].values
            scaled_all = feature_scaler.transform(all_features)
            all_tensor = torch.tensor(scaled_all, dtype=torch.float32)

            model = FloodPredictor(input_dim=len(feature_cols))
            model.load_state_dict(torch.load(model_dir / "flood_model.pt"))
            model.eval()

            with torch.no_grad():
                scaled_all_preds = model(all_tensor).numpy()

            all_preds_meters = target_scaler.inverse_transform(scaled_all_preds)

            last_7_days = features_df.iloc[-7:]
            last_7_dates_full = last_7_days["Date"].tolist()
            last_7_dates = pd.to_datetime(last_7_days["Date"]).dt.strftime("%d-%m-%y").tolist()
            last_7_preds = all_preds_meters[-7:].flatten().tolist()

            current_val = last_7_preds[0]
            peak_val = max(last_7_preds)
            peak_index = last_7_preds.index(peak_val)

            today_date_str = last_7_dates_full[0]
            end_date_str = last_7_dates_full[-1]

            formatted_today = pd.to_datetime(today_date_str).strftime("%d-%m-%y")
            formatted_end = pd.to_datetime(end_date_str).strftime("%d-%m-%y")

            if last_7_preds[1] > current_val + 0.05:
                trend_text = "Rising"
                trend_color = "#E74C3C"
            elif last_7_preds[1] < current_val - 0.05:
                trend_text = "Receding"
                trend_color = "#2ECC71"
            else:
                trend_text = "Stable"
                trend_color = "#A6ACAF"

            if peak_index == 0:
                lead_time_text = "Active Now"
            else:
                lead_time_text = f"{peak_index} Days"

            self.result_label.configure(text=f"Level: {current_val:.2f} m")
            self.date_label.configure(text=f"Date: {formatted_today}")

            self._draw_boundaries(predicted_level=current_val)
            self._build_ward_status_rows(predicted_level=current_val)

            self.period_lbl.configure(text=f"Period: {formatted_today} to {formatted_end}")
            self.trend_lbl.configure(text=f"Trend: {trend_text}", text_color=trend_color)
            self.peak_lbl.configure(text=f"Est. Peak: {peak_val:.2f} m")
            self.lead_lbl.configure(text=f"Lead Time: {lead_time_text}")

            lowest_threshold = min(ward["threshold_m"] for ward in self.wards)
            self._update_trend_graph(last_7_dates, last_7_preds, lowest_threshold)

            flooded_wards_count = 0
            highest_threat_level = 0.0

            for ward in self.wards:
                threshold = ward["threshold_m"]
                if current_val >= threshold:
                    flooded_wards_count += 1
                    threat = (current_val / threshold) * 100
                    if threat > highest_threat_level:
                        highest_threat_level = threat

            if flooded_wards_count > 0:
                self.status_label.configure(text=f"Alert: {flooded_wards_count} Wards Flooded")
                risk_pct = min(100.0, highest_threat_level)
                self.title_label.configure(text=f"Flood Risk: {risk_pct:.1f}%", text_color="#E74C3C")
            else:
                self.status_label.configure(text="Status: Normal")
                risk_pct = min(99.0, (current_val / lowest_threshold) * 100)
                self.title_label.configure(text=f"Flood Risk: {risk_pct:.1f}%", text_color="#2ECC71")

        except Exception as e:
            self.status_label.configure(text="Error processing data")
            print(f"Prediction execution failure: {e}")