import os
import json
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def build_structure(base_path: str = "."):
    # We use "." as the base path so it builds directly inside your open project folder
    base = Path(base_path)

    # 1. Define all necessary directories
    directories = [
        base / "src",
        base / "src" / "core",
        base / "src" / "analysis",
        base / "src" / "models",
        base / "src" / "models" / "deep_learning",
        base / "src" / "evaluation",
        base / "src" / "gui",
        base / "src" / "gui" / "components",
        base / "src" / "gis",
        base / "src" / "utils",
        base / "config",
        base / "config" / "locations",
        base / "data",
        base / "data" / "raw",
        base / "data" / "processed",
        base / "data" / "features",
        base / "data" / "splits",
        base / "data" / "geospatial" / "maps",
        base / "data" / "geospatial" / "boundaries",
        base / "outputs" / "figures",
        base / "outputs" / "logs",
        base / "outputs" / "models",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {directory}")

    # Create empty __init__.py files
    init_files = [
        base / "src" / "__init__.py",
        base / "src" / "core" / "__init__.py",
        base / "src" / "analysis" / "__init__.py",
        base / "src" / "models" / "__init__.py",
        base / "src" / "models" / "deep_learning" / "__init__.py",
        base / "src" / "evaluation" / "__init__.py",
        base / "src" / "gui" / "__init__.py",
        base / "src" / "gui" / "components" / "__init__.py",
        base / "src" / "gis" / "__init__.py",
        base / "src" / "utils" / "__init__.py",
    ]
    for init_file in init_files:
        init_file.touch()

    # 2. Generate Synthetic Monsoon Dataset (CSV) using built-in csv module
    print("Generating synthetic weather and river-level dataset...")

    start_date = datetime(2025, 6, 1)
    raw_data_path = base / "data" / "raw" / "uttarkashi_monsoon_2025.csv"

    current_level = 1.2  # Normal base water level in meters
    random.seed(42)

    with open(raw_data_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(["Date", "Precipitation_mm", "Temperature_C", "River_Level_m"])

        for i in range(122):  # 122 days from June 1 to Sept 30
            current_date = start_date + timedelta(days=i)

            # Simulate rainfall (exponential distribution approximation)
            rain = random.expovariate(1.0 / 5.0)

            # Inject heavy rain storms during monsoon
            if 15 <= i <= 18:
                rain += random.uniform(50, 100)
            elif 45 <= i <= 49:
                rain += random.uniform(70, 120)
            elif 75 <= i <= 78:
                rain += random.uniform(40, 80)
            elif 105 <= i <= 108:
                rain += random.uniform(50, 90)

            rain = max(0.0, min(150.0, rain))  # Cap between 0 and 150mm

            # Simulate temperature
            temp = random.normalvariate(25.0, 3.0) - (rain * 0.1)

            # Hydrological water level simulation (with lag)
            runoff = (rain * 0.04)
            current_level = (current_level * 0.9) + 0.12 + runoff

            writer.writerow([
                current_date.strftime("%Y-%m-%d"),
                round(rain, 1),
                round(temp, 1),
                round(current_level, 2)
            ])

    print(f"Saved synthetic dataset to: {raw_data_path}")

    # 3. Generate Mock GeoJSON for Uttarkashi Wards using built-in json module
    print("Generating mock GeoJSON boundaries...")
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "ward_name": "Gyanesu Riverside Ward",
                    "flood_threshold_m": 2.2,
                    "description": "High-risk residential zone close to the Bhagirathi riverbank."
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.435, 30.728],
                        [78.442, 30.728],
                        [78.440, 30.724],
                        [78.434, 30.724],
                        [78.435, 30.728]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "ward_name": "Joshiyara Valley Ward",
                    "flood_threshold_m": 3.8,
                    "description": "Mixed commercial and residential zone."
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.442, 30.728],
                        [78.450, 30.730],
                        [78.448, 30.725],
                        [78.440, 30.724],
                        [78.442, 30.728]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "ward_name": "Bhatwari Heights Ward",
                    "flood_threshold_m": 5.5,
                    "description": "High elevation safety zone. Designated evacuation center location."
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.434, 30.724],
                        [78.440, 30.724],
                        [78.444, 30.720],
                        [78.432, 30.720],
                        [78.434, 30.724]
                    ]]
                }
            }
        ]
    }

    geojson_path = base / "data" / "geospatial" / "boundaries" / "uttarkashi_wards.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=4)
    print(f"Saved mock GeoJSON to: {geojson_path}")

    # 4. Generate Main Config File
    config_data = """# Main configuration for the Flood Forecasting System
paths:
  raw: "data/raw/"
  processed: "data/processed/"
  features: "data/features/"
  models: "outputs/models/"
  geospatial: "data/geospatial/"

current_location: "uttarkashi"
"""
    config_path = base / "config" / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_data)
    print(f"Saved main configuration to: {config_path}")

    # 5. Generate Location Config File for Uttarkashi
    uttarkashi_config = """# Location Profile for Uttarkashi
name: "Uttarkashi"
river_name: "Bhagirathi River"
map_center: [30.726, 78.441]  # Latitude, Longitude
zoom_level: 14

boundary_file: "boundaries/uttarkashi_wards.geojson"
historical_data: "raw/uttarkashi_monsoon_2025.csv"
"""
    loc_config_path = base / "config" / "locations" / "uttarkashi.yaml"
    with open(loc_config_path, "w", encoding="utf-8") as f:
        f.write(uttarkashi_config)
    print(f"Saved location profile to: {loc_config_path}")

    print("\nSUCCESS: All folders and files have been generated.")


if __name__ == "__main__":
    build_structure()