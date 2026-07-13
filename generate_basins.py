import json
from pathlib import Path

def create_long_ranging_basins():
    root = Path(__file__).resolve().parent
    geojson_dir = root / "data" / "geospatial" / "boundaries"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = geojson_dir / "uttarkashi_basins.geojson"

    # Winding coordinate ribbons tracing the Bhagirathi River path from Northeast to Southwest
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "basin_name": "Inner Basin",
                    "flood_threshold_m": 1.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.456, 30.745], [78.453, 30.738], [78.449, 30.731], [78.442, 30.726], [78.436, 30.725], [78.429, 30.722],
                        [78.427, 30.721], [78.434, 30.724], [78.440, 30.725], [78.447, 30.730], [78.451, 30.737], [78.454, 30.744],
                        [78.456, 30.745]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "basin_name": "Mid Basin",
                    "flood_threshold_m": 3.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.458, 30.746], [78.455, 30.739], [78.451, 30.732], [78.443, 30.727], [78.437, 30.726], [78.430, 30.723],
                        [78.426, 30.719], [78.433, 30.723], [78.439, 30.724], [78.446, 30.729], [78.450, 30.736], [78.452, 30.743],
                        [78.458, 30.746]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "basin_name": "Outer Basin",
                    "flood_threshold_m": 5.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [78.460, 30.747], [78.457, 30.740], [78.453, 30.733], [78.444, 30.728], [78.438, 30.727], [78.431, 30.724],
                        [78.425, 30.717], [78.432, 30.722], [78.438, 30.723], [78.445, 30.728], [78.449, 30.735], [78.450, 30.742],
                        [78.460, 30.747]
                    ]]
                }
            }
        ]
    }

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=4)
    print(f"SUCCESS: Long-ranging river basin GeoJSON compiled at: {geojson_path}")

if __name__ == "__main__":
    create_long_ranging_basins()