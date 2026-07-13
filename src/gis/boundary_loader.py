import json
from pathlib import Path
from src.utils.config_loader import load_config, get_project_root


class BoundaryLoader:
    def __init__(self):
        pass

    def load_boundaries(self) -> list:
        config = load_config()
        root = get_project_root()

        relative_boundary_path = config["location_details"]["boundary_file"]
        absolute_boundary_path = root / "data" / "geospatial" / relative_boundary_path

        if not absolute_boundary_path.exists():
            raise FileNotFoundError(f"Boundary file not found at: {absolute_boundary_path}")

        with open(absolute_boundary_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        parsed_wards = []
        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            ward_name = properties.get("ward_name", "Unknown Ward")
            threshold = properties.get("flood_threshold_m", 0.0)
            description = properties.get("description", "")

            # GeoJSON uses [longitude, latitude]. We swap to (latitude, longitude) for TkinterMapView.
            raw_coords = geometry.get("coordinates", [[]])[0]
            swapped_coords = [(coord[1], coord[0]) for coord in raw_coords]

            parsed_wards.append({
                "name": ward_name,
                "threshold_m": threshold,
                "description": description,
                "coordinates": swapped_coords
            })

        return parsed_wards

    def load_basins(self) -> list:
        """Loads and parses the topographic river basin polygons for dynamic hazard mapping."""
        config = load_config()
        root = get_project_root()

        relative_basin_path = config["location_details"]["basin_file"]
        absolute_basin_path = root / "data" / "geospatial" / relative_basin_path

        if not absolute_basin_path.exists():
            raise FileNotFoundError(f"Basin configuration file not found at: {absolute_basin_path}")

        with open(absolute_basin_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        parsed_basins = []
        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            basin_name = properties.get("basin_name", "Unknown Basin Segment")
            threshold = properties.get("flood_threshold_m", 0.0)

            raw_coords = geometry.get("coordinates", [[]])[0]
            swapped_coords = [(coord[1], coord[0]) for coord in raw_coords]

            parsed_basins.append({
                "name": basin_name,
                "threshold_m": threshold,
                "coordinates": swapped_coords
            })

        return parsed_basins


if __name__ == "__main__":
    print("Running GIS Boundary and Basin Loader Test")
    try:
        loader = BoundaryLoader()
        wards = loader.load_boundaries()
        basins = loader.load_basins()

        print("GIS boundaries and basins loaded successfully.")
        print(f"Total wards loaded: {len(wards)}")
        print(f"Total topographic basin contours loaded: {len(basins)}")
    except Exception as e:
        print(f"Error during loading: {e}")