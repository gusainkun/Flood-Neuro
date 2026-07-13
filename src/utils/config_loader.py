import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute path of the project root folder.

    This resolves to D:/Project/FloodPred/Code/
    regardless of which script runs it.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path) -> dict:
    """Reads a YAML configuration file and returns it as a Python dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    """Loads the main configuration and merges it with the active location profile."""
    root = get_project_root()
    main_config_path = root / "config" / "config.yaml"

    # Load the main configuration file
    config = load_yaml(main_config_path)

    # Identify which location profile to load
    location = config.get("current_location", "uttarkashi")
    location_config_path = root / "config" / "locations" / f"{location}.yaml"

    # If the location file exists, load and merge its details
    if location_config_path.exists():
        location_config = load_yaml(location_config_path)
        config["location_details"] = location_config
    else:
        print(f"⚠️ Warning: Location profile '{location}' not found at {location_config_path}")
        config["location_details"] = {}

    return config


if __name__ == "__main__":
    # This block only runs if you execute this file directly.
    # It acts as a built-in safety test to make sure our loader works!
    print("--- Running Configuration Loader Test ---")
    try:
        project_config = load_config()
        print("\nConfiguration loaded successfully!")
        print(f"Project Root Location: {get_project_root()}")
        print(f"Active Location: {project_config['current_location']}")
        print(f"Location Map Center Coordinate: {project_config['location_details']['map_center']}")
        print(f"Target River: {project_config['location_details']['river_name']}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")