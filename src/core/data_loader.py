import os
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config, get_project_root


def load_raw_data() -> pd.DataFrame:
    """Loads the raw weather and river-level dataset for the active location."""
    config = load_config()
    root = get_project_root()

    # Get the relative path of the dataset from the location configuration
    relative_data_path = config["location_details"]["historical_data"]
    absolute_data_path = root / "data" / relative_data_path

    if not absolute_data_path.exists():
        raise FileNotFoundError(f"❌ Data file not found at: {absolute_data_path}")

    print(f"📖 Loading dataset from: {absolute_data_path}")
    df = pd.read_csv(absolute_data_path)
    return df


if __name__ == "__main__":
    # This block only runs if you execute this file directly.
    # It acts as a built-in safety test to make sure our data loading works!
    print("--- Running Data Loader Test ---")
    try:
        data = load_raw_data()
        print("\nDataset loaded successfully!")
        print(f"Total records loaded: {len(data)} days of monsoon data.")
        print("\nFirst 5 rows of the dataset:")
        print(data.head())  # Displays the first 5 rows in a neat table
    except Exception as e:
        print(f"Failed to load dataset: {e}")