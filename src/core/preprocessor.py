import pandas as pd
import numpy as np
from src.core.data_loader import load_raw_data


class DataPreprocessor:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()

        if 'Date' in cleaned_df.columns:
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

        if 'Precipitation_mm' in cleaned_df.columns:
            cleaned_df['Precipitation_mm'] = pd.to_numeric(cleaned_df['Precipitation_mm'], errors='coerce')
            cleaned_df['Precipitation_mm'] = cleaned_df['Precipitation_mm'].clip(lower=0.0)

        if 'River_Level_m' in cleaned_df.columns:
            cleaned_df['River_Level_m'] = pd.to_numeric(cleaned_df['River_Level_m'], errors='coerce')
            cleaned_df['River_Level_m'] = cleaned_df['River_Level_m'].clip(lower=0.0)

        if 'Temperature_C' in cleaned_df.columns:
            cleaned_df['Temperature_C'] = pd.to_numeric(cleaned_df['Temperature_C'], errors='coerce')

        cleaned_df = cleaned_df.ffill().bfill()
        return cleaned_df


if __name__ == "__main__":
    print("Running Data Preprocessor Test")
    try:
        raw_data = load_raw_data()
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(raw_data)

        print("Data preprocessing completed successfully.")
        print(f"Cleaned dataset shape: {cleaned_data.shape}")
        print("\nData summary statistics:")
        print(cleaned_data.describe())
    except Exception as e:
        print(f"Error during preprocessing: {e}")