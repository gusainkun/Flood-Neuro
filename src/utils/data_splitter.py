import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.core.data_loader import load_raw_data
from src.core.preprocessor import DataPreprocessor
from src.core.feature_engineer import FeatureEngineer


class ChronologicalSplitter:
    def __init__(self, target_column: str = "River_Level_m", train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def split_and_scale(self, df: pd.DataFrame) -> dict:
        # Exclude date and target columns from feature scaling
        feature_cols = [col for col in df.columns if col not in ["Date", self.target_column]]

        X = df[feature_cols].values
        y = df[[self.target_column]].values

        total_len = len(df)
        train_end = int(total_len * self.train_ratio)
        val_end = train_end + int(total_len * self.val_ratio)

        # Split indices chronologically
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Scale features using training statistics only
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Scale target variable independently
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val)
        y_test_scaled = self.target_scaler.transform(y_test)

        return {
            "X_train": X_train_scaled, "y_train": y_train_scaled,
            "X_val": X_val_scaled, "y_val": y_val_scaled,
            "X_test": X_test_scaled, "y_test": y_test_scaled
        }


if __name__ == "__main__":
    print("Running Chronological Splitter Test")
    try:
        raw_data = load_raw_data()
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(raw_data)

        engineer = FeatureEngineer()
        features_data = engineer.create_features(cleaned_data)

        splitter = ChronologicalSplitter()
        splits = splitter.split_and_scale(features_data)

        print("Data splitting and scaling completed successfully.")
        print(f"Train set size: {splits['X_train'].shape[0]} days")
        print(f"Validation set size: {splits['X_val'].shape[0]} days")
        print(f"Test set size: {splits['X_test'].shape[0]} days")
        print(f"Number of engineered features input: {splits['X_train'].shape[1]}")
    except Exception as e:
        print(f"Error during splitting: {e}")