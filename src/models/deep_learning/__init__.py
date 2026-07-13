import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from src.core.data_loader import load_raw_data
from src.core.preprocessor import DataPreprocessor
from src.core.feature_engineer import FeatureEngineer
from src.utils.data_splitter import ChronologicalSplitter
from src.utils.config_loader import get_project_root


class FloodPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super(FloodPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    def __init__(self, input_dim: int, lr: float = 0.001):
        self.model = FloodPredictor(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self, splits: dict, epochs: int = 100, batch_size: int = 16) -> nn.Module:
        X_train = torch.tensor(splits["X_train"], dtype=torch.float32)
        y_train = torch.tensor(splits["y_train"], dtype=torch.float32)
        X_val = torch.tensor(splits["X_val"], dtype=torch.float32)
        y_val = torch.tensor(splits["y_val"], dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)

                epoch_train_loss /= len(train_dataset)

            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:03d}/{epochs:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")
        return self.model


def save_artifacts(model: nn.Module, splitter: ChronologicalSplitter):
    root = get_project_root()
    model_dir = root / "outputs" / "models"

    # Save the model state dict
    torch.save(model.state_dict(), model_dir / "flood_model.pt")

    # Save feature and target scalers
    with open(model_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(splitter.feature_scaler, f)

    with open(model_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(splitter.target_scaler, f)

    print("Model and scaling artifacts saved successfully.")


if __name__ == "__main__":
    print("Running Deep Learning Model Training Test")
    try:
        raw_data = load_raw_data()
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(raw_data)

        engineer = FeatureEngineer()
        features_data = engineer.create_features(cleaned_data)

        splitter = ChronologicalSplitter()
        splits = splitter.split_and_scale(features_data)

        input_features_count = splits["X_train"].shape[1]
        trainer = ModelTrainer(input_dim=input_features_count)

        trained_model = trainer.train_model(splits, epochs=100, batch_size=8)
        save_artifacts(trained_model, splitter)

        # Simulate loading saved files and running offline predictions
        print("\n--- Simulating Offline Prediction on Test Set ---")

        root = get_project_root()
        model_dir = root / "outputs" / "models"

        loaded_model = FloodPredictor(input_dim=input_features_count)
        loaded_model.load_state_dict(torch.load(model_dir / "flood_model.pt"))
        loaded_model.eval()

        with open(model_dir / "target_scaler.pkl", "rb") as f:
            loaded_target_scaler = pickle.load(f)

        X_test_tensor = torch.tensor(splits["X_test"], dtype=torch.float32)
        with torch.no_grad():
            scaled_predictions = loaded_model(X_test_tensor).numpy()

        # Reconstruct prediction values in meters
        predictions_meters = loaded_target_scaler.inverse_transform(scaled_predictions)
        actual_meters = loaded_target_scaler.inverse_transform(splits["y_test"])

        # Output predictions compared to ground truth
        print("\nComparing Actual vs Predicted River Levels (Meters):")
        print(f"{'Day':<5} | {'Actual Level (m)':<18} | {'Predicted Level (m)':<18} | {'Error (m)':<10}")
        print("-" * 60)
        for i in range(10):
            actual = actual_meters[i][0]
            pred = predictions_meters[i][0]
            error = abs(actual - pred)
            print(f"{i + 1:<5} | {actual:<18.2f} | {pred:<18.2f} | {error:<10.2f}")

    except Exception as e:
        print(f"Error during training and saving test: {e}")