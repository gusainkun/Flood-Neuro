import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.core.data_loader import load_raw_data
from src.core.preprocessor import DataPreprocessor
from src.core.feature_engineer import FeatureEngineer
from src.utils.data_splitter import ChronologicalSplitter


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
        print("Model training pipeline test completed successfully.")
    except Exception as e:
        print(f"Error during training test: {e}")