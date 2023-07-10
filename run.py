import torch
from torch.utils.data import DataLoader
from data.dataloader import BasketballDataset
from model.train import train
from model.model import LineupPredictor
import config

def main():
    # Load data
    print("Loading data...")
    dataset = BasketballDataset(config.DATA_PATH)
    train_dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    # Sample dataloader
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Initialize model
    print("Initializing model...")
    model = LineupPredictor(config.MODEL_PARAMS)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    train(model)
    # Implement your training loop here

    # Evaluate model
    print("Evaluating model...")
    # Implement your evaluation loop here

    print("Done.")

if __name__ == "__main__":
    main()