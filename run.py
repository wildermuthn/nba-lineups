import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataloader import BasketballDataset
from model.train import train_loop
from model.test import test_loop
from model.model import LineupPredictor
import config
import wandb


def main():
    wandb.init(project="nba-lineups",
               config=config.MODEL_PARAMS)

    print("Loading data...")
    dataset = BasketballDataset(config.DATA_PATH)


    train_dataset, eval_dataset = dataset.split(train_fraction=0.8)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  pin_memory=True)
    test_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.BATCH_SIZE,
                                 pin_memory=True)
    # print size of datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(eval_dataset)}")
    n_players = len(dataset.player_ids)
    n_ages = len(dataset.player_ages_set)

    # # Sample dataloader
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # Initialize model
    print("Initializing model...")
    model = LineupPredictor(config.MODEL_PARAMS, n_players, n_ages)
    wandb.watch(model, log='all', log_freq=100)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.MODEL_PARAMS['lr'])

    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        test_loop(test_dataloader, model, loss_fn)

    print("Done.")

if __name__ == "__main__":
    main()