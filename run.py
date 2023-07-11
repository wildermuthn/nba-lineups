import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataloader import BasketballDataset
from model.train import train_loop
from model.test import test_loop
from model.model import LineupPredictor, LineupPredictorTransformer
import config
import wandb
import pickle
from utils.utils import get_latest_file


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def save_model_configuration(filename, config, model_init_params):
    # Remove pth from filename
    filename = filename[:-4]
    # merge config and model_init_params into one dict
    config_dict = {**config, **model_init_params}
    # save dict to file as a pickle, named afer the model
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(config_dict, f, pickle.HIGHEST_PROTOCOL)


def load_model_configurations(model_filepath):
    # Remove pth from filename
    model_filepath = model_filepath[:-4]
    # load dict from file as a pickle, named afer the model
    with open(model_filepath + '.pkl', 'rb') as f:
        config_dict = pickle.load(f)
    return config_dict


def initialize_saved_model(model_filepath):
    # Load model configuration
    cfg = load_model_configurations(model_filepath)

    # Initialize model
    print("Initializing model...")
    n_players = cfg['n_players']
    n_ages = cfg['n_ages']
    model = None
    if config.MODEL_PARAMS['model'] == 'LineupPredictorTransformer':
        model = LineupPredictorTransformer(cfg.MODEL_PARAMS, n_players, n_ages)
    if config.MODEL_PARAMS['model'] == 'LineupPredictor':
        model = LineupPredictor(cfg.MODEL_PARAMS, n_players, n_ages)

    if config.MODEL_PARAMS['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MODEL_PARAMS['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.MODEL_PARAMS['lr'])

    load_checkpoint(model_filepath, model, optimizer)
    return model, optimizer


def main():
    wandb.init(project="nba-lineups",
               config=config.MODEL_PARAMS)

    print("Loading data...")
    dataset = BasketballDataset(config.DATA_PATH)

    g = torch.Generator()
    g.manual_seed(42)

    train_dataset, eval_dataset = dataset.split(train_fraction=0.8)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  pin_memory=True,
                                  generator=g
                                  )
    test_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True,
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
    model = None
    if config.MODEL_PARAMS['model'] == 'LineupPredictorTransformer':
        model = LineupPredictorTransformer(config.MODEL_PARAMS, n_players, n_ages)
    if config.MODEL_PARAMS['model'] == 'LineupPredictor':
        model = LineupPredictor(config.MODEL_PARAMS, n_players, n_ages)

    if config.MODEL_PARAMS['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.MODEL_PARAMS['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.MODEL_PARAMS['lr'])

    # Load checkpoint if it exists
    # checkpoint_path = 'checkpoint.pth'
    # if os.path.exists(checkpoint_path):
    #     start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    # else:
    #     start_epoch = 0

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    wandb.watch(model, log='all', log_freq=100)
    loss_fn = nn.MSELoss()
    model.to(device)

    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        test_loop(test_dataloader, model, loss_fn)
        checkpoint_path = f"checkpoints/{wandb.run.name}__{epoch}.pth"
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
        }, checkpoint_path)
        save_model_configuration(
            checkpoint_path,
            config,
            { 'n_players': n_players, 'n_ages': n_ages }
        )

    print("Done.")


def eval():
    model, optimizer = initialize_saved_model(get_latest_file('checkpoints'))
    model.eval()
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Loading data...")
    dataset = BasketballDataset(config.DATA_PATH)


if __name__ == "__main__":
    main()
    # elements = an array of 10 zeros
    # elements = [0] * 10
    # replacement = 1
    # total = []
    # for i in range(1, 2 ** len(elements) - 1):
    #     print(bin(i)[2:])
    #     new_elements = [replacement if (i & (1 << j)) else elements[j] for j in range(len(elements))]
    #     print(new_elements)

