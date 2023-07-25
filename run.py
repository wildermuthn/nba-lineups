import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataloader import BasketballDataset
from model.train import train_loop
from model.test import test_loop
from model.model import LineupPredictor, LineupPredictorTransformer, LineupPredictorJustEmbedding
import config
import wandb
import pickle
from tqdm import tqdm
import os
import numpy as np
import optuna
from optuna.trial import TrialState


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def save_model_configuration(filename, cfg, model_init_params):
    filename = filename[:-4]
    config_dict = {'PARAMS': cfg.PARAMS, **model_init_params}
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(config_dict, f, pickle.HIGHEST_PROTOCOL)


def load_model_configurations(model_filepath):
    # Remove pth from filename
    model_filepath = model_filepath[:-4]
    # load dict from file as a pickle, named afer the model
    with open(model_filepath + '.pkl', 'rb') as f:
        config_dict = pickle.load(f)
    return config_dict


def initialize_model(model_filepath, dataset):
    # Initialize model
    print("Initializing model...")
    saved_config = None
    if model_filepath is not None:
        saved_config = load_model_configurations(model_filepath)
        params = saved_config['PARAMS']
        n_players = saved_config['n_players']
        n_ages = saved_config['n_ages']
    else:
        params = config.PARAMS
        n_players = len(dataset.player_ids)
        n_ages = len(dataset.player_ages_set)

    model = None
    if config.PARAMS['model'] == 'LineupPredictorTransformer':
        model = LineupPredictorTransformer(params, n_players, n_ages)
    if config.PARAMS['model'] == 'LineupPredictor':
        model = LineupPredictor(params, n_players, n_ages)
    if config.PARAMS['model'] == 'LineupPredictorJustEmbedding':
        model = LineupPredictorJustEmbedding(params, n_players, n_ages)

    if config.PARAMS['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

    if model_filepath is not None:
        load_checkpoint(model_filepath, model, optimizer)
    return model, optimizer, saved_config


def main(trial):

    config.PARAMS['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    config.PARAMS['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    config.PARAMS['player_embedding_dim'] = trial.suggest_categorical('player_embedding_dim', [16, 32, 64, 128])
    config.PARAMS['n_head'] = trial.suggest_categorical('n_head', [2, 4, 8, 16])
    config.PARAMS['n_layers'] = trial.suggest_categorical('n_layers', [2, 4, 8, 16])
    config.PARAMS['optimizer'] = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    config.PARAMS['transformer_dropout'] = trial.suggest_float('transformer_dropout', 0.0, 0.5)
    config.PARAMS['xavier_init'] = trial.suggest_categorical('xavier_init', [True, False])

    group = 'optuna_3'

    wandb.init(
        project="nba-lineups",
        config=config.PARAMS,
        group=group,
        name=f"{group}_trial_{trial.number}"
    )

    print("Loading data...")

    dataset = BasketballDataset(config)

    g = torch.Generator()
    g.manual_seed(42)
    train_dataset, eval_dataset = dataset.split(train_fraction=0.9)
    if config.PARAMS['augment_with_generic_players']:
        indices = train_dataset.dataset.augment_with_generic_players()
        # add indices to current subset indices, and shuffle
        train_dataset.indices = np.concatenate((train_dataset.indices, indices))
        np.random.shuffle(train_dataset.indices)



    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.PARAMS['batch_size'],
                                  shuffle=True,
                                  pin_memory=True,
                                  generator=g,
                                  )
    test_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.PARAMS['batch_size'],
                                 shuffle=True,
                                 pin_memory=True)
    # print size of datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(eval_dataset)}")
    n_players = len(dataset.player_ids)
    n_ages = len(dataset.player_ages_set)

    # Initialize model
    print("Initializing model...")
    model = None
    if config.PARAMS['model'] == 'LineupPredictorTransformer':
        model = LineupPredictorTransformer(config.PARAMS, n_players, n_ages)
    if config.PARAMS['model'] == 'LineupPredictor':
        model = LineupPredictor(config.PARAMS, n_players, n_ages)
    if config.PARAMS['model'] == 'LineupPredictorJustEmbedding':
        model = LineupPredictorJustEmbedding(config.PARAMS, n_players, n_ages)

    if config.PARAMS['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.PARAMS['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.PARAMS['lr'])

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    wandb.watch(model, log='all', log_freq=100)
    loss_fn = nn.MSELoss()
    model.to(device)
    scores = dataset.scores
    scores_data = [[s] for s in scores]
    table = wandb.Table(data=scores_data, columns=["scores"])
    wandb.log({'plus_minus_per_minute_histogram_raw': wandb.plot.histogram(
        table, "scores",
        title="Score Distribution (raw)"
    )})

    scores = dataset.scores_z_scaled
    scores_data = [[s] for s in scores]
    table = wandb.Table(data=scores_data, columns=["scores"])
    wandb.log({'plus_minus_per_minute_histogram_z_scaled': wandb.plot.histogram(
        table, "scores",
        title="Score Distribution (z-scaled)"
    )})

    scores = dataset.scores_min_max_scaled
    scores_data = [[s] for s in scores]
    table = wandb.Table(data=scores_data, columns=["scores"])
    wandb.log({'plus_minus_per_minute_histogram_min_max': wandb.plot.histogram(
        table, "scores",
        title="Score Distribution (min-max scaled)"
    )})

    epochs = config.PARAMS['n_epochs']
    loss = None
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        last_step, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        test_loss = test_loop(test_dataloader, model, loss_fn, epoch, step=last_step)
        checkpoint_path = f"checkpoints/{wandb.run.name}__{epoch}.pth"
        if epoch % config.PARAMS['epochs_per_checkpoint'] == 0:
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
        loss = test_loss
        trial.report(loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
    wandb.finish()
    return loss

    print("Done.")

def eval_lineups(filepath):
    batch_size = config.PARAMS['batch_size']
    print("Loading data...")
    dataset = BasketballDataset(config)
    g = torch.Generator()
    g.manual_seed(42)
    train_dataset, eval_dataset = dataset.split(train_fraction=0.9)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=g,
    )

    model, optimizer, saved_config = initialize_model(filepath, dataset)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    player_info = dataset.player_info
    lineup_preds = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            # X.shape = (batch_size, 10, 2)
            print('X', X.shape)
            y = y.float().to(device)
            # y.shape = (batch_size, 1)
            print('y', y.shape)
            pred = model(X)
            print('pred', pred.shape)
            # pred.shape = (batch_size, 1)

            # Take the first element of X from its 3rd dimension
            X = X[:, :, 0]
            # reshape X to (batch_size, 10)
            X = X.reshape(X.shape[0], X.shape[1])
            lineup_preds.append(torch.cat((X, y, pred), dim=1))
    lineup_preds = torch.cat(lineup_preds, dim=0)
    lineup_preds = lineup_preds.cpu().numpy()
    # lineup_preds.shape = (n_lineups, 12)
    # replace first 10 tokens in each lineup with the actual player names,
    # using dataset.player_index_to_player_info
    lineup_pred_clean = []
    player_total_points = {}
    for i in range(lineup_preds.shape[0]):
        lineup = lineup_preds[i]
        clean_lineup = []
        for j in range(10):
            player_points = lineup[-1]
            player_index = int(lineup[j])
            player_info = dataset.player_index_to_player_info[player_index]
            if j > 4:
                player_points = 1 - player_points
            if player_info['DISPLAY_FIRST_LAST'] not in player_total_points:
                player_total_points[player_info['DISPLAY_FIRST_LAST']] = [player_points]
            else:
                player_total_points[player_info['DISPLAY_FIRST_LAST']].append(player_points)
            clean_lineup.append(player_info['DISPLAY_FIRST_LAST'])
        clean_lineup.append(lineup[-2])
        clean_lineup.append(lineup[-1])
        lineup_pred_clean.append(clean_lineup)
    # sort by predicted points per game
    lineup_pred_clean = sorted(lineup_pred_clean, key=lambda x: x[-1])
    # print top 10
    for i in range(10):
        lineup = lineup_pred_clean[-i-1]
        print(f"{i+1} Home. {lineup[0]} | {lineup[1]} | {lineup[2]} | {lineup[3]} | {lineup[4]}")
        print(f"{i+1} Away. {lineup[5]} | {lineup[6]} | {lineup[7]} | {lineup[8]} | {lineup[9]}")
        print(f"Actual points: {lineup[-2]}")
        print(f"Predicted points: {lineup[-1]}")
        print()
    player_average_points = {}
    # For each player in player_total_points, average their points
    for player, points_arr in player_total_points.items():
        player_average_points[player] = np.mean(points_arr)
    # sort by average points
    player_average_points = sorted(player_average_points.items(), key=lambda x: x[1])
    # print top 10
    for i in range(100):
        player = player_average_points[-i-1]
        print(f"{i+1} {player[0]}: {player[1]}")

    print("Done.")


def eval(filepath=None):
    dataset = BasketballDataset(config, transform=None)
    # filepath = 'checkpoints/avid-waterfall-148__4.pth'
    model, optimizer, saved_config = initialize_model(filepath, dataset)
    model.eval()
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Loading data...")
    player_info = dataset.player_info
    generic_players = torch.tensor(
        [[dataset.get_player_tensor_indexes({'IS_GENERIC': True}, 0) for i in range(10)]]
    ).to(device)
    pred = model(generic_players)
    pps = pred.item()
    print(f"Predicted points per game: {pps}")
    player_preds = {}

    # filter player_info for 'GAMES_PLAYED_CURRENT_SEASON_FLAG' == 'Y'
    player_info = {k: v for k, v in player_info.items() if v['TOTAL_SECONDS'] >
                   config.PARAMS['player_total_seconds_threshold'] and
                   v['TO_YEAR'] == 2023
                   }

    # loop over key values of player_info dict with tqdm
    for player_id, player in tqdm(player_info.items()):
        # replace first element in generic_players with player
        player_id_age = dataset.get_player_tensor_indexes(player, 0)
        player_id_age = torch.tensor(player_id_age).to(device)
        generic_players[0][0] = player_id_age
        # generic_players[0][1] = player_id_age
        # generic_players[0][2] = player_id_age
        # generic_players[0][3] = player_id_age
        # generic_players[0][4] = player_id_age
        # generic_players[0][5] = player_id_age
        # generic_players[0][6] = player_id_age
        # generic_players[0][7] = player_id_age
        # generic_players[0][8] = player_id_age
        # generic_players[0][9] = player_id_age
        pred = model(generic_players)
        pps = pred.item()
        player_preds[player['DISPLAY_FIRST_LAST']] = pps

    sorted_players = sorted(player_preds.items(), key=lambda x: x[1], reverse=True)
    # delete player_predictions.txt if it exists
    if os.path.exists('player_predictions.txt'):
        os.remove('player_predictions.txt')

    with open('player_predictions.txt', 'a') as f:
        for i, player in enumerate(sorted_players):
            print(f"{i+1}. {player[0]}: {player[1]}")
            # Write to new line in file
            f.write(f"{i+1}. {player[0]}: {player[1]}\n")

    print('done')


def eval_simple(filepath=None):
    dataset = BasketballDataset(config, transform=None)
    # filepath = 'checkpoints/avid-waterfall-148__4.pth'
    model, optimizer, saved_config = initialize_model(filepath, dataset)
    model.eval()
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Loading data...")
    player_info = dataset.player_info
    player_info = {k: v for k, v in player_info.items() if v['TOTAL_SECONDS'] >
                   config.PARAMS['player_total_seconds_threshold'] and
                   v['TO_YEAR'] == 2023
                   }
    player_preds = {}
    # loop over key values of player_info dict with tqdm
    for player_id, player in tqdm(player_info.items()):
        # replace first element in generic_players with player
        player_id_index, player_age_index = dataset.get_player_tensor_indexes(player, 0)
        player_embedding = model.get_player_embedding(player_id_index)
        player_preds[player['DISPLAY_FIRST_LAST']] = player_embedding.item()

    sorted_players = sorted(player_preds.items(), key=lambda x: x[1], reverse=True)
    # delete player_predictions.txt if it exists
    if os.path.exists('player_predictions.txt'):
        os.remove('player_predictions.txt')

    with open('player_predictions.txt', 'a') as f:
        for i, player in enumerate(sorted_players):
            print(f"{i+1}. {player[0]}: {player[1]}")
            # Write to new line in file
            f.write(f"{i+1}. {player[0]}: {player[1]}\n")

    print('done')


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(main, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # eval_simple('checkpoints/swept-dust-292__500.pth')
