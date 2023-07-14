# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 24000

EPOCHS_PER_CHECKPOINT = 5

# Model parameters
MODEL_PARAMS = {
    'lr': 0.01,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 12,
    'player_embedding_dim': 12,
    'model': 'LineupPredictorTransformer',
    'n_head': 4,
    'n_layers': 2,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 30,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': True,
    'gradient_clipping': False,
    'transformer_dropout': 0.4,
    'xavier_init': True,
    # Add any other parameters your model needs
}

