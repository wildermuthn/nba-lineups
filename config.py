# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 10000

# Model parameters
MODEL_PARAMS = {
    'lr': 0.00001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 128,
    'player_embedding_dim': 128,
    'model': 'LineupPredictorTransformer',
    'n_head': 8,
    'n_layers': 4,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 120,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': True,
    'gradient_clipping': False,
    # Add any other parameters your model needs
}

