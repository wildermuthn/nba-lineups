# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 48000

EPOCHS_PER_CHECKPOINT = 50

# Model parameters
MODEL_PARAMS = {
    'lr': 0.001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 4,
    'player_embedding_dim': 4,
    'model': 'LineupPredictorTransformer',
    'n_head': 2,
    'n_layers': 1,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 60,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': True,
    'gradient_clipping': False,
    'transformer_dropout': 0.4,
    'xavier_init': False,
    'specific_init': 50.0
    # Add any other parameters your model needs
}

