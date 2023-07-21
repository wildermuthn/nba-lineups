# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 12000

EPOCHS_PER_CHECKPOINT = 100

# Model parameters
MODEL_PARAMS = {
    'lr': 0.0001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 64,
    'player_embedding_dim': 64,
    'model': 'LineupPredictorTransformer',
    'n_head': 8,
    'n_layers': 4,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 90,
    'lineup_abs_point_max_threshold_per_60': 10,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': True,
    'z_score_target': False,
    'gradient_clipping': False,
    'transformer_dropout': 0.5,
    'xavier_init': True,
    # 'specific_init': 50.0
    # Add any other parameters your model needs
}

