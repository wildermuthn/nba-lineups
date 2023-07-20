# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 24000

EPOCHS_PER_CHECKPOINT = 5

# Model parameters
MODEL_PARAMS = {
    'lr': 0.00001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 12,
    'player_embedding_dim': 12,
    'model': 'LineupPredictorTransformer',
    'n_head': 4,
    'n_layers': 2,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 60,
    'lineup_abs_point_max_threshold_per_60': 10,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': False,
    'z_score_target': True,
    'gradient_clipping': False,
    'transformer_dropout': 0,
    'xavier_init': True,
    # 'specific_init': 50.0
    # Add any other parameters your model needs
}

