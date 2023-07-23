# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 34000

EPOCHS_PER_CHECKPOINT = 100

# Model parameters
MODEL_PARAMS = {
    'lr': 0.0001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 12,
    'player_embedding_dim': 12,
    'model': 'LineupPredictorTransformer',
    'n_head': 4,
    'n_layers': 2,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 90,
    'lineup_abs_point_max_threshold_per_60': 10,
    'player_total_seconds_threshold': 25*60*82,
    'min_max_target': True,
    'z_score_target': False,
    'gradient_clipping': False,
    'transformer_dropout': 0,
    'xavier_init': True,
    'augment_with_generic_players': True,
    # 'specific_init': 50.0
    # Add any other parameters your model needs
}

