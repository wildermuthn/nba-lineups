# Path to the data
DATA_PATH = 'data/raw'

# Batch size for the DataLoader
BATCH_SIZE = 640

# Model parameters
MODEL_PARAMS = {
    'lr': 0.001,
    'batch_size': BATCH_SIZE,
    'linear_embedding_dim': 64,
    'player_embedding_dim': 64,
    'model': 'LineupPredictorTransformer',
    'n_head': 16,
    'n_layers': 16,
    'optimizer': 'Adam',
    # Add any other parameters your model needs
}

