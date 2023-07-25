PARAMS = {
    # Meta parameters
    'batch_size': 24000,
    'n_epochs': 5,
    'epochs_per_checkpoint': 100,
    'data_path': 'data/raw',

    # Model parameters
    'lr': 0.001,
    'linear_embedding_dim': 32,
    'player_embedding_dim': 32,
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
    'transformer_dropout': 0.4,
    'xavier_init': True,
    'augment_with_generic_players': False,
    'augment_every_n_samples': 100,
    'train_specific_season': None,
    'specific_init': None,
    'log_scores': False,
    # Add any other parameters your model needs
}

