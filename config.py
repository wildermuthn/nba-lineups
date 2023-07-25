PARAMS = {
    # Meta parameters
    'batch_size': 512,
    'n_epochs': 1000000,
    'epochs_per_checkpoint': 100,
    'data_path': 'data/raw',

    # Model parameters
    'lr': 2.7510162796840593e-05,
    'linear_embedding_dim': 32,
    'player_embedding_dim': 32,
    'model': 'LineupPredictorTransformer',
    'n_head': 8,
    'n_layers': 16,
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 30,
    'lineup_abs_point_max_threshold_per_60': 30,
    'player_total_seconds_threshold': 10*60*82,
    'min_max_target': True,
    'z_score_target': False,
    'gradient_clipping': False,
    'transformer_dropout': 0.21036599520792046,
    'xavier_init': False,
    'augment_with_generic_players': False,
    'augment_every_n_samples': 100,
    'train_specific_season': None,
    'specific_init': None,
    'log_scores': True,
    'log_all': True,
    # Add any other parameters your model needs
}

