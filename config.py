PARAMS = {
    # Meta parameters
    'batch_size': 4096,
    'n_epochs': 1000000,
    'epochs_per_checkpoint': 100,
    'data_path': 'data/raw',

    # Model parameters
    'lr': .000343,
    'player_embedding_dim': 16,
    'n_layers': 16,
    'n_head': 2,
    'transformer_dropout': 0.1824,
    'xavier_init': True,
    'model': 'LineupPredictorTransformer',
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 30,
    'lineup_abs_point_max_threshold_per_60': 3,
    'player_total_seconds_threshold': 10*60*82,
    'min_max_target': True,
    'z_score_target': False,
    'gradient_clipping': False,
    'augment_with_generic_players': False,
    'augment_every_n_samples': 100,
    'train_specific_season': None,
    'specific_init': None,
    'log_scores': True,
    'log_all': True,
    'linear_embedding_dim': 16,
    # Add any other parameters your model needs
}

