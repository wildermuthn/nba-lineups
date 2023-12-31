PARAMS = {
    # Meta parameters
    'batch_size': 32768,
    'n_epochs': 100,
    'epochs_per_checkpoint': 100,
    'eval_table_per_checkpoint': 100,
    'data_path': 'data/raw',

    # Model parameters
    'lr': 0.009889,
    'player_embedding_dim': 1,
    'n_layers': 16,
    'n_head': 2,
    'transformer_dropout': 0.1824,
    'xavier_init': True,
    'model': 'LineupPredictorTransformer',
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 60,
    'lineup_abs_point_max_threshold_per_60': 14,
    'lineup_abs_point_min_threshold_per_60': 0.25,
    'player_total_seconds_threshold': 82*15*60,
    'min_max_target': True,
    'z_score_target': False,
    'log_target': False,
    'gradient_clipping': False,
    'augment_with_generic_players': True,
    'augment_n_per_sample': 100,
    'augment_every_n_samples': 1,
    'train_specific_season': 0,
    'specific_init': None,
    'log_scores': True,
    'log_all': False,
    'linear_embedding_dim': 16,
    'max_starting_score_diff': 24,
    'game_type': 'regular',
    # Add any other parameters your model needs
}

