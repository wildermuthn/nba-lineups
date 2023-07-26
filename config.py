PARAMS = {
    # Meta parameters
    'batch_size': 2048,
    'n_epochs': 5000,
    'epochs_per_checkpoint': 100,
    'eval_table_per_checkpoint': 5,
    'data_path': 'data/raw',

    # Model parameters
    'lr': 0.007753,
    'player_embedding_dim': 1,
    'n_layers': 16,
    'n_head': 2,
    'transformer_dropout': 0.1824,
    'xavier_init': True,
    'model': 'LineupPredictorJustEmbedding',
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 30,
    'lineup_abs_point_max_threshold_per_60': 5.5,
    'lineup_abs_point_min_threshold_per_60': 0.25,
    'player_total_seconds_threshold': 10*60*82,
    'min_max_target': True,
    'z_score_target': False,
    'gradient_clipping': False,
    'augment_with_generic_players': True,
    'augment_every_n_samples': 100,
    'train_specific_season': 0,
    'specific_init': None,
    'log_scores': False,
    'log_all': False,
    'linear_embedding_dim': 16,
    'max_starting_score_diff': 10,
    # Add any other parameters your model needs
}

