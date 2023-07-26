PARAMS = {
    # Meta parameters
    'batch_size': 4096,
    'n_epochs': 5,
    'epochs_per_checkpoint': 100,
    'eval_table_per_checkpoint': 5,
    'data_path': 'data/raw',

    # Model parameters
    'lr': .01,
    'player_embedding_dim': 16,
    'n_layers': 2,
    'n_head': 4,
    'transformer_dropout': 0.0,
    'xavier_init': True,
    'model': 'LineupPredictorTransformerV2',
    'optimizer': 'Adam',
    'lineup_time_played_threshold': 30,
    'lineup_abs_point_max_threshold_per_60': 5.5,
    'lineup_abs_point_min_threshold_per_60': 0.25,
    'player_total_seconds_threshold': 49200,
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

