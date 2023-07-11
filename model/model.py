import torch
from torch import nn


class LineupPredictor(torch.nn.Module):

    def __init__(self, params, n_players, n_ages):
        print(params)
        super(LineupPredictor, self).__init__()
        player_embedding_dim = params['player_embedding_dim']
        linear_embedding_dim = params['linear_embedding_dim']

        self.player_embedding = nn.Embedding(n_players, player_embedding_dim)
        self.away_team_embedding = nn.Embedding(1, player_embedding_dim)
        self.home_team_embedding = nn.Embedding(1, player_embedding_dim)
        self.age_embedding = nn.Embedding(n_ages, player_embedding_dim)

        self.linear1 = torch.nn.Linear(player_embedding_dim*10, linear_embedding_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(linear_embedding_dim, 1)

    def forward(self, x):
        # apply embedding to player ids and ages
        (player_ids, player_ages) = x.split(1, dim=2)
        player_ids = self.player_embedding(player_ids)
        player_ages = self.age_embedding(player_ages)
        # Add player age embedding to player id embedding
        x = player_ids + player_ages
        # Add away team embedding to first five players
        x[:, :5, :] += self.away_team_embedding.weight
        # Add home team embedding to last five players
        x[:, 5:, :] += self.home_team_embedding.weight
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x