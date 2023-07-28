import torch
from torch import nn
import torch.nn.init as init

import config


class LineupPredictorJustEmbedding(torch.nn.Module):

    def __init__(self, params, n_players, n_ages):
        super(LineupPredictorJustEmbedding, self).__init__()
        self.gradient_clipping = params['gradient_clipping']
        self.generic_player_id = torch.tensor(n_players + 1, dtype=torch.int64).to('cuda')
        self.batch_size = params['batch_size']
        player_embedding_dim = params['player_embedding_dim']
        total_players = n_players
        self.player_embedding = nn.Embedding(total_players + 5, player_embedding_dim)
        self.age_embedding = nn.Embedding(n_ages + 5, player_embedding_dim)
        self.home_embedding = torch.nn.Parameter(torch.randn(1, player_embedding_dim))

    def get_player_embedding(self, player_id):
        t = torch.tensor(player_id, dtype=torch.int64).to('cuda')
        return self.player_embedding(t)

    def forward(self, x):
        (player_ids, player_ages) = x.split(1, dim=2)
        x = self.player_embedding(player_ids) # * self.age_embedding(player_ages)
        home_x = x[:, :5]
        away_x = x[:, 5:]
        sum_home_x = torch.sum(home_x, dim=1) #  + self.home_embedding
        sum_away_x = torch.sum(away_x, dim=1)
        # Resize dimensions
        sum_home_x = sum_home_x.view(sum_home_x.shape[0], -1)
        sum_away_x = sum_away_x.view(sum_away_x.shape[0], -1)
        pred = torch.cat((sum_home_x, sum_away_x), dim=1)
        return pred


class LineupPredictor(torch.nn.Module):

    def __init__(self, params, n_players, n_ages):
        print(params)
        super(LineupPredictor, self).__init__()
        player_embedding_dim = params['player_embedding_dim']
        linear_embedding_dim = params['linear_embedding_dim']
        total_players = n_players + 2 # For General Players
        self.gradient_clipping = params['gradient_clipping']
        self.generic_player_id = 1400

        self.player_embedding = nn.Embedding(total_players, player_embedding_dim)
        self.away_team_embedding = nn.Embedding(1, player_embedding_dim)
        self.home_team_embedding = nn.Embedding(1, player_embedding_dim)
        self.age_embedding = nn.Embedding(n_ages + 1, player_embedding_dim)

        self.linear1 = torch.nn.Linear(player_embedding_dim*10, linear_embedding_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(linear_embedding_dim, 1)

        # initialize weights
        init.xavier_uniform_(self.player_embedding.weight)
        init.xavier_uniform_(self.away_team_embedding.weight)
        init.xavier_uniform_(self.home_team_embedding.weight)
        init.xavier_uniform_(self.age_embedding.weight)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        # apply embedding to player ids and ages
        (player_ids, player_ages) = x.split(1, dim=2)
        player_ids_embedded = self.player_embedding(player_ids)
        player_ages_embedded = self.age_embedding(player_ages)

        # Add player age embedding to player id embedding
        x = player_ids_embedded * player_ages_embedded

        generic_player_mask = (player_ids == self.generic_player_id)
        if generic_player_mask.any():
            # Calculate the average embedding of all players
            avg_player_embedding = self.player_embedding.weight.mean(dim=0, keepdim=True)
            avg_age_embedding = self.age_embedding.weight.mean(dim=0, keepdim=True)
            avg_embedding = avg_player_embedding + avg_age_embedding
            # Replace the embeddings of generic players with the average embedding
            x = torch.where(generic_player_mask.unsqueeze(-1), avg_embedding, x)

        # Add away team embedding to first five players
        x[:, :5, :] += self.away_team_embedding.weight
        # Add home team embedding to last five players
        x[:, 5:, :] += self.home_team_embedding.weight
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class LineupPredictorTransformer(nn.Module):
    def __init__(self, params, n_players, n_ages):
        super(LineupPredictorTransformer, self).__init__()
        player_embedding_dim = params['player_embedding_dim']
        self.gradient_clipping = params['gradient_clipping']
        self.generic_player_id = torch.tensor(n_players + 1, dtype=torch.int64).to('cuda')

        self.player_embedding = nn.Embedding(n_players+5, player_embedding_dim)
        self.age_embedding = nn.Embedding(n_ages+5, player_embedding_dim)

        self.away_team_embedding = nn.Embedding(1, player_embedding_dim)
        self.home_team_embedding = nn.Embedding(1, player_embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=player_embedding_dim,
                nhead=params['n_head'],
                dropout=params['transformer_dropout']
            ),
            num_layers=params['n_layers'],
        )

        self.linear = torch.nn.Linear(int(player_embedding_dim*5), 1)

        # initialize weights
        if params['xavier_init']:
            init.xavier_uniform_(self.player_embedding.weight)
            init.xavier_uniform_(self.age_embedding.weight)
            init.xavier_uniform_(self.away_team_embedding.weight)
            init.xavier_uniform_(self.home_team_embedding.weight)
        elif params['specific_init']:
            self.init_weights(config.PARAMS['specific_init'])

    def init_weights(self, init_range) -> None:
        self.player_embedding.weight.data.uniform_(-init_range, init_range)
        self.age_embedding.weight.data.uniform_(-init_range, init_range)
        self.away_team_embedding.weight.data.uniform_(-init_range, init_range)
        self.home_team_embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        # apply embedding to player ids and ages
        player_ids, player_ages = x.split(1, dim=2)
        player_ids_embedded = self.player_embedding(player_ids)
        player_ages_embedded = self.age_embedding(player_ages)

        # Add player age embedding to player id embedding
        x = player_ids_embedded * player_ages_embedded

        # Check if any player_id is the generic player_id
        generic_player_mask = (player_ids == self.generic_player_id)
        if generic_player_mask.any():
            # Calculate the average embedding of all players
            avg_player_embedding = self.player_embedding.weight.mean(dim=0, keepdim=True)
            avg_age_embedding = self.age_embedding.weight.mean(dim=0, keepdim=True)
            avg_embedding = avg_player_embedding + avg_age_embedding
            # Replace the embeddings of generic players with the average embedding
            x = torch.where(generic_player_mask.unsqueeze(-1), avg_embedding, x)

        # Reshape x to have 3 dimensions
        x = x.view(x.shape[0], x.shape[1], -1)

        # # Add home team embedding to first five players
        x[:, :5, :] += self.home_team_embedding.weight
        # # Add away team embedding to last five players
        x[:, 5:, :] += self.away_team_embedding.weight

        # Get the first five token (home) and shuffle them
        home_tokens = x[:, :5, :]
        home_tokens = home_tokens[:, torch.randperm(home_tokens.size()[1]), :]
        # Get the last five token (away) and shuffle them
        away_tokens = x[:, 5:, :]
        away_tokens = away_tokens[:, torch.randperm(away_tokens.size()[1]), :]
        # Concatenate the home and away tokens
        x = torch.cat((home_tokens, away_tokens), dim=1)

        # Reshape x to meet the input requirements of TransformerEncoder
        x = x.permute(1, 0, 2)

        # Pass the sequence through the Transformer encoder
        x = self.transformer_encoder(x)

        # Sum the home team
        x_home = x[:5, :, :]
        x_home = x_home.view(x_home.shape[1], -1)
        x_home = self.linear(x_home)
        # Sum the away team
        x_away = x[5:, :, :]
        x_away = x_away.view(x_away.shape[1], -1)
        x_away = self.linear(x_away)

        # concat the two outputs
        pred = torch.cat((x_home, x_away), dim=1)

        return pred
