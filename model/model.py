import torch
from torch import nn
import torch.nn.init as init


class LineupPredictorJustEmbedding(torch.nn.Module):

    def __init__(self, params, n_players, n_ages):
        print(params)
        super(LineupPredictorJustEmbedding, self).__init__()
        self.gradient_clipping = params['gradient_clipping']
        self.generic_player_id = torch.tensor(n_players + 1, dtype=torch.int64).to('cuda')
        player_embedding_dim = params['player_embedding_dim']
        linear_embedding_dim = params['linear_embedding_dim']
        total_players = n_players + 2 # For General Players

        self.player_embedding = nn.Embedding(total_players, player_embedding_dim)
        self.linear2 = torch.nn.Linear(player_embedding_dim*10, 1)

    def forward(self, x):
        # apply embedding to player ids and ages
        (player_ids, player_ages) = x.split(1, dim=2)
        player_ids_embedded = self.player_embedding(player_ids)
        x = player_ids_embedded

        generic_player_mask = (player_ids == self.generic_player_id)
        if generic_player_mask.any():
            avg_player_embedding = player_ids_embedded.mean(dim=1, keepdim=True)
            avg_embedding = avg_player_embedding
            x = torch.where(generic_player_mask.unsqueeze(-1), avg_embedding, x)

        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x

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
        x = player_ids_embedded + player_ages_embedded

        generic_player_mask = (player_ids == self.generic_player_id)
        if generic_player_mask.any():
            # Calculate the average embedding of all players
            avg_player_embedding = player_ids_embedded.mean(dim=1, keepdim=True)
            avg_age_embedding = player_ages_embedded.mean(dim=1, keepdim=True)
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
            nn.TransformerEncoderLayer(d_model=player_embedding_dim, nhead=params['n_head']), num_layers=params['n_layers'])

        self.linear = torch.nn.Linear(player_embedding_dim, 1)

        # initialize weights
        # init.xavier_uniform_(self.player_embedding.weight)
        # init.xavier_uniform_(self.age_embedding.weight)
        # self.init_weights()

    def init_weights(self) -> None:
        init_range = 5.0
        self.player_embedding.weight.data.uniform_(-init_range, init_range)
        self.age_embedding.weight.data.uniform_(-init_range, init_range)
        self.away_team_embedding.weight.data.uniform_(-init_range, init_range)
        self.home_team_embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        # apply embedding to player ids and ages
        player_ids, player_ages = x.split(1, dim=2)
        player_ids_embedded = self.player_embedding(player_ids)
        # player_ages_embedded = self.age_embedding(player_ages)

        # Add player age embedding to player id embedding
        x = player_ids_embedded

        # Check if any player_id is the generic player_id
        generic_player_mask = (player_ids == self.generic_player_id)
        if generic_player_mask.any():
            # Calculate the average embedding of all players
            avg_player_embedding = player_ids_embedded.mean(dim=1, keepdim=True)
            # avg_age_embedding = player_ages_embedded.mean(dim=1, keepdim=True)
            avg_embedding = avg_player_embedding
            # Replace the embeddings of generic players with the average embedding
            x = torch.where(generic_player_mask.unsqueeze(-1), avg_embedding, x)

        # Reshape x to have 3 dimensions
        x = x.view(x.shape[0], x.shape[1], -1)

        # # Add home team embedding to first five players
        # x[:, :5, :] += self.home_team_embedding.weight
        # # Add away team embedding to last five players
        # x[:, 5:, :] += self.away_team_embedding.weight

        # Reshape x to meet the input requirements of TransformerEncoder
        x = x.permute(1, 0, 2)

        # Pass the sequence through the Transformer encoder
        x = self.transformer_encoder(x)

        # Use only the output of the last token
        x = x[-1, :, :]

        # Pass the output through the linear layer
        x = self.linear(x)

        return x