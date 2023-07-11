import os
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd

class BasketballDataset(Dataset):
    def __init__(self, directory):
        self.data = []

        # Get lineup diffs
        lineup_dir = os.path.join(directory, 'lineup_diffs')
        self.lineup_diffs = []
        for filename in os.listdir(lineup_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(lineup_dir, filename), 'rb') as f:
                    lineup_diffs = pickle.load(f)
                    for diff in lineup_diffs:
                        diff['game_id'] = filename[:-4]
                        self.lineup_diffs.append(diff)

        # Get player IDs and num players
        self.player_ids = []
        self.calc_num_players()
        print("Number of unique players: {}".format(len(self.player_ids)))

        # Load player data based on parquet files
        self.player_info = {}
        player_dir = os.path.join(directory, 'player_info')
        for filename in os.listdir(player_dir):
            if filename.endswith(".parquet"):
                player_df = pd.read_parquet(os.path.join(player_dir, filename))
                player_data = player_df.iloc[0]
                # convert pandas array to dict
                player_data = player_data.to_dict()
                # add to dict
                self.player_info[player_data['PERSON_ID']] = player_data
        print('Number of players in player_info: {}'.format(len(self.player_info)))

        # Load all seasons game data
        self.game_data = {}
        game_dir = os.path.join(directory, 'seasons')
        for filename in os.listdir(game_dir):
            if filename.endswith(".parquet"):
                season_df = pd.read_parquet(os.path.join(game_dir, filename))
                season_data = season_df.to_dict('records')
                for game in season_data:
                    # Calculate how many years ago the game was played
                    game_date = pd.Timestamp(game['GAME_DATE'])
                    now = pd.Timestamp('now')
                    years_ago = (now - game_date).days / 365
                    years_ago = int(round(years_ago, 0))
                    game['YEARS_AGO'] = years_ago
                    self.game_data[game['GAME_ID']] = game

        print('Number of games in game_data: {}'.format(len(self.game_data)))

        # Get player ages
        self.player_ages_set = set()
        self.player_ages = {}
        self.player_ages_set = set(range(17, 51))
        for player_id in self.player_info:
            birthday = self.player_info[player_id]['BIRTHDATE']
            # Calculate current age based on today's date
            now = pd.Timestamp('now')
            birthday_timestamp = pd.Timestamp(birthday)
            age_in_years = (now - birthday_timestamp).days / 365
            age_in_years = int(round(age_in_years, 0))
            self.player_ages[player_id] = age_in_years
            self.player_info[player_id]['AGE'] = age_in_years
        self.player_age_to_index = {player_age: index for index, player_age in enumerate(self.player_ages_set)}

        # Create player_id_idx
        self.player_id_to_index = None
        self.create_player_id_idx()

        # Create self.data
        for sample in self.lineup_diffs:
            # Get home and away lineups
            try:
                home_lineup = sample['home']
                away_lineup = sample['away']
                # Get plus_minus
                plus_minus = sample['plus_minus']
                time_played = sample['time_played']
                plus_minus_per_second = plus_minus / time_played
                # Get home and away player info
                home_player_info = []
                away_player_info = []
                for player_id in home_lineup:
                    home_player_info.append(self.player_info[player_id])
                for player_id in away_lineup:
                    away_player_info.append(self.player_info[player_id])
                # Add to self.data
                # Ensure there are exactly 5 players on each team, or don't add the data
                if len(home_player_info) != 5 or len(away_player_info) != 5:
                    continue
                game_id = sample['game_id']
                game_info = self.game_data[game_id]
                # Add year difference between game and today
                self.data.append({
                    'home': home_player_info,
                    'away': away_player_info,
                    'plus_minus_per_second': plus_minus_per_second,
                    'years_ago': game_info['YEARS_AGO'],
                })
            except Exception as e:
                continue

    def __len__(self):
        return len(self.data)

    def create_player_id_idx(self):
        self.player_id_to_index = {player_id: index for index, player_id in enumerate(self.player_ids)}

    def calc_num_players(self):
        players = set()
        for sample in self.lineup_diffs:
            players.update(sample['home'])
            players.update(sample['away'])
        self.player_ids = list(players)

    def __getitem__(self, idx):
        sample = self.data[idx]
        years_ago = sample['years_ago']
        # Convert lists of player IDs to tensors
        away = torch.tensor([
            (self.player_id_to_index[player['PERSON_ID']],
             self.player_age_to_index[player['AGE']-years_ago])
            for player in sample['away']]
        )
        home = torch.tensor([
            (self.player_id_to_index[player['PERSON_ID']],
             self.player_age_to_index[player['AGE']-years_ago])
            for player in sample['home']])
        # concat away and home
        players = torch.cat((home, away))
        plus_minus_per_second = torch.tensor([sample['plus_minus_per_second']])
        # Return the sample as a tuple
        return players, plus_minus_per_second

    def split(self, train_fraction):
        train_length = int(train_fraction * len(self))
        eval_length = len(self) - train_length
        return torch.utils.data.random_split(self, [train_length, eval_length])