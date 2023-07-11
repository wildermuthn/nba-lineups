import os
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
import traceback
from tqdm import tqdm

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

        # Get player IDs and num players
        self.player_ids = []
        self.calc_num_players()
        print("Number of unique players: {}".format(len(self.player_ids)))

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
                if time_played == 0:
                    continue
                plus_minus_per_second = plus_minus / time_played
                # Get home and away player info
                home_player_info = []
                away_player_info = []
                for player_id in home_lineup:
                    try:
                        player_info = self.player_info[player_id]
                    except:
                        # Use generic player info
                        player_info = {
                            'PERSON_ID': len(self.player_ids) + 1,
                            'IS_GENERIC': True
                        }
                    home_player_info.append(player_info)
                for player_id in away_lineup:
                    try:
                        player_info = self.player_info[player_id]
                    except:
                        # Use generic player info
                        player_info = {
                            'PERSON_ID': len(self.player_ids) + 1,
                            'IS_GENERIC': True
                        }
                    away_player_info.append(player_info)
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
                print('Got a problem')
                # print stack trace
                traceback.print_exc()
                continue
        # self.augment_with_generic_players()

    def augment_with_generic_players(self):
        # For each sample, add additional samples that replace one or more players with a generic player
        new_data = []
        num_players = 10  # Total number of players in combined home and away lineups

        # Create a generic player
        generic_player = {
            'PERSON_ID': len(self.player_ids) + 1,
            'IS_GENERIC': True
        }

        count = 0
        print(len(self.data))
        for sample in tqdm(self.data):
            count += 1
            if (count % 100) == 0:
                # Combine home and away lineups
                combined_lineup = sample['home'] + sample['away']
                # Get plus_minus and years_ago
                plus_minus_per_second = sample['plus_minus_per_second']
                years_ago = sample['years_ago']

                # Generate all possible combinations using binary numbers
                for i in range(1, 2 ** num_players - 1):
                    new_lineup = [generic_player if (i & (1 << j)) == 0 else combined_lineup[j] for j in range(num_players)]
                    new_home_lineup = new_lineup[:5]
                    new_away_lineup = new_lineup[5:]
                    new_data.append({
                        'home': new_home_lineup,
                        'away': new_away_lineup,
                        'plus_minus_per_second': plus_minus_per_second,
                        'years_ago': years_ago,
                    })
        self.data.extend(new_data)

    def __len__(self):
        return len(self.data)

    def create_player_id_idx(self):
        self.player_id_to_index = {player_id: index for index, player_id in enumerate(self.player_ids)}
        self.player_id_to_index[len(self.player_ids) + 1] = len(self.player_ids) + 1

    def get_player_index(self, player_info):
        if 'IS_GENERIC' in player_info:
            return self.player_id_to_index[len(self.player_ids) + 1]
        else:
            return self.player_id_to_index[player_info['PERSON_ID']]

    def get_player_age_index(self, player_info, years_ago=0):
        if 'IS_GENERIC' in player_info:
            n_player_ages = len(self.player_age_to_index)
            return n_player_ages
        else:
            current_age = player_info['AGE']
            game_age = current_age - years_ago
            return self.player_age_to_index[game_age]

    def get_player_tensor_indexes(self, player_info, years_ago=0):
        player_index = self.get_player_index(player_info)
        player_age_index = self.get_player_age_index(player_info, years_ago)
        return player_index, player_age_index

    def calc_num_players(self):
        players = set()
        # Get all player ids from player_info dict
        for player_id in self.player_info:
            players.add(player_id)
        self.player_ids = list(players)

    def __getitem__(self, idx):
        sample = self.data[idx]
        years_ago = sample['years_ago']
        # Convert lists of player IDs to tensors
        away = torch.tensor(
            [self.get_player_tensor_indexes(player_info, years_ago) for player_info in sample['away']]
        )
        home = torch.tensor(
            [self.get_player_tensor_indexes(player_info, years_ago) for player_info in sample['home']]
        )
        # concat away and home
        players = torch.cat((home, away))
        plus_minus_per_second = torch.tensor([sample['plus_minus_per_second']])
        # Return the sample as a tuple
        return players, plus_minus_per_second

    def split(self, train_fraction):
        train_length = int(train_fraction * len(self))
        eval_length = len(self) - train_length
        return torch.utils.data.random_split(self, [train_length, eval_length])