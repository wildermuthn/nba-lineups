import os
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
import traceback
from tqdm import tqdm
import itertools
import random
import numpy as np


class BasketballDataset(Dataset):
    def __init__(self, config):
        self.min_max_target = config.PARAMS['min_max_target']
        self.data = []
        self.num_generic_players = 0
        self.player_total_seconds = {}
        self.player_total_seconds_threshold = config.PARAMS['player_total_seconds_threshold']
        self.lineups_skipped = 0
        self.z_score_target = config.PARAMS['z_score_target']
        self.lineup_abs_point_max_threshold_per_60 = config.PARAMS['lineup_abs_point_max_threshold_per_60']
        self.lineup_abs_point_min_threshold_per_60 = config.PARAMS['lineup_abs_point_min_threshold_per_60']
        self.train_specific_season = config.PARAMS['train_specific_season']
        self.player_index_to_player_info = {}
        directory = config.PARAMS['data_path']
        self.lineup_time_played_threshold = config.PARAMS['lineup_time_played_threshold']
        self.augment_every_n_samples = config.PARAMS['augment_every_n_samples']
        self.max_starting_score_diff = config.PARAMS['max_starting_score_diff']
        self.game_type = config.PARAMS['game_type']
        # Get lineup diffs
        lineup_dir = os.path.join(directory, 'lineup_diffs')
        self.lineup_diffs = []
        for filename in os.listdir(lineup_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(lineup_dir, filename), 'rb') as f:
                    lineup_diffs_load = pickle.load(f)
                    for diff in lineup_diffs_load:
                        diff['game_id'] = filename[:-4]
                        self.lineup_diffs.append(diff)

        if self.train_specific_season is not None:
            self.lineup_diffs = [diff for diff in self.lineup_diffs if diff['season_ago'] == self.train_specific_season]

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

        # Get total seconds per player
        for sample in self.lineup_diffs:
            home_lineup = sample['home']
            away_lineup = sample['away']
            # Get plus_minus
            time_played = sample['time_played']
            # Add player total minutes
            for player_id in home_lineup:
                if player_id not in self.player_total_seconds:
                    self.player_total_seconds[player_id] = 0
                self.player_total_seconds[player_id] += time_played
            for player_id in away_lineup:
                if player_id not in self.player_total_seconds:
                    self.player_total_seconds[player_id] = 0
                self.player_total_seconds[player_id] += time_played

        # Add to player_info
        for player_id in self.player_total_seconds:
            self.player_info[player_id]['TOTAL_SECONDS'] = self.player_total_seconds[player_id]

        print(f'Number of raw lineups: {len(self.lineup_diffs)}')

        # Create self.data
        for sample in self.lineup_diffs:
            # Get home and away lineups
            should_skip_sample = False
            try:
                home_lineup = sample['home']
                away_lineup = sample['away']
                # Get plus_minus
                plus_minus = sample['plus_minus']
                time_played = sample['time_played']
                home_plus = sample['home_plus']
                away_plus = sample['away_plus']
                starting_score_diff = sample['starting_score_diff']
                game_type = sample['game_type']
                season_ago = sample['season_ago']
                season = sample['season']
                # Add player total minutes
                for player_id in home_lineup:
                    player_total_seconds = self.player_total_seconds[player_id]
                    if player_total_seconds < self.player_total_seconds_threshold:
                        should_skip_sample = True
                for player_id in away_lineup:
                    player_total_seconds = self.player_total_seconds[player_id]
                    if player_total_seconds < self.player_total_seconds_threshold:
                        should_skip_sample = True
                if time_played < self.lineup_time_played_threshold:
                    self.lineups_skipped += 1
                    continue
                if starting_score_diff > self.max_starting_score_diff:
                    self.lineups_skipped += 1
                    continue
                if self.game_type is not None and game_type != self.game_type:
                    self.lineups_skipped += 1
                    continue
                home_plus_per_minute = home_plus / time_played * 60
                home_plus_per_game = home_plus / time_played * 48 * 60
                away_plus_per_minute = away_plus / time_played * 60
                away_plus_per_game = away_plus / time_played * 48 * 60
                if self.lineup_abs_point_max_threshold_per_60 is not False:
                    if home_plus_per_minute > self.lineup_abs_point_max_threshold_per_60:
                        should_skip_sample = True
                    if away_plus_per_minute > self.lineup_abs_point_max_threshold_per_60:
                        should_skip_sample = True
                if self.lineup_abs_point_min_threshold_per_60 is not False:
                    if home_plus_per_minute < self.lineup_abs_point_min_threshold_per_60:
                        should_skip_sample = True
                    if away_plus_per_minute < self.lineup_abs_point_min_threshold_per_60:
                        should_skip_sample = True
                if should_skip_sample:
                    self.lineups_skipped += 1
                    continue
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
                        self.num_generic_players += 1
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
                        self.num_generic_players += 1
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
                    'home_plus_per_minute': home_plus_per_minute,
                    'home_plus_per_game': home_plus_per_game,
                    'away_plus_per_minute': away_plus_per_minute,
                    'away_plus_per_game': away_plus_per_game,
                    'years_ago': game_info['YEARS_AGO'],
                    'season_ago': season_ago,
                    'game_type': game_type,
                })
            except Exception as e:
                print('Got a problem')
                # print stack trace
                traceback.print_exc()
                self.lineups_skipped += 1
                continue

        all_home_plus_per_minute = [sample['home_plus_per_minute'] for sample in self.data]
        all_home_plus_per_game = [sample['home_plus_per_game'] for sample in self.data]
        all_away_plus_per_minute = [sample['away_plus_per_minute'] for sample in self.data]
        all_away_plus_per_game = [sample['away_plus_per_game'] for sample in self.data]
        all_plus_per_minute = all_home_plus_per_minute + all_away_plus_per_minute
        all_plus_per_game = all_home_plus_per_game + all_away_plus_per_game

        self.min_plus_per_minute = min(all_plus_per_minute)
        self.min_plus_per_game = min(all_plus_per_game)
        self.max_plus_per_minute = max(all_plus_per_minute)
        self.max_plus_per_game = max(all_plus_per_game)
        self.mean_score = np.mean(all_plus_per_minute)
        self.mean_score_per_game = np.mean(all_plus_per_game)
        self.std_score = np.std(all_plus_per_minute)
        self.std_score_per_game = np.std(all_plus_per_game)

        print(f"Number of generic players: {self.num_generic_players}")
        print(f"Number of lineups skipped: {self.lineups_skipped}")

        # self.scores_rest = None
        # self.scores_rest_z_scaled = None
        # self.scores_rest_min_max_scaled = None
        # get random 200,000 items from scores
        if len(all_plus_per_minute) > 200000:
            self.scores = random.sample(all_plus_per_minute, 200000)
            # self.scores_rest = all_plus_per_minute[200000:]
        else:
            self.scores = all_plus_per_minute

        self.scores_z_scaled = [(score - self.mean_score) / self.std_score for score in self.scores]
        self.scores_min_max_scaled = [(score - self.min_plus_per_minute) / (self.max_plus_per_minute - self.min_plus_per_minute) for score in self.scores]
        # if self.scores_rest is not None:
        #     self.scores_rest_z_scaled = [(score - self.mean_score) / self.std_score for score in self.scores_rest]
        #     self.scores_rest_min_max_scaled = [(score - self.min_plus_per_minute) / (self.max_plus_per_minute - self.min_plus_per_minute) for score in self.scores_rest]
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
        new_samples_count = 0
        print(len(self.data))
        for sample in tqdm(self.data):
            count += 1
            if (count % self.augment_every_n_samples) == 0:
                # Combine home and away lineups
                combined_lineup = sample['home'] + sample['away']
                # Get plus_minus and years_ago
                home_plus_per_minute = sample['home_plus_per_minute']
                away_plus_per_minute = sample['away_plus_per_minute']
                years_ago = sample['years_ago']

                # Generate all possible combinations using binary numbers
                for i in range(1, 2 ** num_players - 1):
                    new_lineup = [generic_player if (i & (1 << j)) == 0 else combined_lineup[j] for j in range(num_players)]
                    new_home_lineup = new_lineup[:5]
                    new_away_lineup = new_lineup[5:]
                    new_data.append({
                        'home': new_home_lineup,
                        'away': new_away_lineup,
                        'years_ago': years_ago,
                        'home_plus_per_minute': home_plus_per_minute,
                        'away_plus_per_minute': away_plus_per_minute,
                        'game_type': sample['game_type'],
                        'season_ago': sample['season_ago'],
                    })
        new_samples_count += len(new_data)
        self.data.extend(new_data)
        # return indices of the new data
        return list(range(len(self.data) - new_samples_count, len(self.data)))

    def __len__(self):
        return len(self.data)

    def create_player_id_idx(self):
        self.player_id_to_index = {player_id: index for index, player_id in enumerate(self.player_ids)}
        for player_id, index in self.player_id_to_index.items():
            player_info = self.player_info[player_id]
            self.player_index_to_player_info[index] = player_info
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
        away = torch.tensor(
            [self.get_player_tensor_indexes(player_info, years_ago) for player_info in sample['away']]
        )
        home = torch.tensor(
            [self.get_player_tensor_indexes(player_info, years_ago) for player_info in sample['home']]
        )
        players = torch.cat((home, away))
        if self.min_max_target:
            home_plus_per_minute = torch.tensor([(sample['home_plus_per_minute'] - self.min_plus_per_minute) / (self.max_plus_per_minute - self.min_plus_per_minute)])
            away_plus_per_minute = torch.tensor([(sample['away_plus_per_minute'] - self.min_plus_per_minute) / (self.max_plus_per_minute - self.min_plus_per_minute)])
        elif self.z_score_target:
            home_plus_per_minute = (sample['home_plus_per_minute'] - self.mean_score) / self.std_score
            home_plus_per_minute = torch.tensor([home_plus_per_minute])
            away_plus_per_minute = (sample['away_plus_per_minute'] - self.mean_score) / self.std_score
            away_plus_per_minute = torch.tensor([away_plus_per_minute])
        else:
            home_plus_per_minute = torch.tensor([sample['home_plus_per_minute']])
            away_plus_per_minute = torch.tensor([sample['away_plus_per_minute']])

        # concat home and away
        plus_per_minute = torch.cat((home_plus_per_minute, away_plus_per_minute))

        return players, plus_per_minute

    def split(self, train_fraction):
        train_length = int(train_fraction * len(self))
        eval_length = len(self) - train_length
        return torch.utils.data.random_split(self, [train_length, eval_length])
