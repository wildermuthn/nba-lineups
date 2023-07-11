# from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, gamerotation, boxscoresummaryv2, boxscoretraditionalv2, commonplayerinfo
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType, SeasonTypePlayoffs
from nba_api.stats.endpoints import playbyplay
import os
from enum import Enum
import pandas as pd
import itertools
import datetime
from operator import itemgetter
import pickle
import time
import random
import traceback
from tqdm import tqdm

pd.set_option('display.max_colwidth', 250)
pd.set_option('display.max_rows', 250)


class EventMsgType(Enum):
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROW = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13

def load_game_rotation(game_id):
    # set parquet path
    path_home = os.path.join(os.getcwd(), 'data', 'raw', 'game_rotations', game_id + '_home.parquet')
    path_away = os.path.join(os.getcwd(), 'data', 'raw', 'game_rotations', game_id + '_away.parquet')
    # check if parquet file exists
    if os.path.exists(path_home) and os.path.exists(path_away):
        return pd.read_parquet(path_home), pd.read_parquet(path_away)
    rotations = gamerotation.GameRotation(game_id=game_id).get_data_frames()
    # save rotations file to parquet, ensuring directory exists
    os.makedirs(os.path.dirname(path_home), exist_ok=True)
    os.makedirs(os.path.dirname(path_away), exist_ok=True)
    rotations[0].to_parquet(path_away)
    rotations[1].to_parquet(path_home)
    return rotations


def get_game_pbp(game_id):
    path = 'data/raw/pbp/' + str(game_id) + '.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        pbp_response = playbyplay.PlayByPlay(game_id)
        pbp = pbp_response.get_data_frames()[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pbp.to_parquet(path, index=False)
    return pbp


def get_game_box_score(game_id):
    path = 'data/raw/box_score/' + str(game_id) + '.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    box_score = boxscoresummaryv2.BoxScoreSummaryV2(game_id).get_data_frames()[0]
    # Save box score to parquet
    os.makedirs(os.path.dirname(path), exist_ok=True)
    box_score.to_parquet(path, index=False)
    return box_score


def get_season_games(season=Season.default):
    # Check to see if df is already saved (parquet)
    path = 'data/raw/seasons/season_' + season + '_games.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    gamefinder_regular_season = leaguegamefinder.LeagueGameFinder(league_id_nullable='00',
                                                                  season_nullable=season,
                                                                  season_type_nullable=SeasonType.regular)
    # get games from the playoffs too
    gamefinder_playoffs = leaguegamefinder.LeagueGameFinder(league_id_nullable='00',
                                                            season_nullable=season,
                                                            season_type_nullable=SeasonTypePlayoffs.playoffs)

    df_regular_season = gamefinder_regular_season.get_data_frames()[0]
    df_playoffs = gamefinder_playoffs.get_data_frames()[0]
    # combine the two dataframes
    df = pd.concat([df_regular_season, df_playoffs])
    df = df.sort_values(by=['GAME_ID'])
    df = df.drop_duplicates(subset=['GAME_ID'])

    # Save df to load later, checking if the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

    return df


def get_last_game(team):
    # Get the team id column cell value from team df
    team_id = team['id'].values[0]
    # Check
    df = get_season_games()
    # get the last game for the team
    game = df[df['TEAM_ID'] == team_id].iloc[-1]
    game_id = game['GAME_ID']
    game_matchup = game['MATCHUP']

    print(f'Searching through {len(df)} game(s) for the game_id of {game_id} where {game_matchup}')
    return game


def get_teams():
    # Check to see if df is already saved
    if os.path.exists('data/raw/nba_teams.parquet'):
        return pd.read_parquet('data/raw/nba_teams.parquet')
    nba_teams = teams.get_teams()
    # Save to load later (using parquet), checking if the directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    pd.DataFrame(nba_teams).to_parquet('data/raw/nba_teams.parquet')
    return teams


def get_team(abbv):
    nba_teams = get_teams()

    # nba_teams is a dataframe. We can filter on it to get the row with the team we want.
    team = nba_teams[nba_teams['abbreviation'] == abbv]
    return team


def add_score_margins(df):
    # find index of row that contains the first non-null score_margin
    prev_score_margin = 0
    for i, row in df.iterrows():
        # if score margin is null, set it to previous score margin
        if pd.isnull(row['SCOREMARGIN']):
            df.at[i, 'SCOREMARGIN'] = prev_score_margin
        elif row['SCOREMARGIN'] == 'TIE':
            df.at[i, 'SCOREMARGIN'] = 0
            prev_score_margin = 0
        else:
            prev_score_margin = row['SCOREMARGIN']
    return df


def remove_duplicate_time_rows(df):
    # remove rows with duplicate PCTIMESTRING values
    df = df.drop_duplicates(subset=['SECONDS_ELAPSED'], keep='last')
    return df


def remove_duplicate_rows(df):
    # remove rows with duplicate PCTIMESTRING values
    df = df.drop_duplicates(subset=['PCTIMESTRING'], keep='last')
    return df


def parse_lineups(game_rotations):
    away_rotations = game_rotations[0]
    home_rotations = game_rotations[1]

    home_team_name = home_rotations['TEAM_NAME'].iloc[0]

    rotations = pd.concat([home_rotations, away_rotations])

    # Create a list of players with time in or time out
    player_events = []
    for i, row in rotations.iterrows():
        player_events.append((row['PERSON_ID'], row['IN_TIME_REAL'], 'IN', row['TEAM_NAME']))
        player_events.append((row['PERSON_ID'], row['OUT_TIME_REAL'], 'OUT', row['TEAM_NAME']))

    # Sort the list by time IN
    player_events.sort(key=lambda x: x[1])

    # Group events by time
    player_events_by_time = []
    for key, group in itertools.groupby(player_events, lambda x: x[1]):
        player_events_by_time.append(list(group))

    # Map over the list of events and create a dictionary for each item
    # <team_name1>: [list of players on the court]
    # <team_name2>: [list of players on the court]
    # time: <time in seconds> that event occurred
    player_events_by_time_parsed = []

    current_players = {}
    current_players['home'] = []
    current_players['away'] = []
    for i, event in enumerate(player_events_by_time):
        time = event[0][1] # (time is seconds*10)
        # Convert time to period remaining time (there are 12 minutes to each period)
        time_second = round(time / 10)
        util_period, util_time = get_time_period(time_second)
        is_overtime = False
        if time_second > 2880:
            is_overtime = True
        time_seconds_for_period = time_second % 720
        time_seconds_remaining_in_period = 720 - time_seconds_for_period
        period_time = str(datetime.timedelta(seconds=time_seconds_remaining_in_period))
        # remove 0s from the start of the string period_time (e.g. 03:00 -> 3:00)
        # format period time to display m:ss
        period_time = period_time[2:]
        if period_time[0] == '0':
            period_time = period_time[1:]
        full_game_in_seconds = 12 * 60 * 4
        if time_second <= full_game_in_seconds:
            period_number = time_second // 720 + 1
        else:
            overtime_period_seconds = time_second - full_game_in_seconds
            period_number = overtime_period_seconds // 300 + 5
        last_players = {key: value[:] for key, value in current_players.items()}
        for player_event in event:
            name, time, event_type, team_name = player_event
            k = 'home' if team_name == home_team_name else 'away'
            if event_type == 'IN':
                last_players[k].append(name)
            else:
                last_players[k].remove(name)
        last_players_copy = last_players.copy()
        player_events_by_time_parsed.append({'seconds_elapsed': time_second,
                                             'home': last_players_copy['home'],
                                             'away': last_players_copy['away']})
        current_players = last_players

    return player_events_by_time_parsed


def get_time_period(seconds_elapsed):
    # Each period is 12 minutes or 720 seconds
    regulation_period_length = 720
    overtime_period_length = 300

    # In regulation time
    if seconds_elapsed <= 2880:
        period = (seconds_elapsed // regulation_period_length) + 1
        remaining_seconds = regulation_period_length - (seconds_elapsed % regulation_period_length)
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        return period, f"{int(minutes)}:{str(int(seconds)).zfill(2)}"
    else:
        period = 5
        seconds_elapsed -= 2880
        period += (seconds_elapsed // overtime_period_length)
        remaining_seconds = overtime_period_length - (seconds_elapsed % overtime_period_length)
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        return period, f"{int(minutes)}:{str(int(seconds)).zfill(2)}"


def add_elapsed_time(df):
    # Add a column for elapsed time in seconds
    for i, row in df.iterrows():
        total_seconds_elapsed = 0
        period_time = row['PCTIMESTRING']
        period_number = row['PERIOD']
        minutes, seconds = period_time.split(':')
        minutes = int(minutes)
        seconds = int(seconds)

        if period_number <= 4:
            total_seconds_elapsed += (period_number - 1) * 720
            period_seconds_elapse = 720 - ((minutes * 60) + seconds)
        if period_number > 4:
            total_seconds_elapsed += (4 * 720)
            total_seconds_elapsed += (period_number - 5) * 300
            period_seconds_elapse = 300 - ((minutes * 60) + seconds)

        total_seconds_elapsed += period_seconds_elapse
        df.at[i, 'SECONDS_ELAPSED'] = total_seconds_elapsed
    return df


class NoMatchingSecondsElapsedError(Exception):
    pass


def calc_lineup_next_diff(current_lineup, next_lineup, pbp, second_mod=0):
    home, away, current_seconds_elapsed = itemgetter('home', 'away', 'seconds_elapsed')(current_lineup)
    _home, _away, next_seconds_elapsed = itemgetter('home', 'away', 'seconds_elapsed')(next_lineup)

    offsets = [0, -1, 1]

    for offset in offsets:
        try:
            current_pbp = pbp[pbp['SECONDS_ELAPSED'] == current_seconds_elapsed + offset]
            current_score_margin = int(current_pbp['SCOREMARGIN'].iloc[0])
            break
        except IndexError:
            continue
    else:
        print('raising exception')
        raise NoMatchingSecondsElapsedError(f"No matching seconds_elapsed found for {next_seconds_elapsed} ± 1")

    for offset in offsets:
        try:
            next_pbp = pbp[pbp['SECONDS_ELAPSED'] == next_seconds_elapsed + offset]
            next_score_margin = int(next_pbp['SCOREMARGIN'].iloc[0])
            break
        except IndexError:
            continue
    else:
        print('raising exception')
        raise NoMatchingSecondsElapsedError(f"No matching seconds_elapsed found for {next_seconds_elapsed} ± 1")

    # get difference between score margins
    score_margin_diff = next_score_margin - current_score_margin
    diff_seconds = next_seconds_elapsed - current_seconds_elapsed
    return {'home': home,
            'away': away,
            'plus_minus': score_margin_diff,
            'subbed_at': current_seconds_elapsed,
            'time_played': diff_seconds}

def get_lineup_point_differential(lineups, pbp):
    lineup_diffs = []

    for i, current_lineup in enumerate(lineups):
        # if next item in lineup exists
        if i + 1 < len(lineups):
            next_lineup = lineups[i + 1]
        else:
            break
        try:
            diff = calc_lineup_next_diff(current_lineup, next_lineup, pbp)
            lineup_diffs.append(diff)
        except NoMatchingSecondsElapsedError:
            print('FAILED')
            continue
    return lineup_diffs


def process_game(game_id):

    # Load parsed file it if exists
    path = f'data/raw/lineup_diffs/{game_id}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f), False
    try:
        pbp = get_game_pbp(game_id)
    except Exception as e:
        print(f'Error getting pbp {game_id}: {e}')
        print(traceback.format_exc())
        return {}, False
    pbp = add_score_margins(pbp)
    pbp = add_elapsed_time(pbp)
    pbp = remove_duplicate_time_rows(pbp)
    game_rotations = load_game_rotation(game_id)

    try:
        player_lineups = parse_lineups(game_rotations)
        lineup_diffs = get_lineup_point_differential(player_lineups, pbp)
    except Exception as e:
        print(f'Error processing game {game_id}: {e}')
        print(traceback.format_exc())
        return {}, False

    # Save pickled lineup diffs to file, ensuring directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(lineup_diffs, f)

    return lineup_diffs, True

def process_last_bulls_game():
    team = get_team('CHI')
    game = get_last_game(team)
    game_id = game['GAME_ID']
    game_id = '0022200552'  # overtime
    lineup_diffs = process_game(game_id)
    print('Go Bulls!')


def process_season(season):
    print(f'{season}...')
    games_df = get_season_games(season)
    # iterate over df rows
    for i, row in games_df.iterrows():
        game_id = row['GAME_ID']
        print(f'{season} - {game_id} - {i}/{len(games_df)}')
        lineup_diffs, do_sleep = process_game(game_id)
        if do_sleep:
            time_sleep = random.uniform(0.9, 1.5)
            # print(f'Sleeping for {time_sleep} seconds...')
            time.sleep(time_sleep)


def process_n_seasons(n=1):
    start_year = 23
    for i in range(n):
        season = f'20{start_year - (i + 1)}-{start_year - i}'
        print(f'Processing season {season}...')
        process_season(season)

def scrape_season_games():
    n_seasons = 10
    max_retries = 10000000
    retry_delay = 30  # seconds

    for retry in range(max_retries):
        try:
            process_n_seasons(n_seasons)
            print('Finished')
            break  # exit the loop if successful
        except Exception as e:
            msg = str(e)
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            if msg != 'cannot unpack non-iterable NoneType object':
                if retry < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # retry_delay += retry_delay  # exponential backoff
                else:
                    print("Max retries exceeded. Exiting.")


def get_player_info(player_id):
    path = f'data/raw/player_info/{player_id}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path), False
    player = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    player.to_parquet(path)
    return player, True


def scrape_players():
    directory = 'data/raw/lineup_diffs'
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            with open(os.path.join(directory, filename), 'rb') as f:
                data.extend(pickle.load(f))
    players = set()
    for sample in data:
        players.update(sample['home'])
        players.update(sample['away'])

    retry_delay = 5  # seconds

    for player_id in tqdm(players):
        try:
            player, do_sleep = get_player_info(player_id)
            if do_sleep:
                time_sleep = random.uniform(0.9, 1.5)
                time.sleep(time_sleep)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            time.sleep(retry_delay)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scrape_players()
    scrape_season_games()



# Year Old (development)
# Year in the NBA (rookies)
# Ignore injuries? Number of years missed?