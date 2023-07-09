# from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, gamerotation, boxscoresummaryv2, boxscoretraditionalv2
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
    path_home = os.path.join(os.getcwd(), 'data', 'game_rotations', game_id + '_home.parquet')
    path_away = os.path.join(os.getcwd(), 'data', 'game_rotations', game_id + '_away.parquet')
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

def parse_game_rotations(dfs, score_margins, period_margins, box_score):
    home_team_id = box_score['HOME_TEAM_ID'][0]
    # TODO use home and away team, since the margins are for the home team
    a_df, b_df = dfs
    a_tmp_id = a_df.iloc[0]['TEAM_ID']
    if a_tmp_id != home_team_id:
        tmp_df = a_df
        a_df = b_df
        b_df = tmp_df

    a_id = a_df.iloc[0]['TEAM_ID']
    b_id = b_df.iloc[0]['TEAM_ID']
    a_starters = a_df[a_df['IN_TIME_REAL'] == 0.0]['PLAYER_LAST'].to_list()
    b_starters = b_df[b_df['IN_TIME_REAL'] == 0.0]['PLAYER_LAST'].to_list()

    subs = pd.concat([a_df, b_df])
    subs = subs.sort_values(by=['OUT_TIME_REAL', 'IN_TIME_REAL'], ascending=[True, False])

    # Iterate through subs, calculating the people on the floor, the plus-minus, and the elapsed time
    player_ids = [[a_starters, b_starters]]
    plus_minus = []
    elapsed_time = []
    i = 0

    subs_made_at = []
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 7200].empty)  # 1st quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 14400].empty)  # 2nd quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 21600].empty)  # 3rd quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 28800].empty)  # 4th quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 36000].empty)  # OT
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 43200].empty)  # OT2
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 50400].empty)  # OT3
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 57600].empty)  # OT4
    period_end = int(box_score['LIVE_PERIOD'])
    subs_made_at[period_end - 1] = True

    prev_out_time = 0
    sub_times = []
    subs_dedup = subs.drop_duplicates(subset='OUT_TIME_REAL', keep="first")
    for _, row in subs.iterrows():
        out_time = row['OUT_TIME_REAL']
        if prev_out_time == out_time:
            continue
        sub_times.append(out_time)
        if i == 0:
            elapsed_time.append(out_time)
        else:
            elapsed_time.append(out_time - prev_out_time)
        prev_out_time = out_time

        a_subbed_out = subs[(subs['OUT_TIME_REAL'] == out_time) & (subs['TEAM_ID'] == a_id)]['PLAYER_LAST'].to_list()
        a_subbed_in = subs[(subs['IN_TIME_REAL'] == out_time) & (subs['TEAM_ID'] == a_id)]['PLAYER_LAST'].to_list()
        b_subbed_in = subs[(subs['IN_TIME_REAL'] == out_time) & (subs['TEAM_ID'] == b_id)]['PLAYER_LAST'].to_list()
        b_subbed_out = subs[(subs['OUT_TIME_REAL'] == out_time) & (subs['TEAM_ID'] == b_id)]['PLAYER_LAST'].to_list()

        a_player_ids, b_player_ids = player_ids[i].copy()
        new_a_player_ids = [x for x in a_player_ids if x not in a_subbed_out]
        new_a_player_ids.extend(a_subbed_in)
        new_b_player_ids = [x for x in b_player_ids if x not in b_subbed_out]
        new_b_player_ids.extend(b_subbed_in)
        # append if both lists are not empty
        if new_a_player_ids and new_b_player_ids:
            player_ids.append([new_a_player_ids, new_b_player_ids])
        i += 1

    prev_score_margin = 0
    for i, _ in enumerate(sub_times):
        time = sub_times[i]
        if time % 7200 == 0 and time <= 28800:
            period_index = int(time / 7200) - 1
            if subs_made_at[period_index]:
                score_margin = period_margins[period_index]
                plus_minus.append(score_margin - prev_score_margin)
                prev_score_margin = score_margin
                continue
        elif time > 28800 and (time - 28800) % 3000 == 0:
            period_index = int(3 + ((time - 28800) / 3000))
            if subs_made_at[period_index]:
                score_margin = period_margins[period_index]
                plus_minus.append(score_margin - prev_score_margin)
                prev_score_margin = score_margin
                continue
        else:
            score_margin = score_margins.pop(0)
            if i == 0:
                plus_minus.append(score_margin)
            else:
                plus_minus.append(score_margin - prev_score_margin)
            prev_score_margin = score_margin

    return zip(player_ids, plus_minus, elapsed_time)


def get_game_pbp(game_id):
    path = 'data/pbp/' + str(game_id) + '.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        pbp_response = playbyplay.PlayByPlay(game_id)
        pbp = pbp_response.get_data_frames()[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pbp.to_parquet(path, index=False)
    return pbp


def get_game_box_score(game_id):
    path = 'data/box_score/' + str(game_id) + '.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    box_score = boxscoresummaryv2.BoxScoreSummaryV2(game_id).get_data_frames()[0]
    # Save box score to parquet
    os.makedirs(os.path.dirname(path), exist_ok=True)
    box_score.to_parquet(path, index=False)
    return box_score


def get_season_games(season=Season.default):
    # Check to see if df is already saved (parquet)
    path = 'data/season_' + season + '_games.parquet'
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
    if os.path.exists('data/nba_teams.parquet'):
        return pd.read_parquet('data/nba_teams.parquet')
    nba_teams = teams.get_teams()
    # Save to load later (using parquet), checking if the directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    pd.DataFrame(nba_teams).to_parquet('data/nba_teams.parquet')
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


def get_lineup_point_differential(lineups, pbp):
    lineup_diffs = []

    for i, current_lineup in enumerate(lineups):
        # if next item in lineup exists
        try:
            if i + 1 < len(lineups):
                next_lineup = lineups[i + 1]
            else:
                break
            home, away, current_seconds_elapsed = itemgetter('home', 'away', 'seconds_elapsed')(current_lineup)
            _home, _away, next_seconds_elapsed = itemgetter('home', 'away', 'seconds_elapsed')(next_lineup)
            # find pbp rows for seconds_elapsed and next_seconds_elapsed
            current_pbp = pbp[pbp['SECONDS_ELAPSED'] == current_seconds_elapsed]
            next_pbp = pbp[pbp['SECONDS_ELAPSED'] == next_seconds_elapsed]
            # get SCOREMARGIN for each pbp row
            current_score_margin = int(current_pbp['SCOREMARGIN'].iloc[0])
            next_score_margin = int(next_pbp['SCOREMARGIN'].iloc[0])
            # get difference between score margins
            score_margin_diff = next_score_margin - current_score_margin
            diff_seconds = next_seconds_elapsed - current_seconds_elapsed
            lineup_diffs.append({'home': home,
                                 'away': away,
                                 'plus_minus': score_margin_diff,
                                 'subbed_at': current_seconds_elapsed,
                                 'time_played': diff_seconds})
        except IndexError:
            print('-------------------')
            print(current_lineup)
            print(next_lineup)
            continue
    return lineup_diffs


def process_game(game_id):

    # Load parsed file it if exists
    path = f'data/lineup_diffs/{game_id}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    pbp = get_game_pbp(game_id)
    pbp = add_score_margins(pbp)
    pbp = add_elapsed_time(pbp)
    pbp = remove_duplicate_time_rows(pbp)
    game_rotations = load_game_rotation(game_id)
    player_lineups = parse_lineups(game_rotations)
    lineup_diffs = get_lineup_point_differential(player_lineups, pbp)

    # Save pickled lineup diffs to file, ensuring directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(lineup_diffs, f)

    return lineup_diffs

def process_last_bulls_game():
    team = get_team('CHI')
    game = get_last_game(team)
    game_id = game['GAME_ID']
    game_id = '0022200552'  # overtime
    lineup_diffs = process_game(game_id)
    print('Go Bulls!')

def process_last_season():
    games_df = get_season_games()
    # iterate over df rows
    for i, row in games_df.iterrows():
        game_id = row['GAME_ID']
        print(f'Processing game {game_id}...')
        lineup_diffs = process_game(game_id)
        time.sleep(5)

def main():
    process_last_season()
    print('finished')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
