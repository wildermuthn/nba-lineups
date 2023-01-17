from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, gamerotation, boxscoresummaryv2, boxscoretraditionalv2
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
from nba_api.stats.endpoints import playbyplay

import pandas as pd
pd.set_option('display.max_colwidth',250)
pd.set_option('display.max_rows',250)

from enum import Enum

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

def get_game_rotation(game_id, score_margins, period_margins, box_score):
    dfs = gamerotation.GameRotation(game_id=game_id).get_data_frames()
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
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 7200].empty) # 1st quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 14400].empty) # 2nd quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 21600].empty) # 3rd quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 28800].empty) # 4th quarter
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 36000].empty) # OT
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 43200].empty) # OT2
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 50400].empty) # OT3
    subs_made_at.append(not subs[subs['IN_TIME_REAL'] == 57600].empty) # OT4
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
    for i, _ in enumerate(player_ids):
        time = sub_times[i]
        if time % 7200 == 0 and time <= 28800:
            period_index = int(time / 7200)-1
            if subs_made_at[period_index]:
                score_margin = period_margins[period_index]
                plus_minus.append(score_margin - prev_score_margin)
                prev_score_margin = score_margin
                continue
        elif time > 28800 and (time - 28800) % 3000 == 0:
            period_index = int(3 + ((time - 28800)/3000))
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



    return player_ids, plus_minus, elapsed_time

def get_game_pbp(game_id):
    pbp = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
    box_score = boxscoresummaryv2.BoxScoreSummaryV2(game_id).get_data_frames()[0]
    return pbp, box_score

def get_last_game(team_id):
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id,
                                                  season_nullable=Season.default,
                                                  season_type_nullable=SeasonType.regular)

    df = gamefinder.get_data_frames()[0]
    game = df.iloc[0]
    game_id = game['GAME_ID']
    game_matchup = game['MATCHUP']

    print(f'Searching through {len(df)} game(s) for the game_id of {game_id} where {game_matchup}')
    return game

def get_team_id(abbv):
    nba_teams = teams.get_teams()

    # Select the dictionary for the Pacers, which contains their team ID
    selected_team = [team for team in nba_teams if team['abbreviation'] == abbv][0]
    team_id = selected_team['id']
    print(f'{selected_team["full_name"]}: {team_id}')
    return selected_team

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

def remove_duplicate_rows(df):
    # remove rows with duplicate PCTIMESTRING values
    df = df.drop_duplicates(subset=['PCTIMESTRING'])
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    team = get_team_id('CHI')
    game = get_last_game(team['id'])
    # game_id = game['GAME_ID']
    game_id = '0022200552'
    pbp, box_score = get_game_pbp(game_id)
    margins = add_score_margins(pbp)
    subs = pbp[(pbp['EVENTMSGTYPE'] == EventMsgType.SUBSTITUTION.value)]
    periods = pbp[(pbp['EVENTMSGTYPE'] == EventMsgType.PERIOD_END.value)]
    # remove duplicate rows
    subs = remove_duplicate_rows(subs)
    # get list of score margins from subs
    score_margins = subs['SCOREMARGIN'].to_list()
    period_margins = periods['SCOREMARGIN'].to_list()
    # convert score margins to ints
    score_margins = [int(x) for x in score_margins]
    period_margins = [int(x) for x in period_margins]
    player_ids, plus_minus, elapsed_time = get_game_rotation(game_id, score_margins, period_margins, box_score)
    print('finished')


