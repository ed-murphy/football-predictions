
def create_qb_features(team_games, plays):
    """
    Use play-level data to add QB EPA to game-level data
    """

    pass_plays = (
        plays[plays['passer_player_name'].notnull()]
        .sort_values(['game_id', 'posteam', 'play_id'])
    )

    starting_qbs = (
        pass_plays.groupby(['game_id', 'posteam'])['passer_player_name']
        .first()
        .reset_index()
        .rename(columns={'posteam': 'team', 'passer_player_name': 'starting_qb'})
    )

    all_qbs = (
        pass_plays.groupby(['game_id', 'posteam'])['passer_player_name']
        .unique()
        .reset_index()
        .explode('passer_player_name')
        .rename(columns={'posteam': 'team', 'passer_player_name': 'qb_name'})
    )

    qb_epa = (
        plays.loc[
            (plays['passer_player_name'].notna()) |  # passes
            (plays['qb_dropback'] == 1) |           # dropbacks (scrambles/sacks)
            ((plays['rusher_player_name'].notna()) & 
            (plays['rusher_player_name'].isin(all_qbs['qb_name'])))  # non-dropback QB runs
        ]
        .assign(qb=lambda df: df['passer_player_name'].fillna(df['rusher_player_name']))
        .groupby(['game_id', 'posteam', 'qb'])['epa']
        .mean()
        .reset_index()
        .rename(columns={'posteam': 'team', 'qb': 'qb_name', 'epa': 'qb_avg_epa'})
    )

    qb_epa = qb_epa.sort_values(['qb_name', 'game_id']).reset_index(drop=True)

    qb_epa['qb_avg_epa_rolling'] = (
        qb_epa.groupby('qb_name')['qb_avg_epa']
        .apply(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    team_games = team_games.merge(starting_qbs, on=['game_id', 'team'], how='left')

    team_games = team_games.merge(
        qb_epa[['game_id', 'team', 'qb_name', 'qb_avg_epa', 'qb_avg_epa_rolling']],
        left_on=['game_id', 'team', 'starting_qb'],
        right_on=['game_id', 'team', 'qb_name'],
        how='left'
    ).drop(columns=['qb_name'])

    home_qb_features = (
        team_games.loc[team_games['is_home'] == 1, 
                    ['game_id', 'qb_avg_epa', 'qb_avg_epa_rolling', 'starting_qb']]
        .rename(columns={
            'qb_avg_epa': 'home_qb_avg_epa',
            'qb_avg_epa_rolling': 'home_rolling_avg_qb_epa',
            'starting_qb': 'home_starting_qb'
        })
    )

    away_qb_features = (
        team_games.loc[team_games['is_home'] == 0, 
                    ['game_id', 'qb_avg_epa', 'qb_avg_epa_rolling', 'starting_qb']]
        .rename(columns={
            'qb_avg_epa': 'away_qb_avg_epa',
            'qb_avg_epa_rolling': 'away_rolling_avg_qb_epa',
            'starting_qb': 'away_starting_qb'
        })
    )

    team_games = team_games.merge(home_qb_features, on='game_id', how='left')
    team_games = team_games.merge(away_qb_features, on='game_id', how='left')

    print("QB EPA features created.")

    return team_games
