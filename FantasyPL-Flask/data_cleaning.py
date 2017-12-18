# # Imports
# import pandas as pd
# from matplotlib import pyplot as plt
# import numpy as np
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.cross_validation import cross_val_score, ShuffleSplit
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.model_selection import train_test_split
#
#
#
# # Read in player data and team ranks
# fantasy_data_file = './resources/player_data.csv'
# league_ranks_file = './resources/team_ranks.csv'
# league_ranks = pd.read_csv(league_ranks_file)
# fantasy_data = pd.read_csv(fantasy_data_file)
#
# # Only take players that have played over 0 minutes in each game, and separate into positions for models
# Reliable_players = fantasy_data.loc[fantasy_data['minutes'] > 0]
# Goalkeepers = Reliable_players.loc[Reliable_players['pos'] == 'Goalkeeper']
# Defenders = Reliable_players.loc[Reliable_players['pos'] == 'Defender']
# Midfielders = Reliable_players.loc[Reliable_players['pos'] == 'Midfielder']
# Forwards = Reliable_players.loc[Reliable_players['pos'] == 'Forward']
#
# # Rename and drop unwanted Columns
# league_ranks.rename(columns={'More': 'round'}, inplace=True)
# league_ranks.rename(columns={'Club': 'team'}, inplace=True)
# league_ranks.drop(['Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points'], axis=1, inplace=True)
#
# # Position values show previous position, get rid of this and keep the original position
# league_ranks['Position'] = league_ranks['Position'].str[0:2]
#
# # Give the league ranks a round value, this is the gameweek that each ranking belongs to
# x = 1
# for i in range(0, 760, 20):
#     j = i + 20
#     league_ranks.iloc[i:j, league_ranks.columns.get_loc('round')] = x
#     x = x + 1
#
# # Merge the two DataFrames so that we have individual player data with opponent teams position in the table
# DefenderModal = pd.merge(Defenders, league_ranks, how='left', left_on = ['round','team'], right_on = ['round','team'])
#
# DefenderModal.drop(['Unnamed: 0', 'saves','ict_index','big_chances_created',
#                     'selected','transfers_in','transfers_out'], axis=1, inplace=True)
#
# DefenderModal.rename(columns={'Position': 'team_rank'}, inplace=True)
# league_ranks.rename(columns={'team': 'opponents'}, inplace=True)
#
# DefenderModal = pd.merge(DefenderModal, league_ranks, how='left', left_on = ['round','opponents'], right_on = ['round','opponents'])
#
# DefenderModal.rename(columns={'Position': 'opponent_team_rank'}, inplace=True)
#
# DefenderModal = DefenderModal[['player_id', 'name', 'team', 'pos', 'round', 'opponents', 'venue',
#        'team_goals', 'opposition_goals', 'minutes',
#        'goals_scored', 'assists', 'clean_sheets', 'bonus', 'value',
#        'team_rank', 'opponent_team_rank', 'total_points']]
#
# # DefenderModal.to_csv('./resources/DefenderModal.csv', sep=',', encoding='utf-8')
#
# DefenderModal.drop(['value'], axis=1, inplace=True)
# DefenderModal.drop(['pos'], axis=1, inplace=True)
# DefenderModal.drop(['team'], axis=1, inplace=True)
# DefenderModal.drop(['name'], axis=1, inplace=True)
# DefenderModal.drop(['opponents'], axis=1, inplace=True)
#
# DefenderModal.columns = ['player_id', 'round', 'home',
#        'team_goals', 'opposition_goals', 'minutes', 'goals', 'assists',
#        'clean_sheets', 'bonus', 'team_rank', 'opponent_team_rank',
#        'total_points']
#
# DefenderModal[['round','team_rank', 'opponent_team_rank']] = \
#     DefenderModal[['round','team_rank', 'opponent_team_rank']].apply(pd.to_numeric)
#
# home_away = {'H': True, 'A': False}
# DefenderModal['home'] = DefenderModal['home'].map(home_away)
#
# DefenderModal.rename(columns={'total_points': 'prediction_points'}, inplace=True)
# for index, row in DefenderModal.iterrows():
#     if DefenderModal.loc[index, "prediction_points"] < 6:
#         DefenderModal.loc[index, "prediction_points"] = False
#     else:
#         DefenderModal.loc[index, "prediction_points"] = True
#
# # UNIVARIATE SELECTION
# def univariate_selection():
#     array = DefenderModal.values
#     X = array[:, 0:12]
#     Y = array[:, 12]
#
#     test = SelectKBest(score_func=chi2, k=8)
#     fit = test.fit(X, Y.astype(int))
#
#     np.set_printoptions(precision=3)
#     print(fit.scores_)
#
# def model_based_ranking():
#     array = DefenderModal.values
#     X = array[:, 0:12]
#     Y = array[:, 12]
#     names = DefenderModal.columns
#
#     rf = RandomForestRegressor(n_estimators=20, max_depth=4)
#     scores = []
#     for i in range(X.shape[1]):
#         score = cross_val_score(rf, X[:, i:i + 1], Y, scoring="r2",
#                                 cv=ShuffleSplit(len(X), 3, .3))
#         scores.append((round(np.mean(score), 3), names[i]))
#
#     print(sorted(scores, reverse=True))
#
# def recursive_feature_elimination():
#     array = DefenderModal.values
#     X = array[:,0:12]
#     Y = array[:,12]
#     # feature extraction
#     model = LinearRegression()
#     rfe = RFE(model, 9)
#     fit = rfe.fit(X, Y)
#     names = DefenderModal.columns
#
#     print('Features sorted by their rank:')
#     print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
#
# print("RECURSIVE FEATURE ELIMINATION:")
# recursive_feature_elimination()
#
# def PCA():
#     array = DefenderModal.values
#     X = array[:, 0:12]
#     Y = array[:, 12]
#
#     reg = LinearRegression()
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
#     reg.fit(x_train, y_train)
#     reg.score(x_test, y_test)
#
# def SVD():
#     array = DefenderModal.values
#     X = array[:, 0:12]
#     Y = array[:, 12]
#     svd = TruncatedSVD(n_components=8)
#     x = svd.fit(X).transform(X)
#     reg = LinearRegression()
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
#     reg.fit(x_train, y_train)
#
# # Load dataframe value into array for best feature selection
# # array = DefenderModal.values
# # X = array[:, 0:12]
# # Y = array[:, 12]
# #
# # test = SelectKBest(score_func=chi2, k=10)
# # fit = test.fit(X, Y.astype(int))
# #
# # np.set_printoptions(precision=3)

# Imports
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
def clean_defender_data():
    # Read in player data and team ranks
    fantasy_data_file = './resources/player_data.csv'
    league_ranks_file = './resources/team_ranks.csv'
    league_ranks = pd.read_csv(league_ranks_file)
    fantasy_data = pd.read_csv(fantasy_data_file)

    # Only take players that have played over 0 minutes in each game, and separate into positions for models
    Reliable_players = fantasy_data.loc[fantasy_data['minutes'] > 0]
    # Goalkeepers = Reliable_players.loc[Reliable_players['pos'] == 'Goalkeeper']

    # For the moment we'll only look at Defender data for the prototype
    Defenders = Reliable_players.loc[Reliable_players['pos'] == 'Defender']
    # Midfielders = Reliable_players.loc[Reliable_players['pos'] == 'Midfielder']
    # Forwards = Reliable_players.loc[Reliable_players['pos'] == 'Forward']

    # Rename and drop unwanted Columns
    league_ranks.rename(columns={'More': 'round'}, inplace=True)
    league_ranks.rename(columns={'Club': 'team'}, inplace=True)
    league_ranks.drop(['Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points'], axis=1, inplace=True)

    # Position values show previous position, get rid of this and keep the original position
    league_ranks['Position'] = league_ranks['Position'].str[0:2]

    # Give the league ranks a round value, this is the gameweek that each ranking belongs to
    x = 1
    for i in range(0, 760, 20):
        j = i + 20
        league_ranks.iloc[i:j, league_ranks.columns.get_loc('round')] = x
        x = x + 1

    # Merge the two DataFrames so that we have individual player data with opponent teams position in the table
    DefenderModal = pd.merge(Defenders, league_ranks, how='left', left_on = ['round','team'], right_on = ['round','team'])

    DefenderModal.drop(['Unnamed: 0', 'saves','big_chances_created',
                        'selected','transfers_in','transfers_out'], axis=1, inplace=True)

    DefenderModal.rename(columns={'Position': 'team_rank'}, inplace=True)
    league_ranks.rename(columns={'team': 'opponents'}, inplace=True)

    DefenderModal = pd.merge(DefenderModal, league_ranks, how='left', left_on = ['round','opponents'], right_on = ['round','opponents'])

    DefenderModal.rename(columns={'Position': 'opponent_team_rank'}, inplace=True)

    DefenderModal = DefenderModal[['player_id', 'name', 'team', 'pos', 'round', 'opponents', 'venue',
           'team_goals', 'opposition_goals', 'minutes',
           'goals_scored', 'assists', 'clean_sheets', 'bonus', 'ict_index', 'value',
           'team_rank', 'opponent_team_rank', 'total_points']]

    # DefenderModal.to_csv('./resources/DefenderModal.csv', sep=',', encoding='utf-8')

    DefenderModal.drop(['value'], axis=1, inplace=True)
    DefenderModal.drop(['pos'], axis=1, inplace=True)
    DefenderModal.drop(['minutes'], axis=1, inplace=True)

    DefenderModal.columns = ['player_id', 'name', 'team', 'round', 'opponents', 'venue',
           'team_goals', 'opposition_goals', 'goals', 'assists',
           'clean_sheets', 'bonus', 'ict_index', 'team_rank', 'opponent_team_rank',
           'total_points']


    DefenderModal.drop(['name'], axis=1, inplace=True)
    DefenderModal.drop(['team'], axis=1, inplace=True)
    DefenderModal.drop(['round'], axis=1, inplace=True)
    DefenderModal.drop(['opponents'], axis=1, inplace=True)

    return DefenderModal.head(5)
