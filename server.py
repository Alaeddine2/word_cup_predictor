from flask import Flask
import pyodbc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from decimal import Decimal
from flask import request, jsonify

app = Flask(__name__)
server = 'DESKTOP-RHOJ5I4'
db1 = 'World_cup'
tcon = 'yes'
uname = 'jnichol3'
pword = '**my-password**'

conn = pyodbc.connect(driver='{SQL Server}', host=server, database=db1,
                      trusted_connection=tcon, user=uname, password=pword)
# check if connection is successful
if conn:
    print("Connection Successful!")
else:
    print("Connection Failed!")
# Fetching the data from the selected table using SQL query
RawData= pd.read_sql_query('''select * from [DBO].[DimMatch]''', conn)
RawData.columns = [c.replace(' ', '_') for c in RawData.columns]
RawData.head()
RawData.dtypes
RawData.sort_values("Year").tail()
RawData.Home_Team_Name.value_counts()
rank= pd.read_sql_query('''select * from [DBO].[DimRank]''', conn)
rank.sort_values("year").tail()
df_wc_ranked = RawData.merge(rank[["country_abrv","country_full", "total_points", "previous_points", "rank", "rank_change", "year"]], left_on=["Year", "Home_Team_Initials"], right_on=["year", "country_abrv"]).drop(["year", "country_abrv"], axis=1)
df_wc_ranked = df_wc_ranked.merge(rank[["country_abrv","country_full", "total_points", "previous_points", "rank", "rank_change", "year"]], left_on=["Year", "Away_Team_Initials"], right_on=["year", "country_abrv"], suffixes=("_home", "_away")).drop(["year", "country_abrv"], axis=1)
df_wc_ranked[(df_wc_ranked.Home_Team_Name == "Brazil") | (df_wc_ranked.Away_Team_Name == "Brazil")].tail(10)
#print(df_wc_ranked[(df_wc_ranked.Home_Team_Name == "Brazil") | (df_wc_ranked.Away_Team_Name == "Brazil")].tail(10))
df = df_wc_ranked
def result_finder(home, away):
    if home > away:
        return pd.Series([0, 3, 0])
    if home < away:
        return pd.Series([1, 0, 3])
    else:
        return pd.Series([2, 1, 1])
results = df.apply(lambda x: result_finder(x["Home_Team_Goals"], x["Away_Team_Goals"]), axis=1)
df[["result", "home_team_points", "away_team_points"]] = results
#print(df[["total_points_home", "rank_home", "total_points_away", "rank_away"]])
df["rank_dif"] = df["rank_home"].astype(int) - df["rank_away"].astype(int)
df["sg"] = df["Home_Team_Goals"].astype(int) - df["Away_Team_Goals"].astype(int)
df["points_home_by_rank"] = df["home_team_points"]/df["rank_away"].astype(int)
df["points_away_by_rank"] = df["away_team_points"]/df["rank_home"].astype(int)
#print(df[["total_points_home", "rank_home", "total_points_away", "rank_away", "sg", "rank_dif", "points_home_by_rank", "away_team_points"]])
home_team = df[["Year", "Home_Team_Name", "Home_Team_Goals", "Away_Team_Goals", "rank_home", "rank_away","rank_change_home", "total_points_home", "result", "rank_dif", "points_home_by_rank", "home_team_points"]]

away_team = df[["Year", "Away_Team_Name", "Away_Team_Goals", "Home_Team_Goals", "rank_away", "rank_home","rank_change_away", "total_points_away", "result", "rank_dif", "points_away_by_rank", "away_team_points"]]
home_team.columns = [h.replace("Home_", "").replace("home_", "").replace("_home", "").replace("Away_", "suf_").replace("_away", "_suf") for h in home_team.columns]

away_team.columns = [a.replace("Away_", "").replace("away_", "").replace("_away", "").replace("Home_", "suf_").replace("_home", "_suf") for a in away_team.columns]
team_stats = team_stats = pd.concat([home_team, away_team])
team_stats_raw = team_stats.copy()

stats_val = []

for index, row in team_stats.iterrows():
    team = row["Team_Name"]
    date = row["Year"]
    past_games = team_stats.loc[(team_stats["Team_Name"] == team) & (team_stats["Year"] < date)].sort_values(by=['Year'], ascending=False)
    last5 = past_games.head(5)
    
    goals = past_games["Team_Goals"].mean()
    goals_l5 = last5["Team_Goals"].mean()
    
    goals_suf = past_games["suf_Team_Goals"].mean()
    goals_suf_l5 = last5["suf_Team_Goals"].mean()
    
    rank = past_games["rank_suf"].mean()
    rank_l5 = last5["rank_suf"].mean()
    if len(last5) > 0:
        points = pd.to_numeric(past_games["total_points"].values[0]) - pd.to_numeric(past_games["total_points"].values[-1])#qtd de pontos ganhos
        points_l5 = pd.to_numeric(last5["total_points"].values[0]) - pd.to_numeric(last5["total_points"].values[-1])
    else:
        points = 0
        points_l5 = 0
        
    gp = past_games["team_points"].mean()
    gp_l5 = last5["team_points"].mean()
    
    gp_rank = past_games["points_by_rank"].mean()
    gp_rank_l5 = last5["points_by_rank"].mean()
    
    stats_val.append([goals, goals_l5, goals_suf, goals_suf_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank, gp_rank_l5])
#print(stats_val)  
stats_cols = ["goals_mean", "goals_mean_l5", "goals_suf_mean", "goals_suf_mean_l5", "rank_mean", "rank_mean_l5", "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5", "game_points_rank_mean", "game_points_rank_mean_l5"]
stats_df = pd.DataFrame(stats_val, columns=stats_cols)
full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)
home_team_stats = full_df.iloc[:int(full_df.shape[0]/2),:]
away_team_stats = full_df.iloc[int(full_df.shape[0]/2):,:]
home_team_stats.columns[-12:]
home_team_stats = home_team_stats[home_team_stats.columns[-12:]]
away_team_stats = away_team_stats[away_team_stats.columns[-12:]]
home_team_stats.columns = ['home_'+str(col) for col in home_team_stats.columns]
away_team_stats.columns = ['away_'+str(col) for col in away_team_stats.columns]
match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)
full_df = pd.concat([df, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)
full_df.columns
full_df["is_friendly"] = 0
full_df = pd.get_dummies(full_df, columns=["is_friendly"])
full_df.columns
base_df = full_df[["Year", "Home_Team_Name", "Away_Team_Name", "rank_home", "rank_away","Home_Team_Goals", "Away_Team_Goals","result", "rank_dif", "rank_change_home", "rank_change_away", 'home_goals_mean',
       'home_goals_mean_l5', 'home_goals_suf_mean', 'home_goals_suf_mean_l5',
       'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean',
       'home_points_mean_l5', 'away_goals_mean', 'away_goals_mean_l5',
       'away_goals_suf_mean', 'away_goals_suf_mean_l5', 'away_rank_mean',
       'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5','home_game_points_mean', 'home_game_points_mean_l5',
       'home_game_points_rank_mean', 'home_game_points_rank_mean_l5','away_game_points_mean',
       'away_game_points_mean_l5', 'away_game_points_rank_mean',
       'away_game_points_rank_mean_l5',
       'is_friendly_0']]
base_df.isna().sum()
base_df_no_fg = base_df.dropna()
df = base_df_no_fg
def no_draw(x):
    if x == 2:
        return 1
    else:
        return x
    
df["target"] = df["result"].apply(lambda x: no_draw(x))
def create_db(df):
    columns = ["Home_Team_Name", "Away_Team_Name", "target", "rank_dif", "home_goals_mean", "home_rank_mean", "away_goals_mean", "away_rank_mean", "home_rank_mean_l5", "away_rank_mean_l5", "home_goals_suf_mean", "away_goals_suf_mean", "home_goals_mean_l5", "away_goals_mean_l5", "home_goals_suf_mean_l5", "away_goals_suf_mean_l5", "home_game_points_rank_mean", "home_game_points_rank_mean_l5", "away_game_points_rank_mean", "away_game_points_rank_mean_l5","is_friendly_0"]
    
    base = df.loc[:, columns]
    base.loc[:, "goals_dif"] = base["home_goals_mean"] - base["away_goals_mean"]
    base.loc[:, "goals_dif_l5"] = base["home_goals_mean_l5"] - base["away_goals_mean_l5"]
    base.loc[:, "goals_suf_dif"] = base["home_goals_suf_mean"] - base["away_goals_suf_mean"]
    base.loc[:, "goals_suf_dif_l5"] = base["home_goals_suf_mean_l5"] - base["away_goals_suf_mean_l5"]
    base.loc[:, "goals_per_ranking_dif"] = (base["home_goals_mean"] / base["home_rank_mean"]) - (base["away_goals_mean"] / base["away_rank_mean"])
    base.loc[:, "dif_rank_agst"] = base["home_rank_mean"] - base["away_rank_mean"]
    base.loc[:, "dif_rank_agst_l5"] = base["home_rank_mean_l5"] - base["away_rank_mean_l5"]
    base.loc[:, "dif_points_rank"] = base["home_game_points_rank_mean"] - base["away_game_points_rank_mean"]
    base.loc[:, "dif_points_rank_l5"] = base["home_game_points_rank_mean_l5"] - base["away_game_points_rank_mean_l5"]
    
    model_df = base[["Home_Team_Name", "Away_Team_Name", "target", "rank_dif", "goals_dif", "goals_dif_l5", "goals_suf_dif", "goals_suf_dif_l5", "goals_per_ranking_dif", "dif_rank_agst", "dif_rank_agst_l5", "dif_points_rank", "dif_points_rank_l5", "is_friendly_0"]]
    return model_df
# Creating the pridection database : 
model_db = create_db(df)
model_db
# Starting the ML modelisation
X = model_db.iloc[:, 3:]
y = model_db[["target"]]
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the GradientBoostingClassifier and the parameters for grid search
gb = GradientBoostingClassifier(random_state=5)
params = {
    "learning_rate": [0.01, 0.1, 0.5],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
    "max_depth": [3, 5, 10],
    "max_features": ["sqrt"],
    "n_estimators": [100, 200]
}

# Perform grid search with cross-validation
gb_cv = GridSearchCV(gb, params, cv=3, n_jobs=-1, verbose=False)
#gb_cv.fit(X_train.values, np.ravel(y_train))
gb_cv.fit(X_train_scaled, np.ravel(y_train))
gb = gb_cv.best_estimator_
gb

def find_stats(team_1):
    past_games = team_stats_raw[(team_stats_raw["Team_Name"] == team_1)].sort_values("Year")
    last5 = team_stats_raw[(team_stats_raw["Team_Name"] == team_1)].sort_values("Year").tail(5)
    #print(past_games)
    team_1_rank = past_games["rank"].values[-1]
    team_1_goals = past_games.Team_Goals.mean()
    team_1_goals_l5 = last5.Team_Goals.mean()
    team_1_goals_suf = past_games.suf_Team_Goals.mean()
    team_1_goals_suf_l5 = last5.suf_Team_Goals.mean()
    team_1_rank_suf = past_games.rank_suf.mean()
    team_1_rank_suf_l5 = last5.rank_suf.mean()
    team_1_gp_rank = past_games.points_by_rank.mean()
    team_1_gp_rank_l5 = last5.points_by_rank.mean()

    return [team_1_rank, team_1_goals, team_1_goals_l5, team_1_goals_suf, team_1_goals_suf_l5, team_1_rank_suf, team_1_rank_suf_l5, team_1_gp_rank, team_1_gp_rank_l5]

def find_features(team_1, team_2):
    
    rank_dif = pd.to_numeric(team_1[0]) - pd.to_numeric(team_2[0])
    goals_dif = pd.to_numeric(team_1[1]) - pd.to_numeric(team_2[1])
    goals_dif_l5 = pd.to_numeric(team_1[2]) - pd.to_numeric(team_2[2])
    goals_suf_dif = pd.to_numeric(team_1[3]) - pd.to_numeric(team_2[3])
    goals_suf_dif_l5 = pd.to_numeric(team_1[4]) - pd.to_numeric(team_2[4])
    goals_per_ranking_dif = (pd.to_numeric(team_1[1])/pd.to_numeric(team_1[5])) - (pd.to_numeric(team_2[1])/pd.to_numeric(team_2[5]))
    dif_rank_agst = pd.to_numeric(team_1[5]) - pd.to_numeric(team_2[5])
    dif_rank_agst_l5 = pd.to_numeric(team_1[6]) - pd.to_numeric(team_2[6])
    dif_gp_rank = pd.to_numeric(team_1[7]) - pd.to_numeric(team_2[7])
    dif_gp_rank_l5 = pd.to_numeric(team_1[8]) - pd.to_numeric(team_2[8])

    return [rank_dif, goals_dif, goals_dif_l5, goals_suf_dif, goals_suf_dif_l5, goals_per_ranking_dif, dif_rank_agst, dif_rank_agst_l5, dif_gp_rank, dif_gp_rank_l5, 1, 0]

def estimate_goal_numbers(team_1_prob, team_2_prob):
    rounded_prob1 = round(team_1_prob, 2)
    rounded_prob2 = round(team_2_prob, 2)
    # Estimate number of goals based on probabilities
    team_1_goals = round(team_1_prob * 3)  # Assuming maximum of 3 goals
    team_2_goals = round(team_2_prob * 3)  # Assuming maximum of 3 goals
    if(rounded_prob1 == rounded_prob2):
        return [team_1_goals, team_1_goals]
    if(team_1_prob > 0.6):
        team_1_goals = team_1_goals + 1
    if(team_1_prob > 0.70):
        team_1_goals = team_1_goals + 1
    if(team_2_prob > 0.60):
        team_2_goals = team_2_goals + 1
    if(team_2_prob > 0.70):
        team_2_goals = team_2_goals + 1
    if(team_1_prob < 0.3):
        team_1_goals = 0
    if(team_2_prob < 0.3):
        team_2_goals = 0
    return [team_1_goals,team_2_goals]


def predict_match(team1, team2):
    teamSelected1 = team1  # Argentina | Germany | Russia "Tunisia"
    teamSelected2 = team2 #Bolivia | Belgium | Japan
    team_1 = find_stats(teamSelected1)
    team_2 = find_stats(teamSelected2)
    features_g1 = find_features(team_1, team_2)
    features_g2 = find_features(team_2, team_1)
    features_g1 = [str(element)[:-5] if 'e' in str(element) else str(element) for element in features_g1]
    features_g2 = [str(element)[:-5] if 'e' in str(element) else str(element) for element in features_g2]
    features_g1 = np.array(features_g1[:-1], dtype=np.float32)  # Remove the last element
    features_g2 = np.array(features_g2[:-1], dtype=np.float32)  # Remove the last element


    probs_g1 = gb.predict_proba([features_g1])
    probs_g2 = gb.predict_proba([features_g2])

    team_1_prob_g1 = probs_g1[0][0]
    team_1_prob_g2 = probs_g2[0][1]
    team_2_prob_g1 = probs_g1[0][1]
    team_2_prob_g2 = probs_g2[0][0]

    team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2
    team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2

    rounded_prob1 = round(team_1_prob, 2)
    rounded_prob2 =round(team_2_prob, 2)

    rounded_prob1 = round(team_1_prob, 2)
    rounded_prob2 = round(team_2_prob, 2)

    goals = estimate_goal_numbers(team_1_prob, team_2_prob)


    if rounded_prob1 == rounded_prob2:
        print("It's a draw with a probability of %.2f" % team_1_prob)
    elif team_1_prob > team_2_prob:
        print("Team: %s wins with a probability of %.2f" % (teamSelected1, team_1_prob))
    else:
        print("Team: %s wins with a probability of %.2f" % (teamSelected2, team_2_prob))
    return "Match Score: %s: %d goals %s: %d goals" % (teamSelected1, goals[0], teamSelected2, goals[1])

# Fetching the data from the selected table using SQL query
RawData= pd.read_sql_query('''select * from [DBO].[Dim_event]''', conn)
RawData.columns = [c.replace(' ', '_') for c in RawData.columns]
RawData.head()
PlayerData= pd.read_sql_query('''select * from [DBO].[DimPlayer]''', conn)
PlayerData.columns = [c.replace(' ', '_') for c in PlayerData.columns]
PlayerData.head()
# Group the data by Player_Name and aggregate the matches and score numbers
grouped_data = RawData.groupby('Player_Name').agg({
    'MatchID': 'nunique',   # Number of unique matches
    'Event': lambda x: x.str.count('G').sum(),   # Count 'G' characters in all events
    'Line-up': lambda x: (x == 'S').sum(),   # Count line-up matches (Line-up field = 'S')
})

# Calculate Yellow_Card_Number: Count all player yellow cards
grouped_data['Yellow_Cards'] = RawData['Event'].str.count('Y').groupby(RawData['Player_Name']).sum()

# Calculate Red_Card_Number: Count all player red cards
grouped_data['Red_Cards'] = RawData['Event'].str.count('RSY|R').groupby(RawData['Player_Name']).sum()

# Reset the index to make Player_Name a column
grouped_data = grouped_data.reset_index()

# Rename the columns
grouped_data.columns = ['Player_Name', 'Played_Matches', 'Score_Numbers', 'Line_Up_Matches', 'Yellow_Cards', 'Red_Cards']

merged_data = pd.merge(PlayerData, grouped_data, left_on='Name', right_on='Player_Name', how='left')
merged_data = merged_data.drop('Player_Name', axis=1)
merged_data = merged_data.drop('PKPlayer', axis=1)
merged_data['Goal_Per_match'] = merged_data['Score_Numbers'] / merged_data['Played_Matches']
# Count the number of players for each number of played matches
matches_count = merged_data['Played_Matches'].value_counts().reset_index()
matches_count.columns = ['Played_Matches', 'Player_Count']
# To make this analysis more realitic we will consider players who have played at least 3 matches (because the others can not be real)
filtered_data = merged_data[merged_data['Played_Matches'] >= 3]
# Count the number of players for each number of played matches
matches_count = filtered_data['Played_Matches'].value_counts().reset_index()
matches_count.columns = ['Played_Matches', 'Player_Count']
filtered_data = filtered_data
clone = filtered_data
clone = clone.set_index("Name")
clone_filtered_data = clone
clone_filtered_data.drop(columns=clone_filtered_data.columns[:3],axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Standard Scaler (mean = 0 and standard deviation = 1)
scaler = StandardScaler()

# fit_transform
players_scaled = pd.DataFrame(scaler.fit_transform(clone_filtered_data),columns=clone_filtered_data.columns)
players_scaled.shape
clone_filtered_data.insert(0, 'name', clone_filtered_data.index)

def  groupby_cluster(cluster_col, head_size):
    g = clone_filtered_data.groupby([cluster_col]).apply(lambda x: x.sort_values(["Played_Matches"], ascending = False)).reset_index(drop=True)
    # select top N rows within each cluster

from sklearn.cluster import KMeans

# Define function to perform the kmeans clustering on the given data
def kmeans_clustering(num_clusters, max_iterations,input_df,output_df, output_col):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations)
    kmeans.fit(input_df)
    # assign the label to the output column
    output_df[output_col] = kmeans.labels_ 
clones = clone_filtered_data.drop(clone_filtered_data.columns[0], axis=1)

ALL_COLUMN_NAMES = list(clones.columns)
print(ALL_COLUMN_NAMES)
# New output column to create for the cluster label
kmeans_label = 'cluster_kmeans'

# K-means clustering
cluster = kmeans_clustering(10,50,players_scaled[ALL_COLUMN_NAMES],clone_filtered_data,kmeans_label)

# View few entries from each cluster
groupby_cluster(kmeans_label,3)

grouper = clone_filtered_data.sort_values(["Score_Numbers"], ascending = False)[['name',kmeans_label]].groupby([kmeans_label])
cluster_df = pd.concat([pd.Series(v['name'].tolist(), name=k) for k, v in grouper], axis=1)
cluster_df.fillna('',inplace=True)
print(cluster_df)

def find_player_cluster(player_name):
    #player_name = "MATA"  # Example player name

    # Find the cluster of the player name
    cluster_name = cluster_df[cluster_df == player_name].dropna(how="all").dropna(axis=1, how="all").columns[0]
    print(cluster_name)
    print(cluster_df[cluster_name])

    # Get the player names in the cluster
    return cluster_df[cluster_name].tolist()[:10]

# find player data:

def find_player_data(player_name):
    player_data = merged_data[merged_data['Name'] == player_name]
    return player_data


def get_players_data():
    return merged_data
    

# APIS 
@app.route('/predict/', methods=['GET', 'POST'])
def welcome():
    if request.method == 'GET':
        return "Welcome to the World Cup 2026 prediction API. To predict a match, please use the following format: /predict/Team1/Team2"
    else:
        FirstTeam = request.form.get('FirstTeam')
        SecondTeam = request.form.get('SecondTeam')
        return {
            "result": "success",
            "code": 200,
            "data": predict_match(FirstTeam, SecondTeam),
        }
        

@app.route('/getlike/', methods=['GET', 'POST'])
def cluster():
    if request.method == 'GET':
        return "Welcome to the World Cup 2026 cluster API."
    else:
        player = request.form.get('Player')
        # get for every player in the list player data
        top_ten_names = find_player_cluster(player)
        top_ten_data = []
        print(top_ten_names)
        for name in top_ten_names:
            print(name)
            top_ten_data.append(find_player_data(name))

        # Convert DataFrames to JSON-serializable format
        top_ten_data_json = [df.to_json() for df in top_ten_data]

        print(top_ten_data_json)
        return {
            "result": "success",
            "code": 200,
            "data": top_ten_data_json,
        }

@app.route('/players/', methods=['GET', 'POST'])
def players():
    if request.method == 'GET':
        merged_data = get_players_data()
        print(merged_data['Name'])
        # Convert DataFrames to JSON-serializable format
        data = merged_data['Name'].tolist()
       # merged_data_json = [df.to_json() for df in merged_data]
        return {
            "result": "success",
            "code": 200,
            "data": data,
        }
    else:
        merged_data_json = [df.to_json() for df in get_players_data()]
        return {
            "result": "success",
            "code": 200,
            "data": merged_data_json,
        }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)