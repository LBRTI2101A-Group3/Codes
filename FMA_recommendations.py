# -*- coding: utf-8 -*-
"""
Created on Nov 2020
Code source : https://medium.com/deep-learning-turkey/build-your-own-spotify-playlist-of-best-playlist-recommendations-fc9ebe92826a
https://github.com/smyrbdr/make-your-own-Spotify-playlist-of-playlist-recommendations/blob/master/Make_Your_Own_Playlist_of_Recs-with_PCA%2Btf-idf%2BDT_on_Blues.ipynb
project by : Juliette Glorieux, Martin Baufayt, Boris Norgaard & Colin Scherpereel
LBRTI2101B - Data Science in bioscience engineering (partim B)
Academic year : 2020 - 2021
"""
#%%

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import numpy as np
import pandas as pd
import random as rd

#%%

############################################
# Retrieving the data from the FMA dataset #
############################################

feat_names = ['acousticness','danceability','energy','instrumentalness',
              'liveness','speechiness','tempo','valence','album_date','album_name',
              'artist_latitude','artist_location','artist_longitude','artist_name',
              'release','artist_discovery_rank','artist_familiarity_rank',
              'artist_hotttnesss_rank','song_currency_rank','song_hotttnesss_rank',
              'artist_discovery','artist_familiarity','artist_hotttnesss']
track_names = ['song_title','listens','album','artist','duration']

tracks_doc = pd.read_csv('tracks.csv').iloc[2:13131,[52,47,11,26,38]]
tracks_doc.columns = track_names
tracks_features = pd.read_csv('echonest.csv').iloc[3:,1:24]
tracks_features.columns = feat_names

listindex = []
for number in range(0,len(tracks_doc)):
    listindex.append(number)
tracks_doc.index = listindex

# Assembling the data
list_artist_song = []
list_artists = list(tracks_doc["artist"])
list_songs = list(tracks_doc["song_title"])
for i in range(len(list(tracks_doc["artist"]))):
    list_artist_song.append((list_artists[i] + " || " + list_songs[i]))
tracks_features.index = list_artist_song
durations = []
for duration in list(tracks_doc["duration"]):
    durations.append(int(duration)*1000)
tracks_features["duration_ms"] = durations

# Converting to floats and keeping only the relevant features
col_title = ["acousticness",
  "danceability", "duration_ms", "energy", "instrumentalness", "liveness",
  "speechiness", "tempo", "valence"]
for title in col_title:
    column = tracks_features[title]
    tracks_features[title] = pd.to_numeric(column)

tracks_features = tracks_features[["acousticness",
  "danceability", "duration_ms", "energy", "instrumentalness", "liveness",
  "speechiness", "tempo", "valence"]]

#%%

####################################
# Getting Spotify's authorizations #
####################################

cid = 'clientID' # Client ID; copy this from your app 
secret = 'clientSecret' # Client Secret; copy this from your app
username = 'Username' # Your Spotify username

#for avaliable scopes see https://developer.spotify.com/web-api/using-scopes/
scope = 'user-library-read playlist-modify-public playlist-read-private'

redirect_uri= 'https://developer.spotify.com/dashboard/applications/3f097271dcb64549b226e9b335e68fc3' # Paste your Redirect URI here

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

#%%

############################################
# Loading a source playlist as a dataframe #
############################################

#Create a dataframe of your playlist including tracks' names and audio features
import pandas as pd

sourcePlaylistID = '76eGMBYZMCC6P26VnNKOFs'
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID);
tracks = sourcePlaylist["tracks"];
songs = tracks["items"];

track_ids = []
track_names = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None: # Removes the local tracks in your playlist if there is any
        track_ids.append(songs[i]['track']['id'])
        track_names.append(songs[i]['track']['name'])

features = []
for i in range(0,len(track_ids)):
    audio_features = sp.audio_features(track_ids[i])
    for track in audio_features:
        features.append(track)
        
playlist_df = pd.DataFrame(features, index = track_names)

#%%

# Selecting only a few useful parameters for future recommandations
playlist_df = playlist_df[["id", "acousticness",
  "danceability", "duration_ms", "energy", "instrumentalness", "liveness",
  "speechiness", "tempo", "valence"]]

#%%

# TF-IDF implementation
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(sublinear_tf = True, ngram_range = (1, 6), max_features = 10000)
X_names_sparse = v.fit_transform(track_names)
X_names_sparse.shape

#%%

# Attributing a rate to each track of the training playlist dataframe
playlist_df['ratings'] = [8,7,7,1,9,8,9,9,2,1,7,7,1,1,2,1,1,1,8,7,6,7,8,7,7,7,
                          8,8,7,6,4,3,6,8,7,8,7,1,1,6,9,8,7,6,2,7,10,10,10,10,
                          10,10,10,10,10,10,10,7,5,3,8,8,6,7,7,8,9,8,9,8,1,2,8,10,4,10]

# Defining the training set
X_train = playlist_df.drop(['id', 'ratings'], axis=1)
y_train = playlist_df['ratings']

#%%

#################################
# Analyzing feature importances #
#################################

from sklearn.ensemble.forest import RandomForestClassifier

X_train = playlist_df.drop(['id', 'ratings'], axis = 1)
y_train = playlist_df['ratings']
forest = RandomForestClassifier(random_state = 42, max_depth = 5, max_features = 9) # Set by GridSearchCV below
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature rankings
print("Feature ranking:")
  
for f in range(len(importances)):
    print("%d. %s %f " % (f + 1, 
            X_train.columns[f], 
            importances[indices[f]]))

#%%

###############################
# PCA for dimension reduction # (Possibility to do it with R for visualization)
###############################

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
import numpy as np

X_scaled = StandardScaler().fit_transform(X_train)
pca = decomposition.PCA().fit(X_scaled)

# Graph to choose the number of PC to keep 95% of the explained variability
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(7, c='b')
plt.axhline(0.95, c='r')
plt.show();

# Fit your dataset to the optimal pca
pca = decomposition.PCA(n_components = 7)
X_pca = pca.fit_transform(X_scaled)

#%%

#################################
# t-SNE for dimension reduction #
#################################

from sklearn.manifold import TSNE
#from bioinfokit.visuz import cluster

tsne = TSNE(random_state = 17)
X_tsne = tsne.fit_transform(X_scaled)
#X_tsne = TSNE(n_components = 2, perplexity = 12, n_iter = 1000, verbose = 1).fit_transform(X_train)
#cluster.tsneplot(score = X_tsne)
#cluster.tsneplot(score = X_tsne)

#%%

########################################
# Preparing the data for further steps #
########################################

from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings('ignore')

# Initialize a stratified split for the validation process
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

# Check with X_tsne + X_names_sparse also
X_train_last = csr_matrix(hstack([X_pca, X_names_sparse]))
X_train_last_2 = csr_matrix(hstack([X_tsne, X_names_sparse]))
X_test_names = v.transform(list(tracks_doc["song_title"]))

#%%

##################
# Decision trees #
##################

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}
tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)
tree_grid.fit(X_train_last, y_train)
print(tree_grid.best_estimator_, tree_grid.best_score_)

#%%

############################
# Random forest classifier #
############################

parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                             n_jobs=-1, oob_score=True)
gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv1.fit(X_train_last, y_train)
print(gcv1.best_estimator_, gcv1.best_score_)

#%%

#######################
# K-nearest neighbors #
#######################

from sklearn.neighbors import KNeighborsClassifier

knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)
knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
print(knn_grid.best_params_, knn_grid.best_score_)

#%%

######################
# Making predictions #
######################

tree_grid.best_estimator_.fit(X_train_last, y_train)
tracks_features_scaled = StandardScaler().fit_transform(tracks_features)
X_test_pca = pca.transform(tracks_features_scaled)
X_test_names = v.transform(list(tracks_doc["song_title"]))
X_test_last = csr_matrix(hstack([X_test_pca, X_test_names]))
y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

tracks_features['ratings'] = y_pred_class
tracks_features = tracks_features.sort_values('ratings', ascending = False)
tracks_features = tracks_features.reset_index()

# Pick the top ranking tracks (rating >= 8 or 9) to add your new playlist
if len(tracks_features[tracks_features['ratings']>=9]) < 10:
    if len(tracks_features[tracks_features['ratings']>=8]) < 10:
        recs_to_add = tracks_features[tracks_features['ratings']>=7]['index'].tolist()
    else:
        recs_to_add = tracks_features[tracks_features['ratings']>=8]['index'].tolist()
else:
    recs_to_add = tracks_features[tracks_features['ratings']>=9]['index'].tolist()

# Check the part of recommended tracks to add
len(list(tracks_doc["song_title"])), tracks_features.shape, len(recs_to_add)

rec_array = np.array(recs_to_add)

#%%

#######################################
# Creating a new recommended playlist #
#######################################

maximum_size = 20
new_rec_playlist = {}

if len(recs_to_add) <= maximum_size:
    for song in recs_to_add:
        songtitle = song.split(" || ")[1]
        artistname = song.split(" || ")[0]
        if artistname in new_rec_playlist.keys():
            new_rec_playlist[artistname].append(songtitle)
        else:
            new_rec_playlist[artistname] = [songtitle]     
else:
    for best_song in list(tracks_features[tracks_features['ratings'] == max(tracks_features['ratings'])]['index']):
        best_songtitle = best_song.split(" || ")[1]
        best_artistname = best_song.split(" || ")[0]
        if best_artistname in new_rec_playlist.keys():
            new_rec_playlist[best_artistname].append(best_songtitle)
        else:
            new_rec_playlist[best_artistname] = [best_songtitle]
    if len(new_rec_playlist.values()) > maximum_size:
        for key in new_rec_playlist.keys():
            new_rec_playlist[key] = [new_rec_playlist[key][0]]
        print("The playlist has been adjusted (1)")
    if len(new_rec_playlist.values()) > maximum_size:
        corr_rec_playlist = {}
        for i in range(maximum_size):
            artist_chosen = rd.choice(list(new_rec_playlist.keys()))
            song_chosen = new_rec_playlist[artist_chosen]
            del new_rec_playlist[artist_chosen]
            corr_rec_playlist[artist_chosen] = song_chosen
        print("The playlist has been adjusted (2)")
        new_rec_playlist = corr_rec_playlist
    else:
        for i in range(maximum_size - len(new_rec_playlist)):
            okay = False
            while not okay:
                chosen = rd.choice(recs_to_add)
                artist_chosen = chosen.split(" || ")[0]
                song_chosen = chosen.split(" || ")[1]
                if artist_chosen not in new_rec_playlist.keys():
                    okay = True
                elif song_chosen not in new_rec_playlist[artist_chosen]:
                    okay = True
            if artist_chosen in new_rec_playlist.keys():
                new_rec_playlist[artist_chosen].append(song_chosen)
            else:
                new_rec_playlist[artist_chosen] = [song_chosen]