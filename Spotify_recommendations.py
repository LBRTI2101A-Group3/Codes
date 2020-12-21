# -*- coding: utf-8 -*-
"""
Created on Oct 2020
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

# Group 1 : '7ovE5Vf0xvGQqTbPkEbOK6'
# Group 2 : ‘7fyFIBf3n8Z8YPdAN57OAi’
# Group 4 : ‘7G7IG1gZsdBcPoNWaRUHJb’
# Group 5 : ‘76eGMBYZMCC6P26VnNKOFs’
# Group 6 : ‘2YMSHQSRSmSET7ZyNozGR4’

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
  "danceability", "duration_ms", "energy", "instrumentalness",  "key", "liveness",
  "loudness", "mode", "speechiness", "tempo", "valence"]]

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
forest = RandomForestClassifier(random_state = 42, max_depth = 5, max_features = 12) # Set by GridSearchCV below
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
plt.axvline(9, c='b')
plt.axvline(10, c='b')
plt.axhline(0.95, c='r')
plt.show();

# Fit your dataset to the optimal pca
pca = decomposition.PCA(n_components = 9)
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

##############################
# Getting recommended tracks #
##############################

# Generate a new dataframe for recommended tracks
# Set recommendation limit as you wish
# Check documentation for  recommendations; https://beta.developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/

rec_tracks = []
for i in playlist_df['id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i], limit = 10)['tracks'];

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0,len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)
        
rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
#rec_playlist_df.head()

X_test_names = v.transform(rec_track_names)
rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms", 
                         "energy", "instrumentalness",  "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]]

#%%

######################
# Making predictions #
######################

tree_grid.best_estimator_.fit(X_train_last, y_train)
rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
X_test_pca = pca.transform(rec_playlist_df_scaled)
X_test_names = v.transform(rec_track_names)
X_test_last = csr_matrix(hstack([X_test_pca, X_test_names]))
y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

rec_playlist_df['ratings']=y_pred_class
rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
rec_playlist_df = rec_playlist_df.reset_index()

# Pick the top ranking tracks (rating >= 8) to add your new playlist 
recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=8]['index'].values.tolist()

# Check the part of recommended tracks to add
len(rec_tracks), rec_playlist_df.shape, len(recs_to_add)

rec_array = np.array(recs_to_add)

#%%

#######################################
# Creating a new recommended playlist #
#######################################

new_rec_playlist = {}
count = 0
for index_URI in recs_to_add:
    new_track = sp.track(str(index_URI))
    new_song = new_track['name']
    new_artist = new_track['artists'][0]['name']
    #new_album = new_track['album']['name']
    if new_artist in new_rec_playlist.keys():
        if new_song not in new_rec_playlist[new_artist]:
            new_rec_playlist[new_artist].append(new_song)
    else:
        new_rec_playlist[new_artist] = [new_song]

#%%

import random as rd

keys = list(new_rec_playlist.keys())

reco_playlist = []
if len(new_rec_playlist.keys()) > 20:
    while len(reco_playlist) < 20:
        choice = rd.choice(keys)
        if new_rec_playlist[choice] not in reco_playlist:
            reco_playlist.append([choice,new_rec_playlist[choice]])
    