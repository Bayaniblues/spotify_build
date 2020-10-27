import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fastapi import FastAPI, Request
import uvicorn
import aiofiles
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import json

# Intantiate our app
app = FastAPI()

# Spotify API credentials / instantiate spotipy object
client_id = 'ac249f8bfc274671a2b90cd8fcc4c4ca'
secret_id = 'fe6dae5f957f4a47bc6657113af3e236'
credentials = SpotifyClientCredentials(client_id=client_id,
                                       client_secret=secret_id)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Pre-tuned model parameters from Unit 4 guys
model = NearestNeighbors(n_neighbors=10, 
                         algorithm='kd_tree', 
                         metric='euclidean', 
                         leaf_size=50, 
                         n_jobs=-1)

# This section defines the data that we will often reference in our routes.
data = pd.read_csv('data.csv.zip')
data = data.drop(columns=['explicit', 'mode', 'release_date', 'popularity',
                          'year', 'name'])
data = data[['acousticness', 'danceability', 'duration_ms',
             'energy', 'instrumentalness', 'key', 'liveness',
             'loudness', 'speechiness', 'tempo', 'valence', 
             'artists', 'id']]


def KNN(model, track_id, data): 
    ''' 
    This function uses our model to find similar
    songs in our database
    '''
    model.fit(data[data.columns[0:10]])

    obs = data.index[data['id'] == track_id]
    series = data.iloc[obs, 0:10].to_numpy()

    neighbors = model.kneighbors(series)

    suggestion_index = neighbors[1][0][1:16].tolist()
    suggestion_ids = data.loc[suggestion_index, 'id'].tolist()
    return json.dumps(suggestion_ids)


@app.get('/')
def home():
    return {'Home page working'}


@app.get('/{artist}/{track}')
def suggestions(artist, track):
    '''
    This route requires an artist and track name.  It 
    Pulls the features for the selected song from spotify api
    and uses the KNN function to suggest songs
    '''
    results = sp.search(q= f'{track} artist:{artist}' , type='track', limit=1)

    track_id = results['tracks']['items'][0]['id']

    artist_name = results['tracks']['items'][0]['album']['artists'][0]['name']
    
    track_features_decoy = list(enumerate(sp.audio_features([track_id])))
    track_features_decoy2 = list(enumerate(track_features_decoy[0]))
    track_features = track_features_decoy2[1][1]
    features = list([track_features['acousticness'],
                    track_features['danceability'],
                    track_features['duration_ms'],
                    track_features['energy'],
                    track_features['instrumentalness'],
                    track_features['key'],
                    track_features['liveness'],
                    track_features['loudness'],
                    track_features['speechiness'],
                    track_features['tempo'],
                    track_features['valence'],
                    artist_name,
                    track_id])
    features_df = pd.DataFrame([features], columns=['acousticness', 'danceability', 'duration_ms',
                                                    'energy', 'instrumentalness', 'key', 'liveness',
                                                    'loudness', 'speechiness', 'tempo', 'valence', 
                                                    'artists', 'id'])

    normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))

    appended_data = data.append(features_df, ignore_index=True)

    data_nums = appended_data.select_dtypes(include=[np.number])

    normal_data = normalizer.fit_transform(data_nums)
    normal_data = pd.DataFrame(normal_data, columns=['acousticness', 'danceability', 'duration_ms',
                                                     'energy', 'instrumentalness', 'key', 'liveness',
                                                     'loudness', 'speechiness', 'tempo', 'valence'])

    normal_data['artists'] = appended_data['artists']
    normal_data['id'] = appended_data['id']
  
    return track_id, KNN(model, track_id, normal_data)


@app.get('/{artist}/{track}/art')
def album_art(artist, track):
    '''
    This function uses the artist and track name
    to return three urls for different sizes of
    album art
    '''
    
    results = sp.search(q= f'{track} artist:{artist}' , type='track', limit=1)
    img_addresses = results['tracks']['items'][0]['album']['images']
    img_600 = img_addresses[0]['url']
    img_300 = img_addresses[1]['url']
    img_240 = img_addresses[2]['url']
    images = [img_600, img_300, img_240]

    return images


@app.get('/{artist}/{track}/clip')
def song_clip(artist, track):
    '''
    This function uses artist and track name to 
    oull a 30 second mp3 preview of the track
    '''
    results = sp.search(q= f'{track} artist:{artist}' , type='track', limit=1)
    clip = results['tracks']['items'][0]['preview_url']

    return clip


    
      
    


    

    




