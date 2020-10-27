from typing import Optional

# Local imports
from functions.graphs import create_plot, search_id
from functions.spotifyroute import track_id, get_stuff,get_track_id


from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pickle
import pandas as pd

import os
import psutil
import json


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

Check_ram = lambda: print("RAM"+str(psutil.virtual_memory().percent)+"%")

model = NearestNeighbors(n_neighbors=10,
                         algorithm='kd_tree',
                         metric='euclidean',
                         leaf_size=50,
                         n_jobs=-1)


# Base function using a knn model input and a track id
def KNN(model, track_id):
    data = pd.read_csv('notebooks/spotify_kaggle/spotify3.csv')
    model.fit(data[data.columns[0:12]])

    obs = data.index[data['id'] == track_id]
    series = data.iloc[obs, 0:12].to_numpy()

    neighbors = model.kneighbors(series)

    new_obs = neighbors[1][0][1:16]
    print(new_obs)
    print(list(data.loc[new_obs, 'id']))
    return list(data.loc[new_obs, 'id'])


def KNN3(model, track_id, data):
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
    return suggestion_ids


@app.get('/api/v3/spotify/{artist}/{track}')
def suggestions(artist, track):
    '''
    This route requires an artist and track name.  It
    Pulls the features for the selected song from spotify api
    and uses the KNN function to suggest songs
    '''
    data = pd.read_csv('notebooks/spotify_kaggle/datav3update.csv')
    data = data.drop(columns=['explicit', 'mode', 'release_date', 'popularity',
                              'year', 'name'])
    data = data[['acousticness', 'danceability', 'duration_ms',
                 'energy', 'instrumentalness', 'key', 'liveness',
                 'loudness', 'speechiness', 'tempo', 'valence',
                 'artists', 'id']]

    client_id = os.getenv("CLIENT_ID")
    secret_id = os.getenv("SECRET_ID")

    credentials = SpotifyClientCredentials(client_id=client_id,
                                           client_secret=secret_id)
    sp = spotipy.Spotify(client_credentials_manager=credentials)

    results = sp.search(q=f'{track} artist:{artist}', type='track', limit=1)

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

    normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1))

    appended_data = data.append(features_df, ignore_index=True)

    data_nums = appended_data.select_dtypes(include=[np.number])

    normal_data = normalizer.fit_transform(data_nums)
    normal_data = pd.DataFrame(normal_data, columns=['acousticness', 'danceability', 'duration_ms',
                                                     'energy', 'instrumentalness', 'key', 'liveness',
                                                     'loudness', 'speechiness', 'tempo', 'valence'])

    normal_data['artists'] = appended_data['artists']
    normal_data['id'] = appended_data['id']

    return {"Searched Track": track_id, "Suggestions": KNN3(model, track_id, normal_data)}


@app.get('/api/v3/spotify/{artist}/{track}/art')
def album_art(artist, track):
    '''
    This function uses the artist and track name
    to return three urls for different sizes of
    album art
    '''
    client_id = os.getenv("CLIENT_ID")
    secret_id = os.getenv("SECRET_ID")

    credentials = SpotifyClientCredentials(client_id=client_id,
                                           client_secret=secret_id)
    sp = spotipy.Spotify(client_credentials_manager=credentials)

    results = sp.search(q=f'{track} artist:{artist}', type='track', limit=1)
    img_addresses = results['tracks']['items'][0]['album']['images']
    img_600 = img_addresses[0]['url']
    img_300 = img_addresses[1]['url']
    img_240 = img_addresses[2]['url']
    images = [img_600, img_300, img_240]

    return images


@app.get('api/v3/spotify/{artist}/{track}/clip')
def song_clip(artist, track):
    '''
    This function uses artist and track name to
    oull a 30 second mp3 preview of the track
    '''
    client_id = os.getenv("CLIENT_ID")
    secret_id = os.getenv("SECRET_ID")

    credentials = SpotifyClientCredentials(client_id=client_id,
                                           client_secret=secret_id)
    sp = spotipy.Spotify(client_credentials_manager=credentials)
    results = sp.search(q=f'{track} artist:{artist}', type='track', limit=1)
    clip = results['tracks']['items'][0]['preview_url']

    return clip


@app.get("/api/v2/search/trackid/{track_id}")
def run_model(track_id: str):
    """
    Production ready model, takes in a track_id, and returns
    an list of 9 track id's based off of nearest neighbors.
    :param track_id:
    :return:
    """
    filename = 'test_1.sav'
    model = pickle.load(open(filename, 'rb'))
    output = KNN(model, track_id)
    Check_ram()
    return output


@app.get("/api/v1/search/byfeature/{acousticness}/{danceability}/{duration_ms}/{energy}/{instrumentalness}/{liveness}/{loudness}/{speechiness}/{valence}/{tempo}")
def search_by_feature(acousticness,danceability,duration_ms,energy,instrumentalness,liveness,loudness,speechiness,valence,tempo):
    """
    An naive, overfitted model that searches music based off of 10 music features, returns track_id
    :param acousticness:
    :param danceability:
    :param duration_ms:
    :param energy:
    :param instrumentalness:
    :param liveness:
    :param loudness:
    :param speechiness:
    :param valence:
    :param tempo:
    :return:
    """
    feature_array = [acousticness,danceability,duration_ms,energy,instrumentalness,liveness,loudness,speechiness,valence,tempo]

    def idk(inpute):
        feature_array = inpute
        filename = 'notebooks/neighbors.pickle'

        infile = open(filename, 'rb')
        model = pickle.load(infile)
        infile.close()

        output = model.kneighbors([feature_array])
        df = pd.read_csv('notebooks/spotify_kaggle/spotify3.csv')
        track_id = df.values.tolist()[0][-1]
        Check_ram()
        return output

    def iterate_this(in_put):
        df = pd.read_csv('notebooks/spotify_kaggle/spotify3.csv')
        state = []
        for i, x in enumerate(in_put[1][0]):
            track_id = df.values.tolist()[x][-1]
            state.append(track_id)
        return state

    return iterate_this(idk(feature_array))


@app.get('/api/v1/spotify/artist/{artist}/track/{track}')
def api_artist_track(artist, track):
    """
    API Router: Connects to the Spotify api and search via Artist & track
    to return data features.
    :param artist:
    :param track:
    :return:
    """
    try:
        return track_id(artist, track)
    except:
        return {"error": "error, did you enter the correct artist/track pair?"}


@app.get('/api/v1/spotify/track_id/{track_id}')
def api_query_track_id(track_id):
    """
    API Router: Connects to the Spotify api Inputs track_id parameter and queries an track_id and returns
    everything from that track id.
    :param artist:
    :return:
    """
    return {"Track Suggestions": get_track_id(track_id)}


@app.get('/api/v1/sql/artist/{artist}')
def sql_query_artist(artist):
    """
    Inputs Artist parameter and queries an artist name and returns
    a Artist that contains that word.
    :param artist:
    :return:
    """
    return {"Artist Suggestions": get_stuff(artist)}


@app.get('/api/v1/csv/track/{track_id}')
def csv_search(track_id):
    data = pd.read_csv('notebooks/spotify_kaggle/spotify3.csv')
    search = data[data.eq(track_id).any(1)].values.tolist()
    return search


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    bar = create_plot()
    return templates.TemplateResponse("home.html", {
        "plot": bar,
        "request": request,
        "id": id})


@app.get("/search/id", response_class=HTMLResponse)
async def ask_id(request: Request):
    return templates.TemplateResponse('idsearch.html',{"request": request,})


@app.get("/")
def read_root():
    return {"Hello": "World"}

