from typing import Optional

# Local imports
from functions.graphs import create_plot, search_id
from functions.spotifyroute import track_id, get_stuff

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pickle
import pandas as pd


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


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
    return KNN(model, track_id)


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
    filename = 'neighbors,pickle'

    infile = open(filename, 'rb')
    model = pickle.load(infile)
    infile.close()

    output = model.kneighbors([feature_array])

    df = pd.read_csv('notebooks/spotify_kaggle/spotify3.csv')

    return output


@app.get('/api/v1/spotify/artist/{artist}/track/{track}')
def api_artist_track(artist, track):
    """
    Connects to the Spotify api and search via Artist & track
    to return data features.
    :param artist:
    :param track:
    :return:
    """
    try:
        return track_id(artist, track)
    except:
        return {"error": "error, did you enter the correct artist/track pair?"}



@app.get('/api/v1/sql/artist/{artist}')
def sql_query_artist(artist):
    """
    Inputs Artist parameter and queries an artist name and returns
    a Artist that contains that word.
    :param artist:
    :return:
    """
    return {"Artist Suggestions": get_stuff(artist)}


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

