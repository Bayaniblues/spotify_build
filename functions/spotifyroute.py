from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
import urllib.parse as up
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()


def track_id(artist, track):
    client_id = os.getenv("CLIENT_ID")
    secret_id = os.getenv("SECRET_ID")
    credentials = SpotifyClientCredentials(client_id=client_id,
                                           client_secret=secret_id)
    sp = spotipy.Spotify(client_credentials_manager=credentials)

    results = sp.search(q=f'{track} artist:{artist}', type='track', limit=1)
    track_id = results['tracks']['items'][0]['id']
    track_features_decoy = list(enumerate(sp.audio_features([track_id])))
    track_features_decoy2 = list(enumerate(track_features_decoy[0]))
    track_features = track_features_decoy2[1][1]
    labels = ['acousticness', 'danceability', 'duration_ms',
              'energy', 'intrumentalness', 'key', 'liveness',
              'loudness', 'speechiness', 'tempo', 'valence', 'track_id']
    features = [track_features['acousticness'],
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
                track_id]

    my_dict = []

    for i, x in enumerate(labels):
        my_dict.append({labels[i]: features[i]})

    return {"Features": my_dict,
            "Results": results}


def get_stuff(input_value):
    url = up.urlparse(os.getenv("POSTGRES"))
    conn = psycopg2.connect(database=url.path[1:],
                            user=url.username,
                            password=url.password,
                            host=url.hostname,
                            port=url.port
                            )

    cursor = conn.cursor()

    query = f"SELECT * FROM suggestion_bank WHERE artists LIKE '%{input_value}%'"

    test1 = pd.read_sql_query(query, conn)

    cursor.close()
    conn.close()

    return test1

if __name__ == "__main__":
    track_id("Beatles", "Stawberryfeilds")