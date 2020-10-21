import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import pickle

import plotly.graph_objects as go
import plotly.offline as pyo

def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def search_id(input_id):
    """Inputs a song id as it appears in the kaggle dataset,
     then run model, outputs a radio graph comparison"""
    # Get meta data
    df = pd.read_csv('../notebooks/spotify_kaggle/data.csv')
    df.head()

    # Get scaled data
    new_df = pd.read_csv('../notebooks/spotify2.csv')
    new_df.head()

    # get model
    filename = '../notebooks/neighbors'
    infile = open(filename, 'rb')
    model = pickle.load(infile)
    infile.close()

    def value_monad(a):
        return new_df.values.tolist()[a]

    def heigher_order_features(input_y):
        """A helper function for compare_this function, it creates
        a list with a specific row input"""
        state = []
        for i, x in enumerate(new_df.columns.tolist()):
            a = new_df[str(x)][input_y]
            state.append(a)

        return state

    def compare_this(a, b):
        categories = new_df.columns.tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=heigher_order_features(a),
            theta=categories,
            fill='toself',
            name='Product A'
        ))
        fig.add_trace(go.Scatterpolar(
            r=heigher_order_features(b),
            theta=categories,
            fill='toself',
            name='Product B'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def search_id_monad(a):
        b = model.kneighbors([value_monad(a)])
        return b[1]

    def run_model(a):
        monad = search_id_monad(a)
        data = [monad[0][0], monad[0][1]]
        meta_data = [
            df.values.tolist()[data[0]],
            df.values.tolist()[data[1]]]
        return meta_data

    return run_model(input_id)

