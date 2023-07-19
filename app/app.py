
from __future__ import print_function

import os
import json
import time
import sys


import pandas as pd
import numpy as np  
import seaborn as sn
import gradio as gr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors



import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

df = pd.read_csv('spotify_data.csv')


df = df.drop(columns=['Unnamed: 0', "Unnamed: 0.1", "pos", "artist_uri", "album_uri", "duration_ms_x", "album_name", "name", "type", "id", "track_href", "analysis_url", "duration_ms_y", "time_signature", "artist_pop", "track_pop"])

df.drop_duplicates(subset=['uri'], inplace=True)
df.reset_index(drop=True, inplace=True)
df_num = df.select_dtypes(include = ['float64', 'int64'])


numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
categorical_cols = ['key', 'mode']



# Create the preprocessing pipeline
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Apply the preprocessing pipeline to your DataFrame
df_processed = preprocessing_pipeline.fit_transform(df_num)

num_cols_transformed = numeric_cols
cat_cols_transformed = preprocessing_pipeline.named_transformers_['cat'].get_feature_names_out(categorical_cols)

# Combine the transformed column names

all_cols_transformed = num_cols_transformed + cat_cols_transformed.tolist()

# Convert the processed NumPy array back to a DataFrame
df_processed = pd.DataFrame(df_processed, columns=all_cols_transformed)


def transform_query(track_uri):
    audio_features = sp.audio_features(track_uri)[0]
    track_data = []
    track_dict = {
        'acousticness': audio_features['acousticness'],
        'danceability': audio_features['danceability'],
        'energy': audio_features['energy'],
        'instrumentalness': audio_features['instrumentalness'],
        'liveness': audio_features['liveness'],
        'loudness': audio_features['loudness'],
        'speechiness': audio_features['speechiness'],
        'tempo': audio_features['tempo'],
        'valence': audio_features['valence'],
        'key': audio_features['key'],
        'mode': audio_features['mode']
    }
    
    track_data.append(track_dict)
    query_data = pd.DataFrame(track_data)
    return query_data


knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
knn_model.fit(df_processed) # I'm using all the data for KNN

# Function to find similar songs to the input URI
def find_similar_songs(track_uri):

    query_data = transform_query(track_uri)
    
    # Scale the query data using the same scaler
    query_data_scaled = preprocessing_pipeline.transform(query_data)
    query_data_scaled_df = pd.DataFrame(query_data_scaled, columns=all_cols_transformed)

    # Find the most similar songs using the KNN model
    distances, indices = knn_model.kneighbors(query_data_scaled_df, n_neighbors=10)

    # Retrieve the Artist Name, Song Name, and Track URI of the most similar songs
    similar_songs = []
    for index in indices[0]:
        artist_name = df.iloc[index]['artist_name']
        song_name = df.iloc[index]['track_name']
        similar_uri = df.iloc[index]['uri']
        
        track_id = similar_uri.split(":")[-1]
        full_url = f"https://open.spotify.com/track/{track_id}"

        similar_songs.append((artist_name, song_name, full_url))
        
    return similar_songs


similar_songs = find_similar_songs('https://open.spotify.com/track/6rDaCGqcQB1urhpCrrD599?si=2ac7add2ea054ab2')


def format_output(similar_songs):
    output = []
    for song in similar_songs:
        output.append({"Artist Name": song[0], "Song Name": song[1], "Spotify Track URL": song[2]})
    return pd.DataFrame(output)

# Create the Gradio interface
iface = gr.Interface(
    fn=find_similar_songs,  # Your find_similar_songs function
    inputs=gr.Textbox(label="Enter Spotify Track URL"),
    outputs=gr.Dataframe(headers=["Artist Name", "Song Name", "Spotify Track URL"]),
    live=True
)


iface.launch("share=True")