# Spotify Similar Song Finder
This project allows you to find similar songs on Spotify by providing a track URL as input. 

It uses the Spotify API to extract audio features of songs, applies k-Nearest Neighbors (KNN) algorithm to find similar songs based on those features, and presents the results through a user-friendly Gradio interface.

## Demo

You can try the Spotify Similar Song Finder by visiting the following link: [Spotify Similar Song Finder Demo](https://huggingface.co/spaces/yusufc/Spotify-SimilarSongFinder)

**Input Track:** Enter the name of the track for which you want to find similar songs.

**Results:** The model will analyze the input track and provide a list of similar songs based on their audio features and metadata.

## Technologies Used
- Python
- Spotipy (Spotify API wrapper)
- K-Nearest Neighbors model
- Gradio
- Hugging Face Spaces


## How it Works
The Spotify Similar Song Finder uses the _**K-Nearest Neighbors**_ machine learning model trained on a large dataset of songs and their audio features.

When you provide the Track URL, the model processes this information, extracts audio features, and compares them with other tracks in its database. 

The model then returns a list of tracks that have similar audio characteristics and are likely to be enjoyed by fans of the input track.

## Local Development
If you wish to run this project locally, you can find the code and notebooks in the GitHub repository. 

The repository contains Jupyter notebooks and Python scripts for data preprocessing, model training, and deployment. You can also find instructions on how to set up the necessary environment variables and run the application.

## Credits
This project was created by Yusuf Coskun. It utilizes the Spotipy library for accessing the Spotify API and Hugging Face for deployment. The model training data was collected from the Spotify API and publicly available datasets.

Special thanks to the developers of Gradio and the Spotify API for providing the tools and data that made this project possible.

## Disclaimer
The Spotify Similar Song Finder is a hobby project and should be used for entertainment purposes only. The accuracy of the model's predictions may vary, and the results should not be considered as professional music recommendations. Always listen to tracks from legal and authorized sources. The creators of this project are not responsible for any unauthorized use of copyrighted material.
