import streamlit as st
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
interaction_matrix = load_npz(r"data\interaction_matrix.npz")
track_ids = np.load(r"data\track_ids.npy", allow_pickle=True)  # Raw string
df = pd.read_csv(r"data\processed\collab_filtered_data.csv")  # Raw string

# Transformed data path
transformed_data_path = "data/processed/transformed_data.npz"

# Cleaned data path
cleaned_data_path = "data/processed/cleaned_data.csv"

# Load the data
data = pd.read_csv(cleaned_data_path)

# Load the transformed data
transformed_data = load_npz(transformed_data_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)

# Lowercase the input
song_name = song_name.lower()

# k recommendations
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

# Function for content-based recommendation
def content_recommendation(song_name, songs_data, transformed_data, k=10):
    # Convert song name to lowercase
    song_name = song_name.lower()
    # Filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name)]
    # Get the index of the song
    song_index = song_row.index[0]
    # Generate the input vector
    input_vector = transformed_data[song_index].reshape(1, -1)
   
    similarity_scores = cosine_similarity(input_vector, transformed_data)
    # Get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # Get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # Print the top k songs
    top_k_list = top_k_songs_names[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
    return top_k_list

# Function for collaborative recommendation
def collaborative_recommendation(song_name, track_ids, songs_data, interaction_matrix, k=5):
    # Convert song name to lowercase
    song_name = song_name.lower()

    # Fetch the row(s) matching the song name
    song_row = songs_data.loc[songs_data["name"].str.lower() == song_name]

    # Handle case when song is not found
    if song_row.empty:
        raise ValueError(f"Song '{song_name}' not found in dataset.")

    # Ensure only one track_id is selected
    if len(song_row) > 1:
        print(f"Warning: Multiple entries found for '{song_name}', selecting the first one.")
    
    input_track_id = song_row.iloc[0]["track_id"]  # Get first matching track_id

    # Find the index of input_track_id in track_ids
    ind = np.where(track_ids == input_track_id)[0]
    if len(ind) == 0:
        raise ValueError(f"Track ID '{input_track_id}' not found in track_ids array.")

    ind = ind.item()  # Convert single index to scalar
    
    # Fetch the input vector
    input_array = interaction_matrix[ind]

    # Get similarity scores
    similarity_scores = cosine_similarity(input_array.reshape(1, -1), interaction_matrix)

    # Get indices of top-k recommendations
    recommendation_indices = np.argsort(similarity_scores.ravel())[-(k+1):][::-1]

    # Get track IDs of recommendations
    recommendation_track_ids = track_ids[recommendation_indices]

    # Get top scores
    top_scores = np.sort(similarity_scores.ravel())[-(k+1):][::-1]

    # Get song names for recommendations
    scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(), "score": top_scores})
    
    top_k_songs = (
        songs_data.loc[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )
    
    return top_k_songs

# Dropdown to select recommendation type
recommendation_type = st.selectbox('Select recommendation type:', ['Content-Based', 'Collaborative'])

# Button to get recommendations
if st.button('Get Recommendations'):
    if (data["name"] == song_name).any():
        st.write('Recommendations for', f"**{song_name}**")
        
        # Choose the recommendation type based on dropdown selection
        if recommendation_type == 'Content-Based':
            recommendations = content_recommendation(song_name, data, transformed_data, k)
        elif recommendation_type == 'Collaborative':
            recommendations = collaborative_recommendation(song_name, track_ids, df, interaction_matrix, k)
        
        # Display Recommendations
        for ind, recommendation in recommendations.iterrows():
            song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            elif ind == 1:
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
    else:
        st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
