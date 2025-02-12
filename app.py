import streamlit as st
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# transformed data path
transformed_data_path = "data/processed/transformed_data.npz"

# cleaned data path
cleaned_data_path = "data/processed/cleaned_data.csv"

# load the data
data = pd.read_csv(cleaned_data_path)

# load the transformed data
transformed_data = load_npz(transformed_data_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)
# lowercase the input
song_name = song_name.lower()

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)
def content_recommendation(song_name,songs_data, transformed_data, k=10):
    
    # convert song name to lowercase
    song_name = song_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name)]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
   
    similarity_scores = cosine_similarity(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list
# Button
if st.button('Get Recommendations'):
    if (data["name"] == song_name).any():
        st.write('Recommendations for', f"**{song_name}**")
        recommendations = content_recommendation(song_name,data,transformed_data,k)

        # Display Recommendations
        for ind , recommendation in recommendations.iterrows():
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

