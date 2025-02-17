import streamlit as st

from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.hybrid_recommendation import HybridRecommenderSystem
#from hybrid_recommendation import HybridRecommenderSystem
import numpy as np

# load the data
cleaned_data_path = r"data/processed/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data_path)

# load the transformed data
transformed_data_path = r"data/processed/transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

# load the track ids
track_ids_path = r"data/track_ids.npy"
track_ids = load(track_ids_path,allow_pickle=True)

# load the filtered songs data
filtered_data_path = r"data/processed/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

# load the interaction matrix
interaction_matrix_path = r"data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

# load the transformed hybrid data
transformed_hybrid_data_path = r"data/processed/transformed_hybrid_data.npz"
transformed_hybrid_data = load_npz(transformed_hybrid_data_path)
def calculate_similarity_scores(input_vector, data):
    
    # calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, data)
    
    return similarity_scores
def content_recommendation(song_name,artist_name,songs_data, transformed_data, k=10):
    
    # convert song name to lowercase
    song_name = song_name.lower()
    # convert the artist name to lowercase
    artist_name = artist_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list
# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)
# artist name
artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', artist_name)
# lowercase the input
song_name = song_name.lower()
artist_name = artist_name.lower()

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():   
    # type of filtering
    filtering_type = "Hybrid Recommender System"

    # diversity slider
    diversity = st.slider(label="Diversity in Recommendations",
                        min_value=1,
                        max_value=9,
                        value=5,
                        step=1)

    content_based_weight = 1 - (diversity / 10)
    
    # plot a bar graph
    chart_data = pd.DataFrame({
        "type" : ["Personalized", "Diverse"],
        "ratio": [10 - diversity, diversity]
    })
    
    st.bar_chart(chart_data,x="type",y="ratio")
    
else:
    # type of filtering
    filtering_type = 'Content-Based Filtering'

# Button
if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((songs_data["name"] == song_name) & (songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = content_recommendation(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=songs_data,
                                                     transformed_data=transformed_data,
                                                     k=k)
            
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
            
elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):
        st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
        recommender = HybridRecommenderSystem(
                                                number_of_recommendations= k,
                                                weight_content_based= content_based_weight
                                                )
                                
        # get the recommendations
        recommendations = recommender.give_recommendations(song_name= song_name,
                                                        artist_name= artist_name,
                                                        songs_data= filtered_data,
                                                        transformed_matrix= transformed_hybrid_data,
                                                        track_ids= track_ids,
                                                        interaction_matrix= interaction_matrix)
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