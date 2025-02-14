import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import os

# Define base directory
base_dir = os.path.abspath(os.path.join("..", "Spotify-Recommendation-System", "data"))

# Define paths correctly using os.path.join()
user_listening_history = os.path.join(base_dir, "UserHistory.csv")
songs_data_path = os.path.join(base_dir, "processed", "cleaned_data.csv")
filtered_data_save_path = os.path.join(base_dir, "processed", "collab_filtered_data.csv")
track_ids_save_path = os.path.join(base_dir, "track_ids.npy")
interaction_matrix_save_path = os.path.join(base_dir, "interaction_matrix.npz")

# Ensure directories exist before writing files
os.makedirs(os.path.dirname(filtered_data_save_path), exist_ok=True)
os.makedirs(os.path.dirname(track_ids_save_path), exist_ok=True)
os.makedirs(os.path.dirname(interaction_matrix_save_path), exist_ok=True)

def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids
    """
    # filter data based on track_ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    # sort the data by track id
    filtered_data.sort_values(by="track_id", inplace=True)
    # rest index
    filtered_data.reset_index(drop=True, inplace=True)
    # save the data
    save_pandas_data_to_csv(filtered_data, save_df_path)

def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file
    """
    data.to_csv(file_path, index=False)
def create_interaction_matrix(history_data:dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    # make a copy of data
    df = history_data.copy()
    
    # convert the playcount column to float
    df['playcount'] = df['playcount'].astype(np.float64)
    
    # convert string column to categorical
    df = df.categorize(columns=['user_id', 'track_id'])
    
    # Convert user_id and track_id to numeric indices
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    
    # get the list of track_ids
    track_ids = df['track_id'].cat.categories.values
    
    # save the categories
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    
    # add the index columns to the dataframe
    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping
    )
    
    # create the interaction matrix
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    
    # compute the matrix
    interaction_matrix = interaction_matrix.compute()
    
    # get the indices to form sparse matrix
    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    
    # get the shape of sparse matrix
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()
    
    # create the sparse matrix
    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    
    # save the sparse matrix
    save_sparse_matrix(interaction_matrix, save_matrix_path)
def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    save_npz(file_path, matrix)
def main():
    ##Load data
    user_data=dd.read_csv(user_listening_history)

    ##get the unique track ids
    unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()

    songs_data=pd.read_csv(songs_data_path)
    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    # create the interaction matrix
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)



if __name__=='__main__':
    main()