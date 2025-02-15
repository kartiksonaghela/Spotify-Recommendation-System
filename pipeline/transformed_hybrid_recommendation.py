import pandas as pd
from content_filtering import transform_data,save_transformed_data

filtered_data=r'..\Spotify-Recommendation-System\data\processed\collab_filtered_data.csv'

##save path
save_path = r"..\Spotify-Recommendation-System\data\processed\transformed_hybrid_data.npz"

def main(data_path,save_path):
    # load the filtered data
    filtered_data = pd.read_csv(data_path)
    filtered_data=filtered_data.drop(columns=["track_id","name","spotify_preview_url"])
    transformed_data=transform_data(filtered_data)
    save_transformed_data(transformed_data,save_path)

if __name__=='__main__':
    main(filtered_data,save_path)