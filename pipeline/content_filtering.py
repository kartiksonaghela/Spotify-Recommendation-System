import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
cleaned_data="..\Spotify-Recommendation-System\data\processed\cleaned_data.csv"
# cols to transform
frequency_enode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]

def data_for_content_filtering(data):
    return (
        data
        .drop(columns=["track_id","name","spotify_preview_url"])
    )
def transform_data(data):
    
    # load the transformer
    transformer = joblib.load("transformer.joblib")
    
    # transform the data
    transformed_data = transformer.transform(data)
    
    return transformed_data
def train_transformer(data):
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_enode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    # fit the transformer
    transformer.fit(data)

    # save the transformer
    joblib.dump(transformer, "transformer.joblib")
def save_transformed_data(transformed_data,save_path):
        # save the transformed data
    save_npz(save_path, transformed_data)
def main(cleaned_data):
    data=pd.read_csv(cleaned_data)
    data_content_filtering=data_for_content_filtering(data)
    train_transformer(data_content_filtering)
    transformed_data = transform_data(data_content_filtering)
    #save transformed data
    save_transformed_data(transformed_data,"data/processed/transformed_data.npz")



if __name__=='__main__':
    main(cleaned_data)