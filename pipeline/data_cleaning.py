import pandas as pd
data_path='..\Spotify-Recommendation-System\data\Music Info.csv'
def cleaned_data(data):
    return (
        data
        .drop_duplicates(subset=["spotify_id","year","duration_ms"])
        .drop(columns=["genre","spotify_id"])
        .fillna({"tags":"no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
        .reset_index(drop=True)
    )
def main(path):
    data=pd.read_csv(path)
    cleaneddata=cleaned_data(data)
    cleaneddata.to_csv("data/processed/cleaned_data.csv",index=False)

if __name__=='__main__':
    main(data_path)
