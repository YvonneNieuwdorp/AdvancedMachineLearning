import os
import pandas as pd
import kagglehub
import re


RAW_PATH = "song_lyrics.csv"
OUTPUT_PATH = "clean_song_lyrics.csv"

RUN_PREPROCESSING = True  # Zet naar True als je dit nog eens wil uitvoeren

def clean_artist(artist):
    if not isinstance(artist, str):
        return artist
    # Voeg spatie toe voor "Featuring" als die er niet is
    artist = re.sub(r'(?i)([a-z])([Ff]eaturing)', r'\1 \2', artist)
    # Dan pas splitten
    artist = re.split(r'(?i)\bfeaturing\b|\bwith\b|&|,|[xX](?=[A-Z])', artist)[0]
    return artist.strip().lower()

def clean_title(title): 
    if not isinstance(title, str): 
        return title
    title = re.sub(r'\(.*?\)', '', title)   # verwijder haakjes
    title = re.sub(r'[^\w\s]', '', title)   # verwijder leestekens
    return title.strip().lower()

def download_and_clean_data():

    print("Starting Kaggle dataset download...")

    # Download dataset to local cache folder
    path = kagglehub.dataset_download(
        "carlosgdcj/genius-song-lyrics-with-language-information"
    )

    print(f"Dataset downloaded to cache: {path}")

    import glob

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the downloaded dataset at {path}")
    raw_path = csv_files[0] 
    print(f'CSV file found: {raw_path}')
    print("loading dataset...")

    df = pd.read_csv(
        raw_path,
        usecols=["title", "tag", "artist", "year", "lyrics", "language"]
    )

    print(f"Data loaded!")

    
    # CLEANING 
    print("\nCleaning data...")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    print("Year converted")

    df = df[df["language"] == "en"]
    print(f"Language filtered: {df.shape}")

    df = df[(df["year"] >= 2003) & (df["year"] <= 2023)]
    print(f"Year filtered: {df.shape}")

    df = df[df["tag"] != "misc"]
    print(f"Genre filtered: {df.shape}")

    df = df.dropna(subset=["year", "lyrics", "title", "artist"])
    print(f"Missing values removed: {df.shape}")

    billboard_df = pd.read_csv("billboard_year_end_2003_2023.csv")
    
    df['title_clean'] = df['title'].apply(clean_title)
    df['artist_clean'] = df['artist'].apply(clean_artist)
    billboard_df['title_clean'] = billboard_df['title'].apply(clean_title)
    billboard_df['artist_clean'] = billboard_df['artist'].apply(clean_artist)
    
    n_genius = len(df)
    n_billboard = len(billboard_df)
    n_billboard_unique = billboard_df[["title_clean", "artist_clean"]].drop_duplicates().shape[0]

    df = df.merge(
        billboard_df[["title_clean", "artist_clean"]],
        on=["title_clean", "artist_clean"],
        how="inner"
    )

    # Na de merge
    n_matched = len(df)
    n_unique_matched = df[["title_clean", "artist_clean"]].drop_duplicates().shape[0]

    print("\n=== MERGE RESULTAAT ===")
    print(f"Genius rijen voor merge:        {n_genius}")
    print(f"Billboard unieke titel+artiest: {n_billboard_unique}")
    print(f"Genius rijen na merge:          {n_matched}")
    print(f"Unieke nummers gematcht:        {n_unique_matched}")
    print(f"Match rate (billboard):         {n_unique_matched / n_billboard_unique * 100:.1f}%")

      
    # SAVE
    print("Saving cleaned dataset...")

    df.to_csv(OUTPUT_PATH, index=False)

    print("Clean dataset saved.")
    print(df.head())



if __name__ == "__main__":

    if RUN_PREPROCESSING:
        print("Running preprocessing pipeline...")
        download_and_clean_data()
    else:
        print("Pipeline disabled (RUN_PREPROCESSING=False)")