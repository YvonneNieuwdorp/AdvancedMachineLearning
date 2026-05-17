import pandas as pd

def check_billboard_dataset(path):
    df = pd.read_csv(path)

    print("\n=== BASIC INFO ===")
    print(df.head())
    print(df.info())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== ARTIST NUMERIC CHECK (should be empty) ===")
    numeric_artists = df[df["artist"].astype(str).str.isnumeric()]
    print(numeric_artists.head(10))
    print(f"Found {len(numeric_artists)} numeric artist rows")

    print("\n=== YEAR DISTRIBUTION ===")
    print(df["year"].value_counts().sort_index().head())

    print("\n=== RANK CHECK ===")
    print(df["rank"].describe())

    print("\n=== TOP 10 CHECK ===")
    print(df[df["rank"] <= 10].head())

    print("\n=== UNIQUE ARTISTS SAMPLE ===")
    print(df["artist"].dropna().unique()[:20])

    print("\n=== DONE CHECK ===")

    return df

check_billboard_dataset("billboard_year_end_2003_2023.csv")