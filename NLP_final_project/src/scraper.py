"""
scraper.py
----------
Scrapes the Billboard Year-End Hot 100 charts (2003-2023).

Can be run standalone:
    python src/scraper.py

Output: data/billboard_year_end_2003_2023.csv
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os

RUN_SCRAPER = False  # Set to True to re-run the scraper

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")
START_YEAR = 2003
END_YEAR = 2023


def scrape_chart_data() -> None:
    """
    Scrapes Billboard Year-End Hot 100 charts for each year in [START_YEAR, END_YEAR].

    For each year, extracts song title, artist, rank (position on page), and a
    binary top_10 flag. Results are concatenated and saved to BILLBOARD_OUTPUT_PATH.
    Sleeps 1 second between requests to avoid rate-limiting.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    all_data: list[pd.DataFrame] = []

    for year in range(START_YEAR, END_YEAR + 1):
        url = f"https://www.billboard.com/charts/year-end/{year}/hot-100-songs/"
        print(f"Scraping Year-End chart: {year}")

        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"  Skipping {year} — page not found (status {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            entries = soup.find_all("div", class_=re.compile("o-chart-results-list-row-container"))

            year_songs: list[dict] = []
            for entry in entries:
                title_tag = entry.find("h3")
                labels = entry.find_all("span", class_=re.compile("c-label"))
                artist_tag = labels[1] if len(labels) >= 2 else (labels[0] if labels else None)

                if title_tag and artist_tag:
                    title = re.sub(r"\s+", " ", title_tag.get_text(strip=True))
                    artist = re.sub(r"\s+", " ", artist_tag.get_text(strip=True))
                    if title and artist:
                        year_songs.append({"year": year, "title": title, "artist": artist})

            df_year = pd.DataFrame(year_songs).drop_duplicates()
            df_year["rank"] = range(1, len(df_year) + 1)
            df_year["top_10"] = (df_year["rank"] <= 10).astype(int)
            df_year = df_year.head(100)

            all_data.append(df_year)
            print(f"  Collected {len(df_year)} songs for {year}")
            time.sleep(1)

        except Exception as e:
            print(f"  Error for year {year}: {e}")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(BILLBOARD_OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {final_df.shape[0]} rows to {BILLBOARD_OUTPUT_PATH}")


if __name__ == "__main__":
    if RUN_SCRAPER:
        scrape_chart_data()
    else:
        print("Scraper disabled (RUN_SCRAPER=False). Set to True to re-run.")
