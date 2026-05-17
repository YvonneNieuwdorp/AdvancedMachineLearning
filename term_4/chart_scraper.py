import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

RUN_SCRAPER = True  # Zet naar False als je de scraper niet nog eens wil uitvoeren

def scrape_chart_data():
    headers = {
            "User-Agent": "Mozilla/5.0"
        }

    start_year = 2003
    end_year = 2023 # nummers van afgelopen 20 jaar sinds kaggle dataset

    all_data = []

    for year in range(start_year, end_year + 1):

        # Billboard Year-End chart URL
        url = f"https://www.billboard.com/charts/year-end/{year}/hot-100-songs/"

        print(f"Scraping Year-End chart: {year}") # Debugje

        try:
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print(f"Skipping {year}, page not found")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Year-end chart items 
            entries = soup.find_all("div", class_=re.compile("o-chart-results-list-row-container"))

            year_songs = []

            for entry in entries:

                title_tag = entry.find("h3")

                # pak alle labels en neem de juiste 
                labels = entry.find_all("span", class_=re.compile("c-label"))

                artist_tag = None

                if len(labels) >= 2:
                    artist_tag = labels[1]
                elif len(labels) == 1:
                    artist_tag = labels[0]

                if title_tag and artist_tag:

                    title = title_tag.get_text(strip=True)
                    artist = artist_tag.get_text(strip=True)

                    title = re.sub(r"\s+", " ", title)
                    artist = re.sub(r"\s+", " ", artist)

                    if title and artist:

                        year_songs.append({
                            "year": year,
                            "title": title,
                            "artist": artist
                        })

            df_year = pd.DataFrame(year_songs).drop_duplicates()

            # rank = volgorde op pagina
            df_year["rank"] = range(1, len(df_year) + 1)

            # top 10 label, 1 als wel, 0 anders
            df_year["top_10"] = (df_year["rank"] <= 10).astype(int)

            # houd top 100 (year-end chart is meestal al 100 maar toch ff zeker)
            df_year = df_year.head(100)

            all_data.append(df_year)

            print(f"Collected {len(df_year)} songs for {year}")

            time.sleep(1)

        except Exception as e:
            print(f"Error for year {year}: {e}")

    # alles combineren
    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_csv("billboard_year_end_2003_2023.csv", index=False)

    print("\nDone!")
    print(final_df.head())
    print(final_df.shape)


        
if __name__ == "__main__":

    if RUN_SCRAPER:
        print("Running preprocessing pipeline...")
        scrape_chart_data()
    else:
        print("Pipeline disabled (RUN_SCRAPER=False)")