# WEEK 2
# Billboard Hot 100 scraper
# Scrapes weekly Hot 100 charts from 8 years ago until 3 years ago
# Goal: create dataset for NLP + ML hit prediction project

# pip install -r requirements.txt

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path


# SETTINGS

# START_DATE = datetime.now() - timedelta(weeks=52 * 8) # Acht jaar terug
# END_DATE = datetime.now() - timedelta(weeks=52 * 3) # Drie jaar terug (want database is drie jaar oud, dus heb je 5 jaar aan chart data)

# OUTPUT_FILE = "billboard_hot100_historical.csv"

# # User-Agent header
# headers = {
#     "User-Agent": "Mozilla/5.0"
# }


# GENERATE WEEKLY DATES

# dates = []

# current_date = START_DATE

# while current_date <= END_DATE:
#     dates.append(current_date.strftime("%Y-%m-%d"))
#     current_date += timedelta(weeks=1)

# print(f"Dates: \n{dates}")
# print(f"Total chart weeks to scrape: {len(dates)}")

csv_path = Path("term_4") / "song_lyrics.csv"
df = pd.read_csv(csv_path)
print(df.head())