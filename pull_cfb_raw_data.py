import os
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set up API key
api_key = os.environ.get('CFBD_API_KEY')
if not api_key:
    raise ValueError("CFBD_API_KEY not found in .env file. Ensure it's set and retry.")
headers = {"Authorization": f"Bearer {api_key}"}

# Base API URL
base_url = "https://api.collegefootballdata.com"

# Years to pull
years = [2022, 2023, 2024]

# --------------------
# Pull Play-by-Play Data (filtered for FBS)
# --------------------
all_plays = []
for year in tqdm(years, desc="Pulling plays by year"):
    week_range = range(1, 9) if year == 2024 else range(1, 16)
    for week in tqdm(week_range, desc=f"Year {year} weeks", leave=False):
        params = {
            "year": year,
            "week": week,
            "seasonType": "both",
            "classification": "fbs"  # Filter for FBS games
        }
        try:
            response = requests.get(f"{base_url}/plays", params=params, headers=headers)
            response.raise_for_status()
            plays_data = response.json()
            if plays_data:
                # Add year and week to each play
                for play in plays_data:
                    play['year'] = year
                    play['week'] = week
                all_plays.extend(plays_data)
        except requests.exceptions.RequestException as e:
            print(f"Error pulling plays for {year} week {week}: {e}")
            continue

# Save raw plays
if all_plays:
    plays_df = pd.DataFrame(all_plays)
    plays_df.to_csv('raw_plays_2022_2024.csv', index=False)
    print(f"Saved {len(plays_df)} raw plays to raw_plays_2022_2024.csv")
else:
    print("No play data pulled.")
    plays_df = pd.DataFrame()

# --------------------
# Pull Head Coaches Data
# --------------------
all_coaches = []
for year in tqdm(years, desc="Pulling coaches by year"):
    params = {"year": year}
    try:
        response = requests.get(f"{base_url}/coaches", params=params, headers=headers)
        response.raise_for_status()
        coaches_data = response.json()
        if coaches_data:
            for coach in coaches_data:
                coach['year'] = year
            all_coaches.extend(coaches_data)
    except requests.exceptions.RequestException as e:
        print(f"Error pulling coaches for {year}: {e}")
        continue

# Save raw coaches
if all_coaches:
    coaches_df = pd.DataFrame(all_coaches)
    coaches_df.to_csv('raw_coaches_2022_2024.csv', index=False)
    print(f"Saved {len(coaches_df)} raw coach records to raw_coaches_2022_2024.csv")
else:
    print("No coaches data pulled.")