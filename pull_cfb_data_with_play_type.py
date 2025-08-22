import os
import requests
import pandas as pd
import re
from tqdm import tqdm
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load .env file
load_dotenv()

# Set up API key
api_key = os.environ.get('CFBD_API_KEY')
if not api_key:
    raise ValueError("CFBD_API_KEY not found in environment variables. Ensure it's set in .env file and retry.")
headers = {"Authorization": f"Bearer {api_key}"}

# Base API URL
base_url = "https://api.collegefootballdata.com"

# Years to pull (2022-2024 as requested)
years = [2022, 2023, 2024]

# --------------------
# Function to Parse Player Names Using play_type
# --------------------
def parse_play_text(play_text, play_type):
    """
    Extract player names from play_text based on play_type.
    Returns dict with passer_name, receiver_name, rusher_name, other_name (None if not applicable).
    """
    if not play_text or pd.isna(play_text) or not play_type or pd.isna(play_type):
        return {"passer_name": None, "receiver_name": None, "rusher_name": None, "other_name": None}
    
    play_text = str(play_text).lower().strip()
    play_type = str(play_type).lower().strip()
    result = {"passer_name": None, "receiver_name": None, "rusher_name": None, "other_name": None}
    
    # Define patterns
    pass_pattern = r"(\w+\.?\s+\w+)\s+pass\s+(?:complete\s+to\s+(\w+\.?\s+\w+)|incomplete)"
    rush_pattern = r"(\w+\.?\s+\w+)\s+(?:run|rush)\s+for"
    sack_pattern = r"(\w+\.?\s+\w+)\s+sacked\s+by\s+(\w+\.?\s+\w+)"
    fumble_pattern = r"(\w+\.?\s+\w+)\s+fumbled,\s+recovered\s+by\s+\w+\s+(\w+\.?\s+\w+)"
    
    # Match based on play_type
    if "pass completion" in play_type or "pass reception" in play_type:
        match = re.search(pass_pattern, play_text)
        if match:
            result["passer_name"] = match.group(1).title() if match.group(1) else None
            result["receiver_name"] = match.group(2).title() if match.group(2) else None
    elif "pass incompletion" in play_type:
        match = re.search(pass_pattern, play_text)
        if match:
            result["passer_name"] = match.group(1).title() if match.group(1) else None
    elif "rush" in play_type:
        match = re.search(rush_pattern, play_text)
        if match:
            result["rusher_name"] = match.group(1).title() if match.group(1) else None
    elif "sack" in play_type:
        match = re.search(sack_pattern, play_text)
        if match:
            result["passer_name"] = match.group(1).title() if match.group(1) else None
            result["other_name"] = match.group(2).title() if match.group(2) else None  # Tackler
    elif "fumble" in play_type:
        match = re.search(fumble_pattern, play_text)
        if match:
            result["rusher_name"] = match.group(1).title() if match.group(1) else None  # Assume rusher fumbled
            result["other_name"] = match.group(2).title() if match.group(2) else None  # Recoverer
    
    return result

# --------------------
# Pull Play-by-Play Data
# --------------------
all_plays = []
for year in tqdm(years, desc="Pulling plays by year"):
    # For 2024, limit to weeks 1-8 (per SP+ data up to week 8)
    week_range = range(1, 9) if year == 2024 else range(1, 16)
    for week in tqdm(week_range, desc=f"Year {year} weeks", leave=False):
        params = {"year": year, "week": week, "seasonType": "both"}
        try:
            response = requests.get(f"{base_url}/plays", params=params, headers=headers)
            response.raise_for_status()
            plays_data = response.json()
            if plays_data:
                all_plays.extend(plays_data)
        except requests.exceptions.RequestException as e:
            print(f"Error pulling plays for {year} week {week}: {e}")
            continue

# Convert to DataFrame and parse player names
if all_plays:
    plays_df = pd.DataFrame(all_plays)
    # Apply parsing using play_type
    parsed_players = plays_df.apply(
        lambda row: parse_play_text(row['play_text'], row['play_type']), axis=1
    )
    plays_df[['passer_name', 'receiver_name', 'rusher_name', 'other_name']] = pd.DataFrame(
        parsed_players.tolist(), index=plays_df.index
    )
    print(f"Pulled {len(plays_df)} plays")
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

# Process coaches to get primary coach per team-year
if all_coaches:
    coaches_df = pd.DataFrame(all_coaches)
    seasons_data = []
    for _, row in coaches_df.iterrows():
        for season in row['seasons']:
            if season['year'] in years:
                seasons_data.append({
                    'coach_name': f"{row['first_name']} {row['last_name']}",
                    'year': season['year'],
                    'team': season['school'],
                    'games': season['games']
                })
    seasons_df = pd.DataFrame(seasons_data)
    primary_coaches = seasons_df.groupby(['team', 'year']).apply(
        lambda x: x.loc[x['games'].idxmax()]
    ).reset_index(drop=True)[['team', 'year', 'coach_name']]
    coaches_df.to_csv('coaches_2022_2024.csv', index=False)
    print(f"Saved {len(coaches_df)} coach records to coaches_2022_2024.csv")
else:
    print("No coaches data pulled.")
    primary_coaches = pd.DataFrame()

# --------------------
# Merge Coaches into Plays (for offense; can add defense if needed)
# --------------------
if not plays_df.empty and not primary_coaches.empty:
    plays_df = plays_df.merge(
        primary_coaches,
        left_on=['offense', 'season'],
        right_on=['team', 'year'],
        how='left'
    )
    plays_df = plays_df.drop(columns=['team', 'year'])
    plays_df.rename(columns={'coach_name': 'head_coach'}, inplace=True)
    plays_df.to_csv('plays_2022_2024.csv', index=False)
    print(f"Saved {len(plays_df)} plays with coaches to plays_2022_2024.csv")
elif not plays_df.empty:
    plays_df.to_csv('plays_2022_2024.csv', index=False)
    print(f"Saved {len(plays_df)} plays (no coaches merged) to plays_2022_2024.csv")

# Summary
print("\nDataset Summary:")
print(f"Plays: {len(plays_df)} rows")
print(f"Columns: {list(plays_df.columns)}")
if not plays_df.empty:
    print(f"Sample play with parsed players and coach:")
    sample = plays_df[
        plays_df[['passer_name', 'receiver_name', 'rusher_name', 'other_name']].notna().any(axis=1)
    ].iloc[0][['play_type', 'play_text', 'passer_name', 'receiver_name', 'rusher_name', 'other_name', 'head_coach']]
    print(sample)