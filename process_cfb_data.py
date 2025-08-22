import pandas as pd
import re
from tqdm import tqdm
import ast

# --------------------
# Load and Inspect Raw Plays
# --------------------
try:
    plays_df = pd.read_csv('raw_plays_2022_2024.csv')
    print(f"Loaded {len(plays_df)} raw plays")
except FileNotFoundError:
    print("Error: raw_plays_2022_2024.csv not found. Run pull_cfb_data_raw.py first.")
    exit()

# Inspect columns and sample
print("\nPlay Data Columns:", list(plays_df.columns))
print("\nSample Play (first row):")
print(plays_df.iloc[0][['gameId', 'offense', 'defense', 'playType', 'playText', 'year', 'week']])

# Verify required columns
required_cols = ['playType', 'playText', 'offense', 'defense', 'year', 'week']
missing_cols = [col for col in required_cols if col not in plays_df.columns]
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit()

# Check FBS filter (should exclude conferences like Big Sky, FCS)
fbs_conferences = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'Mountain West', 'American Athletic', 
                   'Conference USA', 'Mid-American', 'Sun Belt', 'FBS Independents']
non_fbs_plays = plays_df[~plays_df['offenseConference'].isin(fbs_conferences) | 
                         ~plays_df['defenseConference'].isin(fbs_conferences)]
if not non_fbs_plays.empty:
    print(f"Warning: {len(non_fbs_plays)} plays involve non-FBS conferences:")
    print(non_fbs_plays[['offense', 'offenseConference', 'defense', 'defenseConference']].head())

# --------------------
# Parse Player Names Using playType
# --------------------
def parse_play_text(play_text, play_type):
    if not play_text or pd.isna(play_text) or not play_type or pd.isna(play_type):
        return {"passer_name": None, "receiver_name": None, "rusher_name": None, "other_name": None}
    
    play_text = str(play_text).lower().strip()
    play_type = str(play_type).lower().strip()
    result = {"passer_name": None, "receiver_name": None, "rusher_name": None, "other_name": None}
    
    pass_pattern = r"(\w+\.?\s+\w+)\s+pass\s+(?:complete\s+to\s+(\w+\.?\s+\w+)|incomplete)"
    rush_pattern = r"(\w+\.?\s+\w+)\s+(?:run|rush)\s+for"
    sack_pattern = r"(\w+\.?\s+\w+)\s+sacked\s+by\s+(\w+\.?\s+\w+)"
    fumble_pattern = r"(\w+\.?\s+\w+)\s+fumbled,\s+recovered\s+by\s+\w+\s+(\w+\.?\s+\w+)"
    
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
            result["other_name"] = match.group(2).title() if match.group(2) else None
    elif "fumble" in play_type:
        match = re.search(fumble_pattern, play_text)
        if match:
            result["rusher_name"] = match.group(1).title() if match.group(1) else None
            result["other_name"] = match.group(2).title() if match.group(2) else None
    
    return result

# Apply parsing
parsed_players = plays_df.apply(
    lambda row: parse_play_text(row['playText'], row['playType']), axis=1
)
plays_df[['passer_name', 'receiver_name', 'rusher_name', 'other_name']] = pd.DataFrame(
    parsed_players.tolist(), index=plays_df.index
)
print(f"Parsed player names for {len(plays_df)} plays")

# --------------------
# Process Coaches
# --------------------
try:
    coaches_df = pd.read_csv('raw_coaches_2022_2024.csv')
    print(f"Loaded {len(coaches_df)} raw coach records")
except FileNotFoundError:
    print("Warning: raw_coaches_2022_2024.csv not found. Skipping coach merge.")
    coaches_df = pd.DataFrame()

if not coaches_df.empty:
    seasons_data = []
    for _, row in coaches_df.iterrows():
        seasons = ast.literal_eval(row['seasons'])
        for season in seasons:
            if season['year'] in [2022, 2023, 2024]:
                seasons_data.append({
                    'coach_name': f"{row['firstName']} {row['lastName']}",
                    'year': season['year'],
                    'team': season['school'],
                    'games': season['games']
                })
    seasons_df = pd.DataFrame(seasons_data)
    primary_coaches = seasons_df.groupby(['team', 'year']).apply(
        lambda x: x.loc[x['games'].idxmax()]
    ).reset_index(drop=True)[['team', 'year', 'coach_name']]
    print(f"Processed {len(primary_coaches)} primary coaches")

    plays_df = plays_df.merge(
        primary_coaches,
        left_on=['offense', 'year'],
        right_on=['team', 'year'],
        how='left'
    ).drop(columns=['team']).rename(columns={'coach_name': 'head_coach'})
else:
    print("No coaches merged.")

# --------------------
# Merge SP+ Data
# --------------------
try:
    sp_df = pd.read_csv('PregameSPPlus2022_2024_8.csv')
    print(f"Loaded SP+ data with {len(sp_df)} rows")
except FileNotFoundError:
    print("Warning: PregameSPPlus2022_2024_8.csv not found. Skipping SP+ merge.")
    sp_df = pd.DataFrame()

if not sp_df.empty:
    sp_df = sp_df[['team', 'year', 'week', 'RATING', 'OFFENSE', 'DEFENSE']]
    plays_df = plays_df.merge(
        sp_df.rename(columns={'RATING': 'sp_rating_off', 'OFFENSE': 'sp_offense_rating_off', 'DEFENSE': 'sp_defense_rating_off'}),
        left_on=['offense', 'year', 'week'],
        right_on=['team', 'year', 'week'],
        how='left'
    ).drop(columns=['team'])
    plays_df = plays_df.merge(
        sp_df.rename(columns={'RATING': 'sp_rating_def', 'OFFENSE': 'sp_offense_rating_def', 'DEFENSE': 'sp_defense_rating_def'}),
        left_on=['defense', 'year', 'week'],
        right_on=['team', 'year', 'week'],
        how='left'
    ).drop(columns=['team'])

# Save final dataset
plays_df.to_csv('plays_with_sp_2022_2024.csv', index=False)
print(f"Saved processed dataset to plays_with_sp_2022_2024.csv ({len(plays_df)} rows)")

# Summary
print("\nDataset Summary:")
print(f"Columns: {list(plays_df.columns)}")
print(f"Missing SP+ for offense: {plays_df['sp_rating_off'].isna().sum()} plays ({plays_df['sp_rating_off'].isna().sum() / len(plays_df):.2%})")
print(f"Missing SP+ for defense: {plays_df['sp_rating_def'].isna().sum()} plays ({plays_df['sp_rating_def'].isna().sum() / len(plays_df):.2%})")
print("\nSample play with parsed players, coach, and SP+:")
sample = plays_df[
    plays_df[['passer_name', 'receiver_name', 'rusher_name', 'other_name']].notna().any(axis=1)
].iloc[0][['playType', 'playText', 'passer_name', 'receiver_name', 'rusher_name', 'other_name', 'head_coach', 'sp_rating_off', 'sp_rating_def']]
print(sample)