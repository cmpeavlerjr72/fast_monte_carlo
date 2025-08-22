# sim_store.py
import json, hashlib, os
from pathlib import Path
import pandas as pd

def make_signature(meta: dict) -> str:
    s = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()

def save_sim_bundle(run_dir: str, games_df: pd.DataFrame, players_df: pd.DataFrame, meta: dict):
    p = Path(run_dir); p.mkdir(parents=True, exist_ok=True)
    # thin columns for speed/size
    games_keep = ["sim_id","team","opp","pts","opp_pts","margin","total","seed"]
    if "margin" not in games_df: games_df = games_df.assign(margin=games_df.pts - games_df.opp_pts)
    if "total"  not in games_df: games_df  = games_df.assign(total =games_df.pts + games_df.opp_pts)
    games_df[games_keep].to_parquet(p/"games.parquet", index=False)

    players_df.to_parquet(p/"players.parquet", index=False)
    (p/"meta.json").write_text(json.dumps(meta, indent=2))

def load_sim_bundle(run_dir: str):
    p = Path(run_dir)
    games   = pd.read_parquet(p/"games.parquet")
    players = pd.read_parquet(p/"players.parquet")
    meta    = json.loads((p/"meta.json").read_text())
    return games, players, meta
