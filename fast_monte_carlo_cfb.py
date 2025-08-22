# monte_carlo_cfb.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")


import json, math, random
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
import re
from functools import lru_cache
from multiprocessing import Pool, cpu_count

# ---- speed knobs ----
# Track only the top N players (by share) per team in the per-player stats tables.
# Everyone else is grouped into "Other QB/RB/WR". Set to None to track everyone.
TRACK_PLAYERS_PER_TEAM: Optional[int] = 5   # try 5; set to None to disable
AGGREGATE_OTHERS: bool = True              # if True, group non-tracked into "Other ..."

# -----------------------------
# clock constants
# -----------------------------

T_PASS_C   = 26  # completed pass
T_PASS_INC = 10  # incompletion (clock stops)
T_SACK     = 24
T_RUN      = 28
T_FG       = 12
T_PUNT     = 16

# -----------------------------
# Utilities
# -----------------------------

# ==== Play-call policy: use trained model instead of heuristic ====
USE_PLAY_MODEL = True
_PLAY_BOOSTER = None           # xgboost.Booster
_PLAY_FEATURES = None          # list[str] from features.pkl (order-sensitive)
_PLAY_CLASSES = None           # list[str] from label_encoder.pkl
_PLAY_TEMP = 1.0               # from calibration.json
_PLAY_CACHE = {}

# Optional: team -> head coach string (only used if 'head_coach' is in _PLAY_FEATURES)
# If you don't want to maintain this, we'll pass NA and xgboost will route via the Missing branch.
HEAD_COACH_MAP = {
    "Kansas State": "Chris Klieman",
    "Iowa State": "Matt Campbell",
    "Kansas":"Lance Leipold",
    "Fresno State": "Matt Entz"

}


RNG = np.random.default_rng()

def _round1(x): return float(np.round(x, 1))

def _pass_key(df: pd.DataFrame, with_target: bool) -> tuple:
    r = df.iloc[0]
    # coarse bins to boost cache hits (tune as you like)
    dist_b = round(r['distance'] * 2) / 2.0          # 0.5-yd bins
    ytg_b  = int(round(r['yardsToGoal']))            # 1-yd bins
    sec_b  = int(r['seconds_remaining'] // 30)       # 30s bucket
    key = (
        int(r['down']), dist_b, ytg_b,
        int(r['is_red_zone']), int(r['goal_to_go']),
        int(r['fourth_and_short']), int(r['fg_range']),
        int(r['half']), int(r['two_minute']),
        int(r['offenseTimeouts']), int(r['defenseTimeouts']),
        _round1(r['sp_rating_off']),
        _round1(r['sp_offense_rating_off']),
        _round1(r['sp_defense_rating_def']),
        _round1(r['sp_rating_def']),
        str(r['passer_name']),
        str(r['target_name']) if with_target else "",
        sec_b,
    )
    return key

_PASS1_CACHE = {}
_PASS2_CACHE = {}
_PY_CACHE = {}   # pass yards quantiles
_RY_CACHE = {}   # rush yards quantiles
_SY_CACHE = {}   # sack loss quantiles


def softclip(x, lo, hi): return max(lo, min(hi, x))

def sample_categorical(labels: List[str], probs: List[float]) -> str:
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return labels[RNG.choice(len(labels), p=probs)]

def sec_in_half_left(seconds_remaining: int) -> int:
    # seconds_remaining counts down from 3600; halves are 1800s blocks
    return int(seconds_remaining % 1800)

def red_zone(yards_to_goal: float) -> int:
    return int(yards_to_goal <= 20)

def goal_to_go(distance: float, yards_to_goal: float) -> int:
    return int(distance >= (yards_to_goal - 0.5))

def fourth_and_short(down: int, distance: float) -> int:
    return int(down == 4 and distance <= 2.0)

def fg_range(yards_to_goal: float) -> int:
    # heuristic: within ~50 yard attempt
    return int(yards_to_goal <= 33)

def new_team_stats():
    return dict(
        plays=0,
        pass_att=0, comp=0, pass_yds=0.0, pass_td=0, INT=0, sacks=0,
        rush_att=0, rush_yds=0.0, rush_td=0,
        FG=0, FGA=0, punts=0,
        # NEW diagnostics
        rz_trips=0, rz_TD=0,
        fourth_go=0, fourth_conv=0,
        points=0
    )

def _taper(y: float, t1: float = 22.0, t2: float = 42.0, r1: float = 0.60, r2: float = 0.40) -> float:
    """
    Piecewise linear taper:
      <= t1: unchanged
      t1..t2: slope r1
      > t2:   slope r2
    Tunables: t1/t2 (breakpoints), r1/r2 (post-break slopes).
    """
    if y <= t1:
        return y
    if y <= t2:
        return t1 + (y - t1) * r1
    return t1 + (t2 - t1) * r1 + (y - t2) * r2

# --- Player stat helpers ---
COUNT_SACK_AS_ATT = False  # NCAA-style: sacks are not pass attempts

def _new_p_pass(): return dict(att=0, comp=0, yds=0.0, td=0, INT=0, sacks=0)
def _new_p_rush(): return dict(att=0, yds=0.0, td=0)
def _new_p_rec():  return dict(tgt=0, rec=0, yds=0.0, td=0)

def _ensure_player(pstats, team, role, name):
    """
    pstats[team] = {'pass':{}, 'rush':{}, 'rec':{}}
    role in {'pass','rush','rec'}
    """
    if team not in pstats:
        pstats[team] = {'pass':{}, 'rush':{}, 'rec':{}}
    if role not in pstats[team]:
        pstats[team][role] = {}
    if name not in pstats[team][role]:
        if role == 'pass': pstats[team][role][name] = _new_p_pass()
        elif role == 'rush': pstats[team][role][name] = _new_p_rush()
        else: pstats[team][role][name] = _new_p_rec()




# -----------------------------
# SP+ loader (3 columns per team: RATING, OFFENSE, DEFENSE)
# -----------------------------
SP_PATH = "PregameSPPlus2022_2024_8.csv"
_SP_CACHE = None

def _load_sp(path: str = SP_PATH) -> pd.DataFrame:
    global _SP_CACHE
    if _SP_CACHE is None:
        sp = pd.read_csv(path)
        # normalize columns we need
        keep = ['team','RATING','OFFENSE','DEFENSE','year','week','conference']
        sp = sp[keep].copy()
        sp['team'] = sp['team'].astype(str)
        sp['year'] = sp['year'].astype(int)
        sp['week'] = sp['week'].astype(int)
        _SP_CACHE = sp
    return _SP_CACHE

def _lookup_sp(team: str, year: int, week: int) -> Tuple[float,float,float]:
    """Return (RATING, OFFENSE, DEFENSE) for the latest entry at or before week."""
    sp = _load_sp()
    # case-insensitive match on team
    df = sp[(sp['year'] == year) & (sp['team'].str.lower() == team.lower()) & (sp['week'] <= week)]
    if df.empty:
        # fallback: latest in that year
        df = sp[(sp['year'] == year) & (sp['team'].str.lower() == team.lower())]
        if df.empty:
            raise ValueError(f"SP+ not found for team={team}, year={year}.")
    row = df.sort_values('week').iloc[-1]
    return float(row['RATING']), float(row['OFFENSE']), float(row['DEFENSE'])

def _top_names(df: Optional[pd.DataFrame], name_col: str, k: Optional[int]) -> set:
    if df is None or df.empty:
        return {"Unknown"}
    if k is None:
        return set(df[name_col].astype(str))
    d = df.sort_values('share', ascending=False).head(max(1, int(k)))
    return set(d[name_col].astype(str))

def _build_track_sets(qb: pd.DataFrame, ru: pd.DataFrame, tg: pd.DataFrame) -> Dict[str, set]:
    k = TRACK_PLAYERS_PER_TEAM
    return {
        'pass': _top_names(qb, 'passer_name', k),
        'rush': _top_names(ru, 'rusher_name', k),
        'rec' : _top_names(tg, 'receiver_name', k),
    }

def _maybe_alias(tc: "TeamContext", role: str, name: str) -> str:
    # role in {'pass','rush','rec'}
    if not AGGREGATE_OTHERS or TRACK_PLAYERS_PER_TEAM is None:
        return name
    keep = tc.track_sets.get(role, set())
    if name in keep:
        return name
    return "Other QB" if role == 'pass' else ("Other RB" if role == 'rush' else "Other WR")

def _usage_from_focus_or_fallback(team: str, year: int):
    """
    If the team is present in the FOCUS csv, return (qb_df, ru_df, tg_df, track_sets),
    otherwise fall back to the old per-team usage files.
    """
    if team in _FOCUS_USAGE:
        info = _FOCUS_USAGE[team]
        return (
            info["qb_df"].copy(),
            info["ru_df"].copy(),
            info["tg_df"].copy(),
            info["track_pass"], info["track_rush"], info["track_rec"]
        )

    # === fallback to your original usage files ===
    qb = _load_usage_table("usage_qb_share.csv", team, year, "passer_name")
    ru = _load_usage_table("usage_rush_share.csv", team, year, "rusher_name")
    tg = _load_usage_table("usage_target_share.csv", team, year, "receiver_name")
    if qb is None: qb = pd.DataFrame({"passer_name": ["Unknown"], "share": [1.0]})
    if ru is None: ru = pd.DataFrame({"rusher_name": ["Unknown"], "share": [1.0]})
    if tg is None: tg = pd.DataFrame({"receiver_name": ["Unknown"], "share": [1.0]})
    return qb, ru, tg, set(), set(), set()


# -----------------------------
# Team context & usage
# -----------------------------
@dataclass(slots=True)
class TeamContext:
    name: str
    year: int
    week: int
    # raw SP+
    sp_rating: float
    sp_offense: float
    sp_defense: float
    # usage shares (DataFrames with columns: <role_name>, share)
    qb_share: pd.DataFrame | None = None
    rush_share: pd.DataFrame | None = None
    target_share: pd.DataFrame | None = None
    # which players to actually track per-play (names from your CSV)
    track_pass: set | None = None
    track_rush: set | None = None
    track_rec:  set | None = None

def _ensure_play_feature_dtypes(row: pd.DataFrame, offense_team_name: str) -> pd.DataFrame:
    """
    Returns a fresh 1-row DataFrame containing exactly _PLAY_FEATURES
    with the correct dtypes for XGBoost: numeric -> int/float, head_coach -> category.
    """
    assert _PLAY_FEATURES is not None, "Play policy not loaded."

    # Copy to avoid mutating the reusable row
    out = row.copy()

    # --- attach head coach (string) ---
    coach = HEAD_COACH_MAP.get(offense_team_name, "Unknown")
    out.loc[out.index[0], "head_coach"] = coach

    # --- coerce numeric columns ---
    # everything except 'head_coach' is numeric in _PLAY_FEATURES.
    num_cols = [c for c in _PLAY_FEATURES if c != "head_coach"]
    # force to numeric; anything weird becomes NaN
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    # ints (down/flags/counters) → int64; floats → float64
    int_like = [
        "down","is_red_zone","score_diff","seconds_remaining",
        "offenseTimeouts","defenseTimeouts",
        "goal_to_go","fourth_and_short","fg_range","half","two_minute"
    ]
    float_like = [
        "distance","yardsToGoal","sp_rating_off","sp_offense_rating_off",
        "sp_defense_rating_def","sp_rating_def"
    ]
    for c in int_like:
        if c in out:
            out[c] = out[c].fillna(0).astype("int64")
    for c in float_like:
        if c in out:
            out[c] = out[c].fillna(0.0).astype("float64")

    # --- categorical for head_coach ---
    # If you later save a canonical list of coaches from training, set categories to that list.
    # For now, a single-category works and avoids dtype=object.
    if "head_coach" in _PLAY_FEATURES:
        out["head_coach"] = pd.Categorical([str(coach)])

    # Return only the features the model expects, with correct dtypes
    return out[_PLAY_FEATURES]


def _load_play_policy():
    import os, json
    global USE_PLAY_MODEL, _PLAY_BOOSTER, _PLAY_FEATURES, _PLAY_CLASSES, _PLAY_TEMP
    # >>> ADD THIS GUARD <<<
    if _PLAY_BOOSTER is not None and _PLAY_FEATURES is not None and _PLAY_CLASSES is not None:
        return
    if not os.path.exists("play_model.json"):
        USE_PLAY_MODEL = False
        return
    booster = xgb.Booster(); booster.load_model("play_model.json")
    try: booster.set_param({'nthread': 1})
    except: pass
    _PLAY_FEATURES = joblib.load("features.pkl")
    le = joblib.load("label_encoder.pkl")
    _PLAY_CLASSES = [str(c) for c in le.classes_]
    if os.path.exists("calibration.json"):
        with open("calibration.json","r") as f:
            _PLAY_TEMP = float(json.load(f).get("temperature", 1.0))
    _PLAY_BOOSTER = booster


_load_play_policy()# ==== Play-call policy: binary PASS/RUN model ====

def _play_state_key(df: pd.DataFrame) -> tuple:
    r = df.iloc[0]
    dist_b = round(float(r['distance']) * 2) / 2.0
    ytg_b  = int(round(float(r['yardsToGoal'])))
    sec_b  = int(int(r['seconds_remaining']) // 30)
    return (
        int(r['down']), dist_b, ytg_b,
        int(r['is_red_zone']), int(r['goal_to_go']),
        int(r['fourth_and_short']), int(r['fg_range']),
        int(r['half']), int(r['two_minute']),
        int(r['offenseTimeouts']), int(r['defenseTimeouts']),
        _round1(r['sp_rating_off']), _round1(r['sp_offense_rating_off']),
        _round1(r['sp_defense_rating_def']), _round1(r['sp_rating_def']),
        sec_b, int(r['score_diff']),
    )

def _softmax_T(z: np.ndarray, T: float) -> np.ndarray:
    z = z / max(1e-6, T)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

# def play_call_pass_prob_binary(row_full: pd.DataFrame, offense_team_name: str) -> float:
#     """
#     Return P(pass) from the binary model. Falls back to heuristic if model unavailable.
#     """
#     if not USE_PLAY_MODEL or _PLAY_BOOSTER is None or _PLAY_FEATURES is None or _PLAY_CLASSES is None:
#         return pass_prob_v1(
#             int(row_full.at[0, 'down']),
#             float(row_full.at[0, 'distance']),
#             float(row_full.at[0, 'yardsToGoal']),
#             int(row_full.at[0, 'seconds_remaining']),
#             int(row_full.at[0, 'score_diff'])
#         )

#     k = _play_state_key(row_full)
#     hit = _PLAY_CACHE.get(k)
#     if hit is not None:
#         return hit

#     # Build a 1-row frame in the exact training feature order
#     row = pd.DataFrame(columns=_PLAY_FEATURES, dtype=object)
#     for col in _PLAY_FEATURES:
#         if col == 'head_coach':
#             hc = HEAD_COACH_MAP.get(offense_team_name)
#             row.loc[0, col] = hc if hc is not None else np.nan  # unseen coach -> treated as missing
#         else:
#             row.loc[0, col] = row_full.at[0, col]

#     dmat = xgb.DMatrix(row, enable_categorical=True)

#     # For binary training with multi:softprob you still get 2 margins -> softmax
#     margins = _PLAY_BOOSTER.predict(dmat, output_margin=True)  # shape (1, 2)
#     probs = _softmax_T(margins, _PLAY_TEMP)[0]                 # shape (2,)

#     # Map class index to 'pass'
#     classes_low = [c.lower() for c in _PLAY_CLASSES]
#     i_pass = classes_low.index('pass') if 'pass' in classes_low else 0
#     p_pass = float(probs[i_pass])

#     p_pass = softclip(p_pass, 0.02, 0.98)
#     _PLAY_CACHE[k] = p_pass
#     return p_pass

def play_call_pass_prob_binary(row: pd.DataFrame, offense_team_name: str) -> float:
    if _PLAY_BOOSTER is None:  # fallback to heuristic if anything is missing
        return pass_prob_v1(int(row.at[0,'down']), float(row.at[0,'distance']),
                            float(row.at[0,'yardsToGoal']), int(row.at[0,'seconds_remaining']),
                            int(row.at[0,'score_diff']))

    k = _play_state_key(row)
    hit = _PLAY_CACHE.get(k)
    if hit is not None:
        return hit

    feat = _ensure_play_feature_dtypes(row, offense_team_name)
    dmat = xgb.DMatrix(feat, enable_categorical=True)
    margins = _PLAY_BOOSTER.predict(dmat, output_margin=True)
    ez = np.exp((margins/_PLAY_TEMP) - (margins/_PLAY_TEMP).max(axis=1, keepdims=True))
    probs = ez / ez.sum(axis=1, keepdims=True)
    pass_idx = int(np.where(np.array(_PLAY_CLASSES) == "pass")[0][0])
    p = float(probs[0, pass_idx])
    p = softclip(p, 0.02, 0.98)
    _PLAY_CACHE[k] = p
    return p



def matchup_bias(off: TeamContext, deff: TeamContext, k: float = 0.12) -> float:
    # positive when offense > defense
    return k * (off.sp_offense - deff.sp_defense) / 40.0

def yardage_multiplier(off: TeamContext, deff: TeamContext, k: float = 0.10) -> float:
    gap = (off.sp_offense - deff.sp_defense) / 30.0
    return 1.0 + k * math.tanh(gap)


def mismatch_z(off: TeamContext, deff: TeamContext) -> float:
    # Rough standardized gap (40 ≈ big gap)
    return (off.sp_offense - deff.sp_defense) / 40.0

def rz_finish_prob_pass(ytg: float, off: TeamContext, deff: TeamContext, down: int) -> float:
    # Base ~30% at 1st & Goal on the 7 → ~60% on the 1
    base = 0.32 + 0.30 * (max(0.0, 7.0 - ytg) / 7.0)
    base += 0.03 * max(0, 4 - down)           # more downs left ⇒ slightly higher
    tilt = 0.08 * math.tanh((off.sp_offense - deff.sp_defense) / 35.0)
    return float(softclip(base + tilt, 0.22, 0.68))


def rz_finish_prob_run(ytg: float, off: TeamContext, deff: TeamContext, down: int) -> float:
    # Base ~28% at 1st & Goal on the 7 → ~58% on the 1
    base = 0.30 + 0.30 * (max(0.0, 7.0 - ytg) / 7.0)
    base += 0.04 * max(0, 4 - down)
    tilt = 0.07 * math.tanh((off.sp_offense - deff.sp_defense) / 35.0)
    return float(softclip(base + tilt, 0.20, 0.62))





def sack_scale(off: TeamContext, deff: TeamContext) -> float:
    # <1.0 fewer sacks for strong O vs weak D; >1.0 the opposite
    return float(softclip(math.exp(-1.0 * mismatch_z(off, deff)), 0.60, 1.50))

def explosive_prob(off: TeamContext, deff: TeamContext, ytg: float) -> float:
    # Small chance of an explosive when there’s space & mismatch
    base = 0.03 + 0.05 * mismatch_z(off, deff)
    if ytg > 60: base += 0.02
    if ytg > 40: base += 0.01
    return float(softclip(base, 0.01, 0.12))


def redzone_finish_prob(off: TeamContext, deff: TeamContext, down: int, ytg: float):
    # Base ~45% at 1st & Goal on the 5, trending up closer to goal line
    base = 0.45 + 0.10 * max(0, 5 - ytg)   # 0.45..0.95 as ytg goes 5→0
    # More shots left → higher
    downs_left = max(0, 4 - down)
    base += 0.05 * downs_left              # +0..0.15
    # Team strength tilt
    tilt = softclip((off.sp_offense - deff.sp_defense) / 40.0, -0.5, 0.5)
    base += 0.10 * tilt
    return float(softclip(base, 0.30, 0.95))


def _load_usage_table(path: str, team: str, year: int,
                      who_col: str, count_col: str = 'share') -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df = df[(df['offense'] == team) & (df['year'] == year)].copy()
        if df.empty: return None
        col = who_col
        if col not in df.columns:
            return None
        # normalize shares (defensive against rounding issues)
        df = df[[col, 'share']].dropna()
        if df.empty: return None
        s = df['share'].clip(lower=0)
        s = s / s.sum() if s.sum() > 0 else pd.Series([1.0], index=[0])
        df['share'] = s.values
        return df
    except Exception:
        return None


# ---- Focus Players: only track + use the players in a small CSV ----
FOCUS_PLAYERS_CSV = "2025_week1_players.csv"  # change path if needed
OTHER_SENTINEL = "__Other__"

def _build_focus_usage_tables(path: str):
    """
    Expected columns: team, player, pos, usage, stat, yards
    - 'usage' should be the share for that stat bucket. If it's given as a %
      (e.g., 25 for 25%), we'll convert to 0.25.
    - We allow a player (e.g., QB) to appear in multiple stat buckets.
    """
    import pandas as _pd, numpy as _np, os as _os
    if not _os.path.exists(path):
        return {}

    df = _pd.read_csv(path)

    # normalize casing / basic hygiene
    df["team"]   = df["team"].astype(str).str.strip()
    df["player"] = df["player"].astype(str).str.strip()
    df["pos"]    = df["pos"].astype(str).str.upper().str.strip()
    df["stat"]   = df["stat"].astype(str).str.strip().str.lower()

    # coerce usage, drop NaN, clip to nonnegative
    df["usage"]  = _pd.to_numeric(df["usage"], errors="coerce")

    by_team = {}

    def _mk(df2: _pd.DataFrame, name_col: str) -> _pd.DataFrame:
        """
        Build a (name_col, share) frame that sums to 1.0, adding OTHER if needed.
        Handles NaNs, negative values, and percentage → fraction conversion.
        """
        if df2 is None or df2.empty:
            return _pd.DataFrame({name_col: ["Unknown"], "share": [1.0]})

        tmp = df2[["player", "usage"]].copy()
        tmp["usage"] = _pd.to_numeric(tmp["usage"], errors="coerce").fillna(0.0).clip(lower=0.0)

        # If user provided percentages (0–100), convert to 0–1.
        if tmp["usage"].max() > 1.5:
            tmp["usage"] = tmp["usage"] / 100.0

        # Combine duplicates just in case
        tmp = tmp.groupby("player", as_index=False)["usage"].sum()

        s = float(tmp["usage"].sum())

        # If nothing positive, fall back to Unknown
        if not _np.isfinite(s) or s <= 0.0:
            return _pd.DataFrame({name_col: ["Unknown"], "share": [1.0]})

        if s >= 1.0 - 1e-9:
            # normalize to 1.0; no 'Other' remainder
            tmp["share"] = tmp["usage"] / s
        else:
            # leave provided entries as-is, remainder goes to OTHER
            tmp["share"] = tmp["usage"]
            rem = 1.0 - float(tmp["share"].sum())
            if rem > 1e-12:
                tmp.loc[len(tmp)] = {"player": "__Other__", "usage": rem, "share": rem}

        # Rename and clean up
        tmp = tmp.rename(columns={"player": name_col})[[name_col, "share"]]

        # Final renorm (absolute safety) to avoid floating drift
        total = float(tmp["share"].sum())
        if not _np.isfinite(total) or total <= 0.0:
            return _pd.DataFrame({name_col: ["Unknown"], "share": [1.0]})
        tmp["share"] = tmp["share"] / total

        # Non-negative guard
        tmp["share"] = tmp["share"].clip(lower=0.0)

        return tmp

    for team, g in df.groupby("team"):
        # Buckets by stat (players can appear in multiple buckets)
        passing = g[g["stat"] == "pass_yards"][["player","usage"]].copy()
        rushing = g[g["stat"] == "rush_yards"][["player","usage"]].copy()
        rec     = g[g["stat"] == "rec_yards"][["player","usage"]].copy()

        qb_df = _mk(passing, "passer_name")
        ru_df = _mk(rushing, "rusher_name")
        tg_df = _mk(rec,     "receiver_name")

        track_pass = set(passing["player"].astype(str))
        track_rush = set(rushing["player"].astype(str))
        track_rec  = set(rec["player"].astype(str))

        by_team[team] = dict(
            qb_df=qb_df, ru_df=ru_df, tg_df=tg_df,
            track_pass=track_pass, track_rush=track_rush, track_rec=track_rec
        )

    return by_team


# cache (empty dict if file missing)
_FOCUS_USAGE = _build_focus_usage_tables(FOCUS_PLAYERS_CSV)


def build_team_context(team: str, year: int, week: int) -> TeamContext:
    rating, off, deff = _lookup_sp(team, year, week)
    qb, ru, tg, tp, tr, trec = _usage_from_focus_or_fallback(team, year)
    # print(qb)
    # print(ru)
    # print(tg)
    # print(tr)
    # print(trec)
    return TeamContext(
        name=team, year=year, week=week,
        sp_rating=rating, sp_offense=off, sp_defense=deff,
        qb_share=qb, rush_share=ru, target_share=tg,
        track_pass=tp, track_rush=tr, track_rec=trec
    )

# Keep as-is
def sample_qb(tc: TeamContext) -> str:
    df = tc.qb_share
    return str(df.iloc[RNG.choice(len(df), p=df['share'].values)]['passer_name'])

def sample_rusher(tc: TeamContext) -> str:
    df = tc.rush_share
    return str(df.iloc[RNG.choice(len(df), p=df['share'].values)]['rusher_name'])

def sample_target(tc: TeamContext) -> str:
    df = tc.target_share
    return str(df.iloc[RNG.choice(len(df), p=df['share'].values)]['receiver_name'])

# -----------------------------
# Load models (once)
# -----------------------------
# Two-stage pass outcome
PASS1 = xgb.Booster(); PASS1.load_model("pass_stage1_complete_vs_not.json")
PASS2 = xgb.Booster(); PASS2.load_model("pass_stage2_notcomplete.json")
# Keep XGBoost single-threaded inside each process
try:
    PASS1.set_param({'nthread': 1})
    PASS2.set_param({'nthread': 1})
except Exception:
    pass


PASS1_META = joblib.load("pass_stage1_preprocessor.joblib")  # ColumnTransformer pipeline for stage1


PASS2_PRE = joblib.load("pass_stage2_preprocessor.joblib")
PASS2_CLASSES = pd.read_csv("pass_stage2_classes.csv", header=None)[0].tolist()  # order used by model

# Quantile pipelines
PY10 = joblib.load("pass_yards_q10.joblib")
PY50 = joblib.load("pass_yards_q50.joblib")
PY90 = joblib.load("pass_yards_q90.joblib")

RY10 = joblib.load("run_yards_q10.joblib")
RY50 = joblib.load("run_yards_q50.joblib")
RY90 = joblib.load("run_yards_q90.joblib")

SY10 = joblib.load("sack_yards_q10.joblib")
SY50 = joblib.load("sack_yards_q50.joblib")
SY90 = joblib.load("sack_yards_q90.joblib")

# -----------------------------
# Policy (play call) – simple v1 heuristic
# -----------------------------
# -----------------------------
# Outcome models wrappers
# -----------------------------
ST1_FEATURES = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute',
    'passer_name'
]
ST2_FEATURES = ST1_FEATURES[:] + ['target_name']  # same shape; target_name optional for stage2 v2 if you add it

def build_state_row(off: TeamContext, deff: TeamContext, down: int, distance: float,
                    yards_to_goal: float, seconds_remaining: int,
                    offense_timeouts: int, defense_timeouts: int,
                    passer_name: str = "Unknown", target_name: str = "Unknown") -> pd.DataFrame:
    row = {
        'down': down,
        'distance': float(distance),
        'yardsToGoal': float(yards_to_goal),
        'is_red_zone': red_zone(yards_to_goal),
        'score_diff': 0,
        'seconds_remaining': int(seconds_remaining),
        'offenseTimeouts': int(offense_timeouts),
        'defenseTimeouts': int(defense_timeouts),

        # <<< SP+ mapping >>>
        # offense team's numbers populate the "off" fields:
        'sp_rating_off': off.sp_rating,                    # offense team's overall RATING
        'sp_offense_rating_off': off.sp_offense,           # offense team's OFFENSE
        # defense team's numbers populate the "def" fields:
        'sp_defense_rating_def': deff.sp_defense,          # defense team's DEFENSE
        'sp_rating_def': deff.sp_rating,                   # defense team's overall RATING

        'goal_to_go': goal_to_go(distance, yards_to_goal),
        'fourth_and_short': fourth_and_short(down, distance),
        'fg_range': fg_range(yards_to_goal),
        'half': 1 if seconds_remaining > 1800 else 2,
        'two_minute': int(sec_in_half_left(seconds_remaining) <= 120),
        'passer_name': str(passer_name),
        'target_name': str(target_name),
    }
    return pd.DataFrame([row])


# 1) PLAY-CALL POLICY (keep exactly like you have it)
def pass_prob_v1(down, distance, yards_to_goal, seconds_remaining, score_diff):
    base = 0.53  # was 0.48
    if down == 1: base += 0.02 + 0.010 * max(0, distance - 10) / 10
    if down == 2: base += 0.12 + 0.020 * max(0, distance - 7)  / 10
    if down == 3: base += 0.28 + 0.030 * max(0, distance - 5)  / 10
    if down == 4: base += 0.45 + 0.035 * max(0, distance - 3)  / 10
    # red zone leans run a bit
    if yards_to_goal <= 10: base -= 0.05
    if yards_to_goal <=  5: base -= 0.03
    # hurry-up pressure
    two_min = (seconds_remaining % 1800) <= 120
    if two_min and score_diff < 0:
        base += 0.22
    # trailing in last 10:00 of game
    if seconds_remaining < 600 and score_diff < 0:
        base += 0.06
    return float(softclip(base, 0.10, 0.95))



def pass_stage1_proba(dfrow: pd.DataFrame) -> float:
    key = _pass_key(dfrow, with_target=False)
    val = _PASS1_CACHE.get(key)
    if val is not None:
        return val
    X = PASS1_META.transform(dfrow[ST1_FEATURES])
    p = float(PASS1.inplace_predict(X)[0])
    _PASS1_CACHE[key] = p
    return p



def pass_stage2_proba(dfrow: pd.DataFrame) -> Dict[str, float]:
    key = _pass_key(dfrow, with_target=True)
    val = _PASS2_CACHE.get(key)
    if val is None:
        # transform once, predict once
        X = PASS2_PRE.transform(dfrow[ST2_FEATURES])
        raw = PASS2.inplace_predict(X)[0]  # shape (nclass,)
        out = {cls: float(p) for cls, p in zip(PASS2_CLASSES, raw)}
        val = out
        _PASS2_CACHE[key] = val

    # copy + nudge + renorm (unchanged)
    out = dict(val)
    p_inc = max(0.0, out.get("incomplete", 0.0))
    p_int = max(0.0, out.get("intercepted", 0.0))
    p_sck = max(0.0, out.get("sack", 0.0))
    p_sck *= 0.65
    p_int = p_int * 1.20 + 0.004
    s = p_inc + p_int + p_sck or 1.0
    return {"incomplete": p_inc/s, "intercepted": p_int/s, "sack": p_sck/s}

def _pass_yards_key(dfrow: pd.DataFrame) -> tuple:
    # names usually not used by yardage models, but keep if they were
    return _pass_key(dfrow, with_target=True)

def _rush_yards_key(dfrow: pd.DataFrame) -> tuple:
    # same key builder works; passer_name will be "Unknown"
    return _pass_key(dfrow, with_target=False)

def _get_pass_quants(dfrow):
    k = _pass_yards_key(dfrow)
    v = _PY_CACHE.get(k)
    if v is None:
        q10 = float(PY10.predict(dfrow)[0])
        q50 = float(PY50.predict(dfrow)[0])
        q90 = float(PY90.predict(dfrow)[0])
        _PY_CACHE[k] = (q10, q50, q90)
        v = _PY_CACHE[k]
    return v

def _get_rush_quants(dfrow):
    k = _rush_yards_key(dfrow)
    v = _RY_CACHE.get(k)
    if v is None:
        q10 = float(RY10.predict(dfrow)[0])
        q50 = float(RY50.predict(dfrow)[0])
        q90 = float(RY90.predict(dfrow)[0])
        _RY_CACHE[k] = (q10, q50, q90)
        v = _RY_CACHE[k]
    return v

def _get_sack_quants(dfrow):
    # sacks key can reuse without target
    k = _pass_key(dfrow, with_target=False)
    v = _SY_CACHE.get(k)
    if v is None:
        q10 = float(SY10.predict(dfrow)[0])
        q50 = float(SY50.predict(dfrow)[0])
        q90 = float(SY90.predict(dfrow)[0])
        _SY_CACHE[k] = (q10, q50, q90)
        v = _SY_CACHE[k]
    return v




def sample_pass_yards(dfrow: pd.DataFrame) -> float:
    q10, q50, q90 = _get_pass_quants(dfrow)
    ytg = float(dfrow['yardsToGoal'].iloc[0])

    # Slightly dampen when close to the GL (reduces short-field bombs)
    # if ytg < 15:
    #     rz_scale = 0.8 + 0.2 * (ytg / 15.0)   # 0.8 at GL → 1.0 at 15
    #     q10 *= rz_scale; q50 *= rz_scale; q90 *= rz_scale

    sigma = max(0.4, (q90 - q10) / 2.56)
    y = float(RNG.normal(q50, sigma))

    # Taper long gains, then cap by field position
    # y = _taper(y, t1=22.0, t2=42.0, r1=0.60, r2=0.40)
    return softclip(y, 0.0, ytg)


def sample_rush_yards(dfrow: pd.DataFrame) -> float:
    q10, q50, q90 = _get_rush_quants(dfrow)
    ytg = float(dfrow['yardsToGoal'].iloc[0])

    sigma = max(0.35, (q90 - q10) / 2.56)
    y = float(RNG.normal(q50, sigma))

    # Rush taper harsher on the extreme tail
    # y = _taper(y, t1=15.0, t2=30.0, r1=0.60, r2=0.35)
    return softclip(y, -4.0, ytg)




def sample_sack_loss(dfrow: pd.DataFrame) -> float:
    q10, q50, q90 = _get_sack_quants(dfrow)
    sigma = max(0.25, (q90 - q10) / 2.56)
    y = float(RNG.normal(q50, sigma))
    return float(softclip(y, -20.0, 0.0))


# -----------------------------
# Simple special teams v1
# -----------------------------
def field_goal_prob(distance_yd: float) -> float:
    # very rough baseline curve
    # 20-29: 0.95; 30-39: 0.90; 40-49: 0.75; 50-55: 0.45; >55: 0.2
    if distance_yd < 30: return 0.96
    if distance_yd < 40: return 0.92
    if distance_yd < 50: return 0.78
    if distance_yd <= 55: return 0.50
    return 0.25

def attempt_fg(yards_to_goal: float) -> Tuple[bool, int]:
    # NCAA FG distance approx: yards_to_goal + 17 (end zone 10 + 7 snap)
    dist = yards_to_goal + 17
    p = field_goal_prob(dist)
    good = RNG.random() < p
    time = 6

    return good, time

def attempt_punt(yards_to_goal: float) -> Tuple[int, int]:
    """
    Returns (net_yards, time). Models touchbacks when punting near midfield.
    """
    # Base gross and return → net
    gross = max(30.0, float(RNG.normal(43.0, 6.0)))        # gross distance
    ret   = max(0.0, float(RNG.normal(6.0, 3.0)))          # return yards
    net   = gross - ret

    # If punting from near/inside opponent half, chance of touchback goes up
    # ytg <= 60 means LOS is at opp 40 or closer.
    if yards_to_goal <= 60:
        tb_prob = softclip((60.0 - yards_to_goal) / 60.0, 0.10, 0.55)  # 10–55%
        if RNG.random() < tb_prob:
            # Choose a TB: set net so new offense starts at own 25 (ytg=75)
            net = yards_to_goal - 25.0

    # Clamp reasonable limits
    net = float(softclip(net, 15.0, yards_to_goal - 1.0))
    time = 8
    return int(net), time


# -----------------------------
# Game state
# -----------------------------

@dataclass(slots=True)
class GameState:
    offense: TeamContext
    defense: TeamContext
    seconds_remaining: int = 60*60
    down: int = 1
    distance: float = 10.0
    yards_to_goal: float = 75.0
    offense_timeouts: int = 3
    defense_timeouts: int = 3
    period: int = 1
    drive_id: int = 1
    # team-keyed scores (do NOT swap on possession)
    scores: Dict[str, int] = field(default_factory=dict)
    # NEW flags
    in_rz_this_drive: bool = False
    going_for_it: bool = False

    def init_scores(self):
        if not self.scores:
            self.scores = {self.offense.name: 0, self.defense.name: 0}



def first_down_reset(gs: GameState, gained: float):
    gs.down = 1
    gs.distance = 10.0
    gs.yards_to_goal = max(0.0, gs.yards_to_goal - gained)

def advance_down(gs: GameState, gained: float):
    gs.yards_to_goal = max(0.0, gs.yards_to_goal - gained)
    if gained + 1e-6 >= gs.distance:  # first down
        first_down_reset(gs, 0)
    else:
        gs.down += 1
        gs.distance -= gained
        if gs.down > 4:
            # turnover on downs
            change_possession(gs, reason="downs")

def change_possession(gs: GameState, reason="normal", spot_overwrite: Optional[float] = None):
    gs.offense, gs.defense = gs.defense, gs.offense
    gs.down = 1
    gs.distance = 10.0
    gs.drive_id += 1
    gs.in_rz_this_drive = False
    gs.going_for_it = False
    if spot_overwrite is not None:
        gs.yards_to_goal = spot_overwrite
    else:
        gs.yards_to_goal = 100.0 - gs.yards_to_goal


def tick_clock(gs: GameState, base: int):
    gs.seconds_remaining = max(0, gs.seconds_remaining - base)
    old_period = gs.period
    # periods 1..4
    gs.period = 4 - ((gs.seconds_remaining - 1) // 900) if gs.seconds_remaining > 0 else 4

    if gs.period != old_period:
        if gs.period == 3:
            # reset timeouts
            gs.offense_timeouts = 3
            gs.defense_timeouts = 3
            # halftime kickoff to the team that did NOT get the opening kickoff
            change_possession(gs, reason="halftime", spot_overwrite=75.0)


# -----------------------------
# One play simulation
# -----------------------------

# per-process reusable state row
_WORKER_ROW = None

def _get_reusable_row(off: TeamContext, deff: TeamContext) -> pd.DataFrame:
    """Create a 1-row DataFrame once per worker, then mutate in place."""
    global _WORKER_ROW
    if _WORKER_ROW is None:
        _WORKER_ROW = pd.DataFrame([{
            'down': 1, 'distance': 10.0, 'yardsToGoal': 75.0,
            'is_red_zone': 0, 'score_diff': 0, 'seconds_remaining': 3600,
            'offenseTimeouts': 3, 'defenseTimeouts': 3,
            'sp_rating_off': off.sp_rating,
            'sp_offense_rating_off': off.sp_offense,
            'sp_defense_rating_def': deff.sp_defense,
            'sp_rating_def': deff.sp_rating,
            'goal_to_go': 0, 'fourth_and_short': 0, 'fg_range': 0,
            'half': 1, 'two_minute': 0,
            'passer_name': "Unknown", 'target_name': "Unknown"
        }])
    return _WORKER_ROW

def _fill_row(row: pd.DataFrame, off: TeamContext, deff: TeamContext,
              down: int, distance: float, ytg: float, seconds_remaining: int,
              offense_timeouts: int, defense_timeouts: int,
              passer_name: str, target_name: str, score_diff: int):
    # ALWAYS refresh SP+ fields in case possession flipped
    row.at[0, 'sp_rating_off']          = off.sp_rating
    row.at[0, 'sp_offense_rating_off']  = off.sp_offense
    row.at[0, 'sp_defense_rating_def']  = deff.sp_defense
    row.at[0, 'sp_rating_def']          = deff.sp_rating

    # situational fields
    row.at[0, 'down']            = int(down)
    row.at[0, 'distance']        = float(distance)
    row.at[0, 'yardsToGoal']     = float(ytg)
    row.at[0, 'is_red_zone']     = int(ytg <= 20)
    row.at[0, 'score_diff']      = int(score_diff)
    row.at[0, 'seconds_remaining']= int(seconds_remaining)
    row.at[0, 'offenseTimeouts'] = int(offense_timeouts)
    row.at[0, 'defenseTimeouts'] = int(defense_timeouts)
    row.at[0, 'goal_to_go']      = int(distance >= (ytg - 0.5))
    row.at[0, 'fourth_and_short']= int(down == 4 and distance <= 2.0)
    row.at[0, 'fg_range']        = int(ytg <= 33)
    row.at[0, 'half']            = 1 if seconds_remaining > 1800 else 2
    row.at[0, 'two_minute']      = int((seconds_remaining % 1800) <= 120)
    row.at[0, 'passer_name']     = str(passer_name)
    row.at[0, 'target_name']     = str(target_name)




def simulate_play(gs: GameState, stats: Dict[str, dict], score: Dict[str, int], pstats: Dict[str, dict]) -> None:
    if gs.seconds_remaining <= 0:
        return

    team = gs.offense.name
    opp  = gs.defense.name
    dist0 = gs.distance
    ytg0  = gs.yards_to_goal
    was_fourth_go = gs.going_for_it

    # RZ trip detection (first time per drive)
    if not gs.in_rz_this_drive and gs.yards_to_goal <= 20:
        stats[team]["rz_trips"] += 1
        gs.in_rz_this_drive = True

    # policy (ML-based)
    score_diff = score[team] - score[opp]
    row = _get_reusable_row(gs.offense, gs.defense)
    _fill_row(row, gs.offense, gs.defense, gs.down, gs.distance, gs.yards_to_goal,
            gs.seconds_remaining, gs.offense_timeouts, gs.defense_timeouts,
            passer_name="Unknown", target_name="Unknown", score_diff=score_diff)

    p_pass = play_call_pass_prob_binary(row, offense_team_name=gs.offense.name)
    call = sample_categorical(["run", "pass"], [1.0 - p_pass, p_pass])
    stats[team]["plays"] += 1

    # snapshot for 4th-down conversion & RZ-TD logic
    dist0 = gs.distance
    ytg0  = gs.yards_to_goal
    was_fourth_go = gs.going_for_it  # set by handle_fourth when we "go"

    if call == "pass":
        qb = sample_qb(gs.offense)
        wr = sample_target(gs.offense)

        # Only track players that are in the focus list (NEW)
        track_qb = (gs.offense.track_pass and qb in gs.offense.track_pass)
        track_wr = (gs.offense.track_rec  and wr in gs.offense.track_rec)

        # If receiver is synthetic OTHER, feed "Unknown" to the model to avoid OHE issues (NEW)
        wr_for_model = "Unknown" if wr == OTHER_SENTINEL else wr

        if track_qb:
            _ensure_player(pstats, team, 'pass', qb)
        if track_wr:
            _ensure_player(pstats, team, 'rec',  wr)

        # Only credit target count if tracked (NEW)
        if track_wr:
            pstats[team]['rec'][wr]['tgt'] += 1

        # Build features
        row = _get_reusable_row(gs.offense, gs.defense)
        _fill_row(row, gs.offense, gs.defense, gs.down, gs.distance, gs.yards_to_goal,
                gs.seconds_remaining, gs.offense_timeouts, gs.defense_timeouts,
                passer_name=qb, target_name=wr_for_model, score_diff=score_diff)

        row.loc[0, 'score_diff'] = score_diff

        # Stage 1: completion prob + matchup tilt
        p_complete = pass_stage1_proba(row)
        p_complete = softclip(p_complete + matchup_bias(gs.offense, gs.defense), 0.02, 0.98)

        if RNG.random() < p_complete:
            # Completed pass
            yards = sample_pass_yards(row) * yardage_multiplier(gs.offense, gs.defense)

            # Optional red-zone finish boost
            ytg0 = gs.yards_to_goal

            if ytg0 > 25 and RNG.random() < 0.60 * explosive_prob(gs.offense, gs.defense, ytg0):
                # 35–95% boost, bigger when mismatch favors the offense
                yards *= 1.0 + RNG.uniform(0.35, 0.95) * (1.0 + 0.7 * mismatch_z(gs.offense, gs.defense))
                yards = min(yards, ytg0)


            if ytg0 <= 12 and gs.down <= 3 and RNG.random() < rz_finish_prob_pass(ytg0, gs.offense, gs.defense, gs.down):
                yards = ytg0

            # NCAA-style attempts (count only non-sack plays)
            stats[team]["pass_att"] += 1
            # player tracking (only focus names)
            if track_qb:
                pstats[team]['pass'][qb]['att'] += 1

            if yards + 1e-9 >= gs.yards_to_goal:
                # TD pass
                stats[team]["comp"] += 1
                stats[team]["pass_yds"] += gs.yards_to_goal
                stats[team]["pass_td"] += 1
                score[team] += 7
                stats[team]["points"] = score[team]

                # QB and WR credit
                if track_qb:
                    pstats[team]['pass'][qb]['comp'] += 1
                    pstats[team]['pass'][qb]['yds']  += gs.yards_to_goal
                    pstats[team]['pass'][qb]['td']   += 1
                if track_wr:
                    pstats[team]['rec'][wr]['rec']  += 1
                    pstats[team]['rec'][wr]['yds']  += gs.yards_to_goal
                    pstats[team]['rec'][wr]['td']   += 1

                if was_fourth_go:
                    stats[team]["fourth_conv"] += 1
                gs.going_for_it = False
                tick_clock(gs, 20)
                change_possession(gs, reason="kickoff", spot_overwrite=75.0)
                return
            else:
                # Gain but not a TD
                stats[team]["comp"] += 1
                stats[team]["pass_yds"] += yards

                if track_qb:
                    pstats[team]['pass'][qb]['comp'] += 1
                    pstats[team]['pass'][qb]['yds']  += yards
                if track_wr:
                    pstats[team]['rec'][wr]['rec']  += 1
                    pstats[team]['rec'][wr]['yds']  += yards

                if was_fourth_go and (yards + 1e-6 >= dist0):
                    stats[team]["fourth_conv"] += 1
                gs.going_for_it = False
                advance_down(gs, yards)
                tick_clock(gs, T_PASS_C)
                return

        else:
            # Stage 2: sack / INT / incomplete
            p2 = pass_stage2_proba(row)
            outcome = sample_categorical(["incomplete","intercepted","sack"],
                                         [p2["incomplete"], p2["intercepted"], p2["sack"]])

            if outcome == "incomplete":
                # Count an attempt, no completion
                stats[team]["pass_att"] += 1
                if track_qb:
                    pstats[team]['pass'][qb]['att'] += 1
                gs.down += 1
                gs.going_for_it = False
                tick_clock(gs, T_PASS_INC)
                return

            elif outcome == "sack":
                # Sacks do NOT count as attempts by default
                stats[team]["sacks"] += 1
                if track_qb:
                    pstats[team]['pass'][qb]['sacks'] += 1

                loss = -sample_sack_loss(row)
                loss = max(0.0, loss)
                loss = min(loss, 100 - (100 - gs.yards_to_goal))
                gs.yards_to_goal += loss
                gs.distance += loss
                gs.down += 1
                gs.going_for_it = False
                tick_clock(gs, T_SACK)
                return

            else:  # intercepted
                # Count an attempt + INT
                stats[team]["pass_att"] += 1
                stats[team]["INT"] += 1
                if track_qb:
                    pstats[team]['pass'][qb]['att'] += 1
                    pstats[team]['pass'][qb]['INT'] += 1

                ret = float(softclip(RNG.normal(6, 5), 0, gs.yards_to_goal))
                new_ytg_for_new_offense = 100.0 - (gs.yards_to_goal - ret)
                gs.going_for_it = False
                change_possession(gs, reason="INT", spot_overwrite=new_ytg_for_new_offense)
                tick_clock(gs, 12)
                return

    else:  # run

        rb = sample_rusher(gs.offense)
        track_rb = (gs.offense.track_rush and rb in gs.offense.track_rush)

        if track_rb:
            _ensure_player(pstats, team, 'rush', rb)
            pstats[team]['rush'][rb]['att'] += 1
        stats[team]["rush_att"] += 1


        row = _get_reusable_row(gs.offense, gs.defense)
        _fill_row(row, gs.offense, gs.defense, gs.down, gs.distance, gs.yards_to_goal,
                gs.seconds_remaining, gs.offense_timeouts, gs.defense_timeouts,
                passer_name="Unknown", target_name="Unknown", score_diff=score_diff)
        row['rusher_name'] = rb  # leave as-is if your run models read this

        row.loc[0, 'score_diff'] = score_diff

        yards = sample_rush_yards(row) * yardage_multiplier(gs.offense, gs.defense)
        # occasional explosive on runs
        if ytg0 > 25 and RNG.random() < 0.5 * explosive_prob(gs.offense, gs.defense, ytg0):
            yards *= 1.0 + RNG.uniform(0.2, 0.5) * (1.0 + 0.6 * mismatch_z(gs.offense, gs.defense))
            yards = min(yards, ytg0)
        # finish-in-RZ hook
        if ytg0 <= 9 and gs.down <= 3:
            if RNG.random() < rz_finish_prob_run(ytg0, gs.offense, gs.defense, gs.down):
                yards = ytg0

        if yards + 1e-9 >= ytg0:
            stats[team]["rush_yds"] += ytg0
            if track_rb:
                pstats[team]['rush'][rb]['yds'] += gs.yards_to_goal
                pstats[team]['rush'][rb]['td']  += 1
            stats[team]["rush_td"] += 1

            if ytg0 <= 20: stats[team]["rz_TD"] += 1
            score[team] += 7
            stats[team]["points"] = score[team]
            tick_clock(gs, T_RUN)
            change_possession(gs, reason="kickoff", spot_overwrite=75.0)
            if was_fourth_go: stats[team]["fourth_conv"] += 1
            gs.going_for_it = False
            return
        else:
            stats[team]["rush_yds"] += yards
            if track_rb:
                pstats[team]['rush'][rb]['yds'] += yards
            if was_fourth_go and (yards + 1e-6 >= dist0):
                stats[team]["fourth_conv"] += 1
            advance_down(gs, yards)
            tick_clock(gs, T_RUN)
            if not gs.in_rz_this_drive and gs.yards_to_goal <= 20:
                stats[team]["rz_trips"] += 1
                gs.in_rz_this_drive = True
            gs.going_for_it = False
            return
        
PLAYER_COLS = [
    "sim","start","team","opp","player","role",
    "pass_att","pass_comp","pass_yds","pass_td","INT","sacks",
    "rush_att","rush_yds","rush_td",
    "rec","tgt","rec_yds","rec_td"
]

def flatten_player_box_rows(result: dict, sim_id: int, start_flag: str = "") -> List[dict]:
    teams = list(result["box"].keys())
    opp = {teams[0]: teams[1], teams[1]: teams[0]}
    rows = []
    pb = result["players"]
    for team in teams:
        for name, s in pb[team]['pass'].items():
            if name == OTHER_SENTINEL: 
                continue
            rows.append(dict(
                sim=sim_id, start=start_flag, team=team, opp=opp[team], player=name, role="QB",
                pass_att=s['att'], pass_comp=s['comp'], pass_yds=round(s['yds'],1),
                pass_td=s['td'], INT=s['INT'], sacks=s['sacks'],
                rush_att=0, rush_yds=0.0, rush_td=0, rec=0, tgt=0, rec_yds=0.0, rec_td=0
            ))
        for name, s in pb[team]['rush'].items():
            if name == OTHER_SENTINEL:
                continue
            rows.append(dict(
                sim=sim_id, start=start_flag, team=team, opp=opp[team], player=name, role="Rusher",
                pass_att=0, pass_comp=0, pass_yds=0.0, pass_td=0, INT=0, sacks=0,
                rush_att=s['att'], rush_yds=round(s['yds'],1), rush_td=s['td'],
                rec=0, tgt=0, rec_yds=0.0, rec_td=0
            ))
        for name, s in pb[team]['rec'].items():
            if name == OTHER_SENTINEL:
                continue
            rows.append(dict(
                sim=sim_id, start=start_flag, team=team, opp=opp[team], player=name, role="Receiver",
                pass_att=0, pass_comp=0, pass_yds=0.0, pass_td=0, INT=0, sacks=0,
                rush_att=0, rush_yds=0.0, rush_td=0,
                rec=s['rec'], tgt=s['tgt'], rec_yds=round(s['yds'],1), rec_td=s['td']
            ))
    return rows


_WORKER_A = None
_WORKER_B = None
_WORKER_COLLECT = False

def _init_pool(a_ctx: TeamContext, b_ctx: TeamContext, collect: bool):
    # Called once per process
    global _WORKER_A, _WORKER_B, _WORKER_COLLECT
    _WORKER_A = a_ctx
    _WORKER_B = b_ctx
    _WORKER_COLLECT = collect

    _load_play_policy()

    try:
        PASS1.set_param({'nthread': 1})
        PASS2.set_param({'nthread': 1})
    except Exception:
        pass

def _run_pair(i: int):
    r1 = simulate_game(_WORKER_A, _WORKER_B, seed=None)
    r2 = simulate_game(_WORKER_B, _WORKER_A, seed=None)
    if _WORKER_COLLECT:
        p1 = flatten_player_box_rows(r1, sim_id=2*i,   start_flag="A")
        p2 = flatten_player_box_rows(r2, sim_id=2*i+1, start_flag="B")
        return r1, r2, p1, p2
    return r1, r2, None, None



# -----------------------------
# 4th down decision + special teams
# -----------------------------

def go_for_it_prob(ytg: float, dist: float, score_diff: int, seconds_remaining: int) -> float:
    """
    Probability to go on 4th given yards_to_goal (ytg), distance to gain (dist),
    score context, and clock. Returns a number in [0,1].
    """
    # Late game aggression if trailing and < 5 minutes
    if seconds_remaining < 300 and (score_diff < 0):
        # If a FG doesn't tie or take the lead, be very aggressive
        return 0.90 if (ytg > 38) else 0.75

    p = 0.0
    # Buckets by field position (ytg = distance to opponent end zone; smaller = deeper)
    if ytg > 80:            # own 20 or worse
        if dist <= 1:  p = 0.15
        elif dist <= 2: p = 0.05
    elif ytg > 65:          # own 20–35
        if dist <= 1:  p = 0.30
        elif dist <= 2: p = 0.15
    elif ytg > 50:          # own 35–midfield
        if dist <= 1:  p = 0.60
        elif dist <= 2: p = 0.40
        elif dist <= 3: p = 0.20
    elif ytg > 35:          # midfield to opp 35 (no-man’s land)
        if dist <= 1:  p = 0.85
        elif dist <= 2: p = 0.65
        elif dist <= 3: p = 0.40
        elif dist <= 4: p = 0.25
    elif ytg > 20:          # opp 35–20 (fringe FG)
        if dist <= 1:  p = 0.75
        elif dist <= 2: p = 0.50
        elif dist <= 3: p = 0.30
    elif ytg > 10:          # red zone (20–10)
        if dist <= 1:  p = 0.70
        elif dist <= 2: p = 0.45
    else:                   # goal-to-go (≤ 10)
        if dist <= 2:  p = 0.85
        elif dist <= 4: p = 0.40

    # Slight tilt if leading late: be a bit more conservative
    if seconds_remaining < 300 and score_diff > 0:
        p *= 0.85

    return float(softclip(p, 0.0, 1.0))



def handle_fourth(gs: GameState, stats: Dict[str, dict], score: Dict[str, int]) -> bool:
    """Return True if a special-teams play ended the drive; False if we run a normal play (go for it)."""
    if gs.down != 4:
        return False

    team = gs.offense.name
    opp  = gs.defense.name
    ytg  = float(gs.yards_to_goal)
    dist = float(gs.distance)
    score_diff = score[team] - score[opp]

    # 1) Decide whether to go
    p_go = min(1.0, go_for_it_prob(ytg, dist, score_diff, gs.seconds_remaining) * 1.15)
    if RNG.random() < p_go:
            gs.going_for_it = True
            stats[team]["fourth_go"] += 1
            return False  # go for it → run normal play

    # 2) If not going, consider FG (≤ ~55-yard attempt → ytg <= 38)
    if ytg <= 38:
        stats[team]["FGA"] += 1
        good, t = attempt_fg(ytg)
        tick_clock(gs, T_FG)
        if good:
            stats[team]["FG"] += 1
            score[team] += 3
            stats[team]["points"] = score[team]
            change_possession(gs, reason="kickoff", spot_overwrite=75.0)
        else:
            # Miss: opponent takes over at previous LOS
            change_possession(gs, reason="fg_miss", spot_overwrite=100.0 - ytg)
        return True

    # 3) Otherwise punt
    stats[team]["punts"] += 1
    net, t = attempt_punt(ytg)
    tick_clock(gs, T_PUNT)
    new_ytg_for_new_offense = softclip(100.0 - (ytg - net), 1, 99)
    change_possession(gs, reason="punt", spot_overwrite=new_ytg_for_new_offense)
    return True



# -----------------------------
# Full game simulation
# -----------------------------
def simulate_game(team_off: TeamContext, team_def: TeamContext, seed: Optional[int] = None) -> Dict:
    if seed is not None:
        global RNG
        RNG = np.random.default_rng(seed)

    gs = GameState(offense=team_off, defense=team_def)
    # team-scoped stats and points
    stats = {team_off.name: new_team_stats(), team_def.name: new_team_stats()}
    score = {team_off.name: 0, team_def.name: 0}
        # per-player stats
    pstats = {
        team_off.name: {'pass':{}, 'rush':{}, 'rec':{}},
        team_def.name: {'pass':{}, 'rush':{}, 'rec':{}}
    }


    # Start at own 25 after opening KO
    gs.yards_to_goal = 75.0

    while gs.seconds_remaining > 0:
        # handle 4th-down special teams (needs team-scoped stats)
        if handle_fourth(gs, stats, score):
            continue
        simulate_play(gs, stats, score, pstats)
    # stats[team_off.name]["points"] = score[team_off.name]
    # stats[team_def.name]["points"] = score[team_def.name]

    # Build a backward-compatible return plus a boxscore
    return {
        "offense": team_off.name, "defense": team_def.name,
        "off_score": score[team_off.name], "def_score": score[team_def.name],
        "box": {
            team_off.name: stats[team_off.name],
            team_def.name: stats[team_def.name],
        },
        "players": pstats
    }


def simulate_matchup(teamA: TeamContext, teamB: TeamContext, n: int = 100,
                     seed: int | None = None, show_progress: bool = True,
                     collect_players: bool = False,
                     players_csv: Optional[str] = None,
                     processes: Optional[int] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    # default to 4 on your i7 (or min(CPU, 6) if you want)
    if processes is None:
        processes = min(4, max(1, (cpu_count() or 4)))

    rows = []
    player_rows = []



    if processes <= 1:
        iterator = range(n)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, desc=f"{teamA.name} vs {teamB.name} sims", unit="game")
        for i in iterator:
            r1 = simulate_game(teamA, teamB, seed=seed)
            r2 = simulate_game(teamB, teamA, seed=seed)
            rows.append({"team": r1["offense"], "opp": r1["defense"], "pts": r1["off_score"], "opp_pts": r1["def_score"]})
            rows.append({"team": r2["offense"], "opp": r2["defense"], "pts": r2["off_score"], "opp_pts": r2["def_score"]})
            if collect_players:
                player_rows.extend(p1)
                player_rows.extend(p2)
    else:
        cs = max(8, (n // (processes * 8)) or 1)
        with Pool(processes=processes, initializer=_init_pool,
                initargs=(teamA, teamB, collect_players)) as pool:
            it = pool.imap_unordered(_run_pair, range(n), chunksize=cs)
            pbar = tqdm(total=n, desc=f"{teamA.name} vs {teamB.name} sims (x{processes})",
                        unit="game") if show_progress and tqdm is not None else None
            for i, (r1, r2, p1, p2) in enumerate(it):
                rows.append({"team": r1["offense"], "opp": r1["defense"], "pts": r1["off_score"], "opp_pts": r1["def_score"]})
                rows.append({"team": r2["offense"], "opp": r2["defense"], "pts": r2["off_score"], "opp_pts": r2["def_score"]})
                if collect_players:
                    player_rows.append(p1); player_rows.append(p2)
                if pbar: pbar.update(1)
            if pbar: pbar.close()

    sims_df = pd.DataFrame(rows)

    players_df = None
    if collect_players:
        players_df = pd.DataFrame(player_rows) if player_rows else pd.DataFrame(columns=PLAYER_COLS)
        if players_csv:
            if players_csv.lower().endswith(".parquet"):
                players_df.to_parquet(players_csv, index=False)
            else:
                players_df.to_csv(players_csv, index=False)


    return sims_df, players_df



def print_boxscore(result):
    box = result["box"]
    a, b = list(box.keys())

    def line(team):
        s = box[team]
        pts = int(s.get("points", 0))

        att = int(s.get("pass_att", 0))
        comp = int(s.get("comp", 0))
        pass_yds = float(s.get("pass_yds", 0.0))
        pass_td = int(s.get("pass_td", 0))
        itc = int(s.get("INT", 0))
        sacks = int(s.get("sacks", 0))

        rush_att = int(s.get("rush_att", 0))
        rush_yds = float(s.get("rush_yds", 0.0))
        rush_td = int(s.get("rush_td", 0))

        fg = int(s.get("FG", 0))
        fga = int(s.get("FGA", 0))
        punts = int(s.get("punts", 0))

        rz_trips = int(s.get("rz_trips", 0))
        rz_td = int(s.get("rz_TD", 0))
        fourth_go = int(s.get("fourth_go", 0))
        fourth_conv = int(s.get("fourth_conv", 0))

        cmp_pct = (100.0 * comp / att) if att else 0.0
        ypa = (pass_yds / att) if att else 0.0
        ypc = (rush_yds / rush_att) if rush_att else 0.0

        print(
            f"{team}: {pts} pts | "
            f"Pass {comp}/{att} ({cmp_pct:.0f}%) for {pass_yds:.1f} yds (YPA {ypa:.1f}), "
            f"TD {pass_td}, INT {itc}, Sacks {sacks} | "
            f"Rush {rush_att} for {rush_yds:.1f} yds (YPC {ypc:.1f}), TD {rush_td} | "
            f"FG {fg}/{fga}, Punts {punts} | "
            f"RZ {rz_td}/{rz_trips} TD | 4th {fourth_conv}/{fourth_go}"
        )

    for team in (a, b):
        line(team)


# ---------- New: SP+ table helpers for upcoming games ----------
_SP_CACHE_MAP_FLEX: Dict[str, pd.DataFrame] = {}

def _norm_team(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def load_sp_flex(sp_path: str) -> pd.DataFrame:
    """
    Load an SP+ table and normalize to columns: team, RATING, OFFENSE, DEFENSE.
    Supports:
      (A) team,RATING,OFFENSE,DEFENSE,year,week,...
      (B) 'Current SP+' | 'Past SP+' (team names) + 'Rating','Offense Rating','Defense Rating'
    Caches by path.
    """
    global _SP_CACHE_MAP_FLEX
    if sp_path in _SP_CACHE_MAP_FLEX:
        return _SP_CACHE_MAP_FLEX[sp_path]

    raw = pd.read_csv(sp_path)

    # Schema A: already in team / RATING / OFFENSE / DEFENSE
    if {'team','RATING','OFFENSE','DEFENSE'}.issubset(raw.columns):
        sp = raw[['team','RATING','OFFENSE','DEFENSE']].copy()
        sp['team'] = sp['team'].astype(str).str.strip()
        sp['RATING'] = sp['RATING'].astype(float)
        sp['OFFENSE'] = sp['OFFENSE'].astype(float)
        sp['DEFENSE'] = sp['DEFENSE'].astype(float)
    # Schema B: your 2025 file
    elif {'Current SP+','Past SP+','Rating','Offense Rating','Defense Rating'}.issubset(raw.columns):
        a = raw[['Current SP+','Rating','Offense Rating','Defense Rating']].rename(
            columns={'Current SP+':'team','Rating':'RATING','Offense Rating':'OFFENSE','Defense Rating':'DEFENSE'}
        )
        b = raw[['Past SP+','Rating','Offense Rating','Defense Rating']].rename(
            columns={'Past SP+':'team','Rating':'RATING','Offense Rating':'OFFENSE','Defense Rating':'DEFENSE'}
        )
        sp = pd.concat([a, b], ignore_index=True)
        sp = sp.dropna(subset=['team']).copy()
        sp['team'] = sp['team'].astype(str).str.strip()
        # If a team appears in both columns with same numbers, keep first
        sp = sp.drop_duplicates(subset=['team'], keep='first')
        sp['RATING'] = sp['RATING'].astype(float)
        sp['OFFENSE'] = sp['OFFENSE'].astype(float)
        sp['DEFENSE'] = sp['DEFENSE'].astype(float)
    else:
        raise ValueError(
            f"Unrecognized SP+ schema in {sp_path}. "
            f"Expected columns either "
            f"[team,RATING,OFFENSE,DEFENSE,...] or "
            f"['Current SP+','Past SP+','Rating','Offense Rating','Defense Rating']"
        )

    sp['norm_team'] = sp['team'].map(_norm_team)
    _SP_CACHE_MAP_FLEX[sp_path] = sp
    return sp

def lookup_sp_flex(team: str, sp_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Case-insensitive and punctuation-insensitive lookup in normalized 'team'.
    Returns (RATING, OFFENSE, DEFENSE).
    """
    norm = _norm_team(team)
    hit = sp_df[sp_df['norm_team'] == norm]
    if hit.empty:
        # try simple lowercase exact as a fallback
        hit = sp_df[sp_df['team'].str.lower() == team.lower()]
    if hit.empty:
        # final loose fallback: substring contains (guarded)
        cand = sp_df[sp_df['team'].str.lower().str.contains(team.lower(), regex=False)]
        if not cand.empty:
            hit = cand.iloc[:1]
    if hit.empty:
        raise ValueError(f"Team '{team}' not found in provided SP+ table.")

    row = hit.iloc[0]
    return float(row['RATING']), float(row['OFFENSE']), float(row['DEFENSE'])

def build_team_context_from_sp_flex(team: str, year: int, week: int, sp_df: pd.DataFrame) -> TeamContext:
    rating, off, deff = lookup_sp_flex(team, sp_df)
    qb, ru, tg, tp, tr, trec = _usage_from_focus_or_fallback(team, year)
    # print(qb)
    # print(ru)
    # print(tg)
    # print(tr)
    # print(trec)
    return TeamContext(
        name=team, year=year, week=week,
        sp_rating=rating, sp_offense=off, sp_defense=deff,
        qb_share=qb, rush_share=ru, target_share=tg,
        track_pass=tp, track_rush=tr, track_rec=trec
    )

def simulate_upcoming_matchup(teamA: str, teamB: str, *,
                              year: int = 2025, week: int = 1,
                              sp_path: str = "Pregame_SPPlus2025_1.csv",
                              n: int = 1000,
                              show_progress: bool = True,
                              collect_players: bool = True,
                              save_csv: Optional[str] = None,
                              processes: Optional[int] = None):
    import time
    sp_df = load_sp_flex(sp_path)
    A = build_team_context_from_sp_flex(teamA, year, week, sp_df)
    B = build_team_context_from_sp_flex(teamB, year, week, sp_df)

    t0 = time.perf_counter()
    sims_df, players_df = simulate_matchup(
        A, B, n=n, seed=None, show_progress=show_progress,
        collect_players=collect_players, processes=processes
    )
    t1 = time.perf_counter()

    summary = sims_df.groupby("team").agg(
        mean_pts=("pts", "mean"),
        sd_pts=("pts", "std"),
        mean_opp=("opp_pts", "mean"),
        sd_opp=("opp_pts", "std"),
        win_rate=("pts", lambda s: (s.values > sims_df.loc[s.index, "opp_pts"].values).mean()),
    )

    write_time = 0.0
    if save_csv:
        t_w0 = time.perf_counter()
        try:
            if save_csv.lower().endswith(".parquet"):
                sims_df.to_parquet(f"scores_{save_csv}", index=False)
                if players_df is not None:
                    players_df.to_parquet(f"players_{save_csv}", index=False)
            else:
                sims_df.to_csv(f"scores_{save_csv}", index=False)
                if players_df is not None:
                    players_df.to_csv(f"players_{save_csv}", index=False)
        except Exception:
            # Parquet fallback to CSV if pyarrow not installed, etc.
            sims_df.to_csv(f"scores_{save_csv}.csv", index=False)
            if players_df is not None:
                players_df.to_csv(f"players_{save_csv}.csv", index=False)
        write_time = time.perf_counter() - t_w0

    sim_time = t1 - t0
    meta = {
        "sim_time_sec": sim_time,
        "io_time_sec": write_time,
        "total_time_sec": sim_time + write_time,
        "sims": n
    }
    return sims_df, players_df, summary, A, B, meta

def csv_base_from(team_a: str, team_b: str, week: int, ext: str = ".csv") -> str:
    """
    Build a consistent file base: {team_a}_{team_b}_wk{week}.csv
    using the same normalization as _norm_team (lowercase, remove non-alnum).
    """
    return f"{_norm_team(team_a)}_{_norm_team(team_b)}_wk{int(week)}_sims{ext}"


import time

if __name__ == "__main__":

    # games = [['Kansas State','Iowa State'],
    #          ['Kansas', 'Fresno State'],
    #          ['Hawaii','Stanford'],
    #          ['Western Kentucky','Sam Houston']]

    # games = ['Kansas', 'Fresno State']

    # for g in games:


    team1 = "Kansas State"
    team2 = "Iowa State"
    week = 1

    CSV_BASE = csv_base_from(team1,team2,week)
    sims_df, players_df, summary, KSU, ISU, meta = simulate_upcoming_matchup(
        team1, team2,
        year=2025, week=week,
        sp_path="PregameSPPlus2025_1.csv",
        n=500, show_progress=False,
        collect_players=True,
        save_csv=f"{CSV_BASE}.csv",
        processes=4
    )

    pairs = max(1, len(sims_df) // 2)
    per_pair = meta["total_time_sec"]/pairs
    print(f"\nTiming: {meta['total_time_sec']:.2f}s total | {pairs} sims "
        f"| {pairs/meta['total_time_sec']:.2f} sims/sec "
        f"| sim-only {meta['sim_time_sec']:.2f}s | I/O {meta['io_time_sec']:.2f}s "
        f"| {per_pair:.2f}s per sim")






