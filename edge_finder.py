from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import re
import math
import ast
from pathlib import Path

# ---------- file discovery ----------

STAT_ALIASES = {
    "pass_yards": "pass_yds",
    "rush_yards": "rush_yds",
    "rec_yards":  "rec_yds",
    # add more if you ever expand: e.g., "pass_tds": "pass_td"
}

def _norm_team(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _find_saved_csvs(
    csv_base: Optional[str] = None,
    teamA: Optional[str] = None,
    teamB: Optional[str] = None,
    directory: str = "."
) -> Tuple[Path, Optional[Path]]:
    """
    Returns (scores_path, players_path_or_None).
    Priority:
      1) If csv_base is provided, expect scores_<csv_base>, players_<csv_base> (CSV or Parquet).
      2) Else, try to find CSV files that contain normalized team names in any order.
    """
    d = Path(directory)

    def _maybe(path_stem: str) -> Optional[Path]:
        # prefer parquet if present (faster/smaller), else csv
        pqt = d / f"{path_stem}.parquet"
        csv = d / f"{path_stem}.csv"
        if pqt.exists(): return pqt
        if csv.exists(): return csv
        return None

    # 1) explicit base
    if csv_base:
        scores = _maybe(f"scores_{csv_base}")
        if scores is None:
            raise FileNotFoundError(f"Could not find scores_{csv_base}.csv or .parquet")
        players = _maybe(f"players_{csv_base}")
        return scores, players

    # 2) fuzzy search by teams
    if teamA is None or teamB is None:
        raise ValueError("Provide either csv_base or both teamA and teamB for fuzzy search.")

    na, nb = _norm_team(teamA), _norm_team(teamB)
    candidates = sorted(d.glob("scores_*.csv")) + sorted(d.glob("scores_*.parquet"))
    for p in candidates:
        low = p.name.lower()
        if na in low and nb in low:
            stem = p.name.rsplit(".", 1)[0]
            players = _maybe(stem.replace("scores_", "players_"))
            return p, players

    raise FileNotFoundError("Could not find matching scores_* for those teams. "
                            "Pass csv_base if you used a custom filename.")

# ---------- odds helpers ----------

def _prob_to_american(p: float) -> int:
    """
    Convert probability to fair American odds (no vig). Clamps to (1e-6, 1-1e-6).
    """
    p = float(np.clip(p, 1e-6, 1-1e-6))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

def _breakeven_for_minus110() -> float:
    # -110 -> 110/(100+110)
    return 110.0 / 210.0

def _ev_per_100_at_minus110(p_win: float, p_push: float = 0.0) -> float:
    """
    EV per $100 risk at -110 (both sides). Push returns stake (EV=0 on pushes).
    Profit on win per $100 risk = 100/110 * 100 = 90.909...
    Loss on lose = -100.
    """
    win_profit = 100.0 * (100.0 / 110.0)
    lose_loss = 100.0
    return p_win * win_profit - (1.0 - p_win - p_push) * lose_loss

def _infer_role_from_stat(stat: str) -> str:
    stat = stat.lower()
    if stat.startswith("pass_"): return "QB"
    if stat.startswith("rush_"): return "Rusher"
    if stat.startswith("rec_") or stat in {"tgt", "rec"}: return "Receiver"
    return "Receiver"

def _american_implied_prob(price: int) -> float:
    return (-price) / ((-price) + 100) if price < 0 else 100 / (price + 100)

def _ev_per_100(p_win: float, price: int) -> float:
    # EV measured per $100 stake at given American price
    if price < 0:
        profit = 100 * (100 / (-price))    # e.g., -110 → $90.91 profit on $100 stake
    else:
        profit = 100 * (price / 100)       # e.g., +120 → $120 profit on $100 stake
    return p_win * profit - (1 - p_win) * 100

def _best_side_ev(p_over: float, price: int = -110) -> dict:
    # Compare EV of Over vs Under at same price, return the better one
    ev_over  = _ev_per_100(p_over,  price)
    ev_under = _ev_per_100(1 - p_over, price)
    ip = _american_implied_prob(price)
    edge_over  = p_over       - ip
    edge_under = (1 - p_over) - ip
    if ev_over >= ev_under:
        return {"side": "Over",  "ev": ev_over,  "edge": edge_over}
    else:
        return {"side": "Under", "ev": ev_under, "edge": edge_under}


# ---------- load helpers ----------

def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

# ---------- public API: player prop ----------

def _load_players_table(players_path: Path) -> pd.DataFrame:
    """
    Reads players_*.csv saved by the sim. Handles both the normal flat CSV and the
    'columns of dict-strings' format (what you currently have).
    """
    df = pd.read_csv(players_path)
    # Detect 'column of dicts' format (columns like "0","1","2", cells that look like "{'sim':...}")
    looks_packed = all(str(c).isdigit() for c in df.columns)
    if looks_packed:
        rows = []
        for c in df.columns:
            for v in df[c].dropna():
                try:
                    d = ast.literal_eval(v)
                    if isinstance(d, dict):
                        rows.append(d)
                except Exception:
                    continue
        if rows:
            df = pd.DataFrame(rows)

    # Normalize column names if needed
    # Ensure expected columns exist; if not, raise a friendly error
    needed = {"team","player","role"}
    if not needed.issubset(set(map(str.lower, df.columns))):
        # Try light casing normalization
        df.columns = [str(c) for c in df.columns]
    # Convert stat columns to numeric where present
    for col in ["pass_yds","rush_yds","rec_yds","pass_td","rush_td","rec_td","tgt","rec","pass_att","pass_comp","sacks"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Add fast lookups
    df["team_low"]   = df["team"].astype(str).str.lower()
    df["player_low"] = df["player"].astype(str).str.lower()
    df.to_csv('testings.csv')
    return df

def player_prop_odds(
    csv_base: Optional[str],
    team: str,
    player: str,
    stat: str,
    line: float,
    *,
    role: Optional[str] = None,
    directory: str = "."
) -> Dict[str, float | int]:
    scores_path, players_path = _find_saved_csvs(csv_base, teamA=team, teamB=None, directory=directory)
    if players_path is None or not players_path.exists():
        raise FileNotFoundError("players CSV not found. Re-run sims with collect_players=True and save_csv set.")

    df = _load_players_table(Path(players_path))
    # map props stat name -> sim column name
    stat_col = STAT_ALIASES.get(stat, stat)
    if stat_col not in df.columns:
        raise ValueError(f"Stat '{stat}' (mapped to '{stat_col}') not present in {players_path.name}.")

    role = role or _infer_role_from_stat(stat_col)
    mask = (
        (df["team_low"] == team.lower()) &
        (df["player_low"] == player.lower()) &
        (df["role"] == role)
    )
    sub = df.loc[mask]
    # print(sub)
    if sub.empty:
        raise ValueError(f"No rows found for {player} on {team} in {players_path.name}.")

    vals = pd.to_numeric(sub[stat_col], errors="coerce").dropna().to_numpy()
    # print(player)
    # print(stat_col)
    # print(vals)
    if vals.size == 0:
        raise ValueError(f"No numeric values for {player} {stat_col} in saved sims.")

    p_over  = float(np.mean(vals > line))
    p_under = float(np.mean(vals < line))
    p_push  = float(np.mean(np.isclose(vals, line, atol=1e-9)))

    rec = _best_side_ev(p_over, price=-110)

    return {
        "team": team,
        "player": player,
        "role": role,
        "stat": stat_col,
        "line": float(line),
        "samples": int(vals.size),
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "push_rate": round(p_push, 4),
        "american_over": _prob_to_american(p_over),
        "american_under": _prob_to_american(p_under),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "best_side": rec["side"],
        "edge": round(rec["edge"] * 100, 2),    # %
        "ev_per_$100": round(rec["ev"], 2),     # dollars
    }

# ---------- public API: spread & total & moneyline ----------

def _wins_from_scores(sims: pd.DataFrame, team: str, opp: str) -> float:
    sub = sims[(sims['team'].astype(str).str.lower() == team.lower()) &
               (sims['opp'].astype(str).str.lower() == opp.lower())]
    if sub.empty:
        return float('nan')
    return float(np.mean(sub['pts'].values > sub['opp_pts'].values))

def _team_scores_from_scores(sims: pd.DataFrame, team: str, opp: str) -> float:
    sub = sims[(sims['team'].astype(str).str.lower() == team.lower()) &
               (sims['opp'].astype(str).str.lower() == opp.lower())]
    if sub.empty:
        return float('nan')
    return float(np.mean(sub['pts'].values > sub['opp_pts'].values))

def moneyline_from_sims(
    csv_base: Optional[str],
    team: str,
    opp: str,
    directory: str = ".",
) -> Dict[str, Dict[str, float | int]]:
    """
    Fair win probabilities and fair ML odds (no vig) for both teams based on the sims.
    """
    scores_path, _ = _find_saved_csvs(csv_base, teamA=team, teamB=opp, directory=directory)
    sims = _load_table(scores_path)

    p_team = _wins_from_scores(sims, team, opp)
    p_opp  = _wins_from_scores(sims, opp, team)

    # Fallback to complement if one side missing
    if not np.isfinite(p_team) and np.isfinite(p_opp):
        p_team = max(0.0, min(1.0, 1.0 - p_opp))
    if not np.isfinite(p_opp) and np.isfinite(p_team):
        p_opp = max(0.0, min(1.0, 1.0 - p_team))

    return {
        "team": {
            "name": team,
            "p_win": round(p_team, 6),
            "ml_fair": _prob_to_american(p_team),
        },
        "opp": {
            "name": opp,
            "p_win": round(p_opp, 6),
            "ml_fair": _prob_to_american(p_opp),
        }
    }

def game_market_odds(
    csv_base: Optional[str],
    team: str,
    opp: str,
    *,
    spread: Optional[float] = None,
    total: Optional[float] = None,
    directory: str = "."
) -> Dict[str, Dict[str, float | int]]:
    """
    Compute fair probabilities from the scores_* file.
    - spread: line from TEAM's perspective (team -3.5 => spread=-3.5; team +3 => +3.0).
      Cover condition: (margin = pts - opp_pts) > -spread ; push when == -spread.
    - total: game total; Over is (pts + opp_pts) > total.
    """
    scores_path, _ = _find_saved_csvs(csv_base, teamA=team, teamB=opp, directory=directory)
    sims = _load_table(scores_path)

    sims = sims[(sims['team'].astype(str).str.lower() == team.lower()) &
                (sims['opp'].astype(str).str.lower() == opp.lower())]
    if sims.empty:
        raise ValueError("No rows from the TEAM perspective found in scores file.")

    out: Dict[str, Dict[str, float | int]] = {}

    if spread is not None:
        margin = (sims['pts'] - sims['opp_pts']).to_numpy()
        tgt = -float(spread)
        p_cover = float(np.mean(margin > tgt))
        p_notcover = float(np.mean(margin < tgt))
        p_push = float(np.mean(np.isclose(margin, tgt, atol=1e-9)))
        out["spread"] = {
            "team": team, "opp": opp, "spread": float(spread), "samples": int(margin.size),
            "p_cover": round(p_cover, 6), "p_notcover": round(p_notcover, 6), "push_rate": round(p_push, 6),
            "american_cover": _prob_to_american(p_cover), "american_notcover": _prob_to_american(p_notcover),
            "mean_margin": float(np.mean(margin)), "median_margin": float(np.median(margin)),
        }

    if total is not None:
        totals = (sims['pts'] + sims['opp_pts']).to_numpy()
        T = float(total)
        p_over  = float(np.mean(totals > T))
        p_under = float(np.mean(totals < T))
        p_push  = float(np.mean(np.isclose(totals, T, atol=1e-9)))
        out["total"] = {
            "team": team, "opp": opp, "total": float(total), "samples": int(totals.size),
            "p_over": round(p_over, 6), "p_under": round(p_under, 6), "push_rate": round(p_push, 6),
            "american_over": _prob_to_american(p_over), "american_under": _prob_to_american(p_under),
            "mean_total": float(np.mean(totals)), "median_total": float(np.median(totals)),
        }

    if not out:
        raise ValueError("Provide at least one of spread= or total=.")
    return out

# ---------- scanning player props for both teams ----------

def scan_props_for_matchup(
    csv_base: str,
    teamA: str,
    teamB: str,
    prop_sheet_path: str = "2025_week1_players.csv",
    directory: str = ".",
    min_abs_edge_pct: float = 2.0  # show only props with |edge| >= 2%
) -> pd.DataFrame:
    props = pd.read_csv(prop_sheet_path)
    props["team_low"]   = props["team"].astype(str).str.lower()
    props["player_low"] = props["player"].astype(str).str.lower()

    keep = props[props["team_low"].isin({teamA.lower(), teamB.lower()})].copy()
    results = []
    for _, r in keep.iterrows():
        stat_col = STAT_ALIASES.get(str(r["stat"]), str(r["stat"]))
        try:
            out = player_prop_odds(
                csv_base=csv_base,
                team=r["team"],
                player=r["player"],
                stat=stat_col,
                line=float(r["yards"]),
                role=None,
                directory=directory
            )
            results.append({
                "team": r["team"],
                "player": r["player"],
                "stat": stat_col,
                "line": float(r["yards"]),
                "best_side": out["best_side"],
                "p_over": out["p_over"],
                "p_under": out["p_under"],
                "edge_pct": out["edge"],
                "ev_$100": out["ev_per_$100"],
                "mean": out["mean"],
                "median": out["median"],
                "samples": out["samples"],
            })
        except Exception:
            # skip props that don't exist in the sims
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["abs_edge"] = df["edge_pct"].abs()
    df = df.sort_values(["abs_edge","ev_$100"], ascending=[False, False])
    return df[df["abs_edge"] >= min_abs_edge_pct].drop(columns=["abs_edge"])
# ---------- pretty printers ----------

def _odds_str(o: int) -> str:
    return f"{o:+d}"

def print_game_market_readable(res: dict) -> None:
    print("\n— Game Market —")
    if 'spread' in res:
        s = res['spread']
        be = _breakeven_for_minus110()
        ev_cover = _ev_per_100_at_minus110(s["p_cover"], s["push_rate"])
        ev_not   = _ev_per_100_at_minus110(s["p_notcover"], s["push_rate"])
        print(f"Spread: {s['team']} vs {s['opp']}  |  {s['team']} {s['spread']:+.1f}  (samples {s['samples']})")
        print(f"  Cover:     {100*s['p_cover']:.1f}%  fair {_odds_str(s['american_cover'])}  "
              f"edge {(100*(s['p_cover']-be)):.2f}%  EV ${ev_cover:.2f}/$100")
        print(f"  Not cover: {100*s['p_notcover']:.1f}%  fair {_odds_str(s['american_notcover'])}  "
              f"edge {(100*(s['p_notcover']-be)):.2f}%  EV ${ev_not:.2f}/$100")
        print(f"  Push:      {100*s['push_rate']:.1f}%   | mean/median margin {s['mean_margin']:.1f} / {s['median_margin']:.1f}")
    if 'total' in res:
        t = res['total']
        be = _breakeven_for_minus110()
        ev_over  = _ev_per_100_at_minus110(t["p_over"],  t["push_rate"])
        ev_under = _ev_per_100_at_minus110(t["p_under"], t["push_rate"])
        print(f"\nTotal: {t['team']} vs {t['opp']}  |  {t['total']:.1f}  (samples {t['samples']})")
        print(f"  Over:   {100*t['p_over']:.1f}%  fair {_odds_str(t['american_over'])}   "
              f"edge {(100*(t['p_over']-be)):.2f}%  EV ${ev_over:.2f}/$100")
        print(f"  Under:  {100*t['p_under']:.1f}% fair {_odds_str(t['american_under'])}  "
              f"edge {(100*(t['p_under']-be)):.2f}%  EV ${ev_under:.2f}/$100")
        print(f"  Push:   {100*t['push_rate']:.1f}%   | mean/median total {t['mean_total']:.1f} / {t['median_total']:.1f}")

    if 'spread' in res and 'total' in res:
        t = res["total"]
    
        t_avg = t['mean_total']
        t_med = t["median_total"]
        s = res["spread"]
        s_avg = s['mean_margin']
        s_med = s["median_margin"]

        if s_avg > 0:
            team_score_a = ((t_avg - s_avg) / 2) + s_avg
            opp_score_a = (t_avg - s_avg) / 2
        
        if s_avg < 0:
            team_score_a = ((t_avg + s_avg) / 2) - s_avg
            opp_score_a = (t_avg + s_avg) / 2

        if s_med > 0:
            team_score_m = ((t_med - s_med) / 2) + s_med
            opp_score_m = (t_med - s_med) / 2
        
        if s_med < 0:
            team_score_m = ((t_med + s_med) / 2) - s_med
            opp_score_m = (t_med + s_med) / 2

        print(f"\nMean Score: {t['team']} - {team_score_a:.0f} vs {t['opp']} - {opp_score_a:.0f}  |  (samples {t['samples']})")
        print(f"\nMedian Score: {t['team']} - {team_score_m:.0f} vs {t['opp']} - {opp_score_m:.0f}  |  (samples {t['samples']})")



def print_moneyline_readable(ml: dict) -> None:
    print("\n— Fair Moneyline (no vig) —")
    a = ml["team"]; b = ml["opp"]
    print(f"  {a['name']}: {100*a['p_win']:.2f}%   fair {_odds_str(a['ml_fair'])}")
    print(f"  {b['name']}: {100*b['p_win']:.2f}%   fair {_odds_str(b['ml_fair'])}")

def print_prop_table(df: pd.DataFrame) -> None:
    """
    Pretty-print player props grouped by Passing / Rushing / Receiving.
    Works with either schema:
      - columns like ['best_side','best_edge','best_ev','p_over','p_under', ...]
      - or your earlier ['edge_pct','ev_$100', ...]
    Assumes -110 both sides when it has to compute EV as a fallback.
    """
    if df is None or df.empty:
        print("\n— Player Props value (@ -110 both sides) —\n  (no props matched or no value found)")
        return

    out = df.copy()

    # --- helpers ------------------------------------------------------------
    def _grp(stat: str) -> str:
        s = str(stat).lower()
        if s.startswith("pass"):
            return "Passing"
        if s.startswith("rush"):
            return "Rushing"
        if s.startswith("rec") or s in {"tgt", "receptions", "targets"}:
            return "Receiving"
        return "Other"

    def _to_frac(x):
        if pd.isna(x):
            return None
        x = float(x)
        # If someone passed a percent (e.g., 12.3) convert to fraction.
        return x/100.0 if abs(x) > 1.0 else x

    # fair EV per $100 at -110 if we need to compute it: profit on win = $90.91
    def _ev_per_100_from_prob(p):
        return 100.0 * (float(p) * (1/1.1 - 1) - (1 - float(p)))  # = 100*(0.9091*p - (1-p))

    # --- derive unified columns we need -------------------------------------
    # Best side (string): Over/Under
    if "best_side" not in out.columns:
        # pick by larger absolute edge if those columns exist; otherwise by prob
        if {"p_over","p_under"}.issubset(out.columns):
            out["best_side"] = np.where(out["p_over"] >= out["p_under"], "Over", "Under")
        else:
            out["best_side"] = "Over"  # harmless fallback

    # Edge (fraction): -0.20 .. +0.20
    if "best_edge" not in out.columns:
        if "edge_pct" in out.columns:
            out["best_edge"] = out["edge_pct"].apply(_to_frac)
        elif {"edge_over","edge_under"}.issubset(out.columns):
            out["best_edge"] = np.where(
                out["best_side"].str.lower() == "over",
                out["edge_over"].apply(_to_frac),
                out["edge_under"].apply(_to_frac)
            )
        else:
            # Derive from prob if available using -110 threshold 52.38%
            if {"p_over","p_under"}.issubset(out.columns):
                vig_break_even = 1/2.1  # 52.38095%
                over_edge  = out["p_over"]  - vig_break_even
                under_edge = out["p_under"] - vig_break_even
                out["best_edge"] = np.where(
                    out["best_side"].str.lower() == "over", over_edge, under_edge
                )
            else:
                out["best_edge"] = 0.0

    # EV per $100
    if "best_ev" not in out.columns:
        if "ev_$100" in out.columns:
            out["best_ev"] = out["ev_$100"].astype(float)
        elif {"p_over","p_under"}.issubset(out.columns):
            ev_over  = out["p_over"].apply(_ev_per_100_from_prob)
            ev_under = out["p_under"].apply(_ev_per_100_from_prob)
            out["best_ev"] = np.where(
                out["best_side"].str.lower() == "over", ev_over, ev_under
            )
        else:
            out["best_ev"] = 0.0

    # Samples / mean / median safety
    for c in ("samples","mean","median","line"):
        if c not in out.columns:
            out[c] = np.nan

    # Human-friendly group label from stat prefix
    out["__group__"] = out["stat"].map(_grp)

    # Sort strongest edges first within each group
    if "edge_abs" in out.columns:
        out = out.sort_values(["__group__","edge_abs"], ascending=[True, False])
    else:
        out["__edge_abs__"] = out["best_edge"].abs()
        out = out.sort_values(["__group__","__edge_abs__"], ascending=[True, False])

    # --- print --------------------------------------------------------------
    print("\n— Player Props value (@ -110 both sides) —")

    for group_name in ("Passing","Rushing","Receiving","Other"):
        g = out[out["__group__"] == group_name]
        if g.empty:
            continue
        print(f"\n[{group_name}]")
        for _, r in g.iterrows():
            # choose fields robustly
            team   = r.get("team", "")
            player = r.get("player", "")
            stat   = r.get("stat", "")
            side   = r.get("best_side", "Over")
            line   = r.get("line", np.nan)
            edge   = float(_to_frac(r.get("best_edge", 0.0)) or 0.0)
            ev100  = float(r.get("best_ev", 0.0))
            mean   = r.get("mean", np.nan)
            med    = r.get("median", np.nan)
            n      = int(r.get("samples", 0) or 0)

            # number formatting defensively
            line_s = f"{float(line):.1f}" if pd.notna(line) else "—"
            mean_s = f"{float(mean):.1f}" if pd.notna(mean) else "—"
            med_s  = f"{float(med):.1f}"  if pd.notna(med)  else "—"

            print(
                f"{team}: {player}  |  {stat} {side} {line_s}  "
                f"(edge {edge*100:+.2f}%, EV ${ev100:+.2f}/$100, "
                f"mean {mean_s}, med {med_s}, n={n})"
            )

# ---------- one-call convenience ----------

def find_edges(
    *,
    csv_base: Optional[str],
    teamA: str,
    teamB: str,
    spread: Optional[float],
    total: Optional[float],
    props_csv: str = "2025_week1_players.csv",
    directory: str = ".",
    top_props: int = 15,
) -> None:
    """
    One-call helper that prints:
      • Spread & Total edges at -110
      • Fair Moneylines
      • Best player prop edges for both teams (from props_csv)
    """
    gm = game_market_odds(csv_base, teamA, teamB, spread=spread, total=total, directory=directory)
    ml = moneyline_from_sims(csv_base, teamA, teamB, directory=directory)
    props = scan_props_for_matchup(
        csv_base=csv_base,
        teamA=teamA,
        teamB=teamB,
        prop_sheet_path=props_csv,   # <-- this is your "2025_week1_players.csv"
        directory=directory
    )
    print_game_market_readable(gm)
    print_moneyline_readable(ml)
    print_prop_table(props)

def csv_base_from(team_a: str, team_b: str, week: int, ext: str = ".csv") -> str:
    """
    Build a consistent file base: {team_a}_{team_b}_wk{week}.csv
    using the same normalization as _norm_team (lowercase, remove non-alnum).
    """
    return f"{_norm_team(team_a)}_{_norm_team(team_b)}_wk{int(week)}_sims{ext}"

# ---------- example usage ----------

if __name__ == "__main__":
    # Example: you saved as save_csv="ksu_isu_wk1_sims.csv"
    TEAM_A = "Kansas State"
    TEAM_B = "Iowa State"
    WEEK   = 1

    CSV_BASE = csv_base_from(TEAM_A, TEAM_B, WEEK)
    SPREAD = -3      # from TEAM_A perspective (TEAM_A -3.5)
    TOTAL  = 50.5
    PROPS  = f"2025_week{WEEK}_players.csv"  # your earlier CSV with team, player, stat, yards, ...

    find_edges(
        csv_base=CSV_BASE,
        teamA=TEAM_A, teamB=TEAM_B,
        spread=SPREAD, total=TOTAL,
        props_csv=PROPS,
        directory=".",  # where the scores_/players_ files live
        top_props=15
    )
