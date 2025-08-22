import pandas as pd, numpy as np, joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ---------- Load ----------
df = pd.read_csv("ml_run_yards_2022_2024.csv")

# ---------- Features ----------
num_features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
cat_features = ['rusher_name']  # <-- use ball carrier, not head_coach

# Drop only on numeric features + target; fill names separately
df = df.dropna(subset=num_features + ['yardsGained']).copy()
for c in cat_features:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown").astype(str)
    else:
        df[c] = "Unknown"

y = df['yardsGained'].clip(-10, 40)   # cap outliers for stabler quantiles
X = df[num_features + cat_features]

# ---------- Time-safe split if possible ----------
if 'year' in df.columns:
    train_mask = df['year'].isin([2022, 2023])
    test_mask  = df['year'].isin([2024])
    X_tr_all, y_tr_all = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
else:
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Hold out a validation slice from train
X_tr, X_val, y_tr, y_val = train_test_split(X_tr_all, y_tr_all, test_size=0.1, random_state=42)

# ---------- Preprocessor ----------
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_features),
        ('num', 'passthrough', num_features),
    ],
    remainder='drop'
)

def make_q_model(alpha: float) -> Pipeline:
    return Pipeline(steps=[
        ('pre', pre),
        ('gb', GradientBoostingRegressor(
            loss="quantile", alpha=alpha,
            n_estimators=400, max_depth=3, random_state=42
        ))
    ])

# ---------- Train q10 / q50 / q90 ----------
models = {}
for q in [0.1, 0.5, 0.9]:
    m = make_q_model(q)
    m.fit(X_tr, y_tr)
    val_pred = m.predict(X_val)
    print(f"q{int(q*100)} - val MAE:", round(mean_absolute_error(y_val, val_pred), 3))
    models[q] = m

# ---------- Save ----------
for q, m in models.items():
    joblib.dump(m, f"run_yards_q{int(q*100)}.joblib")
print("Saved run_yards_q10/50/90.joblib")
