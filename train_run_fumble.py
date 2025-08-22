# train_run_fumble.py
import pandas as pd, numpy as np, xgboost as xgb, joblib, json, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------- Load ----------
df = pd.read_csv("ml_run_fumbles_2022_2024.csv")

# ---------- Features ----------
num_features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
cat_features = ['rusher_name']   # <-- use ball carrier

target = 'fumble_lost'

# Only drop NaNs on numeric features + target; fill categoricals later
df = df.dropna(subset=num_features + [target]).copy()

# Fill & cast categoricals
for c in cat_features:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown").astype(str)
    else:
        df[c] = "Unknown"

y_all = df[target].astype(int).values
X_all = df[num_features + cat_features]

# ---------- Time-safe split if available, else random ----------
def time_safe_split(df_like, X, y):
    if 'year' in df_like.columns:
        train_mask = df_like['year'].isin([2022, 2023])
        test_mask  = df_like['year'].isin([2024])
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            return (X[train_mask], X[test_mask], y[train_mask], y[test_mask], True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return (X_tr, X_te, y_tr, y_te, False)

X_train_raw, X_test_raw, y_train, y_test, used_time_split = time_safe_split(df, X_all, y_all)

# Validation split from TRAIN
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# ---------- Preprocessor (OHE name, passthrough numerics) ----------
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_features),
        ('num', 'passthrough', num_features),
    ],
    remainder='drop'
)
pre.fit(X_tr_raw)
joblib.dump(pre, "run_fumble_preprocessor.joblib")

def transform(Xdf):
    Xc = pre.transform(Xdf)
    return Xc if sp.isspmatrix_csr(Xc) else Xc.tocsr()

X_tr   = transform(X_tr_raw)
X_val  = transform(X_val_raw)
X_test = transform(X_test_raw)

# ---------- Class weights (rare-event) ----------
classes = np.unique(y_tr)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
cw_map = {int(c): float(w) for c, w in zip(classes, cw)}
sw_tr  = np.vectorize(cw_map.get)(y_tr).astype(float)
sw_val = np.vectorize(cw_map.get)(y_val).astype(float)

# ---------- XGBoost ----------
dtrain = xgb.DMatrix(X_tr,  label=y_tr,  weight=sw_tr)
dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
dtest  = xgb.DMatrix(X_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
}

bst = xgb.train(
    params, dtrain, num_boost_round=1500,
    evals=[(dtrain,"train"), (dval,"val")],
    early_stopping_rounds=100, verbose_eval=False
)

# ---------- Evaluate ----------
proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
print("Used time split:", used_time_split)
print("Prevalence (test):", round(float(y_test.mean()), 6))
print("AUC", round(roc_auc_score(y_test, proba), 4))
print("AP (PR-AUC)", round(average_precision_score(y_test, proba), 4))
print("Test LogLoss", round(log_loss(y_test, proba), 4))

# ---------- Save ----------
bst.save_model("run_fumble.json")
with open("run_fumble_meta.json", "w") as f:
    json.dump({"best_iteration": int(bst.best_iteration)}, f)
print("Saved run_fumble.json and run_fumble_preprocessor.joblib")
