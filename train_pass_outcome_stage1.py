# train_pass_outcome_stage1.py
import pandas as pd, numpy as np, xgboost as xgb, json, joblib, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------- Load ----------
df = pd.read_csv("ml_pass_outcomes_2022_2024.csv")

# ---------- Features ----------
num_features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
cat_features = ['passer_name']   # <-- use QB, not head coach
all_features = num_features + cat_features
target = 'pass_outcome'

# keep rows with all needed columns & target
df = df.dropna(subset=all_features + [target]).copy()
df['passer_name'] = df['passer_name'].astype(str)

# Binary label: complete vs not-complete
y_all = (df[target] == 'complete').astype(int).values
X_all = df[all_features]

# ---------- Train / test split (time-safe if possible) ----------
if 'year' in df.columns:
    train_mask = df['year'].isin([2022, 2023])
    test_mask  = df['year'].isin([2024])
    X_train_raw, y_train = X_all[train_mask], y_all[train_mask]
    X_test_raw,  y_test  = X_all[test_mask],  y_all[test_mask]
else:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# ---------- Preprocess (OHE passer_name) ----------
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_features),
        ('num', 'passthrough', num_features),
    ],
    remainder='drop'
)

pre.fit(X_tr_raw)  # fit ONLY on training fold
# Save preprocessor for inference
joblib.dump(pre, "pass_stage1_preprocessor.joblib")

def transform(Xdf):
    Xc = pre.transform(Xdf)  # scipy sparse
    # Ensure CSR for xgboost
    return Xc if sp.isspmatrix_csr(Xc) else Xc.tocsr()

X_tr  = transform(X_tr_raw)
X_val = transform(X_val_raw)
X_test = transform(X_test_raw)

# ---------- Class weights (make NOT-complete heavier) ----------
pos_ratio = y_tr.mean()  # P(complete)
w_pos = 1.0
# weight negative (not-complete) higher if positives dominate
w_neg = max(1.0, (w_pos * pos_ratio) / max(1e-6, (1 - pos_ratio)))
# invert since we want more weight on the minority class
w_neg = max(1.0, 1.0 / w_neg)

sw_tr  = np.where(y_tr  == 1, w_pos, w_neg).astype(float)
sw_val = np.where(y_val == 1, w_pos, w_neg).astype(float)

# ---------- XGBoost (DMatrix over sparse) ----------
dtrain = xgb.DMatrix(X_tr,  label=y_tr,  weight=sw_tr)
dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
dtest  = xgb.DMatrix(X_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 7,
    "eta": 0.06,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2500,
    evals=[(dtrain,"train"),(dval,"val")],
    early_stopping_rounds=120,
    verbose_eval=False
)

# ---------- Evaluate ----------
p = bst.predict(dtest, iteration_range=(0, bst.best_iteration+1))
y_pred = (p >= 0.5).astype(int)

print("Stage1 Acc:", round(accuracy_score(y_test, y_pred), 4))
print("Stage1 AUC:", round(roc_auc_score(y_test, p), 4))
print("Stage1 LogLoss:", round(log_loss(y_test, p), 4))

# ---------- Save ----------
bst.save_model("pass_stage1_complete_vs_not.json")
with open("pass_stage1_meta.json","w") as f:
    json.dump({"best_iteration": int(bst.best_iteration)}, f)
print("Saved pass_stage1_complete_vs_not.json and pass_stage1_preprocessor.joblib")
