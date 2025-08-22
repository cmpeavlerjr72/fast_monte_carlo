# train_pass_outcome_stage2.py
import pandas as pd, numpy as np, xgboost as xgb, json, joblib, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

# ---------- Load ----------
df = pd.read_csv("ml_pass_outcomes_2022_2024.csv")

# ---------- Features ----------
num_features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
cat_features = []
if 'passer_name' in df.columns: cat_features.append('passer_name')
if 'target_name' in df.columns: cat_features.append('target_name')  # may be missing on sacks; we fill with "Unknown"

target = 'pass_outcome'

# ---------- Keep only not-complete rows for Stage 2 ----------
# Drop NaNs only on the numeric features + target; fill categoricals later.
df = df.dropna(subset=num_features + [target]).copy()

mask_nc = df[target] != 'complete'
sub = df.loc[mask_nc].copy()

if sub.empty:
    raise ValueError("No 'not complete' rows found after filtering; check your input file.")

# Stringify / fill categoricals so OneHotEncoder can handle them
for c in cat_features:
    sub[c] = sub[c].fillna("Unknown").astype(str)

# Build X/y
all_features = num_features + cat_features
X_all = sub[all_features]
le = LabelEncoder()
y_all = le.fit_transform(sub[target].values)   # classes (alphabetical)

# ---------- Train / test split (time-safe if possible, else random) ----------
def time_safe_split(df_like, X, y):
    if 'year' in df_like.columns:
        train_mask = df_like['year'].isin([2022, 2023])
        test_mask  = df_like['year'].isin([2024])
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            return (X[train_mask], X[test_mask], y[train_mask], y[test_mask], True)
    # fallback: random split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return (X_tr, X_te, y_tr, y_te, False)

X_train_raw, X_test_raw, y_train, y_test, used_time_split = time_safe_split(sub, X_all, y_all)

# validation split from TRAIN
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# ---------- Preprocess (OHE players) ----------
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_features),
        ('num', 'passthrough', num_features),
    ],
    remainder='drop'
)
pre.fit(X_tr_raw)                         # fit ONLY on train
joblib.dump(pre, "pass_stage2_preprocessor.joblib")

def transform(Xdf):
    Xc = pre.transform(Xdf)
    return Xc if sp.isspmatrix_csr(Xc) else Xc.tocsr()

X_tr   = transform(X_tr_raw)
X_val  = transform(X_val_raw)
X_test = transform(X_test_raw)

# ---------- Class weights (strong; no softening) ----------
classes = np.unique(y_tr)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
cw_map = {c: float(w) for c, w in zip(classes, cw)}
sw_tr  = np.vectorize(cw_map.get)(y_tr).astype(float)
sw_val = np.vectorize(cw_map.get)(y_val).astype(float)

# ---------- XGBoost (multiclass) ----------
dtrain = xgb.DMatrix(X_tr,  label=y_tr,  weight=sw_tr)
dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
dtest  = xgb.DMatrix(X_test)

params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(le.classes_),
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
    params, dtrain, num_boost_round=2500,
    evals=[(dtrain,"train"),(dval,"val")],
    early_stopping_rounds=120, verbose_eval=False
)

# ---------- Evaluate ----------
proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration+1))
pred  = proba.argmax(axis=1)

print("Stage2 used time split:", used_time_split)
print("Stage2 Acc:", round(accuracy_score(y_test, pred), 4))
print("Stage2 LogLoss:", round(log_loss(y_test, proba), 4))
print(classification_report(y_test, pred, target_names=le.classes_))
print(confusion_matrix(y_test, pred))

# ---------- Save ----------
bst.save_model("pass_stage2_notcomplete.json")
pd.Series(le.classes_).to_csv("pass_stage2_classes.csv", index=False, header=False)
with open("pass_stage2_meta.json","w") as f:
    json.dump({"best_iteration": int(bst.best_iteration)}, f)

print("Saved pass_stage2_notcomplete.json, pass_stage2_classes.csv, and pass_stage2_preprocessor.joblib")
