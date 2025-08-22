import pandas as pd
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import joblib
import warnings
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, brier_score_loss, top_k_accuracy_score
from xgboost.callback import EarlyStopping
import json

# --------------------
# Load Processed Data
# --------------------
try:
    plays_df = pd.read_csv('ml_ready_plays_2022_2024.csv')
    print(f"Loaded {len(plays_df)} plays")
except FileNotFoundError:
    print("Error: plays_with_sp_2022_2024.csv not found. Run process_cfb_data.py first.")
    exit()

# --------------------
# Filter for FBS Plays (both teams have SP+ data)
# --------------------

# Filter to relevant categories (exclude "other")
relevant_categories = ['run', 'pass']
rel_plays_df = plays_df[plays_df['play_category'].isin(relevant_categories)]
print(f"Filtered to {len(plays_df)} plays in relevant categories: {relevant_categories}")

# --------------------
# Save ML-Ready Dataset
# --------------------
rel_plays_df.to_csv('pass_rush_ml_ready_plays_2022_2024.csv', index=False)
print(f"Saved ML-ready dataset to ml_ready_plays_2022_2024.csv ({len(rel_plays_df)} rows)")

# Summary
print("\nML Dataset Summary:")
print(f"Columns: {list(rel_plays_df.columns)}")
print(f"Sample row:")
print(rel_plays_df.iloc[0])
print(f"\nPlay category distribution:")
print(rel_plays_df['play_category'].value_counts())


warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Check XGBoost version
print(f"XGBoost version: {xgb.__version__}")

# --------------------
# Load ML-Ready Data
# --------------------
try:
    df = pd.read_csv('pass_rush_ml_ready_plays_2022_2024.csv')
    print(f"Loaded {len(df)} rows from ml_ready_plays_2022_2024.csv")
except FileNotFoundError:
    print("Error: ml_ready_plays_2022_2024.csv not found. Run clean_for_ml.py first.")
    exit()

# --------------------
# Preprocess Features
# --------------------
# Define features and target
df['head_coach'] = df['head_coach'].astype('category')
features = ['down', 'distance', 'yardsToGoal', 'is_red_zone', 'score_diff', 'seconds_remaining',
            'offenseTimeouts', 'defenseTimeouts', 'sp_rating_off', 'sp_offense_rating_off',
            'sp_defense_rating_def', 'sp_rating_def', 'head_coach']
target = 'play_category'

# Check for missing values
if df[features + [target]].isna().any().any():
    print("Warning: Missing values detected. Ensure clean_for_ml.py handled NaNs.")
    exit()

# Encode categorical target
le = LabelEncoder()
df['play_category_encoded'] = le.fit_transform(df[target])
print(f"Encoded play_category: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-hot encode head_coach
# df = pd.get_dummies(df, columns=['head_coach'], prefix='coach')
# coach_columns = [col for col in df.columns if col.startswith('coach_')]
# features = [f for f in features if f != 'head_coach'] + coach_columns

# Scale numeric features
numeric_features = ['down', 'distance', 'yardsToGoal', 'score_diff', 'seconds_remaining',
                    'offenseTimeouts', 'defenseTimeouts', 'sp_rating_off', 'sp_offense_rating_off',
                    'sp_defense_rating_def', 'sp_rating_def']
# scaler = StandardScaler()
# df[numeric_features] = scaler.fit_transform(df[numeric_features])

# --------------------
# Split Data
# --------------------

# --------------------
# Train XGBoost Model
# --------------------



xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(le.classes_),
    max_depth=6,
    n_estimators=2000,        # big cap
    learning_rate=0.05,       # smaller LR
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    tree_method='hist',
    enable_categorical=True
)

df['goal_to_go'] = (df['distance'] >= df['yardsToGoal'] - 0.5).astype(int)
df['fourth_and_short'] = ((df['down'] == 4) & (df['distance'] <= 2)).astype(int)
# simple FG range heuristic (~50-yd attempt threshold)
df['fg_range'] = (df['yardsToGoal'] <= 33).astype(int)

features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts','sp_rating_off','sp_offense_rating_off',
    'sp_defense_rating_def','sp_rating_def','head_coach',
    'goal_to_go','fourth_and_short','fg_range'
]

X = df[features]
y = df['play_category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")



classes = np.unique(y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# weights from TRAIN ONLY
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
cw_map = {cls: w for cls, w in zip(classes, class_weights)}

# soften weights to avoid overemphasis of rare classes
alpha = 0.5  # sqrt shrink; try 0.6–0.8 if still under-calling minority
cw_map = {k: (v ** alpha) for k, v in cw_map.items()}

# normalize around 1.0 and clip extremes
mean_w = np.mean(list(cw_map.values()))
cw_map = {k: min(3.0, max(0.33, v / mean_w)) for k, v in cw_map.items()}  # clip to [0.33, 3]

# # optional: tamp down 'timeout' specifically if it's still over-predicted
# timeout_idx = int((le.transform(['timeout']))[0])
# cw_map[timeout_idx] = min(cw_map[timeout_idx], 1.2)

# build per-row weights for train/val
sw_tr = y_tr.map(cw_map).values
sw_val = y_val.map(cw_map).values

# Build DMatrices (names preserved from DataFrame columns)
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, enable_categorical=True)
dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val, enable_categorical=True)
dtest  = xgb.DMatrix(X_test, enable_categorical=True)

# XGBoost params (multi-class with probabilities)
params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(le.classes_),
    "max_depth": 6,
    "eta": 0.05,               # learning_rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "seed": 42,
}

watchlist = [(dtrain, "train"), (dval, "val")]

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=watchlist,
    early_stopping_rounds=100,   # works here
    verbose_eval=False
)

print(f"Best iteration (ntree_limit): {bst.best_iteration + 1}")

# --------------------
# Evaluate
# --------------------
# Predict on test at best iteration
val_margin = bst.predict(dval, output_margin=True, iteration_range=(0, bst.best_iteration+1))
val_y = y_val.values

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def nll_with_T(T):
    probs = softmax(val_margin / T)
    # pick prob of true class
    p_true = probs[np.arange(len(val_y)), val_y]
    return -np.mean(np.log(np.clip(p_true, 1e-12, 1.0)))

# simple 1-D search for T
Ts = np.linspace(0.5, 2.0, 16)
bestT = min(Ts, key=nll_with_T)
print(f"Calibrated temperature T={bestT:.2f}")

# when predicting on test:
test_margin = bst.predict(dtest, output_margin=True, iteration_range=(0, bst.best_iteration+1))
y_pred_proba = softmax(test_margin / bestT)

y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
brier = brier_score_loss((y_test == y_pred).astype(int), y_pred_proba.max(axis=1))
# top2 = top_k_accuracy_score(y_test, y_pred_proba, k=2)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Log-Loss: {logloss:.4f}")
# print(f"Top-2 Accuracy: {top2:.4f}")
print(f"Brier (1-vs-rest on predicted class): {brier:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# --------------------
# Feature importance (gain)
# --------------------
imp = bst.get_score(importance_type="gain")  # dict: {feature_name: gain}
feature_importance = (
    pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
      .sort_values("importance", ascending=False)
)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# --------------------
# Save model and preprocessors
# --------------------
bst.save_model("play_model.json")    # Booster format (preferred)

with open("calibration.json", "w") as f:
    json.dump({"temperature": float(bestT)}, f)
joblib.dump(le, "label_encoder.pkl")
joblib.dump(features, "features.pkl")
print("\nSaved model to play_model.json, label encoder to label_encoder.pkl, and features to features.pkl")

