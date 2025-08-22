# train_pass_outcome.py
import pandas as pd, numpy as np, xgboost as xgb, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("ml_pass_outcomes_2022_2024.csv")

target = "pass_outcome"
features = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'head_coach','goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
df = df.dropna(subset=features+[target]).copy()
df['head_coach'] = df['head_coach'].astype('category')

le = LabelEncoder()
y_all = le.fit_transform(df[target].values)
X_all = df[features]

# time-safe split (train 2022–23, test 2024) if available
if 'year' in df.columns:
    train_mask = df['year'].isin([2022, 2023])
    test_mask  = df['year'].isin([2024])
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
else:
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

# small validation from TRAIN
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# class weights from TRAIN ONLY (softened)
classes = np.unique(y_tr)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
cw_map = {c: w for c, w in zip(classes, cw)}
alpha = 0.7
cw_map = {k: (v ** alpha) for k, v in cw_map.items()}
mean_w = np.mean(list(cw_map.values()))
cw_map = {k: np.clip(v / mean_w, 0.33, 3.0) for k, v in cw_map.items()}
sw_tr = np.vectorize(cw_map.get)(y_tr)
sw_val = np.vectorize(cw_map.get)(y_val)

# Booster with early stopping
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, enable_categorical=True)
dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val, enable_categorical=True)
dtest  = xgb.DMatrix(X_test, enable_categorical=True)

params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(le.classes_),
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "seed": 42,
}
bst = xgb.train(params, dtrain, num_boost_round=2000,
                evals=[(dtrain,"train"),(dval,"val")],
                early_stopping_rounds=100, verbose_eval=False)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

# Temperature calibration on VAL
val_margin = bst.predict(dval, output_margin=True, iteration_range=(0, bst.best_iteration+1))
Ts = np.linspace(0.6, 1.6, 21)
def nll_T(T):
    p = softmax(val_margin / T)
    return -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)))
Tbest = float(Ts[np.argmin([nll_T(T) for T in Ts])])

# Evaluate on TEST
test_margin = bst.predict(dtest, output_margin=True, iteration_range=(0, bst.best_iteration+1))
proba = softmax(test_margin / Tbest)
pred  = proba.argmax(axis=1)

print("Acc", round(accuracy_score(y_test, pred), 4))
print("LogLoss", round(log_loss(y_test, proba), 4))
print("Top-2", round(top_k_accuracy_score(y_test, proba, k=2), 4))
print(classification_report(y_test, pred, target_names=le.classes_))
print(confusion_matrix(y_test, pred))

# Save artifacts
bst.save_model("pass_outcome.json")
pd.Series(le.classes_).to_csv("pass_outcome_classes.csv", index=False, header=False)
with open("pass_outcome_calibration.json","w") as f:
    json.dump({"temperature": Tbest, "best_iteration": int(bst.best_iteration)}, f)
print("Saved pass_outcome.json, pass_outcome_classes.csv, pass_outcome_calibration.json")
