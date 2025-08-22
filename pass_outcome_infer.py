# pass_outcome_infer.py
import numpy as np, json, joblib, xgboost as xgb
import pandas as pd
import scipy.sparse as sp

# must match training
NUM_FEATURES = [
    'down','distance','yardsToGoal','is_red_zone','score_diff','seconds_remaining',
    'offenseTimeouts','defenseTimeouts',
    'sp_rating_off','sp_offense_rating_off','sp_defense_rating_def','sp_rating_def',
    'goal_to_go','fourth_and_short','fg_range','half','two_minute'
]
CAT1 = ['passer_name']                      # stage 1
CAT2 = ['passer_name','target_name']        # stage 2 (target_name optional at runtime)

def _ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

class PassOutcomeTwoStage:
    def __init__(self):
        # stage 1
        self.pre1 = joblib.load("pass_stage1_preprocessor.joblib")
        self.b1 = xgb.Booster(); self.b1.load_model("pass_stage1_complete_vs_not.json")
        with open("pass_stage1_meta.json") as f: self.it1 = json.load(f)["best_iteration"]

        # stage 2
        self.pre2 = joblib.load("pass_stage2_preprocessor.joblib")
        self.b2 = xgb.Booster(); self.b2.load_model("pass_stage2_notcomplete.json")
        with open("pass_stage2_meta.json") as f: self.it2 = json.load(f)["best_iteration"]
        # read label order for stage 2 (e.g., ['incomplete','intercepted','sack'])
        self.nc_classes = pd.read_csv("pass_stage2_classes.csv", header=None)[0].tolist()

    def _prep1(self, df):
        X = _ensure_cols(df.copy(), NUM_FEATURES + CAT1)
        X[CAT1] = X[CAT1].astype(str).fillna("Unknown")
        Xc = self.pre1.transform(X)
        return Xc if sp.isspmatrix_csr(Xc) else Xc.tocsr()

    def _prep2(self, df):
        X = _ensure_cols(df.copy(), NUM_FEATURES + CAT2)
        # target_name may be missing/sacks → fill "Unknown"
        for c in CAT2:
            X[c] = X[c].astype(str).fillna("Unknown")
        Xc = self.pre2.transform(X)
        return Xc if sp.isspmatrix_csr(Xc) else Xc.tocsr()

    def predict_proba(self, row_df: pd.DataFrame) -> np.ndarray:
        """
        row_df: DataFrame with a SINGLE row and at least the columns in NUM_FEATURES + CAT1 (+ target_name if you have it)
        returns probs in order: [complete, incomplete, intercepted, sack]
        """
        # stage 1: P(complete)
        d1 = xgb.DMatrix(self._prep1(row_df))
        p_complete = self.b1.predict(d1, iteration_range=(0, self.it1+1)).reshape(-1,1)  # shape (1,1)
        p_not = 1.0 - p_complete

        # stage 2: distribution over not-complete classes
        d2 = xgb.DMatrix(self._prep2(row_df))
        p_nc = self.b2.predict(d2, iteration_range=(0, self.it2+1))  # shape (1,3)

        # compose
        out = np.zeros((len(row_df), 4), dtype=float)
        out[:,0] = p_complete[:,0]  # complete
        # map stage2 labels to columns
        for j, cls in enumerate(self.nc_classes):
            col = {"incomplete":1, "intercepted":2, "sack":3}[cls]
            out[:, col] = p_not[:,0] * p_nc[:, j]

        # tiny smoothing to avoid exact zeros
        eps = 1e-9
        out = out + eps
        out = out / out.sum(axis=1, keepdims=True)
        return out
