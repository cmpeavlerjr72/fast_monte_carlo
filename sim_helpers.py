# sim_helpers.py
import numpy as np, json, xgboost as xgb, joblib

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

class PassOutcomeModel:
    def __init__(self, model="pass_outcome.json", calib="pass_outcome_calibration.json"):
        self.bst = xgb.Booster(); self.bst.load_model(model)
        try:
            with open(calib) as f:
                meta = json.load(f)
            self.T = float(meta["temperature"])
            self.best_it = int(meta["best_iteration"])
        except Exception:
            self.T = 1.0; self.best_it = getattr(self.bst, "best_iteration", None)

    def predict_proba(self, x_df):
        d = xgb.DMatrix(x_df, enable_categorical=True)
        it = (0, (self.best_it or 0)+1) if self.best_it is not None else None
        margin = self.bst.predict(d, output_margin=True, iteration_range=it)
        return softmax(margin / self.T)

class QuantileYards:
    def __init__(self, prefix):
        self.m10 = joblib.load(f"{prefix}_q10.joblib")
        self.m50 = joblib.load(f"{prefix}_q50.joblib")
        self.m90 = joblib.load(f"{prefix}_q90.joblib")

    def sample(self, x_df, lo, hi, noise=0.5):
        q10 = float(self.m10.predict(x_df)[0])
        q50 = float(self.m50.predict(x_df)[0])
        q90 = float(self.m90.predict(x_df)[0])
        u = np.random.rand()
        y = q10 + (q50-q10)*(u/0.5) if u<0.5 else q50 + (q90-q50)*((u-0.5)/0.5)
        return float(np.clip(y + np.random.normal(0, noise), lo, hi))
