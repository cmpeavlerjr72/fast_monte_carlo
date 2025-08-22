# sim_predict.py
import json, numpy as np, xgboost as xgb

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

class PlayPolicy:
    def __init__(self, model_path="play_model.json", calib_path="calibration.json", best_iter=None):
        self.bst = xgb.Booster()
        self.bst.load_model(model_path)
        self.best_iter = best_iter  # or read from a meta file if you store it
        try:
            with open(calib_path) as f:
                self.T = float(json.load(f)["temperature"])
        except Exception:
            self.T = 1.0

    def predict_proba(self, df_row):
        # df_row: pd.DataFrame with one row, same columns/dtypes as training
        dmat = xgb.DMatrix(df_row, enable_categorical=True)
        m = self.bst.predict(dmat, output_margin=True,
                             iteration_range=(0, (self.best_iter or self.bst.best_iteration)+1))
        return softmax(m / self.T)[0]  # 1D array of class probs

    def sample_action(self, df_row, rng=None):
        p = self.predict_proba(df_row)
        # tiny smoothing for safety:
        p = np.maximum(p, 1e-6); p = p / p.sum()
        (rng or np.random).seed()  # or pass a seeded RNG
        return int((rng or np.random).choice(len(p), p=p))
