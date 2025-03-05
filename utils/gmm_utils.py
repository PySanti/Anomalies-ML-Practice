import numpy as np
from sklearn.metrics import f1_score


def custom_scorer(df_val, target):
    def scorer(estimator, X):
        log_probs_val = estimator.score_samples(df_val.drop(target, axis=1))
        best_score = 0
        for a in np.linspace(log_probs_val.min(), log_probs_val.max(), 1000):
            y_pred = (log_probs_val < a).astype(int)
            f1 = f1_score(df_val[target], y_pred, pos_label=1)
            if f1 > best_score:
                best_score = f1
        return best_score
    return scorer

def generate_filename(scaler, pca):
    filename = ["gmm"]
    filename.append("_scaler" if scaler else "_no-scaler")
    filename.append("_pca" if pca else "_no-pca")
    filename.append(".joblib")
    filename = "".join(filename)
    return filename


