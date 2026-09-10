import numpy as np

from config import NUTRIENT_NAMES


def compute_mae_pmae(preds: np.ndarray, targets: np.ndarray, names=NUTRIENT_NAMES):
    results = {}
    pmaes = []
    for i, name in enumerate(names):
        p = preds[:, i]
        t = targets[:, i]
        mae = np.mean(np.abs(p - t))
        mean_gt = np.mean(t)
        pmae = (mae / mean_gt * 100.0) if mean_gt > 0 else float("nan")
        results[name] = {"mae": mae, "pmae": pmae}
        pmaes.append(pmae)
    results["mean_pmae"] = float(np.nanmean(pmaes))
    return results
