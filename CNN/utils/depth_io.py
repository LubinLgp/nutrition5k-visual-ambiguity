import numpy as np

from config import DEPTH_MIDAS_DIR


def midas_depth_path(dish_id: str):
    return DEPTH_MIDAS_DIR / f"{dish_id}.npy"


def save_depth_midas(dish_id: str, depth_m: np.ndarray):
    path = midas_depth_path(dish_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth_m, dtype=np.float32))


def load_depth_midas(dish_id: str) -> np.ndarray:
    return np.load(midas_depth_path(dish_id))
