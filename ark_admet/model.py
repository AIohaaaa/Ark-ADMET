"""CatBoost fit and the optional MiniMol head. One seed is one model."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


class Scaler:
    """Map a regression target to the scale CatBoost sees, then invert predictions."""

    def __init__(self, log: bool = False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset
        if self.log:
            y = np.log10(y + 1.0)
        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset
        if self.log:
            y = np.log10(y + 1.0)
        return self.scaler.transform(y)

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(np.asarray(y_scale).reshape(-1, 1))
        if self.log:
            y = 10.0**y - 1.0
        return y + self.offset


def fit_catboost(x_train, y_train, x_test, cfg: dict, seed: int, device: str) -> np.ndarray:
    tree = cfg["tree"]
    params = {
        "random_strength": 2,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "depth": int(tree["depth"]),
        "iterations": int(tree["iterations"]),
        "learning_rate": float(tree["learning_rate"]),
        "l2_leaf_reg": float(tree["l2_leaf_reg"]),
    }
    if device == "gpu":
        params["task_type"] = "GPU"
        params["devices"] = "0"
        params["gpu_ram_part"] = 0.85
    else:
        params["task_type"] = "CPU"
        params["thread_count"] = 16
    task = cfg["task"]
    if task == "regression":
        y_scaler = Scaler(log=bool(cfg["log_scale"]))
        y_scaler.fit(y_train)
        params["loss_function"] = "MAE"
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_scaler.transform(y_train).reshape(-1))
        pred = y_scaler.inverse_transform(model.predict(x_test)).reshape(-1)
        return pred.astype(float)
    params["loss_function"] = "Logloss"
    weight = float(cfg["scale_pos_weight"])
    if weight != 1.0:
        params["scale_pos_weight"] = weight
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1].astype(float)


def fit_head(x_train, y_train, x_test, cfg: dict, seed: int) -> np.ndarray:
    """One hidden layer of width 128 on the frozen 512-d MiniMol vector."""
    if not np.isfinite(x_train).all():
        raise RuntimeError("MiniMol training rows contain a non-finite value")
    scaler = StandardScaler().fit(x_train)
    x_scale = scaler.transform(x_train)
    common = dict(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=seed,
    )
    pred = np.full(len(x_test), np.nan, dtype=float)
    ok = np.isfinite(x_test).all(axis=1)
    if cfg["task"] == "regression":
        y_scaler = Scaler(log=bool(cfg["log_scale"]))
        y_scaler.fit(y_train)
        model = MLPRegressor(**common)
        model.fit(x_scale, y_scaler.transform(y_train).reshape(-1))
        if ok.any():
            pred[ok] = y_scaler.inverse_transform(model.predict(scaler.transform(x_test[ok]))).reshape(-1)
        return pred
    model = MLPClassifier(**common)
    model.fit(x_scale, y_train.astype(int))
    if ok.any():
        pred[ok] = model.predict_proba(scaler.transform(x_test[ok]))[:, 1]
    return pred


def mix(catboost: np.ndarray, head: np.ndarray, weight: float) -> np.ndarray:
    blended = (1.0 - weight) * catboost + weight * head
    missing = ~np.isfinite(head)
    blended[missing] = catboost[missing]
    return blended
