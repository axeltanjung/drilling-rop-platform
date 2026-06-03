"""Deep learning ROP model wrapper.

Supports two architectures:
  * 'mlp'  — dense network on tabular features
  * 'lstm' — sequence model on per-well sliding windows (sequential drilling)

TensorFlow/Keras backend. Scales inputs with a persisted StandardScaler.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from backend.utils.config import settings, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from backend.utils.features import ENGINEERED_FEATURES
from backend.utils.logger import get_logger
from backend.training.metrics import regression_metrics

log = get_logger("deep_model")

ARTIFACT = settings.model_dir / "deep_rop.keras"
SCALER = settings.model_dir / "deep_scaler.pkl"
META = settings.model_dir / "deep_meta.pkl"

SEQ_LEN = 24  # sliding window length for LSTM


def _onehot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)


class DeepROPModel:
    name = "deep"

    def __init__(self, arch: str = "mlp"):
        assert arch in ("mlp", "lstm")
        self.arch = arch
        self.numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.model = None

    # ---- feature matrix construction ----
    def _matrix(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        cat = _onehot(df)
        X = pd.concat([df[self.numeric].reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
        if fit_scaler:
            self.feature_columns = X.columns.tolist()
        # align to training columns
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        if fit_scaler:
            X[self.numeric] = self.scaler.fit_transform(X[self.numeric])
        else:
            X[self.numeric] = self.scaler.transform(X[self.numeric])
        return X

    def _sequences(self, df: pd.DataFrame, target: str | None, fit_scaler: bool):
        """Build per-well sliding windows for LSTM."""
        X_all, y_all = [], []
        # build feature matrix once (fit scaler on full frame)
        feat = self._matrix(df, fit_scaler=fit_scaler)
        feat["well_id"] = df["well_id"].values
        if target:
            feat["_y"] = df[target].values
        for _, g in feat.groupby("well_id"):
            arr = g.drop(columns=["well_id"] + (["_y"] if target else [])).values
            for i in range(len(g) - SEQ_LEN):
                X_all.append(arr[i:i + SEQ_LEN])
                if target:
                    y_all.append(g["_y"].values[i + SEQ_LEN])
        X = np.asarray(X_all, dtype="float32")
        y = np.asarray(y_all, dtype="float32") if target else None
        return X, y

    def _build(self, input_shape):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(settings.random_seed)
        if self.arch == "mlp":
            m = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ])
        else:  # lstm
            m = models.Sequential([
                layers.Input(shape=input_shape),
                layers.LSTM(96, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(48),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ])
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return m

    def fit(self, train: pd.DataFrame, target: str, valid: pd.DataFrame | None = None,
            epochs: int = 15, batch_size: int = 512):
        import tensorflow as tf

        if self.arch == "mlp":
            Xtr = self._matrix(train, fit_scaler=True).values.astype("float32")
            ytr = train[target].values.astype("float32")
            val = None
            if valid is not None:
                val = (self._matrix(valid).values.astype("float32"), valid[target].values.astype("float32"))
            self.model = self._build((Xtr.shape[1],))
        else:
            Xtr, ytr = self._sequences(train, target, fit_scaler=True)
            val = None
            if valid is not None:
                Xv, yv = self._sequences(valid, target, fit_scaler=False)
                val = (Xv, yv)
            self.model = self._build((SEQ_LEN, Xtr.shape[2]))

        cb = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        self.model.fit(Xtr, ytr, validation_data=val, epochs=epochs,
                       batch_size=batch_size, verbose=0, callbacks=cb)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.arch == "mlp":
            X = self._matrix(df).values.astype("float32")
            return self.model.predict(X, verbose=0).ravel()
        X, _ = self._sequences(df, target=None, fit_scaler=False)
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X, verbose=0).ravel()

    def evaluate(self, df: pd.DataFrame, target: str) -> dict:
        preds = self.predict(df)
        if self.arch == "lstm":
            # align: lstm drops first SEQ_LEN rows per well
            y = []
            for _, g in df.groupby("well_id"):
                y.extend(g[target].values[SEQ_LEN:])
            y = np.asarray(y[: len(preds)])
            return regression_metrics(y, preds[: len(y)])
        return regression_metrics(df[target], preds)

    def save(self, path: Path | None = None):
        path = path or ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        joblib.dump(self.scaler, SCALER)
        joblib.dump({"arch": self.arch, "numeric": self.numeric,
                     "feature_columns": self.feature_columns, "seq_len": SEQ_LEN}, META)
        log.info("Saved Deep model (%s) -> %s", self.arch, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "DeepROPModel":
        import tensorflow as tf

        path = path or ARTIFACT
        meta = joblib.load(META)
        obj = cls(arch=meta["arch"])
        obj.numeric = meta["numeric"]
        obj.feature_columns = meta["feature_columns"]
        obj.scaler = joblib.load(SCALER)
        obj.model = tf.keras.models.load_model(path)
        return obj
