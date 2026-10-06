"""Interpretable kinematic and residual machine-learning next-state predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.simulator import DT_SECONDS


NUMERIC_FEATURES = [
    "x_position_m",
    "y_position_m",
    "heading_deg",
    "speed_mps",
    "yaw_rate_deg_s",
    "distance_to_berth_m",
    "cross_track_error_m",
    "wind_speed_mps",
    "wind_direction_deg",
    "current_speed_mps",
    "current_direction_deg",
    "wave_height_m",
]
CATEGORICAL_FEATURES = ["previous_action", "recommended_action"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["next_x_position_m", "next_y_position_m"]


def kinematic_predict(frame: pd.DataFrame, dt_seconds: float = DT_SECONDS) -> np.ndarray:
    """Predict next x/y from present speed and heading, ignoring disturbances."""
    heading_radians = np.radians(frame["heading_deg"].to_numpy())
    next_x = frame["x_position_m"].to_numpy() + dt_seconds * frame["speed_mps"].to_numpy() * np.sin(heading_radians)
    next_y = frame["y_position_m"].to_numpy() + dt_seconds * frame["speed_mps"].to_numpy() * np.cos(heading_radians)
    return np.column_stack([next_x, next_y])


def build_residual_model(random_state: int = 42) -> Pipeline:
    """Build a small model that learns residual motion beyond kinematics."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("action", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    regressor = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=24,
            l2_regularization=0.2,
            random_state=random_state,
        )
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def fit_residual_model(train_frame: pd.DataFrame, random_state: int = 42) -> Pipeline:
    baseline = kinematic_predict(train_frame)
    residuals = train_frame[TARGETS].to_numpy() - baseline
    model = build_residual_model(random_state=random_state)
    model.fit(train_frame[FEATURES], residuals)
    return model


def ml_predict(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return kinematic_predict(frame) + model.predict(frame[FEATURES])


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    position_error = np.linalg.norm(errors, axis=1)
    return {
        "mae_x_m": float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
        "mae_y_m": float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
        "position_rmse_m": float(np.sqrt(mean_squared_error(actual, predicted) * 2.0)),
        "mean_position_error_m": float(position_error.mean()),
        "median_position_error_m": float(np.median(position_error)),
        "p95_position_error_m": float(np.percentile(position_error, 95)),
    }
