"""Nominal fixed-berth approach trajectory and deviation geometry.

Coordinate convention
---------------------
``x`` points east, ``y`` points north, and maritime headings are measured
clockwise from north (0 degrees = north, 90 degrees = east).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WAYPOINTS = (
    (-900.0, -340.0, 5.0),
    (-650.0, -205.0, 4.6),
    (-400.0, -85.0, 3.8),
    (-180.0, -22.0, 2.5),
    (0.0, 0.0, 0.8),
)


def _heading_deg(dx: float, dy: float) -> float:
    """Return maritime heading for an x/y vector."""
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def angle_difference_deg(angle: float, reference: float) -> float:
    """Signed shortest difference ``angle - reference`` in [-180, 180)."""
    return float((angle - reference + 180.0) % 360.0 - 180.0)


def generate_nominal_trajectory(
    waypoints: tuple[tuple[float, float, float], ...] = DEFAULT_WAYPOINTS,
    spacing_m: float = 5.0,
) -> pd.DataFrame:
    """Interpolate waypoints into a dense, speed-profiled nominal path."""
    records: list[dict[str, float | int]] = []
    cumulative = 0.0
    for segment_id, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        x0, y0, speed0 = start
        x1, y1, speed1 = end
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        samples = max(2, int(np.ceil(length / spacing_m)) + 1)
        fractions = np.linspace(0.0, 1.0, samples, endpoint=segment_id == len(waypoints) - 2)
        for fraction in fractions:
            records.append(
                {
                    "segment_id": segment_id,
                    "x": x0 + fraction * dx,
                    "y": y0 + fraction * dy,
                    "desired_heading_deg": _heading_deg(dx, dy),
                    "desired_speed_mps": speed0 + fraction * (speed1 - speed0),
                    "path_distance_m": cumulative + fraction * length,
                }
            )
        cumulative += length
    trajectory = pd.DataFrame.from_records(records)
    trajectory["along_track_progress"] = trajectory["path_distance_m"] / cumulative
    return trajectory.reset_index(drop=True)


def nearest_trajectory_point(
    x: float, y: float, trajectory: pd.DataFrame
) -> dict[str, float | int]:
    """Return the nearest sampled point and its desired motion state."""
    distances_sq = (trajectory["x"].to_numpy() - x) ** 2 + (
        trajectory["y"].to_numpy() - y
    ) ** 2
    index = int(np.argmin(distances_sq))
    row = trajectory.iloc[index]
    return {
        "index": index,
        "segment_id": int(row["segment_id"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "desired_heading_deg": float(row["desired_heading_deg"]),
        "desired_speed_mps": float(row["desired_speed_mps"]),
        "along_track_progress": float(row["along_track_progress"]),
        "distance_m": float(np.sqrt(distances_sq[index])),
    }


def calculate_cross_track_error(x: float, y: float, trajectory: pd.DataFrame) -> float:
    """Signed lateral error: positive is port/left of the path direction."""
    nearest = nearest_trajectory_point(x, y, trajectory)
    index = int(nearest["index"])
    if index == len(trajectory) - 1:
        p0 = trajectory.iloc[index - 1]
        p1 = trajectory.iloc[index]
    else:
        p0 = trajectory.iloc[index]
        p1 = trajectory.iloc[index + 1]
    tangent_x, tangent_y = float(p1["x"] - p0["x"]), float(p1["y"] - p0["y"])
    norm = max(float(np.hypot(tangent_x, tangent_y)), 1e-9)
    offset_x, offset_y = x - float(nearest["x"]), y - float(nearest["y"])
    return float((tangent_x * offset_y - tangent_y * offset_x) / norm)


def calculate_heading_error(heading_deg: float, desired_heading_deg: float) -> float:
    """Signed heading error in degrees."""
    return angle_difference_deg(heading_deg, desired_heading_deg)


def calculate_distance_to_berth(x: float, y: float, berth: dict | pd.Series) -> float:
    """Euclidean distance from the vessel to the fixed berth target."""
    return float(np.hypot(x - float(berth["target_x"]), y - float(berth["target_y"])))
