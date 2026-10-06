"""Generate deterministic synthetic vessel, environment, and fixed-berth CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import recommend_action
from src.simulator import DT_SECONDS, SCENARIOS, VesselState, generate_environment_series, simulate_step
from src.trajectory import (
    calculate_cross_track_error,
    calculate_distance_to_berth,
    calculate_heading_error,
    generate_nominal_trajectory,
    nearest_trajectory_point,
)


SEED = 2026
N_TRAJECTORIES = 36
MAX_STEPS = 170
SAMPLE_DIR = ROOT / "data" / "sample"

BERTH = {
    "berth_id": "BERTH_FIXED_01",
    "target_x": 0.0,
    "target_y": 0.0,
    "target_heading_deg": 90.0,
    "approach_zone_radius_m": 45.0,
}


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory = generate_nominal_trajectory()
    rng = np.random.default_rng(SEED)
    vessel_rows: list[dict] = []
    environment_rows: list[dict] = []
    scenario_names = tuple(SCENARIOS)

    for episode_id in range(N_TRAJECTORIES):
        episode_seed = SEED + episode_id * 37
        scenario = scenario_names[episode_id % len(scenario_names)]
        environment = generate_environment_series(scenario, MAX_STEPS, episode_seed)
        first = trajectory.iloc[0]
        state = VesselState(
            x=float(first["x"] + rng.normal(0.0, 4.0)),
            y=float(first["y"] + rng.normal(0.0, 4.0)),
            heading_deg=float((first["desired_heading_deg"] + rng.normal(0.0, 2.0)) % 360.0),
            speed_mps=float(np.clip(first["desired_speed_mps"] + rng.normal(0.0, 0.25), 3.8, 5.5)),
        )
        previous_action = "MAINTAIN"
        for step in range(MAX_STEPS):
            env = environment.iloc[step]
            nearest = nearest_trajectory_point(state.x, state.y, trajectory)
            recommended_action, _ = recommend_action(state, env, trajectory, BERTH, previous_action)
            # Mostly follow the recommendation, with bounded exploration to expose
            # the predictor to different control states.
            draw = rng.random()
            if draw < 0.82:
                applied_action = recommended_action
            elif draw < 0.92:
                applied_action = "MAINTAIN"
            else:
                applied_action = str(rng.choice(["CORRECT_PORT", "CORRECT_STARBOARD", "REDUCE_SPEED", "INCREASE_SPEED"]))
            process_noise = (
                float(rng.normal(0.0, 0.025)),
                float(rng.normal(0.0, 0.025)),
                float(rng.normal(0.0, 0.08)),
                float(rng.normal(0.0, 0.012)),
            )
            next_state = simulate_step(state, applied_action, env, DT_SECONDS, process_noise)
            cte = calculate_cross_track_error(state.x, state.y, trajectory)
            distance = calculate_distance_to_berth(state.x, state.y, BERTH)
            vessel_rows.append(
                {
                    "timestamp_s": step * DT_SECONDS,
                    "trajectory_id": f"SYNTH_{episode_id:03d}",
                    "vessel_id": "SYNTHETIC_VESSEL_01",
                    "x_position_m": state.x,
                    "y_position_m": state.y,
                    "heading_deg": state.heading_deg,
                    "speed_mps": state.speed_mps,
                    "yaw_rate_deg_s": state.yaw_rate_deg_s,
                    "distance_to_berth_m": distance,
                    "cross_track_error_m": cte,
                    "heading_error_deg": calculate_heading_error(state.heading_deg, float(nearest["desired_heading_deg"])),
                    "along_track_progress": float(nearest["along_track_progress"]),
                    "previous_action": previous_action,
                    "recommended_action": applied_action,
                    "next_x_position_m": next_state.x,
                    "next_y_position_m": next_state.y,
                    "next_heading_deg": next_state.heading_deg,
                    "next_speed_mps": next_state.speed_mps,
                    "scenario": scenario,
                }
            )
            environment_rows.append(
                {
                    "timestamp_s": step * DT_SECONDS,
                    "trajectory_id": f"SYNTH_{episode_id:03d}",
                    "wind_speed_mps": float(env["wind_speed_mps"]),
                    "wind_direction_deg": float(env["wind_direction_deg"]),
                    "current_speed_mps": float(env["current_speed_mps"]),
                    "current_direction_deg": float(env["current_direction_deg"]),
                    "wave_height_m": float(env["wave_height_m"]),
                    "scenario": scenario,
                }
            )
            state = next_state
            previous_action = applied_action
            if distance <= BERTH["approach_zone_radius_m"]:
                break

    return pd.DataFrame(vessel_rows), pd.DataFrame(environment_rows), pd.DataFrame([BERTH])


def validate(vessel: pd.DataFrame, environment: pd.DataFrame, berth: pd.DataFrame) -> None:
    required_state = [
        "timestamp_s",
        "trajectory_id",
        "x_position_m",
        "y_position_m",
        "heading_deg",
        "speed_mps",
        "distance_to_berth_m",
        "cross_track_error_m",
    ]
    if vessel[required_state].isna().any().any():
        raise ValueError("Required vessel-state columns contain NaNs")
    if environment.isna().any().any() or berth.isna().any().any():
        raise ValueError("Environment or berth data contain NaNs")
    if vessel.duplicated(["trajectory_id", "timestamp_s"]).any():
        raise ValueError("Duplicate trajectory timestamps found")
    if (vessel.groupby("trajectory_id")["timestamp_s"].diff().dropna() != DT_SECONDS).any():
        raise ValueError("Timestamps are not continuous at the configured timestep")
    if not vessel[["next_x_position_m", "next_y_position_m"]].apply(np.isfinite).all().all():
        raise ValueError("Non-finite prediction targets found")
    if not set(vessel["scenario"]).issubset(SCENARIOS):
        raise ValueError("Unexpected scenario label")


def main() -> None:
    vessel, environment, berth = generate_dataset()
    validate(vessel, environment, berth)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    vessel.to_csv(SAMPLE_DIR / "vessel_states.csv", index=False)
    environment.to_csv(SAMPLE_DIR / "environment.csv", index=False)
    berth.to_csv(SAMPLE_DIR / "berth.csv", index=False)
    print(f"Wrote {len(vessel):,} vessel states across {vessel['trajectory_id'].nunique()} trajectories")
    print(f"Wrote {len(environment):,} environment rows and {len(berth)} fixed berth row")


if __name__ == "__main__":
    main()
