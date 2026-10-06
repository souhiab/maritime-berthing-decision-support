"""Short-horizon ranking of abstract, advisory correction actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator import (
    ACTION_EFFECTS,
    DT_SECONDS,
    VesselState,
    generate_environment_series,
    generate_process_noise,
    simulate_step,
)
from src.trajectory import (
    calculate_cross_track_error,
    calculate_distance_to_berth,
    calculate_heading_error,
    nearest_trajectory_point,
)


CANDIDATE_ACTIONS = tuple(ACTION_EFFECTS)

POLICY_LABELS = {
    "maintain": "Maintain only",
    "rule_based": "Rule-based path follower",
    "predictive": "Predictive recommender",
}

RULE_THRESHOLDS = {
    "cross_track_m": 6.0,
    "large_heading_error_deg": 10.0,
    "heading_error_deg": 5.0,
    "speed_above_profile_mps": 0.35,
    "speed_below_profile_mps": -0.45,
    "near_berth_radii": 3.0,
    "near_berth_speed_margin_mps": 0.20,
    "stabilize_yaw_rate_deg_s": 0.20,
}


@dataclass(frozen=True)
class ActionScore:
    action: str
    cost: float
    cross_track_error_m: float
    heading_error_deg: float
    speed_error_mps: float
    progress_m: float
    action_change_penalty: float


def classify_deviation(cross_track_error_m: float, heading_error_deg: float) -> str:
    """Demonstration-only thresholds; not certified maritime safety limits."""
    cte, heading = abs(cross_track_error_m), abs(heading_error_deg)
    if cte <= 8.0 and heading <= 6.0:
        return "ON_TRACK"
    if cte <= 25.0 and heading <= 15.0:
        return "MINOR_DEVIATION"
    return "MAJOR_DEVIATION"


def score_action(
    state: VesselState,
    environment: dict | pd.Series,
    trajectory: pd.DataFrame,
    berth: dict | pd.Series,
    action: str,
    previous_action: str = "MAINTAIN",
    horizon_steps: int = 6,
) -> ActionScore:
    """Roll an action forward and calculate an interpretable terminal cost."""
    simulated = state
    for _ in range(horizon_steps):
        simulated = simulate_step(simulated, action, environment, DT_SECONDS)
    nearest = nearest_trajectory_point(simulated.x, simulated.y, trajectory)
    cte = calculate_cross_track_error(simulated.x, simulated.y, trajectory)
    heading_error = calculate_heading_error(simulated.heading_deg, float(nearest["desired_heading_deg"]))
    speed_error = simulated.speed_mps - float(nearest["desired_speed_mps"])
    before_distance = calculate_distance_to_berth(state.x, state.y, berth)
    after_distance = calculate_distance_to_berth(simulated.x, simulated.y, berth)
    progress = before_distance - after_distance
    lack_of_progress = max(0.0, 0.5 * horizon_steps * DT_SECONDS - progress)
    action_change = 0.0 if action == previous_action else (0.3 if action == "MAINTAIN" else 0.7)
    cost = (
        1.80 * abs(cte)
        + 0.10 * abs(heading_error)
        + 2.2 * abs(speed_error)
        + 0.75 * lack_of_progress
        + action_change
    )
    return ActionScore(
        action=action,
        cost=float(cost),
        cross_track_error_m=float(cte),
        heading_error_deg=float(heading_error),
        speed_error_mps=float(speed_error),
        progress_m=float(progress),
        action_change_penalty=float(action_change),
    )


def recommend_action(
    state: VesselState,
    environment: dict | pd.Series,
    trajectory: pd.DataFrame,
    berth: dict | pd.Series,
    previous_action: str = "MAINTAIN",
) -> tuple[str, pd.DataFrame]:
    """Return the lowest-cost advisory action and the full candidate ranking."""
    scores = [
        score_action(state, environment, trajectory, berth, action, previous_action)
        for action in CANDIDATE_ACTIONS
    ]
    ranking = pd.DataFrame([score.__dict__ for score in scores]).sort_values("cost").reset_index(drop=True)
    return str(ranking.loc[0, "action"]), ranking


def recommendation_reason(ranking: pd.DataFrame) -> str:
    best = ranking.iloc[0]
    return (
        f"{best['action']} has the lowest short-horizon cost ({best['cost']:.2f}); "
        f"expected |cross-track error| is {abs(best['cross_track_error_m']):.1f} m "
        f"with {best['progress_m']:.1f} m progress toward the fixed berth."
    )


def rule_based_action(
    state: VesselState,
    trajectory: pd.DataFrame,
    berth: dict | pd.Series,
) -> str:
    """Choose an action from current errors only, without prediction or look-ahead.

    Positive cross-track error is port/left of the path, so it requires a
    starboard correction. Positive heading error is clockwise/starboard of the
    desired heading, so it requires a port correction.
    """
    nearest = nearest_trajectory_point(state.x, state.y, trajectory)
    cross_track_error = calculate_cross_track_error(state.x, state.y, trajectory)
    heading_error = calculate_heading_error(
        state.heading_deg, float(nearest["desired_heading_deg"])
    )
    speed_error = state.speed_mps - float(nearest["desired_speed_mps"])
    distance_to_berth = calculate_distance_to_berth(state.x, state.y, berth)
    near_berth = (
        distance_to_berth
        <= RULE_THRESHOLDS["near_berth_radii"] * float(berth["approach_zone_radius_m"])
    )

    if abs(heading_error) > RULE_THRESHOLDS["large_heading_error_deg"]:
        return "CORRECT_PORT" if heading_error > 0.0 else "CORRECT_STARBOARD"
    if abs(cross_track_error) > RULE_THRESHOLDS["cross_track_m"]:
        return "CORRECT_STARBOARD" if cross_track_error > 0.0 else "CORRECT_PORT"
    if near_berth and speed_error > RULE_THRESHOLDS["near_berth_speed_margin_mps"]:
        return "REDUCE_SPEED"
    if speed_error > RULE_THRESHOLDS["speed_above_profile_mps"]:
        return "REDUCE_SPEED"
    if not near_berth and speed_error < RULE_THRESHOLDS["speed_below_profile_mps"]:
        return "INCREASE_SPEED"
    if abs(heading_error) > RULE_THRESHOLDS["heading_error_deg"]:
        return "CORRECT_PORT" if heading_error > 0.0 else "CORRECT_STARBOARD"
    if near_berth and abs(state.yaw_rate_deg_s) > RULE_THRESHOLDS["stabilize_yaw_rate_deg_s"]:
        return "STABILIZE"
    return "MAINTAIN"


def run_closed_loop(
    scenario: str,
    trajectory: pd.DataFrame,
    berth: dict | pd.Series,
    policy: str | bool,
    seed: int = 42,
    max_steps: int = 150,
    initial_state: VesselState | None = None,
) -> pd.DataFrame:
    """Run one control policy with deterministic, policy-independent disturbances."""
    if isinstance(policy, bool):
        policy = "predictive" if policy else "maintain"
    if policy not in POLICY_LABELS:
        raise ValueError(f"Unknown policy: {policy}")
    if initial_state is None:
        first = trajectory.iloc[0]
        initial_state = VesselState(
            x=float(first["x"]),
            y=float(first["y"]),
            heading_deg=float(first["desired_heading_deg"]),
            speed_mps=float(first["desired_speed_mps"]),
        )
    environment = generate_environment_series(scenario, max_steps, seed)
    noise = generate_process_noise(max_steps, seed + 10_000)
    state = initial_state
    previous_action = "MAINTAIN"
    records: list[dict] = []
    for step in range(max_steps):
        env = environment.iloc[step]
        nearest = nearest_trajectory_point(state.x, state.y, trajectory)
        cte = calculate_cross_track_error(state.x, state.y, trajectory)
        heading_error = calculate_heading_error(state.heading_deg, float(nearest["desired_heading_deg"]))
        distance = calculate_distance_to_berth(state.x, state.y, berth)
        if policy == "predictive":
            action, ranking = recommend_action(state, env, trajectory, berth, previous_action)
            action_cost = float(ranking.iloc[0]["cost"])
        elif policy == "rule_based":
            action = rule_based_action(state, trajectory, berth)
            action_cost = np.nan
        else:
            action, action_cost = "MAINTAIN", np.nan
        records.append(
            {
                "timestamp_s": step * DT_SECONDS,
                "scenario": scenario,
                "controller": POLICY_LABELS[policy],
                "x_position_m": state.x,
                "y_position_m": state.y,
                "heading_deg": state.heading_deg,
                "speed_mps": state.speed_mps,
                "yaw_rate_deg_s": state.yaw_rate_deg_s,
                "distance_to_berth_m": distance,
                "cross_track_error_m": cte,
                "heading_error_deg": heading_error,
                "speed_error_mps": state.speed_mps - float(nearest["desired_speed_mps"]),
                "along_track_progress": float(nearest["along_track_progress"]),
                "deviation_state": classify_deviation(cte, heading_error),
                "recommended_action": action,
                "action_cost": action_cost,
                "wind_speed_mps": float(env["wind_speed_mps"]),
                "wind_direction_deg": float(env["wind_direction_deg"]),
                "current_speed_mps": float(env["current_speed_mps"]),
                "current_direction_deg": float(env["current_direction_deg"]),
            }
        )
        if distance <= float(berth["approach_zone_radius_m"]):
            break
        state = simulate_step(state, action, env, DT_SECONDS, tuple(noise[step]))
        previous_action = action
    return pd.DataFrame.from_records(records)


def closed_loop_metrics(run: pd.DataFrame, berth: dict | pd.Series) -> dict[str, float | str]:
    """Calculate executed trajectory metrics for one simulation run."""
    final = run.iloc[-1]
    reached = float(final["distance_to_berth_m"]) <= float(berth["approach_zone_radius_m"])
    corrections = (run["recommended_action"] != "MAINTAIN").sum()
    return {
        "scenario": str(final["scenario"]),
        "controller": str(final["controller"]),
        "mean_cross_track_error_m": float(run["cross_track_error_m"].abs().mean()),
        "max_cross_track_error_m": float(run["cross_track_error_m"].abs().max()),
        "p95_cross_track_error_m": float(run["cross_track_error_m"].abs().quantile(0.95)),
        "final_position_error_m": float(final["distance_to_berth_m"]),
        "final_heading_error_deg": abs(
            calculate_heading_error(float(final["heading_deg"]), float(berth["target_heading_deg"]))
        ),
        "time_to_approach_zone_s": float(final["timestamp_s"]) if reached else np.nan,
        "corrective_recommendations": int(corrections),
        "on_track_pct": float(100.0 * (run["deviation_state"] == "ON_TRACK").mean()),
        "major_deviation_pct": float(100.0 * (run["deviation_state"] == "MAJOR_DEVIATION").mean()),
        "reached_approach_zone": bool(reached),
    }
