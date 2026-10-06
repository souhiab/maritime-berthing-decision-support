"""Transparent 2D vessel and environment simulation for research demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DT_SECONDS = 2.0

ACTION_EFFECTS = {
    "MAINTAIN": (0.0, 0.0),
    "CORRECT_PORT": (-1.6, 0.0),
    "CORRECT_STARBOARD": (1.6, 0.0),
    "REDUCE_SPEED": (0.0, -0.22),
    "INCREASE_SPEED": (0.0, 0.16),
    "STABILIZE": (0.0, -0.08),
}

SCENARIOS = {
    "calm": {
        "wind_speed": 2.0,
        "wind_direction": 205.0,
        "current_speed": 0.10,
        "current_direction": 35.0,
        "wave_height": 0.25,
    },
    "crosswind": {
        "wind_speed": 9.0,
        "wind_direction": 355.0,
        "current_speed": 0.22,
        "current_direction": 5.0,
        "wave_height": 0.85,
    },
    "changing_wind_current": {
        "wind_speed": 7.0,
        "wind_direction": 325.0,
        "current_speed": 0.48,
        "current_direction": 350.0,
        "wave_height": 1.15,
    },
}


@dataclass(frozen=True)
class VesselState:
    x: float
    y: float
    heading_deg: float
    speed_mps: float
    yaw_rate_deg_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "heading_deg": self.heading_deg,
            "speed_mps": self.speed_mps,
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
        }


def _vector(speed: float, direction_deg: float) -> tuple[float, float]:
    radians = np.radians(direction_deg)
    return float(speed * np.sin(radians)), float(speed * np.cos(radians))


def generate_environment_series(
    scenario: str,
    n_steps: int,
    seed: int,
    dt_seconds: float = DT_SECONDS,
) -> pd.DataFrame:
    """Generate smoothly varying wind/current conditions plus bounded noise."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    config = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    step = np.arange(n_steps)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    gust = 1.4 * np.exp(-0.5 * ((step - 0.55 * n_steps) / max(8.0, 0.07 * n_steps)) ** 2)
    wind_speed = np.clip(
        config["wind_speed"] + 0.8 * np.sin(step / 22.0 + phase) + gust + rng.normal(0, 0.10, n_steps),
        0.0,
        None,
    )
    wind_direction = (
        config["wind_direction"] + 15.0 * np.sin(step / 45.0 + phase / 2.0) + 0.08 * step
    ) % 360.0
    current_speed = np.clip(
        config["current_speed"]
        + 0.06 * np.sin(step / 35.0 + phase)
        + 0.12 * np.exp(-0.5 * ((step - 0.42 * n_steps) / max(10.0, 0.1 * n_steps)) ** 2),
        0.0,
        None,
    )
    current_direction = (config["current_direction"] + 10.0 * np.sin(step / 55.0 + phase)) % 360.0
    wave_height = np.clip(config["wave_height"] + 0.12 * np.sin(step / 19.0), 0.0, None)
    return pd.DataFrame(
        {
            "timestamp_s": step * dt_seconds,
            "wind_speed_mps": wind_speed,
            "wind_direction_deg": wind_direction,
            "current_speed_mps": current_speed,
            "current_direction_deg": current_direction,
            "wave_height_m": wave_height,
            "scenario": scenario,
        }
    )


def simulate_step(
    state: VesselState,
    action: str,
    environment: dict | pd.Series,
    dt_seconds: float = DT_SECONDS,
    process_noise: tuple[float, float, float, float] | None = None,
) -> VesselState:
    """Advance one timestep using simple control, wind drift, and current drift.

    Environment directions are synthetic *flow-to* directions. Wind contributes
    2.5% of its speed as lateral drift; current contributes its full velocity.
    """
    if action not in ACTION_EFFECTS:
        raise ValueError(f"Unknown action: {action}")
    heading_delta, speed_delta = ACTION_EFFECTS[action]
    if action == "STABILIZE":
        yaw_rate = 0.30 * state.yaw_rate_deg_s
    else:
        commanded_rate = heading_delta / dt_seconds
        yaw_rate = 0.45 * state.yaw_rate_deg_s + 0.55 * commanded_rate
    noise_x, noise_y, noise_heading, noise_speed = process_noise or (0.0, 0.0, 0.0, 0.0)
    heading = (state.heading_deg + yaw_rate * dt_seconds + noise_heading) % 360.0
    speed = float(np.clip(state.speed_mps + speed_delta + noise_speed, 0.35, 6.0))

    vessel_vx, vessel_vy = _vector(speed, heading)
    wind_vx, wind_vy = _vector(0.025 * float(environment["wind_speed_mps"]), float(environment["wind_direction_deg"]))
    current_vx, current_vy = _vector(float(environment["current_speed_mps"]), float(environment["current_direction_deg"]))
    return VesselState(
        x=state.x + dt_seconds * (vessel_vx + wind_vx + current_vx + noise_x),
        y=state.y + dt_seconds * (vessel_vy + wind_vy + current_vy + noise_y),
        heading_deg=heading,
        speed_mps=speed,
        yaw_rate_deg_s=yaw_rate,
    )


def generate_process_noise(n_steps: int, seed: int) -> np.ndarray:
    """Generate shared, deterministic per-step disturbances for fair comparisons."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.normal(0.0, 0.025, n_steps),
            rng.normal(0.0, 0.025, n_steps),
            rng.normal(0.0, 0.08, n_steps),
            rng.normal(0.0, 0.012, n_steps),
        ]
    )
