"""
Synthetic Drilling Telemetry Data Generator
============================================

Generates a realistic, physics-inspired synthetic drilling telemetry dataset
for the Oil & Gas Drilling ROP Prediction & Optimization Platform.

Highlights
----------
* >= 200,000 rows across multiple wells
* Time-series format with monotonic depth progression per well
* Multiple formation types with formation-dependent ROP behavior
* Physics-inspired relationships:
    - hard formations reduce ROP
    - moderate RPM/WOB improve ROP, excessive values cause vibration & wear
    - bit wear gradually reduces efficiency
    - mud flow affects hole cleaning
    - torque fluctuations indicate instability
* Realistic imperfections: sensor noise, outliers, operational drift,
  drilling dysfunction patterns, and simulated missing values.

Run
---
    python backend/data/synthetic_drilling_data_generator.py

Output
------
    data/raw/drilling_telemetry.csv
    docs/dataset_documentation.md (feature dictionary)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a standalone script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.utils.config import settings, FORMATION_TYPES, BIT_TYPES  # noqa: E402
from backend.utils.logger import get_logger  # noqa: E402

log = get_logger("data_generator")

# ---------------------------------------------------------------------------
# Formation & bit physical properties
# ---------------------------------------------------------------------------
# hardness: relative drillability (1=soft/fast, 10=very hard/slow)
FORMATION_PROPS = {
    "Sandstone": {"hardness": 2.5, "abrasiveness": 0.6, "stuck_prone": 0.10},
    "Shale":     {"hardness": 4.0, "abrasiveness": 0.4, "stuck_prone": 0.35},
    "Limestone": {"hardness": 5.5, "abrasiveness": 0.7, "stuck_prone": 0.15},
    "Dolomite":  {"hardness": 7.0, "abrasiveness": 0.8, "stuck_prone": 0.12},
    "Granite":   {"hardness": 9.0, "abrasiveness": 0.9, "stuck_prone": 0.08},
    "Salt":      {"hardness": 3.0, "abrasiveness": 0.3, "stuck_prone": 0.45},
}

# bit efficiency multiplier and wear rate
BIT_PROPS = {
    "PDC":     {"rop_mult": 1.20, "wear_rate": 1.0, "hard_penalty": 1.3},
    "Tricone": {"rop_mult": 1.00, "wear_rate": 1.2, "hard_penalty": 0.9},
    "Diamond": {"rop_mult": 0.90, "wear_rate": 0.6, "hard_penalty": 0.7},
    "Hybrid":  {"rop_mult": 1.10, "wear_rate": 0.9, "hard_penalty": 1.0},
}


def _formation_for_depth(depth: float, well_seed: int) -> str:
    """Assign a formation based on depth bands with per-well variation."""
    rng = np.random.default_rng(well_seed + int(depth // 1500))
    band = int(depth // 1500) % len(FORMATION_TYPES)
    # 80% follow the depth band, 20% random for realism
    if rng.random() < 0.8:
        return FORMATION_TYPES[band]
    return FORMATION_TYPES[rng.integers(0, len(FORMATION_TYPES))]


def _simulate_well(well_idx: int, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a single well's drilling time-series."""
    well_id = f"WELL-{well_idx:03d}"
    well_seed = settings.random_seed + well_idx

    # Starting conditions vary by well
    start_depth = rng.uniform(500, 2500)
    start_time = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
    bit_type = rng.choice(BIT_TYPES, p=[0.45, 0.30, 0.10, 0.15])
    bit = BIT_PROPS[bit_type]
    mud_density = rng.uniform(9.0, 14.5)  # ppg

    records = []
    depth = start_depth
    bit_wear = rng.uniform(0.0, 0.1)         # 0..1 fraction worn
    drilling_hours = rng.uniform(0, 20)
    timestamp = start_time
    # operational drift baselines (slowly varying per well)
    wob_base = rng.uniform(15, 35)            # klbs
    rpm_base = rng.uniform(80, 160)           # rpm
    flow_base = rng.uniform(450, 750)         # gpm

    for _ in range(n_rows):
        formation = _formation_for_depth(depth, well_seed)
        fp = FORMATION_PROPS[formation]
        hardness = fp["hardness"]

        # --- operational parameters with drift + autocorrelated noise ---
        wob = float(np.clip(wob_base + rng.normal(0, 2.5), 5, 55))
        rpm = float(np.clip(rpm_base + rng.normal(0, 8), 40, 220))
        mud_flow_rate = float(np.clip(flow_base + rng.normal(0, 25), 300, 900))

        # --- derived measurements (physics-inspired) ---
        # torque grows with WOB, hardness, and bit wear
        torque = (
            0.9 * wob * (1 + 0.06 * hardness) * (1 + 0.5 * bit_wear)
            + rng.normal(0, 3)
        )
        torque = float(np.clip(torque, 2, 120))

        standpipe_pressure = float(
            np.clip(800 + 1.6 * mud_flow_rate + 8 * mud_density + rng.normal(0, 60), 800, 5000)
        )
        pump_pressure = float(np.clip(standpipe_pressure * rng.uniform(0.92, 1.02), 700, 5200))
        hook_load = float(np.clip(120 + depth * 0.012 + rng.normal(0, 8), 80, 600))
        flow_out = float(np.clip(mud_flow_rate * rng.uniform(0.90, 1.0), 250, 900))
        differential_pressure = float(np.clip(standpipe_pressure - pump_pressure + rng.normal(0, 40), -300, 600))
        temperature = float(np.clip(70 + depth * 0.011 + rng.normal(0, 5), 70, 320))

        # --- vibration: rises with excessive WOB & RPM and hardness ---
        wob_excess = max(0.0, (wob - 30) / 25.0)
        rpm_excess = max(0.0, (rpm - 140) / 80.0)
        vibration_level = float(
            np.clip(
                1.0
                + 3.5 * wob_excess
                + 3.0 * rpm_excess
                + 0.25 * hardness
                + 2.0 * bit_wear
                + rng.normal(0, 0.5),
                0,
                10,
            )
        )

        # --- ROP physics model ---
        # base inversely proportional to hardness; benefits from WOB & RPM
        # with diminishing returns; penalized by bit wear & vibration
        wob_term = np.log1p(wob) * 6.0
        rpm_term = np.sqrt(rpm) * 1.4
        hardness_term = 35.0 / (hardness + 1.0)
        cleaning = np.clip(mud_flow_rate / 600.0, 0.6, 1.25)  # hole cleaning factor
        wear_penalty = 1.0 - 0.55 * bit_wear
        vib_penalty = 1.0 - 0.06 * vibration_level
        hard_bit = 1.0 - (bit["hard_penalty"] - 1.0) * (hardness / 10.0) * 0.1

        rop = (
            (0.45 * wob_term + 0.45 * rpm_term + hardness_term)
            * bit["rop_mult"]
            * cleaning
            * wear_penalty
            * vib_penalty
            * hard_bit
        )
        rop = float(np.clip(rop + rng.normal(0, 2.0), 1.0, 120.0))

        # --- auxiliary risk / efficiency targets (0..1) ---
        vibration_risk = float(np.clip(vibration_level / 10.0 + rng.normal(0, 0.03), 0, 1))
        bit_damage_risk = float(
            np.clip(bit_wear * 0.7 + 0.03 * vibration_level + 0.02 * (hardness) + rng.normal(0, 0.03), 0, 1)
        )
        stuck_pipe_risk = float(
            np.clip(
                fp["stuck_prone"]
                + 0.25 * max(0.0, (0.85 - cleaning))
                + 0.15 * max(0.0, (mud_density - 13) / 3.0)
                + rng.normal(0, 0.03),
                0,
                1,
            )
        )
        # efficiency: high ROP, low risk = high score
        drilling_efficiency_score = float(
            np.clip(
                0.6 * (rop / 120.0)
                + 0.4 * (1 - (vibration_risk + bit_damage_risk + stuck_pipe_risk) / 3.0),
                0,
                1,
            )
        )

        records.append(
            {
                "timestamp": timestamp,
                "well_id": well_id,
                "depth": round(depth, 2),
                "formation_type": formation,
                "weight_on_bit": round(wob, 2),
                "rpm": round(rpm, 2),
                "torque": round(torque, 2),
                "mud_flow_rate": round(mud_flow_rate, 2),
                "standpipe_pressure": round(standpipe_pressure, 2),
                "hook_load": round(hook_load, 2),
                "bit_type": bit_type,
                "bit_wear": round(bit_wear, 4),
                "mud_density": round(mud_density, 2),
                "vibration_level": round(vibration_level, 3),
                "temperature": round(temperature, 2),
                "drilling_hours": round(drilling_hours, 3),
                "pump_pressure": round(pump_pressure, 2),
                "flow_out": round(flow_out, 2),
                "differential_pressure": round(differential_pressure, 2),
                "rate_of_penetration": round(rop, 3),
                "drilling_efficiency_score": round(drilling_efficiency_score, 4),
                "bit_damage_risk": round(bit_damage_risk, 4),
                "vibration_risk": round(vibration_risk, 4),
                "stuck_pipe_risk": round(stuck_pipe_risk, 4),
            }
        )

        # --- advance state ---
        dt_hours = max(0.05, (rop / 30.0) * rng.uniform(0.8, 1.2))  # time to drill an interval
        timestamp = timestamp + pd.Timedelta(hours=dt_hours)
        depth += rop * dt_hours * rng.uniform(0.9, 1.1)
        drilling_hours += dt_hours
        bit_wear = min(1.0, bit_wear + bit["wear_rate"] * dt_hours * 0.0006 * (1 + 0.05 * hardness))

        # occasional bit trip resets wear
        if bit_wear > 0.9 and rng.random() < 0.3:
            bit_wear = rng.uniform(0.0, 0.05)

        # slow operational drift
        wob_base = float(np.clip(wob_base + rng.normal(0, 0.15), 12, 40))
        rpm_base = float(np.clip(rpm_base + rng.normal(0, 0.4), 70, 180))
        flow_base = float(np.clip(flow_base + rng.normal(0, 1.5), 400, 800))

    return pd.DataFrame.from_records(records)


def _inject_imperfections(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add outliers and simulated missing values to mimic real sensors."""
    df = df.copy()
    n = len(df)

    # Outliers on a few numeric sensors (~0.3%)
    for col, scale in [("torque", 1.6), ("vibration_level", 1.8), ("standpipe_pressure", 1.4)]:
        idx = rng.choice(n, size=int(n * 0.003), replace=False)
        df.loc[df.index[idx], col] = df.loc[df.index[idx], col] * scale

    # Missing values (~0.5%) on selected sensors
    for col in ["mud_density", "temperature", "flow_out", "differential_pressure"]:
        idx = rng.choice(n, size=int(n * 0.005), replace=False)
        df.loc[df.index[idx], col] = np.nan

    return df


def generate(n_rows: int | None = None, n_wells: int | None = None, save: bool = True) -> pd.DataFrame:
    n_rows = n_rows or settings.n_rows
    n_wells = n_wells or settings.n_wells
    rng = np.random.default_rng(settings.random_seed)

    rows_per_well = int(np.ceil(n_rows / n_wells))
    log.info("Generating ~%d rows across %d wells (%d rows/well)...", n_rows, n_wells, rows_per_well)

    frames = []
    for w in range(1, n_wells + 1):
        frames.append(_simulate_well(w, rows_per_well, rng))
        if w % 4 == 0:
            log.info("  ...simulated %d/%d wells", w, n_wells)

    df = pd.concat(frames, ignore_index=True)
    df = df.iloc[:n_rows].reset_index(drop=True)
    df = _inject_imperfections(df, rng)

    log.info("Generated dataset: %d rows x %d columns", df.shape[0], df.shape[1])
    log.info("Mean ROP=%.2f ft/hr | Wells=%d | Formations=%d",
             df["rate_of_penetration"].mean(), df["well_id"].nunique(), df["formation_type"].nunique())

    if save:
        settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(settings.raw_data_path, index=False)
        log.info("Saved -> %s", settings.raw_data_path)
        _write_dataset_docs(df)

    return df


def _write_dataset_docs(df: pd.DataFrame) -> None:
    docs_path = settings.root_dir / "docs" / "dataset_documentation.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Drilling Telemetry Dataset Documentation\n",
        f"- **Rows:** {len(df):,}",
        f"- **Wells:** {df['well_id'].nunique()}",
        f"- **Formations:** {', '.join(sorted(df['formation_type'].unique()))}",
        f"- **Bit types:** {', '.join(sorted(df['bit_type'].unique()))}",
        f"- **Time span:** {df['timestamp'].min()} to {df['timestamp'].max()}\n",
        "## Feature Dictionary\n",
        "| Column | Unit | Description |",
        "|---|---|---|",
        "| timestamp | datetime | Measurement time |",
        "| well_id | id | Well identifier |",
        "| depth | ft | Measured drilling depth |",
        "| formation_type | category | Rock formation being drilled |",
        "| weight_on_bit | klbs | Downward force on bit (WOB) |",
        "| rpm | rev/min | Bit rotation speed |",
        "| torque | klb-ft | Rotational resistance |",
        "| mud_flow_rate | gpm | Mud pump flow rate (in) |",
        "| standpipe_pressure | psi | Circulating system pressure |",
        "| hook_load | klbs | Weight at the hook |",
        "| bit_type | category | Drill bit type |",
        "| bit_wear | 0-1 | Fractional bit wear |",
        "| mud_density | ppg | Drilling fluid density |",
        "| vibration_level | 0-10 | Composite vibration index |",
        "| temperature | degF | Downhole temperature |",
        "| drilling_hours | hr | Cumulative drilling hours |",
        "| pump_pressure | psi | Pump discharge pressure |",
        "| flow_out | gpm | Return flow rate |",
        "| differential_pressure | psi | SPP - pump pressure |",
        "| **rate_of_penetration** | ft/hr | **TARGET** — drilling speed |",
        "| drilling_efficiency_score | 0-1 | Derived efficiency target |",
        "| bit_damage_risk | 0-1 | Derived risk target |",
        "| vibration_risk | 0-1 | Derived risk target |",
        "| stuck_pipe_risk | 0-1 | Derived risk target |",
        "\n## Statistical Summary (numeric)\n",
        "```",
        df.describe().round(2).to_string(),
        "```",
    ]
    docs_path.write_text("\n".join(lines))
    log.info("Dataset documentation -> %s", docs_path)


if __name__ == "__main__":
    generate()
