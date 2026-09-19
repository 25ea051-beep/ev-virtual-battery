

import argparse
import math
import os
import sys
import time
import warnings
from copy import deepcopy

# Ensure UTF-8 stdout encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_can_dataset.csv")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "virtual_battery_model.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCV-SOC LOOKUP TABLE  (NMC 811 chemistry, per cell)
# ─────────────────────────────────────────────────────────────────────────────

_OCV_SOC_POINTS = np.array([
    0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00
], dtype=np.float64)

_OCV_VOLTAGE_POINTS = np.array([
    3.00, 3.45, 3.53, 3.60, 3.65, 3.68, 3.71, 3.73,
    3.75, 3.77, 3.79, 3.82, 3.84, 3.87, 3.90, 3.93,
    3.97, 4.00, 4.05, 4.08, 4.13, 4.17, 4.20
], dtype=np.float64)


def get_cell_ocv(soc: float | np.ndarray) -> float | np.ndarray:
    """Interpolate open-circuit voltage for NMC cell(s) at given SOC."""
    soc_clipped = np.clip(soc, 0.0, 1.0)
    result = np.interp(soc_clipped, _OCV_SOC_POINTS, _OCV_VOLTAGE_POINTS)
    return float(result) if np.ndim(soc) == 0 else result


def get_pack_ocv(soc: float, n_series: int = 96) -> float:
    """Pack OCV = cell OCV × number of series cells."""
    return float(get_cell_ocv(soc) * n_series)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TWO-RC THEVENIN CELL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TwoRCThevenin:
    """
    Equivalent circuit:
        V_terminal = OCV(SOC) - I*R0 - V1 - V2

    R0          : ohmic (instantaneous) resistance
    R1, C1      : fast RC pair  (tau1 = R1*C1 ≈ 30–60 s)
    R2, C2      : slow RC pair  (tau2 = R2*C2 ≈ 500–2000 s, diffusion)

    All parameters degrade with age and temperature.
    """

    def __init__(self, soh: float = 1.0, age_years: float = 0.0,
                 temp_c: float = 25.0, n_series: int = 96):
        self.soh = soh
        self.age = age_years
        self.T   = temp_c
        self.n   = n_series

        # Temperature correction (cold increases internal resistance)
        t_factor = 1.0 + max(0.0, (25.0 - temp_c) * 0.015)

        # Per-cell parameters (scale with age and temperature)
        self.R0 = (0.003 + 0.001 * age_years) * t_factor          # Ohms per cell
        self.R1 = (0.001 + 0.0005 * age_years) * t_factor
        self.C1 = max(500.0, 3000.0 - 80.0 * age_years)
        self.R2 = (0.0008 + 0.0003 * age_years) * t_factor
        self.C2 = max(5000.0, 18000.0 - 400.0 * age_years)

        self.V1 = 0.0   # state variable: voltage across RC1
        self.V2 = 0.0   # state variable: voltage across RC2

    def step(self, soc: float, I_pack: float, dt: float = 0.1) -> float:
        """
        Advance model by dt seconds with pack current I_pack (A).
        Returns pack terminal voltage (V).
        """
        I_cell = I_pack  # series string -> identical current through each cell

        # Euler integration of RC transient dynamics
        self.V1 += dt * (I_cell / self.C1 - self.V1 / (self.R1 * self.C1))
        self.V2 += dt * (I_cell / self.C2 - self.V2 / (self.R2 * self.C2))

        cell_ocv = get_cell_ocv(soc)
        V_cell   = cell_ocv - I_cell * self.R0 - self.V1 - self.V2
        V_pack   = V_cell * self.n
        return float(V_pack)

    def pack_resistance(self) -> float:
        """Effective DC pack resistance in Ohms."""
        return (self.R0 + self.R1 + self.R2) * self.n

    @property
    def polarization_voltage(self) -> float:
        """Combined transient polarization voltage across both RC branches."""
        return float(self.V1 + self.V2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARRHENIUS AGING MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ArrheniusAging:
    """
    Models capacity fade and resistance growth using:

    Calendar aging : SEI layer growth — Q_cal  = k_cal  * sqrt(days)
    Cycle aging    : Ah-throughput law — Q_cyc  = k_cyc  * N_cycles * DoD^1.5
    Temperature    : Arrhenius thermal acceleration factor
    Knee-point     : Exponential degradation acceleration after ~80% SOH
    """

    Ea    = 31_500.0   # activation energy (J/mol) — typical NMC chemistry
    R_gas = 8.314      # universal gas constant (J/(mol*K))
    T_ref = 298.15     # 25 degC reference temperature (K)

    def _arrhenius(self, temp_c: float) -> float:
        T_k = temp_c + 273.15
        return math.exp(-self.Ea / self.R_gas * (1.0 / T_k - 1.0 / self.T_ref))

    def capacity_fade(self, days: float, n_cycles: int,
                      avg_temp_c: float = 25.0, avg_dod: float = 0.8) -> float:
        """Returns State of Health (SOH) in [0.60, 1.0]."""
        af = self._arrhenius(avg_temp_c)
        k_cal = 1.5e-4 * af
        k_cyc = 4.0e-5 * af * (avg_dod ** 1.5)

        q_cal = k_cal * math.sqrt(max(days, 0.0))
        q_cyc = k_cyc * n_cycles

        linear_fade = q_cal + q_cyc

        # Knee-point: once linear fade > 15% (SOH < 85%), non-linear aging accelerates
        if linear_fade > 0.15:
            knee_extra = 0.8 * (linear_fade - 0.15) ** 1.6
        else:
            knee_extra = 0.0

        soh = max(0.60, 1.0 - linear_fade - knee_extra)
        return round(soh, 5)

    def resistance_growth(self, soh: float) -> float:
        """Internal resistance grows non-linearly as capacity fades (Ohms/cell)."""
        fade = 1.0 - soh
        return 0.003 + 0.015 * fade + 0.04 * (fade ** 2)

    def cycles_from_age(self, age_years: float, cycle: str = "urban") -> int:
        """Estimate equivalent full cycles from age + drive cycle intensity."""
        cycles_per_year = {
            "urban": 300, "highway": 250, "aggressive": 400, "idle": 50
        }
        return int(age_years * cycles_per_year.get(cycle, 300))


# ─────────────────────────────────────────────────────────────────────────────
# 4. BATTERY PACK WITH CELL IMBALANCE (Vectorized NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class BatteryPack:
    """
    96 cells in series (typical 400 V EV pack).
    Vectorized representation for fast multi-session simulation.
    Captures cell-to-cell variance in SOH, SOC, and internal resistance.
    """

    def __init__(self, n_series: int = 96, age_years: float = 0.0,
                 avg_temp_c: float = 25.0, seed: int = 42,
                 initial_soc: float = 0.80):
        self.n     = n_series
        self.age   = age_years
        self.aging = ArrheniusAging()

        n_cycles = self.aging.cycles_from_age(age_years)
        base_soh = self.aging.capacity_fade(
            days=age_years * 365, n_cycles=n_cycles, avg_temp_c=avg_temp_c
        )
        base_ir = self.aging.resistance_growth(base_soh)

        # Per-cell manufacturing and aging variance
        rng = np.random.default_rng(seed=seed)
        soh_var  = rng.normal(0, 0.003, n_series)
        soc_var  = rng.normal(0, 0.005, n_series)
        ir_scale = rng.normal(1.0, 0.05, n_series)

        self.soh = np.clip(base_soh + soh_var, 0.60, 1.0).astype(np.float64)
        self.soc = np.clip(initial_soc + soc_var, 0.05, 1.0).astype(np.float64)
        self.ir  = np.clip(base_ir * ir_scale, 0.001, 0.10).astype(np.float64)

    # ── public properties & helpers ──────────────────────────────────────────

    def weakest_soh(self) -> float:
        return float(np.min(self.soh))

    def mean_soh(self) -> float:
        return float(np.mean(self.soh))

    def soc_spread(self) -> float:
        return float(np.ptp(self.soc))

    def mean_soc(self) -> float:
        return float(np.mean(self.soc))

    def update_soc(self, I: float, dt: float, capacity_ah: float):
        """Apply current to every cell's SOC (Coulomb counting)."""
        d_soc = (I * dt) / (capacity_ah * 3600.0)
        self.soc = np.clip(self.soc - d_soc, 0.02, 1.0)

    def pack_voltage(self, I: float, v_rc_per_cell: float = 0.0) -> tuple[float, float, float]:
        """
        Returns (pack_voltage, min_cell_v, max_cell_v).
        Incorporates both ohmic drop (I*R) and transient polarization (V_rc).
        """
        ocvs = get_cell_ocv(self.soc)
        cell_vs = ocvs - I * self.ir - v_rc_per_cell
        return float(np.sum(cell_vs)), float(np.min(cell_vs)), float(np.max(cell_vs))

    @property
    def cells(self) -> list[dict]:
        """Backwards-compatible list-of-dicts access to cell states."""
        return [
            {"soh": float(self.soh[i]), "soc": float(self.soc[i]), "ir": float(self.ir[i])}
            for i in range(self.n)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. CAN SENSOR NOISE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CANSensorModel:
    """
    Adds realistic sensor noise, quantization, and drift:
      - Voltage sensor : drift + Gaussian noise + 0.1 V resolution quantization
      - Current sensor : fixed Hall offset + Gaussian noise + 0.5 A resolution
      - Temperature    : thermal drift + Gaussian noise + 0.5 degC resolution
      - SOC (BMS est.) : Coulomb-counting bias integration error
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._v_drift  = 0.0
        self._t_drift  = 0.0
        self._i_offset = float(rng.normal(0, 0.3))   # fixed hall-sensor calibration offset
        self._soc_err  = 0.0
        self._rng      = rng

    def voltage(self, true_v: float, dt: float = 0.1) -> float:
        self._v_drift += float(self._rng.normal(0, 0.002)) * dt
        self._v_drift  = float(np.clip(self._v_drift, -3.0, 3.0))
        noisy = true_v + self._v_drift + float(self._rng.normal(0, 0.15))
        return round(noisy * 10.0) / 10.0          # 0.1 V resolution

    def current(self, true_i: float) -> float:
        noisy = true_i + self._i_offset + float(self._rng.normal(0, 0.4))
        return round(noisy * 2.0) / 2.0            # 0.5 A resolution

    def temperature(self, true_t: float, dt: float = 0.1) -> float:
        self._t_drift += float(self._rng.normal(0, 0.001)) * dt
        self._t_drift  = float(np.clip(self._t_drift, -1.5, 1.5))
        noisy = true_t + self._t_drift + float(self._rng.normal(0, 0.2))
        return round(noisy * 2.0) / 2.0            # 0.5 degC resolution

    def soc_estimate(self, true_soc: float, I: float, dt: float) -> float:
        """Simulate BMS Coulomb-counting drift over prolonged drive cycles."""
        self._soc_err += float(self._rng.normal(0, 5e-6)) * abs(I) * dt
        self._soc_err  = float(np.clip(self._soc_err, -0.05, 0.05))
        return float(np.clip(true_soc + self._soc_err, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRIVE CYCLE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_CYCLES = {
    "urban": {
        "I_rms":     80.0,    # A RMS pack current during discharge
        "I_peak":    180.0,   # A peak load during acceleration
        "freq_hz":   0.05,    # load pattern frequency
        "regen":     0.25,    # probability of regenerative braking event
        "I_regen":   -30.0,   # A during regen (negative = charging)
        "temp_rise": 8.0,     # degC above ambient
        "stress":    1.0,     # aging stress multiplier
    },
    "highway": {
        "I_rms":     120.0,
        "I_peak":    200.0,
        "freq_hz":   0.02,
        "regen":     0.10,
        "I_regen":   -20.0,
        "temp_rise": 12.0,
        "stress":    0.75,
    },
    "aggressive": {
        "I_rms":     200.0,
        "I_peak":    380.0,
        "freq_hz":   0.08,
        "regen":     0.20,
        "I_regen":   -60.0,
        "temp_rise": 22.0,
        "stress":    1.6,
    },
    "idle": {
        "I_rms":     5.0,
        "I_peak":    10.0,
        "freq_hz":   0.001,
        "regen":     0.0,
        "I_regen":   0.0,
        "temp_rise": 1.0,
        "stress":    0.05,
    },
}


def sample_current(cycle: str, t: float, rng: np.random.Generator) -> float:
    """
    Generate instantaneous pack current (A) for a given drive cycle and time.
    Positive = discharge, negative = regen / charging.
    """
    p   = DRIVE_CYCLES[cycle]
    sin = math.sin(2 * math.pi * p["freq_hz"] * t)
    I   = p["I_rms"] + p["I_peak"] * 0.35 * sin
    I  += float(rng.normal(0, p["I_rms"] * 0.06))

    # Occasional regenerative braking
    if rng.random() < p["regen"] * 0.02:
        I = p["I_regen"] + float(rng.normal(0, 3.0))

    return float(I)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYNTHETIC CAN FRAME GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCANGenerator:
    """
    Produces BMS-style CAN frames at a configurable sample rate.
    Combines 2RC Thevenin ECM, Arrhenius aging, and sensor noise.
    """

    def __init__(self, age_years: float = 0.0, ambient_temp: float = 25.0,
                 cycle: str = "urban", n_series: int = 96,
                 nominal_cap_ah: float = 200.0, seed: int = 42,
                 initial_soc: float = 0.80):
        self.age    = age_years
        self.T_amb  = ambient_temp
        self.cycle  = cycle
        self.n      = n_series
        self.cap_ah = nominal_cap_ah

        self._aging  = ArrheniusAging()
        n_cycles     = self._aging.cycles_from_age(age_years, cycle)
        self.soh     = self._aging.capacity_fade(
            days=age_years * 365, n_cycles=n_cycles, avg_temp_c=ambient_temp
        )
        self.cap_now = nominal_cap_ah * self.soh

        # Seed per-pack variation based on run parameters
        pack_seed    = int(seed + age_years * 100) & 0xFFFFFFFF
        self._pack   = BatteryPack(n_series, age_years, ambient_temp, seed=pack_seed, initial_soc=initial_soc)
        self._ecm    = TwoRCThevenin(self.soh, age_years, ambient_temp, n_series)
        self._sensor = CANSensorModel(seed=seed)
        self._rng    = np.random.default_rng(seed)

        self._t      = 0.0
        rise = DRIVE_CYCLES[cycle]["temp_rise"] if cycle in DRIVE_CYCLES else 8.0
        self._temp   = ambient_temp + rise * 0.1

    def next_frame(self, dt: float = 0.1, current: float | None = None) -> dict:
        """Advance simulation by dt seconds and return one CAN-style record."""
        if current is not None:
            I_true = float(current)
        elif self.cycle in DRIVE_CYCLES:
            I_true = sample_current(self.cycle, self._t, self._rng)
        else:
            I_true = 0.0

        # Thermal model: cell temperature converges exponentially toward steady-state
        if self.cycle in DRIVE_CYCLES:
            T_ss = self.T_amb + DRIVE_CYCLES[self.cycle]["temp_rise"]
        else:
            T_ss = self.T_amb + min(35.0, (abs(I_true) / 200.0) ** 2 * 20.0)
        self._temp += (T_ss - self._temp) * 0.001 * dt

        soc_true = self._pack.mean_soc()

        # Advance 2RC equivalent circuit model
        V_true = self._ecm.step(soc_true, I_true, dt)
        v_rc_cell = self._ecm.polarization_voltage

        # Pack imbalance with cell-level polarization alignment
        _, V_min, V_max = self._pack.pack_voltage(I_true, v_rc_per_cell=v_rc_cell)
        V_spread = V_max - V_min

        # Coulomb count cell states
        self._pack.update_soc(I_true, dt, self.cap_now)

        # Apply sensor noise and quantization
        V_meas   = self._sensor.voltage(V_true, dt)
        I_meas   = self._sensor.current(I_true)
        T_meas   = self._sensor.temperature(self._temp, dt)
        SOC_meas = self._sensor.soc_estimate(soc_true, I_true, dt)

        frame = {
            "timestamp":     round(self._t, 2),
            "voltage":       V_meas,
            "current":       I_meas,
            "soc":           round(SOC_meas, 4),
            "soh":           round(self.soh, 4),
            "temp_cell":     T_meas,
            "ir_pack":       round(self._ecm.pack_resistance(), 4),
            "v_cell_min":    round(V_min, 3),
            "v_cell_max":    round(V_max, 3),
            "v_cell_spread": round(V_spread, 3),
            "soc_spread":    round(self._pack.soc_spread(), 4),
            "weakest_soh":   round(self._pack.weakest_soh(), 4),
            "age_years":     self.age,
            "cycle":         self.cycle,
        }

        self._t += dt
        return frame

    def generate_session(self, duration_s: float = 1800.0,
                         dt: float = 0.5) -> pd.DataFrame:
        """Generate a full driving session as a DataFrame."""
        n_steps = int(duration_s / dt)
        frames = [self.next_frame(dt) for _ in range(n_steps)]
        return pd.DataFrame(frames)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE EXTRACTOR  (CAN window → ML features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> dict:
    """
    Derive health-relevant features from a sliding window of CAN frames.
    Dynamically infers sampling rate dt.
    """
    v = df["voltage"]
    i = df["current"]
    t = df["temp_cell"]

    # Calculate actual dt from timestamps
    if len(df) > 1 and "timestamp" in df.columns:
        dt = float(np.median(np.diff(df["timestamp"])))
    else:
        dt = 0.5

    # dV/dQ — indicator of capacity fade and phase transitions
    q = i.cumsum() * dt / 3600.0          # Ah throughput integral
    dv = v.diff()
    dq = q.diff().replace(0, np.nan)
    dvdq = (dv / dq).abs().dropna()
    dvdq_mean = float(dvdq.mean()) if not dvdq.empty and not np.isnan(dvdq.mean()) else 0.0

    # Internal Resistance (IR) estimation from voltage drop during current pulses
    high_i_mask = i.abs() > i.abs().quantile(0.75)
    if high_i_mask.sum() > 5:
        delta_v = v[~high_i_mask].mean() - v[high_i_mask].mean()
        delta_i = i[high_i_mask].mean() - i[~high_i_mask].mean()
        ir_est = delta_v / (delta_i + 1e-9)
    else:
        ir_est = df["ir_pack"].mean() if "ir_pack" in df.columns else 0.0

    features = {
        "mean_voltage":       float(v.mean()),
        "std_voltage":        float(v.std() if len(v) > 1 else 0.0),
        "min_voltage":        float(v.min()),
        "voltage_range":      float(v.max() - v.min()),
        "mean_current":       float(i.mean()),
        "peak_current":       float(i.abs().max()),
        "rms_current":        float(math.sqrt((i ** 2).mean())),
        "mean_temp":          float(t.mean()),
        "max_temp":           float(t.max()),
        "temp_rise":          float(t.max() - t.min()),
        "ir_estimate":        float(abs(ir_est)),
        "soc_swing":          float(df["soc"].max() - df["soc"].min()),
        "dvdq_mean":          float(dvdq_mean),
        "v_cell_spread_mean": float(df["v_cell_spread"].mean() if "v_cell_spread" in df.columns else 0.0),
        "soc_spread_mean":    float(df["soc_spread"].mean() if "soc_spread" in df.columns else 0.0),
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 9. CAN-TO-VIRTUAL-BATTERY MACHINE LEARNING MODEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class VirtualBatteryCANModel:
    """
    Multi-target machine learning model pipeline that converts raw CAN bus
    signals into complete, high-fidelity Virtual Battery Digital Twin states.
    Trained on 1,000 empirical ground-truth battery records.
    """

    FEATURE_COLS = ["Cycle_Count", "Voltage_V", "Current_A", "Temperature_C", "SOC_percent"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "soh": GradientBoostingRegressor(
                n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42
            ),
            "capacity": GradientBoostingRegressor(
                n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42
            ),
            "ir": GradientBoostingRegressor(
                n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42
            ),
            "condition_score": GradientBoostingRegressor(
                n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42
            ),
        }
        self.clf_condition = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
        self.clf_second_life = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
        self.metrics: dict = {}
        self.is_trained: bool = False

    def train(self, dataset: pd.DataFrame | str = DEFAULT_DATASET_PATH, test_size: float = 0.2) -> dict:
        """Train all regression and classification sub-models on the CAN dataset."""
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
        else:
            df = dataset.copy()

        print(f"[VirtualBatteryModel] Training on {len(df)} battery records ...")
        X = df[self.FEATURE_COLS]
        X_scaled = self.scaler.fit_transform(X)

        X_tr, X_te, idx_tr, idx_te = train_test_split(
            X_scaled, df.index, test_size=test_size, random_state=42
        )

        target_map = {
            "soh": "SOH_percent",
            "capacity": "Capacity_Ah",
            "ir": "Internal_Resistance_Ohm",
            "condition_score": "Condition_Score",
        }

        for name, col in target_map.items():
            y = df[col].values
            self.models[name].fit(X_tr, y[idx_tr])
            preds = self.models[name].predict(X_te)
            self.metrics[name] = {
                "r2": float(r2_score(y[idx_te], preds)),
                "mae": float(mean_absolute_error(y[idx_te], preds)),
            }
            print(f"  Target [{name:15s}] -> R^2: {self.metrics[name]['r2']:.4f}, MAE: {self.metrics[name]['mae']:.4f}")

        y_cond = df["Battery_Condition"].values
        self.clf_condition.fit(X_tr, y_cond[idx_tr])
        cond_preds = self.clf_condition.predict(X_te)
        self.metrics["battery_condition"] = {
            "accuracy": float(accuracy_score(y_cond[idx_te], cond_preds))
        }
        print(f"  Target [battery_condition] -> Accuracy: {self.metrics['battery_condition']['accuracy']*100:.2f}%")

        y_sl = df["Second_Life_Status"].values
        self.clf_second_life.fit(X_tr, y_sl[idx_tr])
        sl_preds = self.clf_second_life.predict(X_te)
        self.metrics["second_life_status"] = {
            "accuracy": float(accuracy_score(y_sl[idx_te], sl_preds))
        }
        print(f"  Target [second_life_status]-> Accuracy: {self.metrics['second_life_status']['accuracy']*100:.2f}%")

        self.is_trained = True
        return self.metrics

    def save(self, model_path: str = DEFAULT_MODEL_PATH):
        """Serialize trained model bundle to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "models": self.models,
            "clf_condition": self.clf_condition,
            "clf_second_life": self.clf_second_life,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
            "feature_cols": self.FEATURE_COLS,
        }, model_path)
        print(f"[VirtualBatteryModel] Model successfully saved to: {model_path}")

    def load(self, model_path: str = DEFAULT_MODEL_PATH) -> bool:
        """Load trained model bundle from disk."""
        if not os.path.exists(model_path):
            return False
        try:
            data = joblib.load(model_path)
            self.scaler = data["scaler"]
            self.models = data["models"]
            self.clf_condition = data["clf_condition"]
            self.clf_second_life = data["clf_second_life"]
            self.metrics = data.get("metrics", {})
            self.is_trained = data.get("is_trained", True)
            print(f"[VirtualBatteryModel] Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"[VirtualBatteryModel] Failed loading model: {e}")
            return False

    def generate_virtual_battery(self, can_data: dict, n_series: int = 96) -> dict:
        """
        Generate complete virtual battery digital twin state from CAN signals.
        """
        if not self.is_trained:
            raise RuntimeError("VirtualBatteryCANModel is not trained. Call .train() or .load() first.")

        v = float(can_data.get("voltage", can_data.get("Voltage_V", 350.0)))
        i = float(can_data.get("current", can_data.get("Current_A", 0.0)))
        t = float(can_data.get("temp_cell", can_data.get("Temperature_C", 25.0)))

        soc_raw = float(can_data.get("soc", can_data.get("SOC_percent", 80.0)))
        soc_pct = soc_raw * 100.0 if soc_raw <= 1.0 else soc_raw
        soc_pct = float(np.clip(soc_pct, 0.0, 100.0))

        if "cycle_count" in can_data or "Cycle_Count" in can_data:
            cycles = float(can_data.get("cycle_count", can_data.get("Cycle_Count", 500.0)))
        elif "age_years" in can_data:
            cycles = float(can_data["age_years"]) * 180.0
        else:
            cycles = 500.0

        feat_df = pd.DataFrame([[cycles, v, i, t, soc_pct]], columns=self.FEATURE_COLS)
        feat_scaled = self.scaler.transform(feat_df)

        soh = float(np.clip(self.models["soh"].predict(feat_scaled)[0], 50.0, 100.5))
        cap = float(np.clip(self.models["capacity"].predict(feat_scaled)[0], 25.0, 65.0))
        ir = float(max(0.015, self.models["ir"].predict(feat_scaled)[0]))
        score = float(np.clip(self.models["condition_score"].predict(feat_scaled)[0], 40.0, 100.0))

        cond = str(self.clf_condition.predict(feat_scaled)[0])
        second_life = str(self.clf_second_life.predict(feat_scaled)[0])

        energy_kwh = round((cap * (soc_pct / 100.0) * v) / 1000.0, 2)
        power_kw = round((v * i) / 1000.0, 2)

        rng = np.random.default_rng(int(cycles + v * 10) & 0xFFFFFFFF)
        v_cell_mean = v / n_series
        v_spread_sigma = max(0.004, 0.012 * (1.0 - (soh / 100.0)) + ir * 0.05)
        cell_voltages = np.clip(rng.normal(v_cell_mean, v_spread_sigma, n_series), 2.80, 4.30)
        v_cell_min = float(np.min(cell_voltages))
        v_cell_max = float(np.max(cell_voltages))
        v_cell_spread = float(v_cell_max - v_cell_min)

        cell_soh = np.clip(rng.normal(soh / 100.0, 0.005, n_series), 0.50, 1.0)
        weakest_soh = float(np.min(cell_soh))

        alerts = []
        if soh < 70.0:
            alerts.append({"level": "CRITICAL", "msg": f"SOH Critical Knee Reached: {soh:.1f}%"})
        elif soh < 80.0:
            alerts.append({"level": "WARNING", "msg": f"Battery Degradation Advisory: SOH {soh:.1f}%"})

        if t > 50.0:
            alerts.append({"level": "CRITICAL", "msg": f"Pack Temperature Overtemperature: {t:.1f}°C"})
        elif t > 40.0:
            alerts.append({"level": "WARNING", "msg": f"Elevated Thermal Load: {t:.1f}°C"})

        if soc_pct < 15.0:
            alerts.append({"level": "WARNING", "msg": f"Low Battery Pack SOC: {soc_pct:.1f}%"})

        return {
            "can_input": {
                "voltage_v": round(v, 2),
                "current_a": round(i, 2),
                "temperature_c": round(t, 2),
                "soc_percent": round(soc_pct, 2),
                "cycle_count": int(cycles),
            },
            "virtual_battery": {
                "soh_percent": round(soh, 2),
                "capacity_ah": round(cap, 2),
                "internal_resistance_ohm": round(ir, 4),
                "internal_resistance_mohm": round(ir * 1000.0, 2),
                "energy_remaining_kwh": energy_kwh,
                "power_kw": power_kw,
                "condition_score": round(score, 2),
                "battery_condition": cond,
                "second_life_status": second_life,
                "v_cell_min": round(v_cell_min, 3),
                "v_cell_max": round(v_cell_max, 3),
                "v_cell_spread_mv": round(v_cell_spread * 1000.0, 1),
                "weakest_cell_soh_pct": round(weakest_soh * 100.0, 2),
            },
            "cell_voltages": [round(float(x), 3) for x in cell_voltages],
            "alerts": alerts,
        }


def get_or_train_model(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    force_retrain: bool = False
) -> VirtualBatteryCANModel:
    """Factory helper: loads or trains VirtualBatteryCANModel."""
    model = VirtualBatteryCANModel()
    if not force_retrain and model.load(model_path):
        return model

    if os.path.exists(dataset_path):
        print(f"[VirtualBatteryModel] Training new model on {dataset_path} ...")
        df = pd.read_csv(dataset_path)
        model.train(df)
        model.save(model_path)
    else:
        raise FileNotFoundError(f"CAN Dataset not found at: {dataset_path}")

    return model


class SOHPredictor:
    """
    SOH Predictor utilizing the trained CAN model and window feature extraction.
    """

    def __init__(self):
        try:
            self.can_model = get_or_train_model()
            self.trained = True
        except Exception:
            self.can_model = None
            self.trained = False

    def train(self, dataset: pd.DataFrame | str = DEFAULT_DATASET_PATH):
        if self.can_model is None:
            self.can_model = VirtualBatteryCANModel()
        metrics = self.can_model.train(dataset)
        self.can_model.save()
        self.trained = True
        return metrics

    def predict(self, window_df: pd.DataFrame) -> float:
        """Predict SOH from a DataFrame window of CAN frames."""
        if not self.trained or not self.can_model:
            raise RuntimeError("Model is not trained.")
        v = float(window_df["voltage"].mean())
        i = float(window_df["current"].mean())
        t = float(window_df["temp_cell"].mean())
        soc = float(window_df["soc"].iloc[-1])
        cycles = float(window_df["cycle_count"].iloc[-1]) if "cycle_count" in window_df.columns else 500.0
        can_data = {"voltage": v, "current": i, "temp_cell": t, "soc": soc, "cycle_count": cycles}
        res = self.can_model.generate_virtual_battery(can_data)
        return float(res["virtual_battery"]["soh_percent"] / 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. VIRTUAL BATTERY  (the digital twin class)
# ─────────────────────────────────────────────────────────────────────────────

class VirtualBattery:
    """
    The digital twin. Advances physics and sensor models forward in time,
    maintains telemetry history, performs ML SOH inference, and dispatches
    cooldown-governed alerts.

    Usage:
        twin = VirtualBattery(age_years=3, cycle="urban", ambient_temp=28)
        for _ in range(1000):
            state = twin.tick()
    """

    ALERT_SOH_WARN    = 0.80
    ALERT_SOH_CRIT    = 0.70
    ALERT_TEMP_WARN   = 45.0
    ALERT_TEMP_CRIT   = 55.0
    ALERT_VMIN_WARN   = 3.10   # per cell
    ALERT_SOC_LOW     = 0.15

    def __init__(self, age_years: float = 0.0, cycle: str = "urban",
                 ambient_temp: float = 25.0, dt: float = 0.5,
                 initial_soc: float = 0.80,
                 nominal_cap_ah: float = 200.0,
                 n_series: int = 96,
                 current: float | None = None,
                 predictor: "SOHPredictor | None" = None,
                 can_model: "VirtualBatteryCANModel | None" = None,
                 predict_window_s: int = 300,
                 predict_interval_s: float = 30.0):
        self._gen = SyntheticCANGenerator(
            age_years=age_years, ambient_temp=ambient_temp,
            cycle=cycle, n_series=n_series, nominal_cap_ah=nominal_cap_ah,
            seed=int(age_years * 100), initial_soc=initial_soc
        )
        self.dt                 = dt
        self.dynamic_current    = current
        self._predictor         = predictor
        self.can_model          = can_model
        if self.can_model is None:
            try:
                self.can_model = get_or_train_model()
            except Exception:
                self.can_model = None

        self._predict_window    = max(1, int(predict_window_s / dt))
        self._predict_step_freq = max(1, int(predict_interval_s / dt))
        self._last_predicted_soh = None
        self._step_counter      = 0

        self.history: list[dict] = []
        self.alerts:  list[dict] = []
        self._active_alerts: dict[str, float] = {}

    def set_current(self, current: float | None):
        """Set dynamic load current (A). Positive = discharge, negative = charge/regen, None = auto drive cycle."""
        self.dynamic_current = current

    def set_cycle(self, cycle: str):
        """Change active drive cycle."""
        self._gen.cycle = cycle

    @classmethod
    def from_can(cls, can_data: dict, n_series: int = 96) -> dict:
        """
        Generate complete virtual battery digital twin state directly from CAN telemetry.
        Convenient entrypoint: VirtualBattery.from_can(can_data)
        """
        model = get_or_train_model()
        return model.generate_virtual_battery(can_data, n_series=n_series)

    # ── main tick ──────────────────────────────────────────────────────────

    def tick(self, current: float | None = None) -> dict:
        """Advance simulation by one dt step. Returns current state dict."""
        eff_current = current if current is not None else self.dynamic_current
        frame = self._gen.next_frame(self.dt, current=eff_current)
        self.history.append(frame)
        self._step_counter += 1

        # ML Virtual Battery generation directly from CAN telemetry
        if self.can_model and self.can_model.is_trained:
            try:
                can_in = dict(frame)
                can_in["cycle_count"] = self._gen._aging.cycles_from_age(self._gen.age, self._gen.cycle)
                vb_out = self.can_model.generate_virtual_battery(can_in, n_series=self._gen.n)
                vb = vb_out["virtual_battery"]
                frame["virtual_battery"] = vb
                frame["predicted_soh"] = round(vb["soh_percent"] / 100.0, 4)
                frame["predicted_capacity_ah"] = vb["capacity_ah"]
                frame["predicted_ir_ohm"] = vb["internal_resistance_ohm"]
                frame["condition_score"] = vb["condition_score"]
                frame["battery_condition"] = vb["battery_condition"]
                frame["second_life_status"] = vb["second_life_status"]
                frame["energy_remaining_kwh"] = vb["energy_remaining_kwh"]
            except Exception:
                pass
        elif self._predictor and len(self.history) >= self._predict_window:
            if self._step_counter % self._predict_step_freq == 0 or self._last_predicted_soh is None:
                window = pd.DataFrame(self.history[-self._predict_window:])
                try:
                    self._last_predicted_soh = self._predictor.predict(window)
                except Exception:
                    pass
            frame["predicted_soh"] = self._last_predicted_soh

        if "predicted_soh" not in frame:
            frame["predicted_soh"] = self._last_predicted_soh

        self._check_alerts(frame)
        return frame

    def run(self, n_steps: int) -> pd.DataFrame:
        """Run n_steps ticks and return full history as DataFrame."""
        for _ in range(n_steps):
            self.tick()
        return pd.DataFrame(self.history)

    # ── alerts with cooldown deduplication ──────────────────────────────────

    def _check_alerts(self, frame: dict):
        t  = frame["timestamp"]
        s  = frame["soh"]
        tc = frame["temp_cell"]
        sc = frame["soc"]
        vc = frame.get("v_cell_min", 999.0)

        if s < self.ALERT_SOH_CRIT:
            self._alert(t, "CRITICAL", "soh_crit", f"SOH critically low: {s*100:.1f}%")
        elif s < self.ALERT_SOH_WARN:
            self._alert(t, "WARNING",  "soh_warn", f"SOH below 80%: {s*100:.1f}%")

        if tc > self.ALERT_TEMP_CRIT:
            self._alert(t, "CRITICAL", "temp_crit", f"Cell overtemperature: {tc:.1f}C")
        elif tc > self.ALERT_TEMP_WARN:
            self._alert(t, "WARNING",  "temp_warn", f"Cell temp elevated: {tc:.1f}C")

        if sc < self.ALERT_SOC_LOW:
            self._alert(t, "WARNING",  "soc_low",  f"Low SOC: {sc*100:.1f}%")

        if vc < self.ALERT_VMIN_WARN:
            self._alert(t, "WARNING",  "vmin_low", f"Cell undervoltage: {vc:.3f} V")

    def _alert(self, t: float, level: str, alert_id: str, msg: str, cooldown_s: float = 60.0):
        """Deduplicates alerts by category using a temporal cooldown."""
        last_t = self._active_alerts.get(alert_id, -float("inf"))
        if t - last_t >= cooldown_s:
            self._active_alerts[alert_id] = t
            entry = {"t": t, "level": level, "id": alert_id, "msg": msg}
            self.alerts.append(entry)
            print(f"  [ALERT @ {t:.0f}s] {level}: {msg}")

    # ── helpers ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.history:
            return {}
        df = pd.DataFrame(self.history)
        return {
            "duration_s":     float(df["timestamp"].max()),
            "mean_soh":       float(df["soh"].mean()),
            "final_soc":      float(df["soc"].iloc[-1]),
            "max_temp":       float(df["temp_cell"].max()),
            "mean_ir_pack":   float(df["ir_pack"].mean()),
            "total_alerts":   len(self.alerts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. DATASET GENERATOR  (bulk synthetic data for model training)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    ages       = np.arange(0, 10.5, 0.5),
    cycles     = ("urban", "highway", "aggressive"),
    temps      = (15.0, 25.0, 35.0),
    duration_s = 1800.0,
    dt         = 0.5,
    output_path: str = "synthetic_can_dataset.parquet"
) -> pd.DataFrame:
    """
    Generate a comprehensive multi-condition synthetic dataset.

    Covers:
      - age 0 -> 10 years (0.5 yr steps)
      - 3 drive cycles
      - 3 ambient temperatures
    Total: 21 * 3 * 3 = 189 sessions * 3600 frames each ~= 680K rows
    """
    records = []
    total   = len(ages) * len(cycles) * len(temps)
    done    = 0

    print(f"[DataGen] Generating {total} sessions ...")
    t0 = time.perf_counter()

    for age in ages:
        for cycle in cycles:
            for temp in temps:
                gen = SyntheticCANGenerator(
                    age_years=age, ambient_temp=temp,
                    cycle=cycle, seed=int(age * 1000 + ord(cycle[0]) + int(temp))
                )
                df = gen.generate_session(duration_s=duration_s, dt=dt)
                df["ambient_temp"] = temp
                records.append(df)
                done += 1
                if done % 20 == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    print(f"  {done}/{total} sessions completed ({elapsed:.1f}s) ...")

    dataset = pd.concat(records, ignore_index=True)

    try:
        dataset.to_parquet(output_path, index=False)
        print(f"[DataGen] Saved {len(dataset):,} rows -> {output_path}")
    except (ImportError, Exception):
        csv_path = output_path.replace(".parquet", ".csv")
        dataset.to_csv(csv_path, index=False)
        print(f"[DataGen] Saved as CSV -> {csv_path}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 12. NASA BATTERY DATASET LOADER  (optional validation)
# ─────────────────────────────────────────────────────────────────────────────

def load_nasa_battery(mat_file_path: str) -> pd.DataFrame:
    """
    Load NASA PCoE battery dataset (B0005-B0008 .mat files).

    Download from:
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    Returns DataFrame with columns: cycle, capacity, soh (normalized to first cycle)
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("Install scipy:  pip install scipy")

    mat = scipy.io.loadmat(mat_file_path, simplify_cells=True)
    key = [k for k in mat if not k.startswith("_")][0]
    cycles_raw = mat[key]["cycle"]

    discharge_caps = []
    for cyc in cycles_raw:
        try:
            if cyc["type"] == "discharge":
                cap = float(np.squeeze(cyc["data"]["Capacity"]))
                discharge_caps.append(cap)
        except (KeyError, TypeError):
            continue

    if not discharge_caps:
        raise ValueError(f"No discharge cycles found in {mat_file_path}")

    cap_series = pd.Series(discharge_caps)
    df = pd.DataFrame({
        "cycle":    range(len(cap_series)),
        "capacity": cap_series.values,
        "soh":      cap_series.values / cap_series.iloc[0],
    })
    print(f"[NASA] Loaded {len(df)} discharge cycles from {mat_file_path}")
    return df


def compare_with_nasa(nasa_df: pd.DataFrame, age_years_max: float = 8.0):
    """
    Print MAE between synthetic SOH curve and NASA experimental SOH curve.
    Maps NASA cycle count -> years (assuming 1 cycle/day average).
    """
    aging    = ArrheniusAging()
    n_cycles = int(len(nasa_df))

    synth_soh = []
    nasa_soh  = []

    for i, row in nasa_df.iterrows():
        frac  = i / n_cycles
        days  = frac * age_years_max * 365
        n_cyc = int(frac * n_cycles)
        s_soh = aging.capacity_fade(days=days, n_cycles=n_cyc)
        synth_soh.append(s_soh)
        nasa_soh.append(row["soh"])

    mae = mean_absolute_error(nasa_soh, synth_soh)
    print(f"[Validation] SOH MAE vs NASA dataset: {mae*100:.2f}%")
    return mae


# ─────────────────────────────────────────────────────────────────────────────
# 13. DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _print_frame(frame: dict, idx: int):
    pred = frame.get("predicted_soh")
    pred_str = f"{pred*100:.1f}%" if pred is not None else "  n/a  "
    score = frame.get("condition_score")
    score_str = f"{score:.1f}" if score is not None else "n/a"
    cond = frame.get("battery_condition", "")
    cond_tag = f"[{cond.upper()}]" if cond else ""
    print(
        f"  t={frame['timestamp']:6.1f}s | "
        f"V={frame['voltage']:6.1f}V | "
        f"I={frame['current']:6.1f}A | "
        f"SOC={frame['soc']*100:5.1f}% | "
        f"SOH={frame['soh']*100:5.1f}% | "
        f"T={frame['temp_cell']:5.1f}C | "
        f"IR={frame['ir_pack']*1000:5.1f}mΩ | "
        f"pred_SOH={pred_str} | "
        f"Score={score_str} {cond_tag}"
    )


def run_demo(predictor: "SOHPredictor | None" = None):
    scenarios = [
        {"age_years": 0.0,  "cycle": "urban",      "ambient_temp": 25.0},
        {"age_years": 3.0,  "cycle": "highway",     "ambient_temp": 30.0},
        {"age_years": 7.0,  "cycle": "aggressive",  "ambient_temp": 38.0},
        {"age_years": 10.0, "cycle": "urban",       "ambient_temp": 20.0},
    ]

    print("\n" + "="*90)
    print("  VIRTUAL EV BATTERY DEMO  -  Synthetic CAN Data Simulation")
    print("="*90)

    for sc in scenarios:
        age   = sc["age_years"]
        cycle = sc["cycle"]
        temp  = sc["ambient_temp"]
        print(f"\n> Scenario: age={age}yr | cycle={cycle} | ambient={temp}C")
        print("-" * 90)

        twin = VirtualBattery(
            age_years=age, cycle=cycle, ambient_temp=temp,
            predictor=predictor
        )

        # Simulate 10 minutes (1200 ticks @ dt=0.5s), print every 60s
        for step in range(1200):
            frame = twin.tick()
            if step % 120 == 0:
                _print_frame(frame, step)

        sm = twin.summary()
        print(f"\n  Summary -> mean SOH: {sm['mean_soh']*100:.1f}%  |  "
              f"max temp: {sm['max_temp']:.1f}C  |  "
              f"alerts fired: {sm['total_alerts']}")

    print("\n" + "="*90)
    print("  Demo complete.")
    print("="*90 + "\n")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# 14. INTERACTIVE DYNAMIC DRIVE & CLI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_float(prompt_text: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    while True:
        try:
            raw = input(f"{prompt_text} [default: {default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  [!] Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  [!] Please enter a valid number.")


def _prompt_choice(prompt_text: str, choices: list[str], default_idx: int = 0) -> str:
    default_str = str(default_idx + 1)
    while True:
        try:
            raw = input(f"{prompt_text} [1-{len(choices)}, default: {default_str}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return choices[default_idx]
        if not raw:
            return choices[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  [!] Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            raw_lower = raw.lower()
            for c in choices:
                if raw_lower in c.lower():
                    return c
            print(f"  [!] Invalid choice: {raw}")


def interactive_live_drive(twin: VirtualBattery):
    """
    Dynamic interactive EV pack driving console.
    Allows user to inject throttle (+A), regenerative braking (-A),
    set exact current, idle, or switch driving cycles on the fly.
    """
    print("\n" + "="*80)
    print("  ⚡ EV VIRTUAL BATTERY - LIVE INTERACTIVE DRIVE CONSOLE")
    print("="*80)
    print("  Controls:")
    print("    [a] or [+]       : Accelerate (+25 A discharge)")
    print("    [b] or [-]       : Brake / Regenerate (-35 A charge)")
    print("    [c <amps>]       : Set exact load current (e.g. 'c 120' or 'c -60' for DC charge)")
    print("    [i]              : Idle / Coast (0 A)")
    print("    [cycle <name>]   : Revert to automatic drive cycle (urban, highway, aggressive, idle)")
    print("    [s <n>]          : Step forward n ticks (default: 1 tick = 0.5s)")
    print("    [p]              : Print detailed 96-cell pack diagnostics")
    print("    [q]              : Quit drive session and display summary")
    print("="*80 + "\n")

    current_val = twin.dynamic_current
    while True:
        try:
            curr_str = f"{current_val:+.1f}A" if current_val is not None else f"Auto ({twin._gen.cycle})"
            cmd = input(f"Drive [Current: {curr_str}] (Enter=step, 'h'=help) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting live drive.")
            break

        if cmd == "q":
            break
        elif cmd in ("h", "help"):
            print("  Commands: [Enter/s]=Step, [a]=Accel, [b]=Regen, [c <amps>]=Set Current, [i]=Idle, [cycle <name>], [p]=Pack Info, [q]=Quit")
            continue
        elif cmd in ("a", "+"):
            base = current_val if current_val is not None else 60.0
            current_val = min(400.0, base + 25.0)
            twin.set_current(current_val)
            print(f"  >> Throttle: {current_val:+.1f} A")
        elif cmd in ("b", "-"):
            base = current_val if current_val is not None else 0.0
            current_val = max(-150.0, base - 35.0)
            twin.set_current(current_val)
            print(f"  >> Regen braking: {current_val:+.1f} A")
        elif cmd == "i":
            current_val = 0.0
            twin.set_current(0.0)
            print("  >> Idle: 0.0 A")
        elif cmd.startswith("c "):
            try:
                val = float(cmd.split()[1])
                current_val = val
                twin.set_current(val)
                print(f"  >> Dynamic current set to: {current_val:+.1f} A")
            except Exception:
                print("  [!] Invalid format. Example: c 120  or  c -50")
                continue
        elif cmd.startswith("cycle"):
            parts = cmd.split()
            cyc = parts[1] if len(parts) > 1 else "urban"
            if cyc in DRIVE_CYCLES:
                twin.set_cycle(cyc)
                twin.set_current(None)
                current_val = None
                print(f"  >> Switched to automatic drive cycle: {cyc.upper()}")
            else:
                print(f"  [!] Unknown cycle. Choose from: {list(DRIVE_CYCLES.keys())}")
                continue
        elif cmd == "p":
            pack = twin._gen._pack
            print(f"\n  -- Pack Diagnostics --")
            print(f"  Cells in series    : {pack.n}")
            print(f"  Mean SOH           : {pack.mean_soh()*100:.2f} % (Weakest: {pack.weakest_soh()*100:.2f} %)")
            print(f"  Mean SOC           : {pack.mean_soc()*100:.2f} % (Spread: {pack.soc_spread()*100:.3f} %)")
            print(f"  Pack DC Resistance : {twin._gen._ecm.pack_resistance()*1000:.2f} mΩ\n")
            continue

        steps = 1
        if cmd.startswith("s"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    steps = max(1, int(parts[1]))
                except ValueError:
                    steps = 1

        for _ in range(steps):
            frame = twin.tick()
        _print_frame(frame, len(twin.history))

        if frame["soc"] <= 0.03:
            print("\n  [!] CRITICAL: Battery fully depleted (SOC <= 3%). Halting drive.")
            break


def interactive_cli_session(predictor: "SOHPredictor | None" = None):
    """Interactive startup wizard for configuring and running the virtual battery twin."""
    print("\n" + "="*80)
    print("  ⚡ EV VIRTUAL BATTERY DIGITAL TWIN - DYNAMIC INPUT WIZARD")
    print("="*80)
    print("  Configure your dynamic battery simulation parameters below.")
    print("  (Press ENTER at any prompt to accept the bracketed default value)\n")

    age = _prompt_float("1. Battery Age in years (0.0=Brand new, 10.0=Aged pack)", default=0.0, min_val=0.0, max_val=15.0)
    soc_pct = _prompt_float("2. Initial State of Charge (SOC) in %", default=80.0, min_val=5.0, max_val=100.0)
    temp = _prompt_float("3. Ambient Temperature in °C", default=25.0, min_val=-20.0, max_val=55.0)
    capacity = _prompt_float("4. Nominal Pack Capacity in Ah", default=200.0, min_val=10.0, max_val=1000.0)

    print("\n5. Select Operating Mode / Drive Cycle:")
    cycle_options = [
        "Interactive Live Drive (Dynamic real-time throttle & regen console)",
        "Generate Virtual Battery Twin directly from CAN Telemetry (ML Model)",
        "Urban Cycle (Stop-and-go city traffic, 80A RMS, frequent regen)",
        "Highway Cycle (Steady high-speed cruise, 120A RMS)",
        "Aggressive / Sport Cycle (High dynamic stress, 200A RMS, 380A peak)",
        "Idle / Standby (Auxiliary loads only, 5A)",
        "Custom Constant Load Current (Discharge or DC Fast Charge)"
    ]
    for i, opt in enumerate(cycle_options, 1):
        print(f"    [{i}] {opt}")

    mode_choice = _prompt_choice("Select mode", cycle_options, default_idx=0)
    initial_soc = soc_pct / 100.0

    if "Generate Virtual Battery Twin" in mode_choice:
        print("\n--- CAN Telemetry Input Configuration ---")
        can_v = _prompt_float("  Enter CAN Voltage in V", default=350.0)
        can_i = _prompt_float("  Enter CAN Current in A", default=45.0)
        can_t = _prompt_float("  Enter CAN Cell Temp in °C", default=28.0)
        can_soc = _prompt_float("  Enter CAN SOC in %", default=soc_pct)
        can_cyc = _prompt_float("  Enter BMS Cycle Count", default=800.0)

        can_data = {
            "Voltage_V": can_v,
            "Current_A": can_i,
            "Temperature_C": can_t,
            "SOC_percent": can_soc,
            "Cycle_Count": can_cyc,
        }
        res = VirtualBattery.from_can(can_data)
        vb = res["virtual_battery"]
        inp = res["can_input"]

        print("\n" + "=" * 76)
        print("  ⚡ GENERATED EV VIRTUAL BATTERY DIGITAL TWIN (FROM CAN DATA)")
        print("=" * 76)
        print("  [INPUT CAN BUS TELEMETRY]")
        print(f"    • Pack Voltage         : {inp['voltage_v']:.2f} V")
        print(f"    • Pack Current         : {inp['current_a']:.2f} A")
        print(f"    • Pack Temperature     : {inp['temperature_c']:.2f} °C")
        print(f"    • State of Charge (SOC): {inp['soc_percent']:.1f} %")
        print(f"    • BMS Cycle Count      : {inp['cycle_count']:,} cycles")
        print("\n  [GENERATED VIRTUAL BATTERY TWIN STATE]")
        print(f"    • State of Health (SOH): {vb['soh_percent']:.2f} %")
        print(f"    • Usable Pack Capacity : {vb['capacity_ah']:.2f} Ah (Nominal: 60.0 Ah)")
        print(f"    • Internal Resistance  : {vb['internal_resistance_mohm']:.2f} mΩ ({vb['internal_resistance_ohm']:.4f} Ω)")
        print(f"    • Energy Remaining     : {vb['energy_remaining_kwh']:.2f} kWh")
        print(f"    • Instantaneous Power  : {vb['power_kw']:.2f} kW")
        print(f"    • Overall Health Score : {vb['condition_score']:.2f} / 100")
        print(f"    • Battery Condition    : [{vb['battery_condition'].upper()}]")
        print(f"    • Second-Life Decision : {vb['second_life_status']}")
        print(f"    • 96S Cell Min / Max   : {vb['v_cell_min']:.3f} V / {vb['v_cell_max']:.3f} V (Spread: {vb['v_cell_spread_mv']:.1f} mV)")
        print(f"    • Weakest Cell SOH     : {vb['weakest_cell_soh_pct']:.2f} %")
        print("=" * 76 + "\n")
        return
    elif "Interactive Live Drive" in mode_choice:
        twin = VirtualBattery(
            age_years=age, cycle="urban", ambient_temp=temp,
            initial_soc=initial_soc, nominal_cap_ah=capacity,
            predictor=predictor
        )
        interactive_live_drive(twin)
    elif "Custom Constant Load Current" in mode_choice:
        load_current = _prompt_float("Enter load current in Amperes (+ discharge, - charging)", default=60.0)
        duration_s = _prompt_float("Simulation duration in seconds", default=300.0, min_val=1.0)
        dt = 0.5
        steps = max(1, int(duration_s / dt))
        twin = VirtualBattery(
            age_years=age, cycle="idle", ambient_temp=temp,
            initial_soc=initial_soc, nominal_cap_ah=capacity,
            current=load_current, predictor=predictor, dt=dt
        )
        print(f"\n> Simulating constant load: {steps} steps @ dt={dt}s, Current={load_current:+.1f}A ...")
        print_interval = max(1, steps // 10)
        for s in range(steps):
            f = twin.tick()
            if s % print_interval == 0 or s == steps - 1:
                _print_frame(f, s)
    else:
        cycle_map = {
            "Urban": "urban",
            "Highway": "highway",
            "Aggressive": "aggressive",
            "Idle": "idle"
        }
        selected_cycle = "urban"
        for k, v in cycle_map.items():
            if k in mode_choice:
                selected_cycle = v
                break

        duration_s = _prompt_float("Simulation duration in seconds", default=600.0, min_val=1.0)
        dt = 0.5
        steps = max(1, int(duration_s / dt))
        twin = VirtualBattery(
            age_years=age, cycle=selected_cycle, ambient_temp=temp,
            initial_soc=initial_soc, nominal_cap_ah=capacity,
            predictor=predictor, dt=dt
        )
        print(f"\n> Simulating {selected_cycle.upper()} cycle for {duration_s:.0f}s ({steps} steps) ...")
        print_interval = max(1, steps // 12)
        for s in range(steps):
            f = twin.tick()
            if s % print_interval == 0 or s == steps - 1:
                _print_frame(f, s)

    sm = twin.summary()
    print("\n" + "="*80)
    print("  SIMULATION COMPLETE - SUMMARY REPORT")
    print("="*80)
    print(f"  Duration Simulated : {sm.get('duration_s', 0):.1f} s")
    print(f"  Mean Pack SOH      : {sm.get('mean_soh', 0)*100:.2f} %")
    print(f"  Final Pack SOC     : {sm.get('final_soc', 0)*100:.2f} %")
    print(f"  Peak Temperature   : {sm.get('max_temp', 0):.1f} °C")
    print(f"  Mean Pack IR       : {sm.get('mean_ir_pack', 0)*1000:.2f} mΩ")
    print(f"  Total Alerts Fired : {sm.get('total_alerts', 0)}")
    print("="*80 + "\n")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EV Virtual Battery Simulator with Dynamic Input")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Launch interactive dynamic setup wizard (default when no args given)")
    parser.add_argument("--age", type=float, default=None,
                        help="Battery age in years (0.0 to 15.0)")
    parser.add_argument("--soc", type=float, default=None,
                        help="Initial battery SOC in percent (5.0 to 100.0) or fraction (0.05 to 1.0)")
    parser.add_argument("--temp", type=float, default=None,
                        help="Ambient temperature in °C (-20 to 55)")
    parser.add_argument("--cycle", type=str, default=None,
                        choices=list(DRIVE_CYCLES.keys()) + ["live"],
                        help="Drive cycle profile or 'live' for interactive drive")
    parser.add_argument("--current", type=float, default=None,
                        help="Constant or initial dynamic load current (A, + discharge, - charge)")
    parser.add_argument("--capacity", type=float, default=200.0,
                        help="Nominal battery pack capacity in Ah (default: 200.0)")
    parser.add_argument("--series", type=int, default=96,
                        help="Number of series cells (default: 96)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of simulation steps to run (default: 200)")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Time step in seconds (default: 0.5)")
    parser.add_argument("--demo", action="store_true",
                        help="Run standard hardcoded 4-scenario demo")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic CAN dataset")
    parser.add_argument("--train", action="store_true",
                        help="Train CAN virtual battery model on ground truth dataset")
    parser.add_argument("--generate-from-can", action="store_true",
                        help="Generate a virtual battery twin state from given CAN inputs")
    parser.add_argument("--can-voltage", type=float, default=350.0,
                        help="CAN pack voltage in V (default: 350.0)")
    parser.add_argument("--can-current", type=float, default=45.0,
                        help="CAN pack current in A (default: 45.0)")
    parser.add_argument("--can-temp", type=float, default=28.0,
                        help="CAN pack temperature in °C (default: 28.0)")
    parser.add_argument("--can-soc", type=float, default=65.0,
                        help="CAN pack SOC in percent (default: 65.0)")
    parser.add_argument("--can-cycles", type=float, default=800.0,
                        help="CAN / BMS cycle count (default: 800.0)")
    parser.add_argument("--all", action="store_true",
                        help="Run generate + train + demo")
    args = parser.parse_args()

    predictor = None
    dataset   = None
    can_model = None

    if args.train:
        print("[Train] Training Virtual Battery CAN Model on ground-truth dataset ...")
        can_model = get_or_train_model(force_retrain=True)
        print("[Train] Training complete!")
        return

    if args.generate_from_can:
        can_model = get_or_train_model()
        can_data = {
            "Voltage_V": args.can_voltage,
            "Current_A": args.can_current,
            "Temperature_C": args.can_temp,
            "SOC_percent": args.can_soc,
            "Cycle_Count": args.can_cycles,
        }
        res = can_model.generate_virtual_battery(can_data)
        vb = res["virtual_battery"]
        inp = res["can_input"]

        print("\n" + "=" * 76)
        print("  ⚡ GENERATED EV VIRTUAL BATTERY DIGITAL TWIN (FROM CAN DATA)")
        print("=" * 76)
        print("  [INPUT CAN BUS TELEMETRY]")
        print(f"    • Pack Voltage         : {inp['voltage_v']:.2f} V")
        print(f"    • Pack Current         : {inp['current_a']:.2f} A")
        print(f"    • Pack Temperature     : {inp['temperature_c']:.2f} °C")
        print(f"    • State of Charge (SOC): {inp['soc_percent']:.1f} %")
        print(f"    • BMS Cycle Count      : {inp['cycle_count']:,} cycles")
        print("\n  [GENERATED VIRTUAL BATTERY TWIN STATE]")
        print(f"    • State of Health (SOH): {vb['soh_percent']:.2f} %")
        print(f"    • Usable Pack Capacity : {vb['capacity_ah']:.2f} Ah (Nominal: 60.0 Ah)")
        print(f"    • Internal Resistance  : {vb['internal_resistance_mohm']:.2f} mΩ ({vb['internal_resistance_ohm']:.4f} Ω)")
        print(f"    • Energy Remaining     : {vb['energy_remaining_kwh']:.2f} kWh")
        print(f"    • Instantaneous Power  : {vb['power_kw']:.2f} kW")
        print(f"    • Overall Health Score : {vb['condition_score']:.2f} / 100")
        print(f"    • Battery Condition    : [{vb['battery_condition'].upper()}]")
        print(f"    • Second-Life Decision : {vb['second_life_status']}")
        print(f"    • 96S Cell Min / Max   : {vb['v_cell_min']:.3f} V / {vb['v_cell_max']:.3f} V (Spread: {vb['v_cell_spread_mv']:.1f} mV)")
        print(f"    • Weakest Cell SOH     : {vb['weakest_cell_soh_pct']:.2f} %")
        print("=" * 76 + "\n")
        return

    if args.generate:
        dataset = generate_dataset()

    if args.demo or args.all:
        run_demo(predictor=predictor)
        return

    # Check if custom CLI simulation parameters were provided
    has_custom_params = any(x is not None for x in (args.age, args.soc, args.temp, args.cycle, args.current))

    if has_custom_params:
        age = args.age if args.age is not None else 0.0
        temp = args.temp if args.temp is not None else 25.0
        initial_soc = 0.80
        if args.soc is not None:
            initial_soc = args.soc / 100.0 if args.soc > 1.0 else args.soc

        cycle = args.cycle if args.cycle is not None else "urban"

        print(f"\n[Dynamic Run] Age: {age}yr | Temp: {temp}°C | Initial SOC: {initial_soc*100:.1f}% | Cycle: {cycle}")
        twin = VirtualBattery(
            age_years=age, cycle=cycle if cycle != "live" else "urban",
            ambient_temp=temp, initial_soc=initial_soc,
            nominal_cap_ah=args.capacity, n_series=args.series,
            current=args.current, dt=args.dt, predictor=predictor
        )

        if cycle == "live":
            interactive_live_drive(twin)
        else:
            print(f"> Running {args.steps} steps @ dt={args.dt}s ...")
            print_interval = max(1, args.steps // 10)
            for s in range(args.steps):
                f = twin.tick()
                if s % print_interval == 0 or s == args.steps - 1:
                    _print_frame(f, s)
            sm = twin.summary()
            print(f"\n[Summary] Mean SOH: {sm['mean_soh']*100:.2f}% | Final SOC: {sm['final_soc']*100:.2f}% | Max Temp: {sm['max_temp']:.1f}°C")
    else:
        # Default mode: interactive CLI wizard
        interactive_cli_session(predictor=predictor)


if __name__ == "__main__":
    main()

