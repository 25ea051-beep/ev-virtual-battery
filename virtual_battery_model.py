"""
virtual_battery_model.py  — v2 (Equal-Weight CAN Model)
========================================================
Equal-Influence Multi-Target ML for EV Virtual Battery Digital Twin.

Key improvements over v1:
  1. Physics-aware feature engineering: 10 features (5 raw + 5 cross-interaction)
     that force ALL 5 CAN inputs to influence every prediction.
  2. Local sensitivity (SHAP-style finite-difference) returned per prediction
     so the UI can show each input's actual contribution at that operating point.
  3. OOD (out-of-distribution) detection with physical range guards.
  4. Physics-based fallback: blends physics equations with ML when inputs are OOD,
     ensuring graceful extrapolation far outside training data.
  5. RobustScaler: handles outliers in features better than StandardScaler.
"""

import os, math, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_can_dataset.csv")
DEFAULT_MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model")
DEFAULT_MODEL_PATH   = os.path.join(DEFAULT_MODEL_DIR, "virtual_battery_model.joblib")

# ─── Physical validity ranges (hard clamps, not OOD ranges) ──────────────────
PHYSICAL_LIMITS = {
    "Cycle_Count":   (0,    5000),
    "Voltage_V":     (150,  500),
    "Current_A":     (-100, 200),
    "Temperature_C": (-20,  80),
    "SOC_percent":   (0,    100),
}

# ─── Training data ranges – used to compute OOD score ────────────────────────
TRAINING_RANGES = {
    "Cycle_Count":   (50,   1800),
    "Voltage_V":     (313,  390),
    "Current_A":     (-12,  100),
    "Temperature_C": (10,   55),
    "SOC_percent":   (5,    100),
}

# Input display labels for UI
INPUT_META = {
    "Cycle_Count":   {"label": "Cycle Count",  "unit": "cycles", "icon": "🔄"},
    "Voltage_V":     {"label": "Voltage",       "unit": "V",      "icon": "⚡"},
    "Current_A":     {"label": "Current",       "unit": "A",      "icon": "➡️"},
    "Temperature_C": {"label": "Temperature",   "unit": "°C",     "icon": "🌡️"},
    "SOC_percent":   {"label": "State of Charge","unit": "%",     "icon": "🔋"},
}

NOMINAL_CAPACITY_AH = 60.0
NOMINAL_VOLTAGE_V   = 350.0
CELLS_SERIES        = 96


# ─────────────────────────────────────────────────────────────────────────────
#  Physics helper functions (all 5 inputs used in each formula)
# ─────────────────────────────────────────────────────────────────────────────

def _physics_soh(cycles, voltage, current, temp, soc_pct) -> float:
    """
    Physics-based SOH estimate that uses ALL 5 CAN inputs.
    Based on:
      - Arrhenius thermal degradation (Temperature)
      - C-rate stress (Current / Capacity)
      - High-SOC calendar aging (SOC_percent)
      - Low-voltage stress (Voltage per cell)
      - Primary cycling degradation (Cycle_Count)
    """
    # 1. Cycling degradation (primary driver, ~50% weight)
    cycle_fade = max(0.0, 1.0 - (cycles / 2000.0) * 0.5)

    # 2. Thermal Arrhenius factor (Temperature, ~20% weight)
    t_excess = max(0.0, temp - 25.0)
    thermal_factor = math.exp(-0.0008 * t_excess * (cycles / 500.0 + 1.0))

    # 3. High-SOC calendar aging (SOC_percent, ~15% weight)
    soc_stress = 1.0 - 0.00015 * max(0, soc_pct - 80.0) ** 1.5 * (cycles / 1000.0 + 1.0)

    # 4. C-rate mechanical stress (Current, ~10% weight)
    c_rate = abs(current) / NOMINAL_CAPACITY_AH
    c_stress = 1.0 - 0.002 * max(0.0, c_rate - 1.0) ** 2 * (cycles / 800.0 + 1.0)

    # 5. Low-voltage lithium-plating stress (Voltage, ~5% weight)
    v_cell = voltage / CELLS_SERIES
    v_stress = 1.0 - 0.003 * max(0.0, 3.2 - v_cell) * (cycles / 600.0 + 1.0)

    soh = 100.0 * cycle_fade * thermal_factor * soc_stress * c_stress * v_stress
    return float(np.clip(soh, 40.0, 100.0))


def _physics_ir(cycles, voltage, current, temp, soh) -> float:
    """Physics-based internal resistance (uses all inputs)."""
    base_ir = 0.025 + (1.0 - soh / 100.0) * 0.12
    t_factor = 1.0 + 0.005 * max(0.0, 15.0 - temp)   # cold = higher IR
    c_factor = 1.0 + 0.001 * max(0.0, abs(current) - 40.0)
    v_factor = 1.0 + 0.002 * max(0.0, 330.0 - voltage) / 20.0
    return float(np.clip(base_ir * t_factor * c_factor * v_factor, 0.015, 0.35))


def _physics_condition_score(soh, temp, ir, soc_pct, cycles) -> float:
    """Composite condition score 0-100 using all derived and input values."""
    soh_score  = (soh - 40.0) / 60.0 * 60.0        # SOH contribution 0-60
    temp_score = max(0.0, 20.0 - abs(temp - 25.0)) # Temperature optimality 0-20
    ir_score   = max(0.0, 10.0 - ir * 100.0)       # IR contribution 0-10
    soc_score  = max(0.0, 10.0 * (1.0 - abs(soc_pct - 50.0) / 50.0))  # SOC 0-10
    return float(np.clip(soh_score + temp_score + ir_score + soc_score, 0.0, 100.0))


def _physics_condition_label(soh, score) -> str:
    if soh >= 85.0 and score >= 75.0: return "Healthy"
    if soh >= 75.0 and score >= 60.0: return "Moderate"
    if soh >= 65.0: return "Degraded"
    return "Critical"


def _physics_second_life(soh) -> str:
    if soh >= 80.0: return "Not yet"
    if soh >= 65.0: return "Suitable for evaluation"
    return "Consider second-life/replacement"


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering (5 cross-interaction features)
# ─────────────────────────────────────────────────────────────────────────────

# Engineered feature attribution: which original inputs contribute to each
# cross-feature (used to map importances back to original 5 inputs)
#
# Index: 0=Cycle, 1=Voltage, 2=Current, 3=Temperature, 4=SOC
ATTRIBUTION = np.array([
    #  Cyc   V      I      T      SOC
    [  1.0,  0.0,  0.0,  0.0,  0.0 ],  # feat[0]: Cycle_Count (raw)
    [  0.0,  1.0,  0.0,  0.0,  0.0 ],  # feat[1]: Voltage_V (raw)
    [  0.0,  0.0,  1.0,  0.0,  0.0 ],  # feat[2]: Current_A (raw)
    [  0.0,  0.0,  0.0,  1.0,  0.0 ],  # feat[3]: Temperature_C (raw)
    [  0.0,  0.0,  0.0,  0.0,  1.0 ],  # feat[4]: SOC_percent (raw)
    [  0.0,  0.5,  0.0,  0.0,  0.5 ],  # feat[5]: V_x_SOC  (Voltage × SOC)
    [  0.5,  0.0,  0.0,  0.5,  0.0 ],  # feat[6]: Thermal_Aging (Temp × Cycles)
    [  0.0,  0.0,  1.0,  0.0,  0.0 ],  # feat[7]: C_Rate (Current-derived)
    [  0.0,  0.0,  0.5,  0.5,  0.0 ],  # feat[8]: Power_Thermal (I²×T)
    [  0.5,  0.5,  0.0,  0.0,  0.0 ],  # feat[9]: V_Degradation (V×Cycles)
])

ENG_FEATURE_NAMES = [
    "Cycle_Count", "Voltage_V", "Current_A", "Temperature_C", "SOC_percent",
    "V_x_SOC", "Thermal_Aging", "C_Rate", "Power_Thermal", "V_Degradation",
]
ORIG_FEATURE_NAMES = ["Cycle_Count", "Voltage_V", "Current_A", "Temperature_C", "SOC_percent"]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5 physics-inspired cross-interaction features to enforce equal-weight
    influence from all 5 CAN inputs during model training and inference.
    """
    out = df.copy()
    out["V_x_SOC"]       = df["Voltage_V"] * df["SOC_percent"] / 100.0
    out["Thermal_Aging"] = df["Temperature_C"] * df["Cycle_Count"] / 1000.0
    out["C_Rate"]        = df["Current_A"].abs() / NOMINAL_CAPACITY_AH
    out["Power_Thermal"] = (df["Current_A"] ** 2) * df["Temperature_C"] / 10000.0
    out["V_Degradation"] = (420.0 - df["Voltage_V"]) / 420.0 * (df["Cycle_Count"] / 1000.0)
    return out[ENG_FEATURE_NAMES]


# ─────────────────────────────────────────────────────────────────────────────
#  Main Model Class
# ─────────────────────────────────────────────────────────────────────────────

class VirtualBatteryCANModel:
    """
    Equal-influence multi-target ML model: 5 CAN inputs → Virtual Battery state.
    Supports OOD detection, physics-based fallback, and local sensitivity export.
    """

    FEATURE_COLS = ORIG_FEATURE_NAMES

    def __init__(self):
        self.scaler = RobustScaler()
        self.models = {
            "soh":             GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42),
            "capacity":        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42),
            "ir":              GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42),
            "condition_score": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42),
        }
        self.clf_condition   = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.clf_second_life = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.metrics: dict   = {}
        self.is_trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, dataset, test_size: float = 0.2) -> dict:
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
        else:
            df = dataset.copy()

        print(f"[VirtualBatteryModel] Training on {len(df)} records with 10 engineered features ...")

        # Clamp raw inputs to physical limits before engineering
        for col, (lo, hi) in PHYSICAL_LIMITS.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        X_eng = _engineer_features(df[ORIG_FEATURE_NAMES])
        X_scaled = self.scaler.fit_transform(X_eng)

        X_tr, X_te, idx_tr, idx_te = train_test_split(
            X_scaled, df.index, test_size=test_size, random_state=42
        )

        target_map = {
            "soh":             "SOH_percent",
            "capacity":        "Capacity_Ah",
            "ir":              "Internal_Resistance_Ohm",
            "condition_score": "Condition_Score",
        }
        for name, col in target_map.items():
            y = df[col].values
            self.models[name].fit(X_tr, y[idx_tr])
            preds = self.models[name].predict(X_te)
            self.metrics[name] = {
                "r2":  float(r2_score(y[idx_te], preds)),
                "mae": float(mean_absolute_error(y[idx_te], preds)),
            }
            print(f"  [{name:15s}] R²={self.metrics[name]['r2']:.4f}  MAE={self.metrics[name]['mae']:.4f}")

        y_cond = df["Battery_Condition"].values
        self.clf_condition.fit(X_tr, y_cond[idx_tr])
        acc = float(accuracy_score(y_cond[idx_te], self.clf_condition.predict(X_te)))
        self.metrics["battery_condition"] = {"accuracy": acc}
        print(f"  [battery_condition] Accuracy={acc*100:.2f}%")

        y_sl = df["Second_Life_Status"].values
        self.clf_second_life.fit(X_tr, y_sl[idx_tr])
        acc2 = float(accuracy_score(y_sl[idx_te], self.clf_second_life.predict(X_te)))
        self.metrics["second_life_status"] = {"accuracy": acc2}
        print(f"  [second_life_status] Accuracy={acc2*100:.2f}%")

        self.is_trained = True
        return self.metrics

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, model_path: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler, "models": self.models,
            "clf_condition": self.clf_condition, "clf_second_life": self.clf_second_life,
            "metrics": self.metrics, "is_trained": self.is_trained,
            "feature_cols": self.FEATURE_COLS,
        }, model_path)
        print(f"[VirtualBatteryModel] Saved -> {model_path}")

    def load(self, model_path: str = DEFAULT_MODEL_PATH) -> bool:
        if not os.path.exists(model_path):
            return False
        try:
            d = joblib.load(model_path)
            self.scaler          = d["scaler"]
            self.models          = d["models"]
            self.clf_condition   = d["clf_condition"]
            self.clf_second_life = d["clf_second_life"]
            self.metrics         = d.get("metrics", {})
            self.is_trained      = d.get("is_trained", True)
            print(f"[VirtualBatteryModel] Loaded <- {model_path}")
            return True
        except Exception as e:
            print(f"[VirtualBatteryModel] Load failed: {e}")
            return False

    # ── OOD detection ─────────────────────────────────────────────────────────

    def detect_ood(self, can_data: dict) -> dict:
        """
        Returns per-input OOD flag and a global OOD score [0.0, 1.0].
        0 = fully in-distribution, 1 = completely out of training range.
        """
        keys = {
            "Cycle_Count": can_data.get("cycle_count", can_data.get("Cycle_Count", 500)),
            "Voltage_V":   can_data.get("voltage",     can_data.get("Voltage_V",   350)),
            "Current_A":   can_data.get("current",     can_data.get("Current_A",     0)),
            "Temperature_C": can_data.get("temp_cell", can_data.get("Temperature_C", 25)),
            "SOC_percent": can_data.get("soc",         can_data.get("SOC_percent",  80)),
        }
        flags = {}
        deviations = []
        for name, val in keys.items():
            lo, hi = TRAINING_RANGES[name]
            span = hi - lo
            if val < lo:
                dev = (lo - val) / span
                flags[name] = {"ood": True, "direction": "below", "deviation_pct": round(dev * 100, 1)}
                deviations.append(dev)
            elif val > hi:
                dev = (val - hi) / span
                flags[name] = {"ood": True, "direction": "above", "deviation_pct": round(dev * 100, 1)}
                deviations.append(dev)
            else:
                flags[name] = {"ood": False, "direction": None, "deviation_pct": 0.0}
                deviations.append(0.0)

        ood_score = float(min(1.0, max(deviations))) if deviations else 0.0
        return {"flags": flags, "ood_score": round(ood_score, 3)}

    # ── Local sensitivity (equal-weight influence metric) ────────────────────

    def get_local_sensitivity(self, feat_scaled: np.ndarray, raw_inputs: dict) -> dict:
        """
        Compute finite-difference sensitivity of SOH to each of the 5 CAN inputs.
        Returns a dict: {input_name: influence_pct} that sums to 100.
        Perturbs each input by ±5% and measures the average absolute change in SOH prediction.
        """
        base_soh = float(self.models["soh"].predict(feat_scaled)[0])

        sensitivities = {}
        perturb_pct = 0.10  # 10% perturbation

        raw_df = pd.DataFrame([raw_inputs], columns=ORIG_FEATURE_NAMES)

        for i, col in enumerate(ORIG_FEATURE_NAMES):
            lo_phys, hi_phys = PHYSICAL_LIMITS[col]
            val = raw_inputs[col]
            delta = abs(val) * perturb_pct if abs(val) > 1e-3 else 5.0

            # Perturb UP
            up_df = raw_df.copy()
            up_df.iloc[0, i] = float(np.clip(val + delta, lo_phys, hi_phys))
            up_eng  = _engineer_features(up_df)
            up_sc   = self.scaler.transform(up_eng)
            up_soh  = float(self.models["soh"].predict(up_sc)[0])

            # Perturb DOWN
            dn_df = raw_df.copy()
            dn_df.iloc[0, i] = float(np.clip(val - delta, lo_phys, hi_phys))
            dn_eng  = _engineer_features(dn_df)
            dn_sc   = self.scaler.transform(dn_eng)
            dn_soh  = float(self.models["soh"].predict(dn_sc)[0])

            sensitivities[col] = abs(up_soh - base_soh) + abs(dn_soh - base_soh)

        # Enforce equal minimum visibility: each input always has ≥ 5% share
        total = sum(sensitivities.values()) + 1e-10
        normalized = {k: max(5.0, v / total * 100.0) for k, v in sensitivities.items()}

        # Re-normalize to sum to 100 after minimum enforcement
        total2 = sum(normalized.values())
        return {k: round(v / total2 * 100.0, 1) for k, v in normalized.items()}

    # ── Global feature importance (from trained models) ───────────────────────

    def get_global_importances(self) -> dict:
        """
        Map 10-feature importances back to 5 original CAN inputs via attribution matrix.
        Returns {input_name: influence_pct} summing to 100.
        """
        all_fi = np.zeros(10)
        n_models = 0

        for m in list(self.models.values()) + [self.clf_condition, self.clf_second_life]:
            if hasattr(m, "feature_importances_"):
                all_fi += m.feature_importances_
                n_models += 1

        if n_models > 0:
            all_fi /= n_models

        # Map 10-feature importances to 5 original inputs via attribution matrix
        orig_imp = ATTRIBUTION.T @ all_fi   # shape: (5,)
        orig_imp = np.maximum(orig_imp, 0.05)  # enforce minimum 5% per input
        orig_imp /= orig_imp.sum() + 1e-10

        return {col: round(float(v * 100.0), 1) for col, v in zip(ORIG_FEATURE_NAMES, orig_imp)}

    # ── Main prediction entry point ───────────────────────────────────────────

    def generate_virtual_battery(self, can_data: dict, n_series: int = CELLS_SERIES) -> dict:
        """
        Generate complete virtual battery state from any CAN input values.
        OOD inputs are handled gracefully via physics-based fallback blending.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call .train() or .load() first.")

        # ── Parse & clamp inputs ────────────────────────────────────────────
        v   = float(can_data.get("voltage",   can_data.get("Voltage_V",     350.0)))
        i   = float(can_data.get("current",   can_data.get("Current_A",       0.0)))
        t   = float(can_data.get("temp_cell", can_data.get("Temperature_C",  25.0)))
        soc_raw = float(can_data.get("soc",   can_data.get("SOC_percent",    80.0)))
        soc = soc_raw * 100.0 if soc_raw <= 1.0 else soc_raw
        soc = float(np.clip(soc, 0.0, 100.0))

        if "cycle_count" in can_data or "Cycle_Count" in can_data:
            cyc = float(can_data.get("cycle_count", can_data.get("Cycle_Count", 500.0)))
        elif "age_years" in can_data:
            cyc = float(can_data["age_years"]) * 180.0
        else:
            cyc = 500.0

        # Clamp to physical limits
        v   = float(np.clip(v,   *PHYSICAL_LIMITS["Voltage_V"]))
        i   = float(np.clip(i,   *PHYSICAL_LIMITS["Current_A"]))
        t   = float(np.clip(t,   *PHYSICAL_LIMITS["Temperature_C"]))
        cyc = float(np.clip(cyc, *PHYSICAL_LIMITS["Cycle_Count"]))

        raw_inputs = {
            "Cycle_Count":   cyc,
            "Voltage_V":     v,
            "Current_A":     i,
            "Temperature_C": t,
            "SOC_percent":   soc,
        }

        # ── OOD detection ────────────────────────────────────────────────────
        ood_result = self.detect_ood(can_data)
        ood_score  = ood_result["ood_score"]   # 0 = in-distribution, 1 = fully OOD

        # ── ML prediction ────────────────────────────────────────────────────
        raw_df   = pd.DataFrame([raw_inputs], columns=ORIG_FEATURE_NAMES)
        eng_df   = _engineer_features(raw_df)
        feat_sc  = self.scaler.transform(eng_df)

        soh_ml   = float(np.clip(self.models["soh"].predict(feat_sc)[0], 40.0, 105.0))
        cap_ml   = float(np.clip(self.models["capacity"].predict(feat_sc)[0], 24.0, 66.0))
        ir_ml    = float(max(0.012, self.models["ir"].predict(feat_sc)[0]))
        score_ml = float(np.clip(self.models["condition_score"].predict(feat_sc)[0], 0.0, 100.0))
        cond_ml  = str(self.clf_condition.predict(feat_sc)[0])
        sl_ml    = str(self.clf_second_life.predict(feat_sc)[0])

        # ── Physics estimates (uses ALL 5 inputs) ────────────────────────────
        soh_phy   = _physics_soh(cyc, v, i, t, soc)
        ir_phy    = _physics_ir(cyc, v, i, t, soh_phy)
        cap_phy   = NOMINAL_CAPACITY_AH * soh_phy / 100.0
        score_phy = _physics_condition_score(soh_phy, t, ir_phy, soc, cyc)
        cond_phy  = _physics_condition_label(soh_phy, score_phy)
        sl_phy    = _physics_second_life(soh_phy)

        # ── Blend ML + Physics based on OOD score ────────────────────────────
        ml_w  = max(0.0, 1.0 - ood_score * 1.2)    # ML weight drops with OOD
        phy_w = 1.0 - ml_w                          # physics fills the gap

        soh   = ml_w * soh_ml   + phy_w * soh_phy
        cap   = ml_w * cap_ml   + phy_w * cap_phy
        ir    = ml_w * ir_ml    + phy_w * ir_phy
        score = ml_w * score_ml + phy_w * score_phy
        cond  = cond_ml  if ml_w > 0.5 else cond_phy
        sl    = sl_ml    if ml_w > 0.5 else sl_phy

        # Final physics clamps
        soh   = float(np.clip(soh,   40.0, 100.0))
        cap   = float(np.clip(cap,   24.0, 63.0))
        ir    = float(np.clip(ir,    0.012, 0.35))
        score = float(np.clip(score, 0.0,  100.0))

        # ── Derived quantities ────────────────────────────────────────────────
        energy_kwh = round((cap * (soc / 100.0) * v) / 1000.0, 2)
        power_kw   = round((v * i) / 1000.0, 2)

        # ── Cell-level twin (96 series cells) ────────────────────────────────
        rng = np.random.default_rng(int(cyc + v * 10) & 0xFFFFFFFF)
        v_cell_mean  = v / n_series
        v_spread     = max(0.004, 0.012 * (1.0 - soh / 100.0) + ir * 0.05)
        cell_voltages = np.clip(rng.normal(v_cell_mean, v_spread, n_series), 2.50, 4.30)
        cell_soh_arr  = np.clip(rng.normal(soh / 100.0, 0.005, n_series), 0.40, 1.0)

        # ── Local sensitivity (per-input influence for THIS prediction point) ─
        sensitivity = self.get_local_sensitivity(feat_sc, raw_inputs)

        # ── Alerts ───────────────────────────────────────────────────────────
        alerts = []
        if soh < 70.0:
            alerts.append({"level": "CRITICAL", "msg": f"SOH Critical: {soh:.1f}% – pack near end-of-life"})
        elif soh < 80.0:
            alerts.append({"level": "WARNING",  "msg": f"SOH Degradation Advisory: {soh:.1f}%"})
        if t > 50.0:
            alerts.append({"level": "CRITICAL", "msg": f"Over-Temperature: {t:.1f}°C"})
        elif t > 40.0:
            alerts.append({"level": "WARNING",  "msg": f"Elevated Thermal Load: {t:.1f}°C"})
        if soc < 15.0:
            alerts.append({"level": "WARNING",  "msg": f"Low SOC: {soc:.1f}%"})
        if ood_score > 0.3:
            alerts.append({"level": "INFO",     "msg": f"Inputs partially outside training range – physics blend active ({int(phy_w*100)}%)"})

        return {
            "can_input": {
                "voltage_v":     round(v, 2),
                "current_a":     round(i, 2),
                "temperature_c": round(t, 2),
                "soc_percent":   round(soc, 2),
                "cycle_count":   int(cyc),
            },
            "virtual_battery": {
                "soh_percent":               round(soh, 2),
                "capacity_ah":               round(cap, 2),
                "internal_resistance_ohm":   round(ir, 4),
                "internal_resistance_mohm":  round(ir * 1000.0, 2),
                "energy_remaining_kwh":      energy_kwh,
                "power_kw":                  power_kw,
                "condition_score":           round(score, 2),
                "battery_condition":         cond,
                "second_life_status":        sl,
                "v_cell_min":    round(float(np.min(cell_voltages)), 3),
                "v_cell_max":    round(float(np.max(cell_voltages)), 3),
                "v_cell_spread_mv": round(float(np.max(cell_voltages) - np.min(cell_voltages)) * 1000.0, 1),
                "weakest_cell_soh_pct": round(float(np.min(cell_soh_arr)) * 100.0, 2),
            },
            "cell_voltages":  [round(float(x), 3) for x in cell_voltages],
            "feature_influence": sensitivity,
            "ood_detection":  {
                "ood_score":   ood_score,
                "ml_weight":   round(ml_w, 3),
                "physics_weight": round(phy_w, 3),
                "input_flags": ood_result["flags"],
            },
            "alerts": alerts,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def get_or_train_model(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path:   str = DEFAULT_MODEL_PATH,
    force_retrain: bool = False,
) -> VirtualBatteryCANModel:
    model = VirtualBatteryCANModel()

    # Check if saved model uses RobustScaler (v2). If old model, force retrain.
    if not force_retrain and model.load(model_path):
        if not isinstance(model.scaler, RobustScaler):
            print("[VirtualBatteryModel] Old StandardScaler model detected – retraining with v2 ...")
            force_retrain = True
        else:
            return model

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        model.train(df)
        model.save(model_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("  EV VIRTUAL BATTERY CAN MODEL v2 — Equal-Weight Training")
    print("=" * 70)
    m = get_or_train_model(force_retrain=True)

    # Test with in-distribution sample
    s1 = {"Cycle_Count": 1176, "Voltage_V": 324.25, "Current_A": 76.06, "Temperature_C": 17.81, "SOC_percent": 20.55}
    r1 = m.generate_virtual_battery(s1)
    print(f"\nIn-Distribution (EVB_0001):")
    for k, v in r1["virtual_battery"].items():
        print(f"  {k:<30s}: {v}")
    print(f"  Feature Influence: {r1['feature_influence']}")
    print(f"  OOD Score: {r1['ood_detection']['ood_score']}")

    # Test with OOD sample (very low cycles, cold temp, unusual current)
    s2 = {"Cycle_Count": 5, "Voltage_V": 250.0, "Current_A": 180.0, "Temperature_C": -5.0, "SOC_percent": 50.0}
    r2 = m.generate_virtual_battery(s2)
    print(f"\nOut-of-Distribution (extreme values):")
    for k, v in r2["virtual_battery"].items():
        print(f"  {k:<30s}: {v}")
    print(f"  Feature Influence: {r2['feature_influence']}")
    print(f"  OOD Score: {r2['ood_detection']['ood_score']}")
    print("=" * 70)
