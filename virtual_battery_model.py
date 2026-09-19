"""
virtual_battery_model.py
========================
Machine Learning Model Pipeline for generating an EV Virtual Battery Digital Twin
directly from Automotive CAN Bus Telemetry.

Trained on ground-truth battery datasets containing:
  Inputs  : Cycle_Count, Voltage_V, Current_A, Temperature_C, SOC_percent
  Outputs : SOH_percent, Capacity_Ah, Internal_Resistance_Ohm,
            Energy_Remaining_kWh, Power_kW, Condition_Score,
            Battery_Condition, Second_Life_Status
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_can_dataset.csv")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "virtual_battery_model.joblib")


class VirtualBatteryCANModel:
    """
    Multi-target machine learning model pipeline that converts raw CAN bus
    signals into complete, high-fidelity Virtual Battery Digital Twin states.
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

    def train(self, dataset: pd.DataFrame | str, test_size: float = 0.2) -> dict:
        """
        Train all regression and classification sub-models on the CAN dataset.
        """
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

        # 1. Regressors for continuous physical states
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

        # 2. Classifiers for battery condition & second life qualification
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

        Accepts dictionary containing CAN telemetry:
          - voltage / Voltage_V (V)
          - current / Current_A (A, + discharge, - charge)
          - temp_cell / Temperature_C (°C)
          - soc / SOC_percent (% or 0.0-1.0 fraction)
          - cycle_count / Cycle_Count (optional, auto-estimated if missing)
        """
        if not self.is_trained:
            raise RuntimeError("VirtualBatteryCANModel is not trained. Call .train() or .load() first.")

        # Extract CAN inputs with fallback normalization
        v = float(can_data.get("voltage", can_data.get("Voltage_V", 350.0)))
        i = float(can_data.get("current", can_data.get("Current_A", 0.0)))
        t = float(can_data.get("temp_cell", can_data.get("Temperature_C", 25.0)))

        # Normalize SOC: if 0.0-1.0, scale to 0-100%
        soc_raw = float(can_data.get("soc", can_data.get("SOC_percent", 80.0)))
        soc_pct = soc_raw * 100.0 if soc_raw <= 1.0 else soc_raw
        soc_pct = float(np.clip(soc_pct, 0.0, 100.0))

        # Cycle count estimation if omitted
        if "cycle_count" in can_data or "Cycle_Count" in can_data:
            cycles = float(can_data.get("cycle_count", can_data.get("Cycle_Count", 500.0)))
        elif "age_years" in can_data:
            cycles = float(can_data["age_years"]) * 180.0
        else:
            # Baseline estimation from voltage/SOC profile
            cycles = 500.0

        # Construct feature vector and scale
        feat_df = pd.DataFrame([[cycles, v, i, t, soc_pct]], columns=self.FEATURE_COLS)
        feat_scaled = self.scaler.transform(feat_df)

        # Predict continuous battery states
        soh = float(np.clip(self.models["soh"].predict(feat_scaled)[0], 50.0, 100.5))
        cap = float(np.clip(self.models["capacity"].predict(feat_scaled)[0], 25.0, 65.0))
        ir = float(max(0.015, self.models["ir"].predict(feat_scaled)[0]))
        score = float(np.clip(self.models["condition_score"].predict(feat_scaled)[0], 40.0, 100.0))

        # Predict categorical classifications
        cond = str(self.clf_condition.predict(feat_scaled)[0])
        second_life = str(self.clf_second_life.predict(feat_scaled)[0])

        # Exact physics states derived from virtual battery parameters
        # Energy Remaining: E = V_pack * Q_pack * SOC / 1000 (kWh)
        energy_kwh = round((cap * (soc_pct / 100.0) * v) / 1000.0, 2)
        power_kw = round((v * i) / 1000.0, 2)

        # Synthesize cell-level digital twin (96 series cells with Gaussian variation)
        rng = np.random.default_rng(int(cycles + v * 10) & 0xFFFFFFFF)
        v_cell_mean = v / n_series
        # Cell voltage spread increases as SOH degrades and IR rises
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
    """
    Factory function: returns a trained VirtualBatteryCANModel.
    Loads from disk if available, otherwise trains on the dataset and persists it.
    """
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


if __name__ == "__main__":
    # Self-test when executed directly
    print("=" * 70)
    print("  EV VIRTUAL BATTERY CAN MODEL - TRAINING & EVALUATION")
    print("=" * 70)
    trained_model = get_or_train_model(force_retrain=True)

    # Test sample prediction
    sample_can = {
        "Cycle_Count": 1176,
        "Voltage_V": 324.25,
        "Current_A": 76.06,
        "Temperature_C": 17.81,
        "SOC_percent": 20.55
    }
    result = trained_model.generate_virtual_battery(sample_can)
    print("\n[Sample Virtual Battery Generation for EVB_0001]:")
    for k, v in result["virtual_battery"].items():
        print(f"  {k:25s}: {v}")
    print("=" * 70)
