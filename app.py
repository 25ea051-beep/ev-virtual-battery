"""
app.py
======
Flask backend server linking the Virtual Battery Digital Twin physics & ML model
to the interactive HTML/Tailwind/Canvas front-end dashboard.

Run directly or with Visual Studio / VS Code (F5):
  python app.py
"""

import os
import sys
import webbrowser
import threading
from flask import Flask, render_template, jsonify, request

# Ensure local module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from virtual_battery import (
    VirtualBattery,
    DRIVE_CYCLES,
    get_cell_ocv,
    SOHPredictor
)
from virtual_battery_model import (
    VirtualBatteryCANModel,
    get_or_train_model,
    DEFAULT_DATASET_PATH
)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MANAGER (Stateful Digital Twin Singleton)
# ─────────────────────────────────────────────────────────────────────────────

class TwinManager:
    def __init__(self):
        self.age_years = 0.0
        self.cycle = "urban"
        self.ambient_temp = 25.0
        self.initial_soc = 0.80
        self.manual_current = None
        self.dt = 0.5
        self.predictor = None
        try:
            self.can_model = get_or_train_model()
        except Exception:
            self.can_model = None
        self.twin = None
        self._lock = threading.Lock()
        self.reinit()

    def reinit(self):
        with self._lock:
            self.twin = VirtualBattery(
                age_years=self.age_years,
                cycle=self.cycle,
                ambient_temp=self.ambient_temp,
                initial_soc=self.initial_soc,
                current=self.manual_current,
                dt=self.dt,
                predictor=self.predictor,
                can_model=self.can_model,
                predict_window_s=300,
                predict_interval_s=15.0
            )

    def set_config(self, age_years=None, cycle=None, ambient_temp=None, dt=None, initial_soc=None, manual_current=False):
        changed = False
        if age_years is not None and age_years != self.age_years:
            self.age_years = float(age_years)
            changed = True
        if cycle is not None and cycle in DRIVE_CYCLES and cycle != self.cycle:
            self.cycle = str(cycle)
            changed = True
        if ambient_temp is not None and ambient_temp != self.ambient_temp:
            self.ambient_temp = float(ambient_temp)
            changed = True
        if dt is not None and dt != self.dt:
            self.dt = float(dt)
            changed = True
        if initial_soc is not None:
            soc_val = float(initial_soc)
            if soc_val > 1.0:
                soc_val /= 100.0
            if abs(soc_val - self.initial_soc) > 1e-4:
                self.initial_soc = soc_val
                changed = True

        if manual_current is not False:
            self.manual_current = float(manual_current) if manual_current is not None else None
            if self.twin:
                self.twin.set_current(self.manual_current)

        if changed:
            self.reinit()
        return self.get_status()

    def tick(self, current=None):
        with self._lock:
            eff_current = current if current is not None else self.manual_current
            prev_alerts_len = len(self.twin.alerts)
            frame = self.twin.tick(current=eff_current)
            new_alerts = self.twin.alerts[prev_alerts_len:]

            # Extract 96 individual series cell voltages
            pack = self.twin._gen._pack
            I_now = frame["current"]
            v_rc = self.twin._gen._ecm.polarization_voltage
            ocvs = get_cell_ocv(pack.soc)
            cell_vs = ocvs - I_now * pack.ir - v_rc

            power_kw = round((frame["voltage"] * frame["current"]) / 1000.0, 2)
            vb = frame.get("virtual_battery", {})

            payload = {
                "frame": frame,
                "power_kw": power_kw,
                "cell_voltages": [round(float(v), 3) for v in cell_vs],
                "cell_soh": [round(float(s), 4) for s in pack.soh],
                "cell_soc": [round(float(c), 4) for c in pack.soc],
                "virtual_battery": vb,
                "condition_score": vb.get("condition_score", 95.0),
                "battery_condition": vb.get("battery_condition", "Healthy"),
                "second_life_status": vb.get("second_life_status", "Not yet"),
                "capacity_ah": vb.get("capacity_ah", 60.0),
                "energy_remaining_kwh": vb.get("energy_remaining_kwh", 15.0),
                "new_alerts": new_alerts,
                "summary": self.twin.summary(),
                "manual_current": self.manual_current
            }
            return payload

    def reset(self):
        self.reinit()
        return {"status": "reset", "config": self.get_status()}

    def get_status(self):
        return {
            "age_years": self.age_years,
            "cycle": self.cycle,
            "ambient_temp": self.ambient_temp,
            "initial_soc": self.initial_soc,
            "manual_current": self.manual_current,
            "dt": self.dt,
            "cycles_info": DRIVE_CYCLES
        }


manager = TwinManager()

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTING & REST API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serves the interactive dashboard HTML."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns the current simulation twin status and configuration."""
    return jsonify(manager.get_status())


@app.route("/api/config", methods=["POST"])
def api_config():
    """Updates simulation configuration (age, drive cycle, ambient temp, initial_soc, manual_current)."""
    data = request.get_json() or {}
    status = manager.set_config(
        age_years=data.get("age_years"),
        cycle=data.get("cycle"),
        ambient_temp=data.get("ambient_temp"),
        dt=data.get("dt"),
        initial_soc=data.get("initial_soc"),
        manual_current=data.get("manual_current", False)
    )
    return jsonify({"status": "updated", "config": status})


@app.route("/api/tick", methods=["POST", "GET"])
def api_tick():
    """Advances simulation by one dt step and returns full CAN telemetry."""
    data = (request.get_json(silent=True) or {}) if request.method == "POST" else {}
    current = data.get("current")
    result = manager.tick(current=current)
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Resets digital twin back to initial state."""
    return jsonify(manager.reset())


@app.route("/api/generate_virtual_battery", methods=["POST"])
def api_generate_virtual_battery():
    """Generates complete virtual battery digital twin directly from CAN input signals."""
    data = request.get_json() or {}
    try:
        model = manager.can_model or get_or_train_model()
        result = model.generate_virtual_battery(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/train_model", methods=["POST"])
def api_train_model():
    """Triggers retraining of the CAN-to-Virtual-Battery model on the dataset."""
    try:
        manager.can_model = get_or_train_model(force_retrain=True)
        if manager.twin:
            manager.twin.can_model = manager.can_model
        return jsonify({
            "status": "success",
            "message": "Virtual Battery CAN Model trained successfully.",
            "metrics": manager.can_model.metrics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """Returns metadata and metrics about the CAN virtual battery model, plus dataset sample presets."""
    metrics = manager.can_model.metrics if manager.can_model else {}
    samples = [
        {
            "id": "EVB_0010",
            "name": "EVB_0010 (Healthy Pack)",
            "cycle_count": 171,
            "voltage": 371.93,
            "current": 63.45,
            "temp_cell": 23.5,
            "soc": 88.53,
            "expected_soh": 97.03,
            "expected_cond": "Healthy",
            "expected_second_life": "Not yet"
        },
        {
            "id": "EVB_0003",
            "name": "EVB_0003 (Moderate Aging)",
            "cycle_count": 910,
            "voltage": 378.93,
            "current": 61.32,
            "temp_cell": 33.0,
            "soc": 93.19,
            "expected_soh": 82.32,
            "expected_cond": "Moderate",
            "expected_second_life": "Suitable for evaluation"
        },
        {
            "id": "EVB_0001",
            "name": "EVB_0001 (Degraded Pack)",
            "cycle_count": 1176,
            "voltage": 324.25,
            "current": 76.06,
            "temp_cell": 17.81,
            "soc": 20.55,
            "expected_soh": 75.55,
            "expected_cond": "Degraded",
            "expected_second_life": "Suitable for evaluation"
        },
        {
            "id": "EVB_0007",
            "name": "EVB_0007 (Critical End-of-Life)",
            "cycle_count": 1774,
            "voltage": 377.84,
            "current": 31.52,
            "temp_cell": 48.74,
            "soc": 99.28,
            "expected_soh": 62.87,
            "expected_cond": "Critical",
            "expected_second_life": "Consider second-life/replacement"
        }
    ]
    return jsonify({
        "is_trained": bool(manager.can_model and manager.can_model.is_trained),
        "metrics": metrics,
        "feature_cols": getattr(manager.can_model, "FEATURE_COLS", []),
        "samples": samples
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def open_browser(port):
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        pass


def main():
    port = 5000
    print("=" * 80)
    print("  EV VIRTUAL BATTERY DIGITAL TWIN - FULL STACK SERVER")
    print(f"  Backend: Python / Flask API")
    print(f"  Frontend: HTML5 / Tailwind / Canvas Telemetry Dashboard")
    print(f"  Local Dashboard URL: http://127.0.0.1:{port}")
    print("  Press Ctrl+C in terminal to stop.")
    print("=" * 80)

    # Launch browser after 1 second
    threading.Timer(1.2, open_browser, args=[port]).start()

    # Run Flask development server
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
