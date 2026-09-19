# ⚡ EV Virtual Battery Digital Twin & CAN Bus Simulator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

A multi-physics simulation and machine learning framework for electric vehicle (EV) battery packs. Combines **2RC Thevenin electrochemistry**, **Arrhenius degradation laws**, **96S cell-level imbalance**, and **CAN 2.0B automotive bus emulation** with an interactive full-stack telemetry cockpit and a **GradientBoosting SOH predictor**.

---

## 📁 Repository Structure

```
ev_virtual_battery/
├── app.py                      # Flask REST API backend linking Python twin to HTML UI
├── virtual_battery.py          # Core physics + ML simulator CLI & digital twin
├── virtual_battery_model.py    # Multi-target ML model pipeline (CAN → Virtual Battery)
├── battery_can_dataset.csv     # 1,000-pack ground-truth battery CAN dataset
├── trained_model/              # Serialized ML model artifacts (.joblib)
├── templates/
│   └── index.html              # Interactive cockpit UI with CAN Twin Generator
├── static/
│   ├── css/
│   │   └── style.css           # Custom telemetry themes & scrollbars
│   └── js/
│       └── app.js              # Real-time chart plotting & CAN Generator engine
├── requirements.txt            # Pinned dependencies
├── presentation_guide.md       # Live presentation script & talking points
├── .vscode/
│   └── launch.json             # Visual Studio / VS Code 1-click F5 launch configs
└── .gitignore                  # Git ignore rules for virtualenvs, cache, datasets
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/ev-virtual-battery.git
cd ev-virtual-battery

# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Train the CAN-to-Virtual Battery Model
Train the multi-target Gradient Boosting & Random Forest models on `battery_can_dataset.csv` (1,000 battery records):
```bash
python virtual_battery.py --train
# or
python virtual_battery_model.py
```
> **Performance**: SOH $R^2 \approx 0.983$ (MAE $0.99\%$), Capacity $R^2 \approx 0.983$ (MAE $0.60\text{ Ah}$), IR $R^2 \approx 0.940$ (MAE $3.4\text{ m}\Omega$), Condition Score $R^2 \approx 0.984$ (MAE $0.73$), Condition Acc $92.5\%$, Second-Life Acc $95.0\%$.

### 3. Generate Virtual Battery Twin directly from CAN Telemetry (CLI)
Input raw CAN bus parameters (`Voltage`, `Current`, `Temperature`, `SOC`, `Cycle Count`) and generate the complete virtual battery digital twin instantly:
```bash
python virtual_battery.py --generate-from-can --can-voltage 365.5 --can-current 45.0 --can-temp 26.0 --can-soc 75.0 --can-cycles 300
```

### 4. Launch Interactive Dynamic CLI Simulator
Run the simulator directly to configure pack parameters dynamically via an interactive setup wizard:
```bash
python virtual_battery.py
```

Or pass dynamic inputs directly via CLI arguments:
```bash
# Dynamic custom simulation (Age: 3.5yr, SOC: 70%, Temp: 32°C, Current: 100A, 50 steps)
python virtual_battery.py --age 3.5 --soc 70 --temp 32 --current 100 --steps 50

# Jump straight into the interactive live drive console
python virtual_battery.py --cycle live
```

### 5. Launch Web Dashboard (Flask + Cockpit)
* Run via terminal:
  ```bash
  python app.py
  ```
* Or press **`F5`** in **VS Code** / **Visual Studio**.
* The browser will open to `http://127.0.0.1:5000` with the interactive **CAN Telemetry to Virtual Battery Twin Generator** panel, preset selectors (`EVB_0010`, `EVB_0003`, `EVB_0001`, `EVB_0007`), Condition Score gauge, Second-Life status, and dynamic drive cycle sliders.

---

## 🔬 Physics & Modeling Highlights

1. **CAN-to-Virtual-Battery ML Model**: Multi-target model predicting SOH %, Usable Capacity (Ah), Internal Resistance (m$\Omega$), Condition Score (0–100), Operational Health Condition (`Healthy`, `Moderate`, `Degraded`, `Critical`), and Second-Life Retirement Status (`Not yet`, `Suitable for evaluation`, `Consider replacement`).
2. **NMC 811 OCV-SOC Profile**: 23-point calibrated piecewise linear open-circuit voltage characteristic.
3. **2RC Thevenin ECM**:
   - $R_0$: Instantaneous ohmic internal resistance.
   - $R_1, C_1$: Rapid electrochemical double-layer transient response ($\tau_1 \approx 30\text{--}60\,\text{s}$).
   - $R_2, C_2$: Slow solid-state lithium-ion diffusion dynamics ($\tau_2 \approx 500\text{--}2000\,\text{s}$).
   - Arrhenius temperature correction factor ($+1.5\% / ^\circ\text{C}$ below $25^\circ\text{C}$).
4. **Arrhenius Degradation Model**:
   - **Calendar Aging**: Solid Electrolyte Interphase (SEI) growth: $Q_{\text{cal}} = k_{\text{cal}} \cdot \sqrt{t}$.
   - **Cycle Aging**: Ah-throughput law: $Q_{\text{cyc}} = k_{\text{cyc}} \cdot N_{\text{cycles}} \cdot \text{DoD}^{1.5}$.
   - **Non-Linear Knee Point**: Exponential acceleration after $15\%$ capacity loss due to lithium plating and particle cracking.
5. **Vectorized 96-Cell Pack Imbalance**: Simulates 96 series cells with Gaussian variance in SOH, SOC, and internal resistance.
6. **Automotive CAN Sensor Model**: Quantization ($0.1\,\text{V}$, $0.5\,\text{A}$, $0.5^\circ\text{C}$), Hall-effect current sensor offset, voltage/temperature drift, and Coulomb-counting integration bias.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the HTML telemetry dashboard. |
| `GET` | `/api/status` | Returns current active twin parameters and drive cycle metadata. |
| `GET` | `/api/model_info` | Returns CAN ML model status, performance metrics, and dataset sample presets. |
| `POST` | `/api/generate_virtual_battery` | Ingests CAN bus signals and returns complete generated virtual battery digital twin. |
| `POST` | `/api/train_model` | Retrains the CAN-to-Virtual-Battery model on `battery_can_dataset.csv`. |
| `POST` | `/api/config` | Hot-reconfigures `age_years`, `cycle`, `ambient_temp`, or `dt`. |
| `POST` | `/api/tick` | Advances the twin by $dt$ and returns CAN frame, 96 cell voltages, ML states, and alerts. |
| `POST` | `/api/reset` | Resets the simulation back to initial condition. |

---

## 📤 How to Push to GitHub

Run the following in your project folder:

```bash
git init
git add .
git commit -m "Initial commit: EV Virtual Battery Digital Twin with Flask & Web Dashboard"

# Create a new repo on github.com, then link and push:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ev-virtual-battery.git
git push -u origin main
```

---

## 📜 License
This project is open-source and distributed under the MIT License.
