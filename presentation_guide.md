# EV Virtual Battery Digital Twin — Presentation & Demo Guide

This guide is designed for presenting the project through **Visual Studio** or **VS Code** in technical reviews, academic project presentations, or engineering interviews.

---

## 🎯 Executive Pitch (30 Seconds)
> *"This project is an end-to-end multi-physics digital twin of an electric vehicle traction battery pack. It combines equivalent circuit electrochemistry (2RC Thevenin model), Arrhenius degradation laws with non-linear knee-point acceleration, and cell-to-cell manufacturing imbalance across a 96-series pack. It streams realistic CAN 2.0B telemetry to an interactive real-time dashboard while using machine learning to predict State of Health (SOH) with under 1% error."*

---

## 🖥️ How to Present via Visual Studio / VS Code

### Step 1: Open the Project
1. Launch **Visual Studio** or **Visual Studio Code**.
2. Open the project folder:
   ```
   C:\Users\srira\.gemini\antigravity\scratch\ev_virtual_battery
   ```
3. Open the file tree in the sidebar to show the clean modular architecture:
   - `virtual_battery.py` — Core 13-component simulation & ML engine
   - `app.py` — Flask REST API backend
   - `templates/index.html` & `static/` — Telemetry dashboard
   - `.vscode/launch.json` — 1-click debug launcher

### Step 2: One-Click Launch (F5)
1. In Visual Studio / VS Code, press **F5** (or open the **Run & Debug** panel `Ctrl+Shift+D`).
2. Select **`🚀 Launch Web Dashboard (Flask + HTML)`**.
3. The integrated terminal will start the server and your default browser will automatically open:
   ```
   http://127.0.0.1:5000
   ```

---

## 🎙️ Step-by-Step Live Demonstration Script

| Demo Stage | Action in Dashboard | What to Explain to the Audience |
| :--- | :--- | :--- |
| **1. Nominal Baseline** | Keep Age at `0.0 yr`, Urban cycle. | Point out **SOH: 100%**, internal resistance at **$460\,\text{m}\Omega$**, and nominal cell voltage around $3.74\,\text{V}$. Mention that the 96 cell bars are balanced with only minor Gaussian variance. |
| **2. Dynamic Driving** | Toggle between **Urban**, **Highway**, and **Aggressive**. | Show the **Current Mode badge**: it switches between `DISCHARGE (-)` and `REGEN (+)` as regenerative braking recovers kinetic energy. Point to the Canvas waveform tracking voltage sag. |
| **3. Accelerated Aging** | Move the **Vehicle Age slider** from `0.0` to `5.0` and then `8.0` years. | Explain the **Arrhenius calendar + cycle fade**. Watch the **SOH drop** and **Internal Resistance (IR) climb** from $460\,\text{m}\Omega$ to over $1800\,\text{m}\Omega$. |
| **4. The Aging Knee-Point** | Push Age to `10.0 years` on **Aggressive** mode. | Highlight the **Knee-Point phenomenon**: once capacity loss passes 15%, degradation accelerates exponentially due to lithium plating and active material loss. The badge flips to `CRITICAL KNEE`. |
| **5. 96-Cell Imbalance** | Observe the **96-Cell bar matrix**. | Point out individual cell variance: weaker cells dip into yellow ($<3.4\,\text{V}$) and red ($<3.1\,\text{V}$), proving that pack performance is governed by the weakest cell. |
| **6. CAN Bus & BMS Faults** | Scroll down to **Live CAN Stream** and **Alert Console**. | Show the simulated **CAN 2.0B frames (ID 0x180)** with hex payloads matching automotive standards, and show how the **temporal alert cooldown** prevents alarm thrashing. |

---

## ❓ Anticipated Q&A

### Q: Why use a 2RC Thevenin model instead of 1RC or Rint?
**A:** A simple 1RC model only captures fast electrochemical double-layer polarization ($\sim 30\,\text{s}$). The 2RC model adds a second, slower RC branch ($\tau_2 \approx 500\text{--}2000\,\text{s}$) to capture lithium-ion concentration diffusion in the active particles, essential for realistic voltage sag during prolonged highway cruising and rest recovery.

### Q: How does the SOH machine learning predictor work?
**A:** A `GradientBoostingRegressor` is trained on 5-minute rolling windows of CAN telemetry. Rather than requiring laboratory Coulomb counting, it extracts features accessible on a vehicle CAN bus: $dV/dQ$ peak shifts, dynamic IR sag during acceleration pulses, and temperature gradient rates.

### Q: How is cell imbalance simulated?
**A:** Each of the 96 series cells is assigned an individual capacity, initial SOC, and internal resistance from calibrated Gaussian distributions. The simulation counts Coulombs individually for every cell using vectorized NumPy operations, allowing real-time execution with negligible CPU overhead.
