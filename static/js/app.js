/**
 * app.js
 * Client-side script communicating with Flask Python Digital Twin backend.
 * Handles API polling, real-time Canvas telemetry plotting, and UI events.
 */

// ── Application State ────────────────────────────────────────────────────────
let isRunning = false;
let simSpeed = 1;
let tickTimer = null;
let isFetching = false;
let frameCounter = 0;

// Histories for real-time charting (last 60 frames)
const MAX_HIST = 60;
const histVolt = [];
const histCurr = [];
const histSOC = [];
const histTemp = [];

const nCells = 96;

// ── DOM References ───────────────────────────────────────────────────────────
const btnPlayPause = document.getElementById("btnPlayPause");
const iconPlay = document.getElementById("iconPlay");
const iconPause = document.getElementById("iconPause");
const labelPlay = document.getElementById("labelPlayPause");
const btnStep = document.getElementById("btnStep");
const btnReset = document.getElementById("btnReset");

const sliderAge = document.getElementById("sliderAge");
const valAge = document.getElementById("valAge");
const sliderInitialSOC = document.getElementById("sliderInitialSOC");
const valInitialSOC = document.getElementById("valInitialSOC");
const selectCycle = document.getElementById("selectCycle");
const valCycle = document.getElementById("valCycle");
const cycleDesc = document.getElementById("cycleDesc");
const sliderTemp = document.getElementById("sliderTemp");
const valTemp = document.getElementById("valTemp");

const badgeDynamicMode = document.getElementById("badgeDynamicMode");
const btnModeAuto = document.getElementById("btnModeAuto");
const btnModeManual = document.getElementById("btnModeManual");
const sliderManualCurrent = document.getElementById("sliderManualCurrent");
const valManualCurrent = document.getElementById("valManualCurrent");
const labelCurrentModeDesc = document.getElementById("labelCurrentModeDesc");
const presetButtons = document.querySelectorAll(".btn-preset");
let isManualCurrentMode = false;
let manualCurrentVal = 0.0;

const kpiVoltage = document.getElementById("kpiVoltage");
const kpiVSpread = document.getElementById("kpiVSpread");
const kpiCurrent = document.getElementById("kpiCurrent");
const kpiPower = document.getElementById("kpiPower");
const badgeCurrent = document.getElementById("badgeCurrentMode");
const kpiSOC = document.getElementById("kpiSOC");
const barSOC = document.getElementById("barSOC");
const kpiSOH = document.getElementById("kpiSOH");
const badgeSOH = document.getElementById("badgeSOH");
const kpiWeakSOH = document.getElementById("kpiWeakSOH");
const kpiConditionScore = document.getElementById("kpiConditionScore");
const liveBadgeCondition = document.getElementById("liveBadgeCondition");
const kpiSecondLife = document.getElementById("kpiSecondLife");
const kpiCapacity = document.getElementById("kpiCapacity");
const kpiTemp = document.getElementById("kpiTemp");
const kpiIR = document.getElementById("kpiIR");
const kpiCellIR = document.getElementById("kpiCellIR");

// ── CAN Virtual Battery Generator DOM References ────────────────────────────
const canInputVoltage = document.getElementById("canInputVoltage");
const canInputCurrent = document.getElementById("canInputCurrent");
const canInputTemp = document.getElementById("canInputTemp");
const canInputSOC = document.getElementById("canInputSOC");
const canInputCycles = document.getElementById("canInputCycles");
const btnGenerateTwin = document.getElementById("btnGenerateTwin");
const btnToggleMetrics = document.getElementById("btnToggleMetrics");
const modelMetricsCard = document.getElementById("modelMetricsCard");
const btnRetrainModel = document.getElementById("btnRetrainModel");

const genTwinSOH = document.getElementById("genTwinSOH");
const genTwinScore = document.getElementById("genTwinScore");
const genTwinCondition = document.getElementById("genTwinCondition");
const genTwinSecondLife = document.getElementById("genTwinSecondLife");
const genTwinCapacity = document.getElementById("genTwinCapacity");
const genTwinIR = document.getElementById("genTwinIR");
const genTwinEnergy = document.getElementById("genTwinEnergy");
const genTwinPower = document.getElementById("genTwinPower");
const genTwinTimestamp = document.getElementById("genTwinTimestamp");

const valCellMin = document.getElementById("valCellMin");
const valCellMax = document.getElementById("valCellMax");
const valCellSpread = document.getElementById("valCellSpread");
const cellMatrix = document.getElementById("cellMatrix");

const canLog = document.getElementById("canLog");
const canFrameCount = document.getElementById("canFrameCount");
const alertLog = document.getElementById("alertLog");
const alertCountBadge = document.getElementById("alertCountBadge");
const backendStatus = document.getElementById("backendStatus");

// ── Initialize 96-Cell Bars ──────────────────────────────────────────────────
function initCellMatrix() {
  cellMatrix.innerHTML = "";
  for (let i = 0; i < nCells; i++) {
    const bar = document.createElement("div");
    bar.id = `cell-${i}`;
    bar.className = "cell-bar flex-1 bg-emerald-500 rounded-t-sm";
    bar.style.height = "70%";
    bar.title = `Cell #${i + 1}`;
    cellMatrix.appendChild(bar);
  }
}

// ── API Communication ────────────────────────────────────────────────────────
async function postAPI(endpoint, body = {}) {
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setBackendConnected(true);
    return await res.json();
  } catch (err) {
    console.error(`Error contacting ${endpoint}:`, err);
    setBackendConnected(false);
    return null;
  }
}

function setBackendConnected(connected) {
  if (connected) {
    backendStatus.className = "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    backendStatus.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5"></span> Python Engine Live';
  } else {
    backendStatus.className = "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20";
    backendStatus.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5"></span> Disconnected';
  }
}

// ── Step Digital Twin (Tick) ────────────────────────────────────────────────
async function requestTick() {
  if (isFetching) return;
  isFetching = true;
  const data = await postAPI("/api/tick");
  isFetching = false;

  if (data && data.frame) {
    updateUI(data);
  }
}

// ── Update Dashboard DOM ────────────────────────────────────────────────────
function updateUI(data) {
  const f = data.frame;
  frameCounter++;

  // 1. Digital Gauges
  kpiVoltage.innerText = f.voltage.toFixed(1);
  kpiVSpread.innerText = Math.round(f.v_cell_spread * 1000) + " mV";

  kpiCurrent.innerText = Math.abs(f.current).toFixed(1);
  kpiPower.innerText = Math.abs(data.power_kw).toFixed(1) + " kW";

  if (f.current < -5.0) {
    badgeCurrent.innerText = "REGEN (+)";
    badgeCurrent.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20";
  } else if (f.current > 5.0) {
    badgeCurrent.innerText = "DISCHARGE (-)";
    badgeCurrent.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
  } else {
    badgeCurrent.innerText = "STANDBY";
    badgeCurrent.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-slate-500/10 text-slate-400 border border-slate-500/20";
  }

  const socPct = (f.soc * 100).toFixed(1);
  kpiSOC.innerText = socPct;
  barSOC.style.width = socPct + "%";
  if (f.soc < 0.20) {
    barSOC.className = "bg-rose-500 h-1.5 rounded-full transition-all duration-300";
  } else if (f.soc < 0.40) {
    barSOC.className = "bg-amber-500 h-1.5 rounded-full transition-all duration-300";
  } else {
    barSOC.className = "bg-emerald-500 h-1.5 rounded-full transition-all duration-300";
  }

  const sohPct = (f.soh * 100).toFixed(1);
  kpiSOH.innerText = sohPct;
  if (f.soh < 0.75) {
    badgeSOH.innerText = "CRITICAL KNEE";
    badgeSOH.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20";
  } else if (f.soh < 0.85) {
    badgeSOH.innerText = "DEGRADED";
    badgeSOH.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20";
  } else {
    badgeSOH.innerText = "OPTIMAL";
    badgeSOH.className = "text-[9px] font-semibold px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
  }
  kpiWeakSOH.innerText = (f.weakest_soh * 100).toFixed(1) + "%";

  kpiTemp.innerText = f.temp_cell.toFixed(1);
  kpiIR.innerText = (f.ir_pack * 1000).toFixed(1);
  kpiCellIR.innerText = ((f.ir_pack * 1000) / nCells).toFixed(2) + " mΩ";

  // ML Generated Condition Score & Condition Badge
  const condScore = data.condition_score !== undefined ? data.condition_score : (f.condition_score !== undefined ? f.condition_score : 95.0);
  if (kpiConditionScore) kpiConditionScore.innerText = condScore.toFixed(1);

  const cond = data.battery_condition || (f.virtual_battery && f.virtual_battery.battery_condition) || "Healthy";
  if (liveBadgeCondition) {
    liveBadgeCondition.innerText = cond.toUpperCase();
    if (cond === "Healthy") {
      liveBadgeCondition.className = "text-[8px] font-bold px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    } else if (cond === "Moderate") {
      liveBadgeCondition.className = "text-[8px] font-bold px-1 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20";
    } else if (cond === "Degraded") {
      liveBadgeCondition.className = "text-[8px] font-bold px-1 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20";
    } else {
      liveBadgeCondition.className = "text-[8px] font-bold px-1 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20";
    }
  }

  // Second Life Status & Usable Capacity
  const secondLife = data.second_life_status || (f.virtual_battery && f.virtual_battery.second_life_status) || "Not yet";
  if (kpiSecondLife) {
    kpiSecondLife.innerText = secondLife;
    if (secondLife.includes("replacement")) {
      kpiSecondLife.className = "text-xs font-bold font-mono tracking-tight text-rose-400 truncate";
    } else if (secondLife.includes("evaluation")) {
      kpiSecondLife.className = "text-xs font-bold font-mono tracking-tight text-amber-400 truncate";
    } else {
      kpiSecondLife.className = "text-xs font-bold font-mono tracking-tight text-sky-400 truncate";
    }
  }

  const cap = data.capacity_ah !== undefined ? data.capacity_ah : (f.virtual_battery && f.virtual_battery.capacity_ah ? f.virtual_battery.capacity_ah : 60.0);
  if (kpiCapacity) kpiCapacity.innerText = cap.toFixed(1) + " Ah";

  // 2. Cell Min/Max/Spread
  valCellMin.innerText = f.v_cell_min.toFixed(3) + " V";
  valCellMax.innerText = f.v_cell_max.toFixed(3) + " V";
  valCellSpread.innerText = Math.round(f.v_cell_spread * 1000) + " mV";

  // 3. Render 96 individual series cell voltages
  if (data.cell_voltages && data.cell_voltages.length === nCells) {
    const minV = f.v_cell_min;
    const maxV = f.v_cell_max;
    const range = Math.max(0.001, maxV - minV);

    for (let i = 0; i < nCells; i++) {
      const bar = document.getElementById(`cell-${i}`);
      if (!bar) continue;
      const cv = data.cell_voltages[i];
      const pct = Math.max(15, Math.min(100, ((cv - (minV - 0.05)) / (range + 0.10)) * 100));
      bar.style.height = pct + "%";

      if (cv < 3.10) {
        bar.className = "cell-bar flex-1 bg-rose-500 rounded-t-sm";
      } else if (cv < 3.40) {
        bar.className = "cell-bar flex-1 bg-amber-500 rounded-t-sm";
      } else {
        bar.className = "cell-bar flex-1 bg-emerald-500 rounded-t-sm";
      }
      bar.title = `Cell #${i + 1}: ${cv.toFixed(3)}V (SOH: ${(data.cell_soh[i] * 100).toFixed(1)}%)`;
    }
  }

  // 4. Update CAN Telemetry Hex Stream
  if (frameCounter % 2 === 0) {
    const hexV = Math.floor(f.voltage * 10).toString(16).padStart(4, "0").toUpperCase();
    const hexI = Math.floor((f.current + 500) * 10).toString(16).padStart(4, "0").toUpperCase();
    const hexSOC = Math.floor(f.soc * 200).toString(16).padStart(2, "0").toUpperCase();
    const hexTemp = Math.floor(f.temp_cell + 40).toString(16).padStart(2, "0").toUpperCase();

    const line = document.createElement("div");
    line.className = "flex items-center justify-between text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]";
    line.innerHTML = `
      <span class="text-emerald-400">t=${f.timestamp.toFixed(1)}s</span>
      <span>ID 0x180 [8]: ${hexV.slice(0, 2)} ${hexV.slice(2)} ${hexI.slice(0, 2)} ${hexI.slice(2)} ${hexSOC} ${hexTemp} 00 A1</span>
      <span class="text-[9px] text-[var(--muted-foreground)] font-sans font-medium">${f.voltage.toFixed(1)}V, ${f.current.toFixed(1)}A, ${f.temp_cell.toFixed(1)}°C</span>
    `;
    canLog.insertBefore(line, canLog.firstChild);
    if (canLog.children.length > 20) canLog.removeChild(canLog.lastChild);
    canFrameCount.innerText = `${frameCounter} frames`;
  }

  // 5. Append New Alerts
  if (data.new_alerts && data.new_alerts.length > 0) {
    for (const al of data.new_alerts) {
      const el = document.createElement("div");
      const isCrit = al.level === "CRITICAL";
      el.className = `p-1.5 rounded-lg border ${isCrit ? "bg-rose-500/10 border-rose-500/30 text-rose-400" : "bg-amber-500/10 border-amber-500/30 text-amber-400"} flex items-center justify-between text-[10px]`;
      el.innerHTML = `
        <span><b>[${al.level}]</b> ${al.msg}</span>
        <span class="text-[9px] opacity-75 font-mono">t=${al.t.toFixed(0)}s</span>
      `;
      if (alertLog.children.length > 0 && alertLog.children[0].innerText.includes("No active faults")) {
        alertLog.innerHTML = "";
      }
      alertLog.insertBefore(el, alertLog.firstChild);
      if (alertLog.children.length > 15) alertLog.removeChild(alertLog.lastChild);
    }
  }

  if (data.summary && data.summary.total_alerts > 0) {
    alertCountBadge.innerText = `${data.summary.total_alerts} Alert${data.summary.total_alerts > 1 ? "s" : ""}`;
    alertCountBadge.className = "text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20";
  }

  // 6. Push to Chart Histories
  histVolt.push(f.voltage);
  histCurr.push(f.current);
  histSOC.push(f.soc * 100.0);
  histTemp.push(f.temp_cell);
  if (histVolt.length > MAX_HIST) {
    histVolt.shift();
    histCurr.shift();
    histSOC.shift();
    histTemp.shift();
  }
}

// ── Real-Time Canvas Charts ──────────────────────────────────────────────────
function drawCharts() {
  drawVIChart();
  drawSTChart();
  requestAnimationFrame(drawCharts);
}

function drawVIChart() {
  const cvs = document.getElementById("chartVI");
  const dpr = window.devicePixelRatio || 1;
  const w = cvs.clientWidth, h = cvs.clientHeight;
  if (cvs.width !== w * dpr || cvs.height !== h * dpr) {
    cvs.width = w * dpr; cvs.height = h * dpr;
  }
  const ctx = cvs.getContext("2d");
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (histVolt.length < 2) return;

  // Grid
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = 0.25; y < 1; y += 0.25) { ctx.moveTo(0, h * y); ctx.lineTo(w, h * y); }
  ctx.stroke();

  // Voltage line (200V to 420V)
  const vMin = 200, vMax = 420;
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < histVolt.length; i++) {
    const x = (i / (MAX_HIST - 1)) * w;
    const y = h - ((histVolt[i] - vMin) / (vMax - vMin)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current line (-80A to +380A)
  const iMin = -80, iMax = 380;
  ctx.strokeStyle = "#0ea5e9";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < histCurr.length; i++) {
    const x = (i / (MAX_HIST - 1)) * w;
    const y = h - ((histCurr[i] - iMin) / (iMax - iMin)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawSTChart() {
  const cvs = document.getElementById("chartST");
  const dpr = window.devicePixelRatio || 1;
  const w = cvs.clientWidth, h = cvs.clientHeight;
  if (cvs.width !== w * dpr || cvs.height !== h * dpr) {
    cvs.width = w * dpr; cvs.height = h * dpr;
  }
  const ctx = cvs.getContext("2d");
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (histSOC.length < 2) return;

  // Grid
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = 0.25; y < 1; y += 0.25) { ctx.moveTo(0, h * y); ctx.lineTo(w, h * y); }
  ctx.stroke();

  // SOC line (0 to 100%)
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < histSOC.length; i++) {
    const x = (i / (MAX_HIST - 1)) * w;
    const y = h - (histSOC[i] / 100.0) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Temp line (10C to 60C)
  const tMin = 10, tMax = 60;
  ctx.strokeStyle = "#f43f5e";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < histTemp.length; i++) {
    const x = (i / (MAX_HIST - 1)) * w;
    const y = h - ((histTemp[i] - tMin) / (tMax - tMin)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ── Timing Loop ─────────────────────────────────────────────────────────────
function startLoop() {
  if (tickTimer) clearInterval(tickTimer);
  const interval = 500 / simSpeed;
  tickTimer = setInterval(requestTick, interval);
}

function stopLoop() {
  if (tickTimer) {
    clearInterval(tickTimer);
    tickTimer = null;
  }
}

function togglePlay() {
  isRunning = !isRunning;
  if (isRunning) {
    iconPlay.classList.add("hidden");
    iconPause.classList.remove("hidden");
    labelPlay.innerText = "Pause";
    btnPlayPause.className = "px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs rounded-xl shadow-sm transition flex items-center gap-1.5 cursor-pointer";
    startLoop();
  } else {
    iconPlay.classList.remove("hidden");
    iconPause.classList.add("hidden");
    labelPlay.innerText = "Resume";
    btnPlayPause.className = "px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs rounded-xl shadow-sm transition flex items-center gap-1.5 cursor-pointer";
    stopLoop();
  }
}

// ── Event Listeners ─────────────────────────────────────────────────────────
btnPlayPause.addEventListener("click", togglePlay);

btnStep.addEventListener("click", () => {
  if (isRunning) togglePlay();
  requestTick();
});

btnReset.addEventListener("click", async () => {
  await postAPI("/api/reset");
  histVolt.length = 0;
  histCurr.length = 0;
  histSOC.length = 0;
  histTemp.length = 0;
  canLog.innerHTML = "";
  alertLog.innerHTML = '<div class="text-[var(--muted-foreground)] italic">No active faults. Battery pack operating within nominal SOA window.</div>';
  alertCountBadge.innerText = "All Systems Nominal";
  alertCountBadge.className = "text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
  frameCounter = 0;
  requestTick();
});

document.querySelectorAll(".speed-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".speed-btn").forEach(b => {
      b.className = "speed-btn px-2 py-1 rounded-lg text-[var(--muted-foreground)] hover:text-white cursor-pointer";
    });
    btn.className = "speed-btn px-2 py-1 rounded-lg font-bold bg-emerald-600 text-white cursor-pointer";
    simSpeed = parseInt(btn.dataset.speed, 10);
    if (isRunning) startLoop();
  });
});

// Age Slider
sliderAge.addEventListener("input", async (e) => {
  const age = parseFloat(e.target.value);
  valAge.innerText = age.toFixed(1) + " Years";
  await postAPI("/api/config", { age_years: age });
  requestTick();
});

// Cycle Select
selectCycle.addEventListener("change", async (e) => {
  const cyc = e.target.value;
  valCycle.innerText = cyc.toUpperCase();
  const descriptions = {
    urban: "Stop-and-go city traffic, 25% regenerative braking.",
    highway: "Sustained high-speed cruise with occasional slowdowns.",
    aggressive: "Heavy full-throttle accelerations and harsh braking.",
    idle: "Vehicle parked; low auxiliary cabin load."
  };
  cycleDesc.innerText = descriptions[cyc] || "";
  await postAPI("/api/config", { cycle: cyc });
  requestTick();
});

// Ambient Temperature Slider
sliderTemp.addEventListener("input", async (e) => {
  const temp = parseFloat(e.target.value);
  valTemp.innerText = temp.toFixed(1) + " °C";
  await postAPI("/api/config", { ambient_temp: temp });
  requestTick();
});

// Initial SOC Slider
if (sliderInitialSOC) {
  sliderInitialSOC.addEventListener("input", async (e) => {
    const soc = parseFloat(e.target.value);
    valInitialSOC.innerText = soc.toFixed(0) + " %";
    await postAPI("/api/config", { initial_soc: soc / 100.0 });
    requestTick();
  });
}

// Dynamic Throttle / Load Current Control
function setDynamicControlMode(manual) {
  isManualCurrentMode = manual;
  if (manual) {
    btnModeManual.className = "px-2.5 py-1 rounded-lg font-bold bg-emerald-600 text-white cursor-pointer text-xs transition";
    btnModeAuto.className = "px-2.5 py-1 rounded-lg text-[var(--muted-foreground)] hover:text-white bg-slate-800 border border-[var(--border)] cursor-pointer text-xs transition";
    badgeDynamicMode.innerText = "Manual Throttle Active";
    badgeDynamicMode.className = "px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    sliderManualCurrent.disabled = false;
    sliderManualCurrent.classList.remove("cursor-not-allowed", "opacity-50");
    sliderManualCurrent.classList.add("cursor-pointer");
    labelCurrentModeDesc.innerText = "(Dynamic user throttle override)";
    updateManualCurrent(parseFloat(sliderManualCurrent.value));
  } else {
    btnModeAuto.className = "px-2.5 py-1 rounded-lg font-bold bg-emerald-600 text-white cursor-pointer text-xs transition";
    btnModeManual.className = "px-2.5 py-1 rounded-lg text-[var(--muted-foreground)] hover:text-white bg-slate-800 border border-[var(--border)] cursor-pointer text-xs transition";
    badgeDynamicMode.innerText = "Auto Drive Cycle";
    badgeDynamicMode.className = "px-2 py-0.5 text-[10px] font-semibold rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20";
    sliderManualCurrent.disabled = true;
    sliderManualCurrent.classList.add("cursor-not-allowed", "opacity-50");
    sliderManualCurrent.classList.remove("cursor-pointer");
    labelCurrentModeDesc.innerText = "(Governed by selected drive cycle)";
    valManualCurrent.innerText = "Auto (Cycle)";
    valManualCurrent.className = "font-mono font-bold text-sm text-emerald-400";
    postAPI("/api/config", { manual_current: null });
  }
}

async function updateManualCurrent(val) {
  manualCurrentVal = val;
  sliderManualCurrent.value = val;
  let text = `${val >= 0 ? "+" : ""}${val.toFixed(1)} A`;
  let color = "text-slate-300";
  if (val < -10) {
    text += " (Charge/Regen)";
    color = "text-sky-400";
  } else if (val > 150) {
    text += " (High Discharge)";
    color = "text-rose-400";
  } else if (val > 10) {
    text += " (Discharge)";
    color = "text-emerald-400";
  } else {
    text += " (Idle)";
    color = "text-slate-400";
  }
  valManualCurrent.innerText = text;
  valManualCurrent.className = `font-mono font-bold text-sm ${color}`;

  if (isManualCurrentMode) {
    await postAPI("/api/config", { manual_current: val });
  }
}

if (btnModeAuto && btnModeManual) {
  btnModeAuto.addEventListener("click", () => setDynamicControlMode(false));
  btnModeManual.addEventListener("click", () => setDynamicControlMode(true));
}

if (sliderManualCurrent) {
  sliderManualCurrent.addEventListener("input", (e) => {
    updateManualCurrent(parseFloat(e.target.value));
  });
}

presetButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const val = parseFloat(btn.dataset.current);
    if (!isManualCurrentMode) {
      setDynamicControlMode(true);
    }
    updateManualCurrent(val);
  });
});

// ── CAN-to-Virtual Battery Generator Logic ──────────────────────────────────
const samplePresets = {
  "EVB_0010": { voltage: 371.93, current: 63.45, temp_cell: 23.5, soc: 88.53, cycle_count: 171 },
  "EVB_0003": { voltage: 378.93, current: 61.32, temp_cell: 33.0, soc: 93.19, cycle_count: 910 },
  "EVB_0001": { voltage: 324.25, current: 76.06, temp_cell: 17.81, soc: 20.55, cycle_count: 1176 },
  "EVB_0007": { voltage: 377.84, current: 31.52, temp_cell: 48.74, soc: 99.28, cycle_count: 1774 },
};

document.querySelectorAll(".btn-can-sample").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sample;
    const p = samplePresets[key];
    if (!p) return;
    if (canInputVoltage) canInputVoltage.value = p.voltage;
    if (canInputCurrent) canInputCurrent.value = p.current;
    if (canInputTemp) canInputTemp.value = p.temp_cell;
    if (canInputSOC) canInputSOC.value = p.soc;
    if (canInputCycles) canInputCycles.value = p.cycle_count;
    triggerGenerateTwin();
  });
});

if (btnToggleMetrics && modelMetricsCard) {
  btnToggleMetrics.addEventListener("click", () => {
    modelMetricsCard.classList.toggle("hidden");
  });
}

if (btnRetrainModel) {
  btnRetrainModel.addEventListener("click", async () => {
    btnRetrainModel.disabled = true;
    btnRetrainModel.innerText = "⏳ Retraining...";
    try {
      const res = await postAPI("/api/train_model", {});
      if (res && res.status === "success") {
        alert("Virtual Battery CAN Model retrained successfully!\nAccuracy: " + (res.metrics.second_life_status.accuracy * 100).toFixed(1) + "%");
      }
    } catch (e) {
      console.error(e);
    } finally {
      btnRetrainModel.disabled = false;
      btnRetrainModel.innerText = "⚙️ Retrain Model";
    }
  });
}

async function triggerGenerateTwin() {
  if (!canInputVoltage || !btnGenerateTwin) return;
  const payload = {
    voltage: parseFloat(canInputVoltage.value),
    current: parseFloat(canInputCurrent.value),
    temp_cell: parseFloat(canInputTemp.value),
    soc: parseFloat(canInputSOC.value),
    cycle_count: parseFloat(canInputCycles.value)
  };

  btnGenerateTwin.disabled = true;
  btnGenerateTwin.innerHTML = "<span>⏳ Computing...</span>";

  const res = await postAPI("/api/generate_virtual_battery", payload);
  btnGenerateTwin.disabled = false;
  btnGenerateTwin.innerHTML = "<span>⚡ Generate Twin</span>";

  if (res && res.data && res.data.virtual_battery) {
    renderGeneratedTwin(res.data);
  }
}

function renderGeneratedTwin(data) {
  const vb = data.virtual_battery;
  if (genTwinSOH) genTwinSOH.innerText = vb.soh_percent.toFixed(2) + " %";
  if (genTwinScore) genTwinScore.innerText = vb.condition_score.toFixed(1) + " / 100";
  if (genTwinCapacity) genTwinCapacity.innerText = vb.capacity_ah.toFixed(2) + " Ah";
  if (genTwinIR) genTwinIR.innerText = vb.internal_resistance_mohm.toFixed(1) + " mΩ";
  if (genTwinEnergy) genTwinEnergy.innerText = vb.energy_remaining_kwh.toFixed(2) + " kWh";
  if (genTwinPower) genTwinPower.innerText = vb.power_kw.toFixed(2) + " kW";
  if (genTwinTimestamp) genTwinTimestamp.innerText = `Inferred @ ${new Date().toLocaleTimeString()} • 96S Spread: ${vb.v_cell_spread_mv} mV`;

  if (genTwinCondition) {
    genTwinCondition.innerText = vb.battery_condition.toUpperCase();
    if (vb.battery_condition === "Healthy") {
      genTwinCondition.className = "inline-block px-2.5 py-0.5 text-[11px] font-bold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
    } else if (vb.battery_condition === "Moderate") {
      genTwinCondition.className = "inline-block px-2.5 py-0.5 text-[11px] font-bold rounded-full bg-sky-500/20 text-sky-400 border border-sky-500/30";
    } else if (vb.battery_condition === "Degraded") {
      genTwinCondition.className = "inline-block px-2.5 py-0.5 text-[11px] font-bold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
    } else {
      genTwinCondition.className = "inline-block px-2.5 py-0.5 text-[11px] font-bold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
    }
  }

  if (genTwinSecondLife) {
    genTwinSecondLife.innerText = vb.second_life_status;
    if (vb.second_life_status.includes("replacement")) {
      genTwinSecondLife.className = "inline-block px-2 py-0.5 text-[10px] font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
    } else if (vb.second_life_status.includes("evaluation")) {
      genTwinSecondLife.className = "inline-block px-2 py-0.5 text-[10px] font-semibold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
    } else {
      genTwinSecondLife.className = "inline-block px-2 py-0.5 text-[10px] font-semibold rounded-full bg-sky-500/20 text-sky-400 border border-sky-500/30";
    }
  }

  // Update cell matrix if cell voltages provided
  if (data.cell_voltages && data.cell_voltages.length === nCells) {
    const minV = vb.v_cell_min;
    const maxV = vb.v_cell_max;
    const range = Math.max(0.001, maxV - minV);
    for (let i = 0; i < nCells; i++) {
      const bar = document.getElementById(`cell-${i}`);
      if (!bar) continue;
      const cv = data.cell_voltages[i];
      const pct = Math.max(15, Math.min(100, ((cv - (minV - 0.05)) / (range + 0.10)) * 100));
      bar.style.height = pct + "%";
      bar.title = `Cell #${i + 1}: ${cv.toFixed(3)}V (SOH: ${vb.soh_percent.toFixed(1)}%)`;
    }
    if (valCellMin) valCellMin.innerText = minV.toFixed(3) + " V";
    if (valCellMax) valCellMax.innerText = maxV.toFixed(3) + " V";
    if (valCellSpread) valCellSpread.innerText = vb.v_cell_spread_mv.toFixed(0) + " mV";
  }
}

if (btnGenerateTwin) {
  btnGenerateTwin.addEventListener("click", triggerGenerateTwin);
}

// ── Startup ─────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initCellMatrix();
  drawCharts();
  requestTick();
  // Populate initial generated twin
  triggerGenerateTwin();
  // Auto-start twin loop
  togglePlay();
});
