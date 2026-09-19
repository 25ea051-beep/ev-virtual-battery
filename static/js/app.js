/**
 * EV Virtual Battery Command Center — app.js v2
 * ─────────────────────────────────────────────
 * Controllers:
 *  1. CANInputController  – manages 5 CAN sliders, influence bars, OOD badges
 *  2. GaugeController     – SVG arc SOH gauge with animated transitions
 *  3. RadarController     – Chart.js radar for 5-input feature influence
 *  4. ChartController     – 3 live telemetry line charts (SOC, Voltage, Current)
 *  5. CellMatrixController – 96-cell voltage heatmap
 *  6. SimulationController – tick loop, drive cycle, configuration
 *  7. AlertController     – alert feed management
 */

'use strict';

// ─── Preset CAN configurations ─────────────────────────────────────────────
const PRESETS = {
  EVB_0010: { voltage: 371.93, current: 63.45, temp_cell: 23.5,  soc: 88.53, cycle_count: 171  },
  EVB_0003: { voltage: 378.93, current: 61.32, temp_cell: 33.0,  soc: 93.19, cycle_count: 910  },
  EVB_0001: { voltage: 324.25, current: 76.06, temp_cell: 17.81, soc: 20.55, cycle_count: 1176 },
  EVB_0007: { voltage: 377.84, current: 31.52, temp_cell: 48.74, soc: 99.28, cycle_count: 1774 },
};

// Training ranges for OOD coloring
const TRAINING_RANGES = {
  Voltage_V:     [313,  390],
  Current_A:     [-12,  100],
  Temperature_C: [10,   55 ],
  SOC_percent:   [5,    100],
  Cycle_Count:   [50,   1800],
};

// ─── SVG Arc Gauge helpers ──────────────────────────────────────────────────
function polarToXY(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, startDeg, endDeg) {
  const s = polarToXY(cx, cy, r, startDeg);
  const e = polarToXY(cx, cy, r, endDeg);
  const sweep = endDeg - startDeg;
  if (Math.abs(sweep) < 0.1) return `M ${s.x} ${s.y}`;
  const large = Math.abs(sweep) > 180 ? 1 : 0;
  const dir   = sweep > 0 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} ${dir} ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Gauge Controller
// ═══════════════════════════════════════════════════════════════════════════
const GaugeController = (() => {
  const CX = 100, CY = 105, R = 80;
  const START = 145, TOTAL = 250; // degrees, SVG clockwise
  let currentSoh = 100;

  const trackEl = document.getElementById('gauge-track');
  const fgEl    = document.getElementById('gauge-fg');
  const glowEl  = document.getElementById('gauge-glow');
  const valEl   = document.getElementById('gauge-soh-val');

  function init() {
    trackEl.setAttribute('d', arcPath(CX, CY, R, START, START + TOTAL));
  }

  function colorForSoh(soh) {
    if (soh >= 80) return '#059669';
    if (soh >= 65) return '#d97706';
    return '#dc2626';
  }

  function update(soh) {
    currentSoh = soh;
    const clampedSoh = Math.max(0, Math.min(100, soh));
    const endDeg = START + TOTAL * (clampedSoh / 100);
    const path  = arcPath(CX, CY, R, START, endDeg);
    const color = colorForSoh(clampedSoh);

    fgEl.setAttribute('d', path);
    fgEl.setAttribute('stroke', color);
    glowEl.setAttribute('d', path);
    glowEl.setAttribute('stroke', color);

    valEl.textContent   = clampedSoh.toFixed(1);
    valEl.style.color   = color;
  }

  init();
  update(100);

  return { update };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  Radar Controller
// ═══════════════════════════════════════════════════════════════════════════
const RadarController = (() => {
  const ctx = document.getElementById('chart-radar').getContext('2d');

  const labels = ['Voltage', 'Current', 'Temperature', 'SOC %', 'Cycles'];
  const initData = [20, 20, 20, 20, 20];

  Chart.defaults.color = '#64748b';

  const chart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Input Influence %',
        data: initData,
        backgroundColor: 'rgba(37,99,235,0.10)',
        borderColor: '#2563eb',
        borderWidth: 2,
        pointBackgroundColor: '#2563eb',
        pointBorderColor: '#2563eb',
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 600, easing: 'easeInOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.r.toFixed(1)}% influence`
          }
        }
      },
      scales: {
        r: {
          min: 0,
          max: 55,
          ticks: {
            stepSize: 10,
            color: '#64748b',
            font: { size: 9 },
            backdropColor: 'transparent',
          },
          grid:         { color: '#e2e8f0' },
          angleLines:   { color: '#cbd5e1' },
          pointLabels:  { color: '#334155', font: { size: 11, family: 'Plus Jakarta Sans', weight: '600' } },
        }
      }
    }
  });

  function update(influence) {
    chart.data.datasets[0].data = [
      influence.Voltage_V     || 20,
      influence.Current_A     || 20,
      influence.Temperature_C || 20,
      influence.SOC_percent   || 20,
      influence.Cycle_Count   || 20,
    ];
    chart.update('active');
  }

  return { update };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  Chart Controller (3 live line charts)
// ═══════════════════════════════════════════════════════════════════════════
const ChartController = (() => {
  const MAX_POINTS = 80;

  function makeLineChart(canvasId, color, label, unit) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label,
          data: [],
          borderColor: color,
          backgroundColor: color === '#059669' ? 'rgba(5,150,105,0.08)' : (color === '#2563eb' ? 'rgba(37,99,235,0.08)' : 'rgba(124,58,237,0.08)'),
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index', intersect: false,
            callbacks: { label: ctx => `${ctx.parsed.y?.toFixed(2)} ${unit}` }
          }
        },
        scales: {
          x: {
            display: false,
            grid: { color: '#f1f5f9' },
          },
          y: {
            grid: { color: '#e2e8f0' },
            ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 5 },
            border: { color: 'transparent' },
          }
        }
      }
    });
  }

  const socChart     = makeLineChart('chart-soc',     '#059669', 'SOC', '%');
  const voltChart    = makeLineChart('chart-voltage',  '#2563eb', 'Voltage', 'V');
  const currChart    = makeLineChart('chart-current',  '#7c3aed', 'Current', 'A');

  let step = 0;

  function push(soc_pct, voltage, current) {
    step++;
    const label = `${step}`;

    const push1 = (chart, val) => {
      if (!chart) return;
      chart.data.labels.push(label);
      chart.data.datasets[0].data.push(val);
      if (chart.data.labels.length > MAX_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
      }
      chart.update('none');
    };

    push1(socChart, soc_pct);
    push1(voltChart, voltage);
    push1(currChart, current);
  }

  function reset() {
    step = 0;
    [socChart, voltChart, currChart].forEach(ch => {
      if (!ch) return;
      ch.data.labels = [];
      ch.data.datasets[0].data = [];
      ch.update('none');
    });
  }

  return { push, reset };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  Cell Matrix Controller
// ═══════════════════════════════════════════════════════════════════════════
const CellMatrixController = (() => {
  const grid = document.getElementById('cell-grid');

  function cellColor(voltage) {
    if (voltage >= 3.8) return '#059669';
    if (voltage >= 3.5) return '#2563eb';
    if (voltage >= 3.2) return '#d97706';
    return '#dc2626';
  }

  function cellOpacity(voltage) {
    const norm = Math.max(0, Math.min(1, (voltage - 2.5) / (4.3 - 2.5)));
    return 0.45 + norm * 0.55;
  }

  // Build 96 cells initially
  const cells = [];
  for (let i = 0; i < 96; i++) {
    const el = document.createElement('div');
    el.className = 'cell';
    el.dataset.idx = i;
    grid.appendChild(el);
    cells.push(el);
  }

  function update(voltages) {
    if (!voltages || voltages.length === 0) return;
    voltages.forEach((v, i) => {
      if (i >= cells.length) return;
      const color = cellColor(v);
      const opacity = cellOpacity(v);
      cells[i].style.background = color;
      cells[i].style.opacity = opacity;
      cells[i].dataset.tip = `Cell ${i+1}: ${v.toFixed(3)}V`;
    });
  }

  return { update };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  Alert Controller
// ═══════════════════════════════════════════════════════════════════════════
const AlertController = (() => {
  const list = document.getElementById('alerts-list');
  const MAX_ALERTS = 15;
  let alertCount = 0;

  function add(alerts) {
    if (!alerts || alerts.length === 0) return;
    alerts.forEach(a => {
      alertCount++;
      const existing = list.querySelector('.no-alerts');
      if (existing) existing.remove();

      const el = document.createElement('div');
      el.className = `alert-item ${a.level.toLowerCase()}`;
      el.innerHTML = `<span class="alert-level">${a.level}</span><span class="alert-msg">${a.msg}</span>`;
      list.insertBefore(el, list.firstChild);

      while (list.children.length > MAX_ALERTS) {
        list.removeChild(list.lastChild);
      }
    });
  }

  function clear() {
    alertCount = 0;
    list.innerHTML = '<div class="no-alerts">No active alerts — system nominal</div>';
  }

  return { add, clear };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  CAN Input Controller
// ═══════════════════════════════════════════════════════════════════════════
const CANInputController = (() => {
  const INPUTS = [
    { key: 'voltage',  numId: 'inp-voltage', slId: 'sl-voltage',  oodId: 'ood-voltage',  cardId: 'card-voltage',  infId: 'inf-voltage',  pctId: 'pct-voltage',  rangeKey: 'Voltage_V' },
    { key: 'current',  numId: 'inp-current', slId: 'sl-current',  oodId: 'ood-current',  cardId: 'card-current',  infId: 'inf-current',  pctId: 'pct-current',  rangeKey: 'Current_A' },
    { key: 'temp_cell',numId: 'inp-temp',    slId: 'sl-temp',     oodId: 'ood-temp',     cardId: 'card-temp',     infId: 'inf-temp',     pctId: 'pct-temp',     rangeKey: 'Temperature_C' },
    { key: 'soc',      numId: 'inp-soc',     slId: 'sl-soc',      oodId: 'ood-soc',      cardId: 'card-soc',      infId: 'inf-soc',      pctId: 'pct-soc',      rangeKey: 'SOC_percent' },
    { key: 'cycle_count', numId: 'inp-cycles', slId: 'sl-cycles', oodId: 'ood-cycles',   cardId: 'card-cycles',   infId: 'inf-cycles',   pctId: 'pct-cycles',   rangeKey: 'Cycle_Count' },
  ];

  INPUTS.forEach(inp => {
    const numEl = document.getElementById(inp.numId);
    const slEl  = document.getElementById(inp.slId);

    numEl.addEventListener('input', () => {
      slEl.value = numEl.value;
      checkOod(inp);
    });
    slEl.addEventListener('input', () => {
      numEl.value = slEl.value;
      checkOod(inp);
    });
  });

  function checkOod(inp) {
    const val  = parseFloat(document.getElementById(inp.numId).value);
    const range = TRAINING_RANGES[inp.rangeKey];
    const oodEl = document.getElementById(inp.oodId);
    const cardEl = document.getElementById(inp.cardId);
    const isOod = val < range[0] || val > range[1];
    oodEl.classList.toggle('visible', isOod);
    cardEl.classList.toggle('ood-warning', isOod);
  }

  function getValues() {
    return {
      voltage:     parseFloat(document.getElementById('inp-voltage').value)  || 350,
      current:     parseFloat(document.getElementById('inp-current').value)  || 50,
      temp_cell:   parseFloat(document.getElementById('inp-temp').value)     || 25,
      soc:         parseFloat(document.getElementById('inp-soc').value)      || 80,
      cycle_count: parseFloat(document.getElementById('inp-cycles').value)   || 500,
    };
  }

  function setValues(vals) {
    const setOne = (numId, slId, val) => {
      document.getElementById(numId).value = val;
      document.getElementById(slId).value  = val;
    };
    if (vals.voltage    != null) setOne('inp-voltage', 'sl-voltage', vals.voltage);
    if (vals.current    != null) setOne('inp-current', 'sl-current', vals.current);
    if (vals.temp_cell  != null) setOne('inp-temp',    'sl-temp',    vals.temp_cell);
    if (vals.soc        != null) setOne('inp-soc',     'sl-soc',     vals.soc);
    if (vals.cycle_count!= null) setOne('inp-cycles',  'sl-cycles',  vals.cycle_count);
    INPUTS.forEach(checkOod);
  }

  function updateInfluenceBars(influence) {
    const keyMap = {
      Voltage_V:     { infId: 'inf-voltage',  pctId: 'pct-voltage' },
      Current_A:     { infId: 'inf-current',  pctId: 'pct-current' },
      Temperature_C: { infId: 'inf-temp',     pctId: 'pct-temp'    },
      SOC_percent:   { infId: 'inf-soc',      pctId: 'pct-soc'     },
      Cycle_Count:   { infId: 'inf-cycles',   pctId: 'pct-cycles'  },
    };
    Object.entries(influence).forEach(([key, pct]) => {
      const ids = keyMap[key];
      if (!ids) return;
      const fill = document.getElementById(ids.infId);
      const pctEl = document.getElementById(ids.pctId);
      if (fill)  fill.style.width  = `${pct}%`;
      if (pctEl) pctEl.textContent = `${pct.toFixed(0)}%`;
    });
  }

  return { getValues, setValues, updateInfluenceBars };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  UI Update helper (applies prediction output to all panels)
// ═══════════════════════════════════════════════════════════════════════════
function applyPrediction(result) {
  const vb = result.virtual_battery;
  const fi = result.feature_influence || {};
  const ood = result.ood_detection || {};

  // SOH Gauge
  GaugeController.update(vb.soh_percent);

  // Condition chip
  const cond = vb.battery_condition || 'Unknown';
  const chipCond = document.getElementById('chip-condition');
  const condLower = cond.toLowerCase();
  chipCond.className = `condition-chip ${condLower}`;
  chipCond.textContent = `● ${cond}`;

  // Second life
  document.getElementById('sl-text').textContent = vb.second_life_status || '--';

  // Condition score
  document.getElementById('condition-score-val').textContent = (vb.condition_score || 0).toFixed(1);

  // Mini KPIs (center panel)
  document.getElementById('kpi-capacity').textContent = (vb.capacity_ah || 0).toFixed(1);
  document.getElementById('kpi-ir').textContent       = (vb.internal_resistance_mohm || 0).toFixed(1);
  document.getElementById('kpi-energy').textContent   = (vb.energy_remaining_kwh || 0).toFixed(2);
  document.getElementById('kpi-power').textContent    = (vb.power_kw || 0).toFixed(2);

  // Right sidebar KPIs
  document.getElementById('r-capacity').textContent = (vb.capacity_ah || 0).toFixed(2);
  document.getElementById('r-ir').textContent       = (vb.internal_resistance_mohm || 0).toFixed(2);
  document.getElementById('r-energy').textContent   = (vb.energy_remaining_kwh || 0).toFixed(3);
  document.getElementById('r-power').textContent    = (vb.power_kw || 0).toFixed(3);
  document.getElementById('r-vmin').textContent     = (vb.v_cell_min || 0).toFixed(3);
  document.getElementById('r-vspread').textContent  = (vb.v_cell_spread_mv || 0).toFixed(1);

  // Radar chart
  if (Object.keys(fi).length > 0) {
    RadarController.update(fi);
    CANInputController.updateInfluenceBars(fi);
  }

  // Cell matrix
  if (result.cell_voltages && result.cell_voltages.length > 0) {
    CellMatrixController.update(result.cell_voltages);
  }

  // Alerts
  if (result.alerts && result.alerts.length > 0) {
    AlertController.add(result.alerts);
  }

  // OOD banner
  const banner = document.getElementById('ood-banner');
  const oodScore = ood.ood_score || 0;
  if (oodScore > 0.05) {
    banner.classList.add('visible');
    const physPct = Math.round((ood.physics_weight || 0) * 100);
    document.getElementById('ood-banner-text').textContent =
      `Some inputs are outside training range — physics fallback active (${physPct}% physics, ${100-physPct}% ML)`;
  } else {
    banner.classList.remove('visible');
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Predict from CAN (manual inputs)
// ═══════════════════════════════════════════════════════════════════════════
async function predictFromCAN() {
  const btn = document.getElementById('btn-predict');
  const canData = CANInputController.getValues();

  btn.classList.add('loading');
  btn.textContent = '⏳ Generating...';

  try {
    const resp = await fetch('/api/generate_virtual_battery', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(canData),
    });
    const json = await resp.json();
    if (json.status === 'success') {
      applyPrediction(json.data);
    } else {
      AlertController.add([{ level: 'WARNING', msg: `Prediction error: ${json.message}` }]);
    }
  } catch (e) {
    AlertController.add([{ level: 'WARNING', msg: `Network error: ${e.message}` }]);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = '⚡ Generate Virtual Battery';
  }
}

document.getElementById('btn-predict').addEventListener('click', predictFromCAN);


// ═══════════════════════════════════════════════════════════════════════════
//  Presets
// ═══════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.preset;
    const vals = PRESETS[id];
    if (!vals) return;
    CANInputController.setValues(vals);
    setTimeout(predictFromCAN, 100);
  });
});


// ═══════════════════════════════════════════════════════════════════════════
//  Simulation Controller
// ═══════════════════════════════════════════════════════════════════════════
const SimController = (() => {
  let isRunning = true;
  let intervalId = null;
  let simStep = 0;
  const TICK_MS = 500;

  const playBtn   = document.getElementById('btn-play-pause');
  const playIcon  = document.getElementById('play-icon');
  const playLabel = document.getElementById('play-label');
  const stepBtn   = document.getElementById('btn-step-once');
  const cyclesSel = document.getElementById('sel-drive-cycle');
  const ageSlider = document.getElementById('sl-age');
  const ageDisp   = document.getElementById('age-display');
  const timeDisp  = document.getElementById('sim-time-display');

  async function doTick() {
    try {
      const resp = await fetch('/api/tick', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const data = await resp.json();
      const frame = data.frame || {};

      simStep++;

      // Update charts
      const socPct = (frame.soc ?? 0) * 100;
      ChartController.push(socPct, frame.voltage ?? 0, frame.current ?? 0);

      // Update gauge & KPIs from live tick if no manual prediction active
      const vb = data.virtual_battery || {};
      if (vb.soh_percent != null) {
        GaugeController.update(vb.soh_percent);
        document.getElementById('kpi-capacity').textContent = (vb.capacity_ah||0).toFixed(1);
        document.getElementById('kpi-ir').textContent       = (vb.internal_resistance_mohm||0).toFixed(1);
        document.getElementById('kpi-energy').textContent   = (vb.energy_remaining_kwh||0).toFixed(2);
        document.getElementById('kpi-power').textContent    = (data.power_kw||0).toFixed(2);

        const cond = vb.battery_condition || 'Healthy';
        const chip = document.getElementById('chip-condition');
        chip.className = `condition-chip ${cond.toLowerCase()}`;
        chip.textContent = `● ${cond}`;

        document.getElementById('sl-text').textContent = vb.second_life_status || '--';
        document.getElementById('condition-score-val').textContent = (vb.condition_score||0).toFixed(1);
      }

      if (data.cell_voltages) CellMatrixController.update(data.cell_voltages);
      if (data.new_alerts && data.new_alerts.length) AlertController.add(data.new_alerts);

      // Update header
      document.getElementById('hdr-step').textContent = simStep;
      timeDisp.textContent = `T + ${(simStep * 0.5).toFixed(0)} s`;

      // Radar from live tick feature influence
      if (vb.feature_influence) {
        RadarController.update(vb.feature_influence);
        CANInputController.updateInfluenceBars(vb.feature_influence);
      }
    } catch(e) { /* Network blip, ignore */ }
  }

  function start() {
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(doTick, TICK_MS);
    isRunning = true;
    playIcon.textContent  = '⏸';
    playLabel.textContent = 'Pause';
    playBtn.classList.add('active');
  }

  function pause() {
    if (intervalId) clearInterval(intervalId);
    intervalId = null;
    isRunning = false;
    playIcon.textContent  = '▶';
    playLabel.textContent = 'Play';
    playBtn.classList.remove('active');
  }

  playBtn.addEventListener('click', () => isRunning ? pause() : start());

  stepBtn.addEventListener('click', async () => {
    pause();
    await doTick();
  });

  cyclesSel.addEventListener('change', async () => {
    try {
      await fetch('/api/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cycle: cyclesSel.value }),
      });
      document.getElementById('hdr-cycle').textContent = cyclesSel.value;
    } catch(e) {}
  });

  ageSlider.addEventListener('input', async () => {
    const age = parseFloat(ageSlider.value);
    ageDisp.textContent = `${age} yr`;
    try {
      await fetch('/api/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age_years: age }),
      });
    } catch(e) {}
  });

  // Auto-start
  start();

  return { start, pause };
})();


// ═══════════════════════════════════════════════════════════════════════════
//  Header buttons
// ═══════════════════════════════════════════════════════════════════════════
document.getElementById('btn-reset-twin').addEventListener('click', async () => {
  try {
    await fetch('/api/reset', { method: 'POST' });
    ChartController.reset();
    AlertController.clear();
    GaugeController.update(100);
    document.getElementById('hdr-step').textContent = '0';
  } catch(e) {}
});

document.getElementById('btn-retrain-model').addEventListener('click', async () => {
  const btn = document.getElementById('btn-retrain-model');
  btn.textContent = '⏳ Training...';
  btn.disabled = true;
  try {
    const resp = await fetch('/api/train_model', { method: 'POST' });
    const json = await resp.json();
    AlertController.add([{
      level: 'INFO',
      msg: json.status === 'success'
        ? `Model retrained. SOH R²=${json.metrics?.soh?.r2?.toFixed(4)}`
        : `Retrain failed: ${json.message}`
    }]);
  } catch(e) {
    AlertController.add([{ level: 'WARNING', msg: `Retrain error: ${e.message}` }]);
  } finally {
    btn.textContent = '↺ Retrain Model';
    btn.disabled = false;
  }
});


// ═══════════════════════════════════════════════════════════════════════════
//  Startup: load model info and run initial prediction
// ═══════════════════════════════════════════════════════════════════════════
(async () => {
  try {
    const resp = await fetch('/api/model_info');
    const info = await resp.json();
    if (info.global_importance) {
      RadarController.update(info.global_importance);
      CANInputController.updateInfluenceBars(info.global_importance);
    }
  } catch(e) {}

  // Run initial prediction with defaults
  setTimeout(predictFromCAN, 800);
})();
