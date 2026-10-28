/**
 * ChronoSense | Diachronic Semantic Change Studio
 * Frontend Application & Visualization Engine
 */

// Global State & Chart Instances
let frequencyChart = null;
let driftChart = null;
let compareDriftChart = null;
let compareFreqChart = null;

let trajectoryData = null;
let trajectoryAnimationTimer = null;
let isPlayingTrajectory = false;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSystemStatus();
  initExplorer();
  initComparative();
  initTrajectory();
  initSemEval();
  initCustomCorpus();
  initMethodology();

  // Load default word
  loadWordAnalysis("gay");
});

/* =====================================================================
   1. Tab Navigation
   ===================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      // Tab specific triggers
      if (targetId === "tab-compare" && !compareDriftChart) {
        loadComparison();
      } else if (targetId === "tab-trajectory" && !trajectoryData) {
        loadTrajectory();
      } else if (targetId === "tab-semeval") {
        loadSemEval();
      } else if (targetId === "tab-methodology") {
        loadMethodology();
      }
    });
  });
}

/* =====================================================================
   2. System Status
   ===================================================================== */
async function initSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const statusText = document.getElementById("statusText");
    if (data.embeddings_loaded) {
      statusText.textContent = `HistWords (${data.decades[0]}–${data.decades[data.decades.length - 1]}) Ready`;
    } else {
      statusText.textContent = "Google Books Ready (Embeddings Offline)";
    }
  } catch (err) {
    console.error("Status check failed:", err);
  }
}

/* =====================================================================
   3. Word Explorer
   ===================================================================== */
function initExplorer() {
  const searchBtn = document.getElementById("explorerSearchBtn");
  const searchInput = document.getElementById("explorerSearchInput");
  const corpusSelect = document.getElementById("explorerCorpusSelect");

  const runSearch = () => {
    const word = searchInput.value.trim();
    if (word) {
      loadWordAnalysis(word, corpusSelect.value);
    }
  };

  searchBtn.addEventListener("click", runSearch);
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") runSearch();
  });

  // Quick Chips
  const chips = document.querySelectorAll(".quick-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const word = chip.getAttribute("data-word");
      searchInput.value = word;
      loadWordAnalysis(word, corpusSelect.value);
    });
  });
}

async function loadWordAnalysis(word, corpus = "en-2019") {
  try {
    const url = `/api/analyze?word=${encodeURIComponent(word)}&corpus=${encodeURIComponent(corpus)}`;
    const res = await fetch(url);
    const data = await res.json();
    renderWordAnalysis(data);
  } catch (err) {
    console.error("Failed to load analysis for word:", word, err);
  }
}

function renderWordAnalysis(data) {
  const word = data.word;
  const sm = data.semantic_measurement;
  const ft = data.frequency_trajectory;
  const meta = data.benchmark_metadata;

  // 1. KPI Cards
  const kpiDisp = document.getElementById("kpiDisplacement");
  const kpiRange = document.getElementById("kpiDecadeRange");
  const kpiPeakVel = document.getElementById("kpiPeakVelocity");
  const kpiPeakRate = document.getElementById("kpiPeakRate");
  const kpiFreq = document.getElementById("kpiFrequencyPattern");
  const kpiPeakYear = document.getElementById("kpiPeakYear");
  const kpiClass = document.getElementById("kpiClassification");
  const kpiClassSub = document.getElementById("kpiClassificationSub");

  if (sm) {
    kpiDisp.textContent = sm.total_displacement.toFixed(4);
    const dStart = sm.decades[0];
    const dEnd = sm.decades[sm.decades.length - 1];
    kpiRange.textContent = `Cosine Distance (${dStart}s → ${dEnd}s)`;

    kpiPeakVel.textContent = `${sm.peak_velocity_decade}s`;
    kpiPeakRate.textContent = `Peak velocity: ${sm.peak_velocity.toFixed(3)}/decade`;

    const clf = sm.classification;
    kpiClass.innerHTML = `<span class="badge-shift ${getBadgeClass(clf.primary_category)}">${clf.primary_category}</span>`;
    kpiClassSub.textContent = `Confidence: ${(clf.confidence * 100).toFixed(0)}%`;
  } else {
    kpiDisp.textContent = "N/A";
    kpiRange.textContent = "Word not in historical embeddings";
    kpiPeakVel.textContent = "--";
    kpiPeakRate.textContent = "No vector kinematics";
    kpiClass.textContent = "Unindexed";
    kpiClassSub.textContent = "Ngram frequency only";
  }

  if (ft) {
    kpiFreq.textContent = ft.pattern;
    kpiPeakYear.textContent = `Peak Year: ${ft.peak_year} (${ft.peak_frequency.toExponential(2)})`;
  }

  // 2. Narrative & Senses
  const narrativeBox = document.getElementById("explorerNarrativeBox");
  const narrativeTitle = document.getElementById("narrativeTitle");
  const narrativeContext = document.getElementById("narrativeContext");
  const sense1800s = document.getElementById("sense1800s");
  const sense2000s = document.getElementById("sense2000s");

  if (meta) {
    narrativeBox.style.display = "block";
    narrativeTitle.textContent = `Historical Semantic Evolution: '${word.toUpperCase()}' (${meta.era_shift})`;
    narrativeContext.textContent = meta.historical_context;
    sense1800s.textContent = meta.primary_sense_1800s;
    sense2000s.textContent = meta.primary_sense_2000s;
  } else if (sm && sm.classification) {
    narrativeBox.style.display = "block";
    narrativeTitle.textContent = `Semantic Shift Analysis: '${word.toUpperCase()}'`;
    narrativeContext.textContent = sm.classification.explanation;
    sense1800s.textContent = `Dominant fields: ${sm.classification.top_domains_start.join(", ") || "General English"}`;
    sense2000s.textContent = `Dominant fields: ${sm.classification.top_domains_end.join(", ") || "General English"}`;
  } else {
    narrativeBox.style.display = "none";
  }

  // 3. Render Frequency Chart
  renderFrequencyChart(ft);

  // 4. Render Drift & Velocity Chart
  renderDriftChart(sm);

  // 5. Neighbors Timeline
  renderNeighborsTimeline(data.neighbors_timeline);
}

function getBadgeClass(cat) {
  if (cat.includes("Amelioration")) return "badge-amelioration";
  if (cat.includes("Pejoration")) return "badge-pejoration";
  if (cat.includes("Broadening")) return "badge-broadening";
  if (cat.includes("Narrowing")) return "badge-narrowing";
  return "badge-domain";
}

/* =====================================================================
   Chart.js Visualizations
   ===================================================================== */
function renderFrequencyChart(ft) {
  const ctx = document.getElementById("frequencyChart").getContext("2d");
  if (frequencyChart) {
    frequencyChart.destroy();
  }

  if (!ft || !ft.annual_years) {
    return;
  }

  frequencyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ft.annual_years,
      datasets: [
        {
          label: `Relative Frequency (${ft.ngram})`,
          data: ft.annual_frequencies,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.08)",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#38bdf8",
          fill: true,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#ffffff",
          bodyColor: "#38bdf8",
          borderColor: "rgba(56, 189, 248, 0.3)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => ` Freq: ${ctx.parsed.y.toExponential(3)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", maxTicksLimit: 10 },
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: {
            color: "#64748b",
            callback: (v) => v.toExponential(1),
          },
        },
      },
    },
  });
}

function renderDriftChart(sm) {
  const ctx = document.getElementById("driftChart").getContext("2d");
  if (driftChart) {
    driftChart.destroy();
  }

  if (!sm || !sm.decades) {
    return;
  }

  const decades = sm.decades;
  const cumulative = decades.map((d) => sm.cumulative_drift[d] || 0.0);
  const velocity = decades.map((d) => sm.velocities[d] || 0.0);

  driftChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: decades,
      datasets: [
        {
          label: "Cumulative Semantic Drift (1 - CosSim)",
          data: cumulative,
          borderColor: "#a855f7",
          backgroundColor: "rgba(168, 85, 247, 0.12)",
          borderWidth: 2.5,
          pointBackgroundColor: "#c084fc",
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: true,
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Semantic Velocity (Rate / Decade)",
          data: velocity,
          borderColor: "#fbbf24",
          borderWidth: 1.8,
          borderDash: [5, 4],
          pointBackgroundColor: "#fbbf24",
          pointRadius: 2.5,
          tension: 0.2,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          borderColor: "rgba(168, 85, 247, 0.3)",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b" },
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Cumulative Drift", color: "#a855f7" },
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b" },
          min: 0,
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Velocity", color: "#fbbf24" },
          grid: { drawOnChartArea: false },
          ticks: { color: "#64748b" },
          min: 0,
        },
      },
    },
  });
}

function renderNeighborsTimeline(timeline) {
  const container = document.getElementById("neighborsTimelineRow");
  container.innerHTML = "";

  if (!timeline) return;

  const sampleDecades = [1830, 1860, 1900, 1940, 1970, 1990];
  const activeDecades = sampleDecades.filter((d) => timeline[d] && timeline[d].length > 0);

  activeDecades.forEach((d) => {
    const card = document.createElement("div");
    card.className = "decade-card";

    const header = document.createElement("div");
    header.className = "decade-card-header";
    header.innerHTML = `
      <span class="decade-year">${d}s</span>
      <span class="decade-tag">Historical Era</span>
    `;

    const list = document.createElement("div");
    list.className = "neighbor-list";

    timeline[d].slice(0, 6).forEach((item) => {
      const neighborItem = document.createElement("div");
      neighborItem.className = "neighbor-item";
      neighborItem.innerHTML = `
        <span class="neighbor-word">${item.word}</span>
        <span class="neighbor-sim">${item.similarity.toFixed(2)}</span>
      `;
      list.appendChild(neighborItem);
    });

    card.appendChild(header);
    card.appendChild(list);
    container.appendChild(card);
  });
}

/* =====================================================================
   4. Comparative Studio
   ===================================================================== */
function initComparative() {
  const compareBtn = document.getElementById("compareBtn");
  const compareInput = document.getElementById("compareInput");

  compareBtn.addEventListener("click", () => {
    loadComparison();
  });
  compareInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") loadComparison();
  });
}

async function loadComparison() {
  const words = document.getElementById("compareInput").value;
  try {
    const res = await fetch(`/api/compare?words=${encodeURIComponent(words)}`);
    const data = await res.json();
    renderComparison(data);
  } catch (err) {
    console.error("Comparison load error:", err);
  }
}

function renderComparison(data) {
  const palette = ["#38bdf8", "#a855f7", "#34d399", "#fbbf24", "#f43f5e", "#818cf8"];

  // 1. Superimposed Drift Chart
  const ctxDrift = document.getElementById("compareDriftChart").getContext("2d");
  if (compareDriftChart) compareDriftChart.destroy();

  const driftDatasets = [];
  data.analyses.forEach((analysis, idx) => {
    const sm = analysis.semantic_measurement;
    if (sm && sm.decades) {
      const color = palette[idx % palette.length];
      driftDatasets.push({
        label: `${analysis.word} (Δ ${sm.total_displacement.toFixed(2)})`,
        data: sm.decades.map((d) => sm.cumulative_drift[d] || 0.0),
        borderColor: color,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 2,
      });
    }
  });

  const baseDecades = (data.analyses[0] && data.analyses[0].semantic_measurement)
    ? data.analyses[0].semantic_measurement.decades
    : [1800, 1850, 1900, 1950, 1990];

  compareDriftChart = new Chart(ctxDrift, {
    type: "line",
    data: {
      labels: baseDecades,
      datasets: driftDatasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#64748b" } },
        y: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#64748b" } },
      },
    },
  });

  // 2. Comparative Frequency Chart
  const ctxFreq = document.getElementById("compareFreqChart").getContext("2d");
  if (compareFreqChart) compareFreqChart.destroy();

  const freqDatasets = [];
  let sharedYears = [];
  data.analyses.forEach((analysis, idx) => {
    const ft = analysis.frequency_trajectory;
    if (ft && ft.annual_years) {
      sharedYears = ft.annual_years;
      const color = palette[idx % palette.length];
      freqDatasets.push({
        label: ft.ngram,
        data: ft.annual_frequencies,
        borderColor: color,
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.2,
      });
    }
  });

  compareFreqChart = new Chart(ctxFreq, {
    type: "line",
    data: {
      labels: sharedYears,
      datasets: freqDatasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#64748b", maxTicksLimit: 10 } },
        y: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#64748b" } },
      },
    },
  });

  // 3. Pairwise Cross-Word Similarity Table
  const tableWrapper = document.getElementById("pairwiseTableWrapper");
  const pairwise = data.pairwise_similarities;
  if (!pairwise || Object.keys(pairwise).length === 0) {
    tableWrapper.innerHTML = "<p style='color: var(--text-dim); padding: 1rem;'>No pairwise embeddings available.</p>";
    return;
  }

  const sampleDecs = [1830, 1870, 1910, 1950, 1990];
  let tableHtml = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Word Pair</th>
          ${sampleDecs.map((d) => `<th>${d}s Sim</th>`).join("")}
          <th>Net Trend</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const [pair, sims] of Object.entries(pairwise)) {
    const vals = sampleDecs.map((d) => (sims[d] !== undefined && sims[d] !== null) ? sims[d] : null);
    const valid = vals.filter((v) => v !== null);
    let trend = "--";
    if (valid.length >= 2) {
      const diff = valid[valid.length - 1] - valid[0];
      trend = diff > 0.05 ? "🟢 Converging" : diff < -0.05 ? "🔴 Diverging" : "⚪ Stable";
    }

    tableHtml += `
      <tr>
        <td style="font-weight: 600; color: #fff;">${pair}</td>
        ${vals.map((v) => `<td>${v !== null ? v.toFixed(3) : "<span style='color: var(--text-dim)'>--</span>"}</td>`).join("")}
        <td>${trend}</td>
      </tr>
    `;
  }
  tableHtml += "</tbody></table>";
  tableWrapper.innerHTML = tableHtml;
}

/* =====================================================================
   5. 2D Semantic Time Machine (Canvas Trajectory Engine)
   ===================================================================== */
function initTrajectory() {
  const trajBtn = document.getElementById("trajectoryBtn");
  const slider = document.getElementById("decadeSlider");
  const playBtn = document.getElementById("playTrajectoryBtn");

  trajBtn.addEventListener("click", () => loadTrajectory());
  slider.addEventListener("input", (e) => {
    document.getElementById("sliderYearLabel").textContent = `${e.target.value}s`;
    drawTrajectoryCanvas(parseInt(e.target.value));
  });

  playBtn.addEventListener("click", () => {
    if (isPlayingTrajectory) {
      stopTrajectoryAnimation();
    } else {
      startTrajectoryAnimation();
    }
  });

  // Window resize observer for canvas
  window.addEventListener("resize", () => {
    if (trajectoryData) {
      drawTrajectoryCanvas(parseInt(slider.value));
    }
  });
}

async function loadTrajectory() {
  const words = document.getElementById("trajectoryWordsInput").value;
  try {
    const res = await fetch(`/api/trajectory?words=${encodeURIComponent(words)}`);
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    trajectoryData = data;
    const slider = document.getElementById("decadeSlider");
    drawTrajectoryCanvas(parseInt(slider.value));
  } catch (err) {
    console.error("Trajectory load error:", err);
  }
}

function startTrajectoryAnimation() {
  isPlayingTrajectory = true;
  document.getElementById("playTrajectoryBtn").textContent = "⏸";
  const slider = document.getElementById("decadeSlider");

  trajectoryAnimationTimer = setInterval(() => {
    let cur = parseInt(slider.value) + 10;
    if (cur > parseInt(slider.max)) {
      cur = parseInt(slider.min);
    }
    slider.value = cur;
    document.getElementById("sliderYearLabel").textContent = `${cur}s`;
    drawTrajectoryCanvas(cur);
  }, 600);
}

function stopTrajectoryAnimation() {
  isPlayingTrajectory = false;
  document.getElementById("playTrajectoryBtn").textContent = "▶";
  if (trajectoryAnimationTimer) {
    clearInterval(trajectoryAnimationTimer);
    trajectoryAnimationTimer = null;
  }
}

function drawTrajectoryCanvas(selectedDecade) {
  const canvas = document.getElementById("trajectoryCanvas");
  if (!canvas || !trajectoryData) return;

  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Clear background
  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, w, h);

  // Draw grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  const gridSize = 40;
  for (let x = 0; x < w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw coordinate axes
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.beginPath();
  ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Find min/max coordinates
  let minX = -1, maxX = 1, minY = -1, maxY = 1;
  const allPoints = [];
  for (const word of trajectoryData.words) {
    const traj = trajectoryData.trajectories[word] || [];
    traj.forEach((pt) => {
      allPoints.push(pt);
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    });
  }

  const pad = 60;
  const scaleX = (val) => pad + ((val - minX) / (maxX - minX || 1)) * (w - 2 * pad);
  const scaleY = (val) => h - (pad + ((val - minY) / (maxY - minY || 1)) * (h - 2 * pad));

  const palette = ["#38bdf8", "#a855f7", "#34d399", "#fbbf24", "#f43f5e", "#818cf8", "#f97316"];

  // Draw trails and current positions
  trajectoryData.words.forEach((word, wordIdx) => {
    const traj = trajectoryData.trajectories[word] || [];
    if (traj.length === 0) return;

    const color = palette[wordIdx % palette.length];

    // Filter points up to selected decade
    const visiblePoints = traj.filter((pt) => pt.decade <= selectedDecade);
    if (visiblePoints.length < 1) return;

    // Draw full trajectory trail (faded)
    ctx.strokeStyle = color + "40"; // 25% opacity
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    traj.forEach((pt, i) => {
      const cx = scaleX(pt.x);
      const cy = scaleY(pt.y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    // Draw active trail up to selected decade (glowing)
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    visiblePoints.forEach((pt, i) => {
      const cx = scaleX(pt.x);
      const cy = scaleY(pt.y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    // Draw waypoints
    visiblePoints.forEach((pt) => {
      const cx = scaleX(pt.x);
      const cy = scaleY(pt.y);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw active marker at selected decade
    const currentPoint = visiblePoints[visiblePoints.length - 1];
    const cx = scaleX(currentPoint.x);
    const cy = scaleY(currentPoint.y);

    // Glowing circle
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0; // reset

    // Word label
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px Outfit, sans-serif";
    ctx.fillText(`${word} (${currentPoint.decade}s)`, cx + 10, cy - 8);
  });
}

/* =====================================================================
   6. SemEval-2020 Benchmark Suite
   ==================================================================== */
function initSemEval() {
  const refreshBtn = document.getElementById("refreshSemevalBtn");
  refreshBtn.addEventListener("click", () => loadSemEval());
}

async function loadSemEval() {
  try {
    const res = await fetch("/api/semeval");
    const data = await res.json();
    renderSemEval(data);
  } catch (err) {
    console.error("SemEval load error:", err);
  }
}

function renderSemEval(data) {
  if (data.error) {
    console.error(data.error);
    return;
  }

  document.getElementById("semevalRho").textContent = `ρ = ${data.spearman_rank_correlation.toFixed(3)}`;
  document.getElementById("semevalF1").textContent = data.binary_classification.f1_score.toFixed(3);
  document.getElementById("semevalAcc").textContent = `${(data.binary_classification.accuracy * 100).toFixed(1)}%`;
  document.getElementById("semevalCount").textContent = `${data.targets_evaluated} lemmas`;

  const tbody = document.querySelector("#semevalTable tbody");
  tbody.innerHTML = "";

  data.targets.forEach((t) => {
    const tr = document.createElement("tr");
    const isChanged = t.gold_binary_label === 1;
    tr.innerHTML = `
      <td style="font-weight: 600; color: #fff;">${t.lemma}</td>
      <td style="font-family: var(--font-mono); color: var(--text-dim);">${t.pos}</td>
      <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${t.predicted_displacement.toFixed(4)}</td>
      <td style="font-family: var(--font-mono);">${t.gold_graded_score.toFixed(3)}</td>
      <td>
        <span class="badge-shift ${isChanged ? 'badge-domain' : 'badge-amelioration'}">
          ${isChanged ? 'Changed (1)' : 'Stable (0)'}
        </span>
      </td>
      <td style="color: var(--text-muted); font-size: 0.82rem;">${t.category}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =====================================================================
   7. Custom Diachronic Corpus Lab
   ===================================================================== */
function initCustomCorpus() {
  const runBtn = document.getElementById("runCustomCorpusBtn");
  const presetBtn = document.getElementById("loadCorpusPresetBtn");

  presetBtn.addEventListener("click", () => {
    document.getElementById("corpusTextT1").value = `
The magnetic electric battery and telegraphic wire communicate intelligence across great distances.
The electric battery and galvanic power transmit electric power to the recording instrument and apparatus.
The steam engine and locomotive carriage carry passengers and mail over iron rails.
The telegraph clerk records the signal intelligence and message upon paper tape by the mechanical instrument.
The power of the steam engine and mechanical apparatus creates industrial production.
The communication of intelligence by wire and battery revolutionized commerce and public news.
`;
    document.getElementById("corpusTextT2").value = `
The modern electric battery and wireless network communicate intelligence across cloud platforms.
The electric battery and power grid transmit electric power to the computing instrument and digital device.
The electric engine and automated vehicle carry passengers and delivery packages over urban roads.
The software system records the signal intelligence and message upon cloud storage by the software program.
The power of the computing engine and algorithmic apparatus creates digital production.
The communication of intelligence by wireless network and battery revolutionized commerce and social media.
`;
  });

  runBtn.addEventListener("click", async () => {
    const t1 = document.getElementById("corpusTextT1").value;
    const t2 = document.getElementById("corpusTextT2").value;

    if (!t1.trim() || !t2.trim()) {
      alert("Please supply text in both Period 1 and Period 2 text boxes.");
      return;
    }

    runBtn.textContent = "Processing & Aligning...";
    try {
      const res = await fetch("/api/custom-corpus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corpus_t1: t1,
          corpus_t2: t2,
          label_t1: "19th Century",
          label_t2: "21st Century",
        }),
      });
      const data = await res.json();
      renderCustomCorpusResults(data);
    } catch (err) {
      console.error("Custom corpus error:", err);
    } finally {
      runBtn.textContent = "Process, Align & Measure Shift";
    }
  });
}

function renderCustomCorpusResults(data) {
  const wrapper = document.getElementById("customCorpusResultsWrapper");
  const tbody = document.querySelector("#customCorpusTable tbody");
  tbody.innerHTML = "";

  if (data.error) {
    alert(data.error);
    return;
  }

  wrapper.style.display = "block";
  data.words.forEach((w) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight: 600; color: #fff;">${w.word}</td>
      <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${w.displacement.toFixed(4)}</td>
      <td style="font-family: var(--font-mono);">${w.frequency_t1}</td>
      <td style="font-family: var(--font-mono);">${w.frequency_t2}</td>
      <td style="font-size: 0.8rem; color: #94a3b8;">${w.neighbors_t1.map((n) => n[0]).join(", ") || "--"}</td>
      <td style="font-size: 0.8rem; color: #c084fc;">${w.neighbors_t2.map((n) => n[0]).join(", ") || "--"}</td>
      <td style="font-family: var(--font-mono);">${(w.neighborhood_turnover * 100).toFixed(0)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =====================================================================
   8. Methodology & Laws
   ===================================================================== */
function initMethodology() {
  // Triggered on tab activation
}

async function loadMethodology() {
  try {
    const res = await fetch("/api/laws");
    const data = await res.json();
    if (data.confirmed) {
      document.getElementById("lawConformityResult").textContent = `Confirmed (ρ = ${data.spearman_rho.toFixed(3)})`;
      document.getElementById("lawConformityStats").textContent = 
        `Evaluated on ${data.sample_words_evaluated} words (p = ${data.spearman_p_value.toExponential(2)}). Frequent words change more slowly.`;
    }
  } catch (err) {
    console.error("Laws evaluation error:", err);
  }
}
