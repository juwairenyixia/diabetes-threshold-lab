/* ============================================================
   糖尿病风险 · 决策阈值实验室
   纯原生 JS：无框架、无依赖、无网络请求。
   ============================================================ */
(() => {
  "use strict";

  const D = window.DIABETES_MODEL;
  const FEATURES = D.features;
  const META = D.meta;
  const W = D.model.weights;
  const MEANS = D.model.means;
  const SDS = D.model.sds;
  const B0 = D.model.intercept;
  const TEST = D.testScores; // [概率, 真实标签, 性别]
  const N = TEST.length;
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);

  /* ---------------- 状态 ---------------- */
  const state = {
    values: Object.fromEntries(FEATURES.map((f) => [f.key, f.default])),
    threshold: 0.5,
    cost: 5,
    preset: null,
  };

  /* ---------------- 数学 ---------------- */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const sigmoid = (z) => 1 / (1 + Math.exp(-clamp(z, -35, 35)));

  function weightOf(f, v) {
    if (f.type === "select") {
      const opt = f.options.find((o) => o.value === v);
      return opt && opt.term ? (W[opt.term] || 0) : 0;
    }
    return W[f.key] || 0;
  }

  function encodedOf(f, v) {
    switch (f.type) {
      case "range": return (Number(v) - MEANS[f.key]) / SDS[f.key];
      case "toggle": return v ? 1 : 0;
      case "segment": return Number(v);
      case "select": {
        const opt = f.options.find((o) => o.value === v);
        return opt && opt.term ? 1 : 0;
      }
      default: return 0;
    }
  }

  function contributionOf(f, v) {
    return weightOf(f, v) * encodedOf(f, v);
  }

  function predict() {
    let eta = B0;
    FEATURES.forEach((f) => { eta += contributionOf(f, state.values[f.key]); });
    return { p: sigmoid(eta), eta };
  }

  /* ---------------- 测试集索引（二分查找） ---------------- */
  const SORTED = TEST.slice().sort((a, b) => a[0] - b[0]);
  const PROBS = Float64Array.from(SORTED, (s) => s[0]);
  const POS_PREFIX = new Int32Array(N + 1);
  for (let i = 0; i < N; i++) {
    POS_PREFIX[i + 1] = POS_PREFIX[i] + (SORTED[i][1] === 1 ? 1 : 0);
  }
  const TOTAL_POS = POS_PREFIX[N];
  const TOTAL_NEG = N - TOTAL_POS;

  function lowerBound(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function statsAt(t) {
    const k = lowerBound(PROBS, t);
    const posBelow = POS_PREFIX[k];
    const negBelow = k - posBelow;
    const tp = TOTAL_POS - posBelow;
    const fn = posBelow;
    const tn = negBelow;
    const fp = TOTAL_NEG - negBelow;
    return {
      tp, fp, fn, tn,
      sensitivity: tp + fn ? tp / (tp + fn) : 0,
      specificity: tn + fp ? tn / (tn + fp) : 0,
      ppv: tp + fp ? tp / (tp + fp) : 0,
      npv: tn + fn ? tn / (tn + fn) : 0,
      accuracy: (tp + tn) / N,
    };
  }

  const SEX_PROBS = { 0: [], 1: [] };
  TEST.forEach(([p, , s]) => { SEX_PROBS[s].push(p); });
  SEX_PROBS[0].sort((a, b) => a - b);
  SEX_PROBS[1].sort((a, b) => a - b);

  /* ---------------- 数字补间 ---------------- */
  const tweens = new Map();
  function tween(node, to, fmt, dur = 320) {
    if (REDUCED) { node.textContent = fmt(to); return; }
    const from = Number(node.dataset.v ?? to);
    node.dataset.v = to;
    const old = tweens.get(node);
    if (old) cancelAnimationFrame(old);
    if (from === to) { node.textContent = fmt(to); return; }
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt(from + (to - from) * e);
      if (k < 1) tweens.set(node, requestAnimationFrame(step));
    };
    tweens.set(node, requestAnimationFrame(step));
  }

  /* ============================================================
     图表几何
     ============================================================ */
  const ROC_W = 320;
  const ROC_H = 320;
  const ROCPAD = { l: 46, r: 14, t: 16, b: 40 };
  const ROCW = ROC_W - ROCPAD.l - ROCPAD.r;
  const ROCH = ROC_H - ROCPAD.t - ROCPAD.b;
  const ROC_BOTTOM = ROCPAD.t + ROCH;

  const DIST = { w: 440, h: 208, l: 30, r: 10, t: 12, b: 30 };
  const COST = { w: 440, h: 208, l: 38, r: 12, t: 14, b: 30 };

  function rocPoints() {
    let tp = 0, fp = 0;
    const pts = [[0, 0]];
    const step = Math.max(1, Math.floor(N / 320));
    for (let i = 0; i < N; i++) {
      if (SORTED[i][1] === 1) tp++;
      else fp++;
      if ((i + 1) % step === 0) pts.push([fp / TOTAL_NEG, tp / TOTAL_POS]);
    }
    pts.push([1, 1]);
    return pts;
  }

  /* ============================================================
     仪表盘
     ============================================================ */
  function buildGauge() {
    const cx = 150, cy = 150, r = 116;
    const arc = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
    $("gauge").innerHTML = `
      <svg viewBox="0 0 300 168" aria-hidden="true">
        <defs>
          <linearGradient id="gaugeFill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stop-color="#12a8a0"/>
            <stop offset="0.52" stop-color="#d99a2b"/>
            <stop offset="1" stop-color="#d9485f"/>
          </linearGradient>
        </defs>
        <path d="${arc}" fill="none" stroke="#eef3f8" stroke-width="15" stroke-linecap="round"/>
        <path id="gaugeArc" class="gauge__arc" d="${arc}" pathLength="100" fill="none"
              stroke="url(#gaugeFill)" stroke-width="15" stroke-linecap="round"
              stroke-dasharray="0 100"/>
        <g id="gaugeKnob" class="gauge__knob">
          <circle cx="${cx}" cy="${cy - r}" r="7.5" fill="#fff" stroke="#0c1a28" stroke-width="2.5"/>
        </g>
      </svg>
      <div class="gauge__value" aria-live="polite">
        <span class="gauge__num"><span id="gaugeNum">0</span><small>%</small></span>
        <span class="gauge__cap">预测概率</span>
      </div>`;
  }

  function renderGauge(p) {
    $("gaugeArc").style.strokeDasharray = `${clamp(p, 0, 1) * 100} 100`;
    $("gaugeKnob").style.transform = `rotate(${-90 + clamp(p, 0, 1) * 180}deg)`;
    const shown = Math.round(p * 1000) / 10;
    tween($("gaugeNum"), shown, (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v))));
  }

  /* ============================================================
     ROC
     ============================================================ */
  function buildROC() {
    const pts = rocPoints();
    const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => {
      const x = ROCPAD.l + v * ROCW;
      const y = ROC_BOTTOM - v * ROCH;
      return `
        <line x1="${x}" y1="${ROCPAD.t}" x2="${x}" y2="${ROC_BOTTOM}" stroke="#eef3f8" stroke-width="1"/>
        <line x1="${ROCPAD.l}" y1="${y}" x2="${ROCPAD.l + ROCW}" y2="${y}" stroke="#eef3f8" stroke-width="1"/>`;
    }).join("");

    const labels = [0, 0.5, 1].map((v) => {
      const x = ROCPAD.l + v * ROCW;
      const y = ROC_BOTTOM - v * ROCH;
      return `
        <text x="${x}" y="${ROC_BOTTOM + 16}" text-anchor="middle" fill="#93a4b5" font-size="10">${v.toFixed(1)}</text>
        <text x="${ROCPAD.l - 9}" y="${y + 3.5}" text-anchor="end" fill="#93a4b5" font-size="10">${v.toFixed(1)}</text>`;
    }).join("");

    const mapped = pts.map(([f, t]) => [ROCPAD.l + f * ROCW, ROC_BOTTOM - t * ROCH]);
    const areaPath =
      `M ${ROCPAD.l} ${ROC_BOTTOM} ` +
      mapped.map(([x, y]) => `L${x.toFixed(2)} ${y.toFixed(2)}`).join(" ") +
      ` L ${ROCPAD.l + ROCW} ${ROC_BOTTOM} Z`;
    const linePath = mapped.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");

    $("rocChart").innerHTML = `
      <svg viewBox="0 0 ${ROC_W} ${ROC_H}" role="img" aria-label="ROC 曲线，当前阈值工作点">
        <defs>
          <linearGradient id="rocFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="rgba(18,168,160,0.20)"/>
            <stop offset="1" stop-color="rgba(18,168,160,0.01)"/>
          </linearGradient>
        </defs>
        ${grid}
        ${labels}
        <line x1="${ROCPAD.l}" y1="${ROC_BOTTOM}" x2="${ROCPAD.l + ROCW}" y2="${ROCPAD.t}"
              stroke="#c9d6e4" stroke-width="1" stroke-dasharray="4 5"/>
        <path d="${areaPath}" fill="url(#rocFill)"/>
        <path d="${linePath}" fill="none" stroke="#0f8d88" stroke-width="2.2"
              stroke-linejoin="round" stroke-linecap="round"/>
        <line id="rocCrossX" x1="0" y1="0" x2="0" y2="0" stroke="#0c1a28" stroke-width="1"
              stroke-dasharray="3 4" opacity="0.28"/>
        <line id="rocCrossY" x1="0" y1="0" x2="0" y2="0" stroke="#0c1a28" stroke-width="1"
              stroke-dasharray="3 4" opacity="0.28"/>
        <circle id="rocDot" cx="0" cy="0" r="6" fill="#0c1a28" stroke="#fff" stroke-width="2.5"/>
        <text x="${ROCPAD.l + ROCW / 2}" y="${ROC_H - 6}" text-anchor="middle" fill="#93a4b5"
              font-size="10">假阳性率（1 − 特异度）</text>
        <text x="12" y="${ROCPAD.t + ROCH / 2}" text-anchor="middle" fill="#93a4b5" font-size="10"
              transform="rotate(-90 12 ${ROCPAD.t + ROCH / 2})">真阳性率（敏感度）</text>
      </svg>`;
  }

  function renderROC(s) {
    const px = ROCPAD.l + (1 - s.specificity) * ROCW;
    const py = ROC_BOTTOM - s.sensitivity * ROCH;
    const dot = $("rocDot");
    dot.setAttribute("cx", px);
    dot.setAttribute("cy", py);
    const cx = $("rocCrossX");
    cx.setAttribute("x1", px);
    cx.setAttribute("x2", px);
    cx.setAttribute("y1", py);
    cx.setAttribute("y2", ROC_BOTTOM);
    const cy = $("rocCrossY");
    cy.setAttribute("x1", ROCPAD.l);
    cy.setAttribute("x2", px);
    cy.setAttribute("y1", py);
    cy.setAttribute("y2", py);
  }

  /* ============================================================
     混淆矩阵 + 指标
     ============================================================ */
  function buildMatrix() {
    $("matrix").innerHTML = `
      <div class="matrix__corner"></div>
      <div class="matrix__head">预测阳性</div>
      <div class="matrix__head">预测阴性</div>
      <div class="matrix__rowhead">实际<br>患病</div>
      <div class="matrix__cell matrix__cell--tp"><span class="n" id="mTp">0</span><span class="t">真阳性 TP</span></div>
      <div class="matrix__cell matrix__cell--fn"><span class="n" id="mFn">0</span><span class="t">假阴性 FN · 漏诊</span></div>
      <div class="matrix__rowhead">实际<br>未患病</div>
      <div class="matrix__cell matrix__cell--fp"><span class="n" id="mFp">0</span><span class="t">假阳性 FP · 误诊</span></div>
      <div class="matrix__cell matrix__cell--tn"><span class="n" id="mTn">0</span><span class="t">真阴性 TN</span></div>`;

    const metrics = [
      ["敏感度 · 召回", "mSens"],
      ["特异度", "mSpec"],
      ["PPV · 精确率", "mPpv"],
      ["NPV", "mNpv"],
      ["准确率", "mAcc"],
      ["漏诊率", "mMiss"],
    ];
    $("metrics").innerHTML = metrics
      .map(([k, id]) => `<div class="metric"><span class="metric__k">${k}</span><span class="metric__v" id="${id}">—</span></div>`)
      .join("");
  }

  function renderMatrix(s) {
    $("mTp").textContent = s.tp;
    $("mFn").textContent = s.fn;
    $("mFp").textContent = s.fp;
    $("mTn").textContent = s.tn;
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    $("mSens").textContent = pct(s.sensitivity);
    $("mSpec").textContent = pct(s.specificity);
    $("mPpv").textContent = pct(s.ppv);
    $("mNpv").textContent = pct(s.npv);
    $("mAcc").textContent = pct(s.accuracy);
    $("mMiss").textContent = pct(1 - s.sensitivity);
  }

  /* ============================================================
     风险分布
     ============================================================ */
  const BINS = 20;
  const distBins = Array.from({ length: BINS }, () => ({ neg: 0, pos: 0 }));
  TEST.forEach(([p, y]) => {
    const i = Math.min(BINS - 1, Math.floor(p * BINS));
    if (y === 1) distBins[i].pos++;
    else distBins[i].neg++;
  });
  const distMax = Math.max(...distBins.map((b) => b.neg + b.pos), 1);

  function buildDist() {
    const { w, h, l, r, t, b } = DIST;
    const pw = w - l - r;
    const ph = h - t - b;
    const bw = pw / BINS;
    const logTop = Math.max(1, Math.log10(distMax));
    const yv = (v) => t + ph - (Math.log10(Math.max(1, v)) / logTop) * ph;

    let bars = "";
    distBins.forEach((bin, i) => {
      const x0 = l + i * bw;
      const barW = Math.max(2, bw * 0.34);
      const xn = x0 + bw * 0.15;
      const xp = xn + barW + Math.max(1, bw * 0.05);
      bars += `<rect x="${xn.toFixed(1)}" y="${yv(bin.neg).toFixed(1)}" width="${barW.toFixed(1)}"
                 height="${(t + ph - yv(bin.neg)).toFixed(1)}" rx="1.5" fill="#a8c6d8"/>`;
      bars += `<rect x="${xp.toFixed(1)}" y="${yv(bin.pos).toFixed(1)}" width="${barW.toFixed(1)}"
                 height="${(t + ph - yv(bin.pos)).toFixed(1)}" rx="1.5" fill="#d9485f"/>`;
    });

    const ticks = [];
    for (let v = 1; v <= distMax; v *= 10) ticks.push(v);
    const tickLabel = (v) => (v >= 1000 ? `${v / 1000}k` : String(v));
    const gridLines = ticks.map((v) => `
      <line x1="${l}" y1="${yv(v).toFixed(1)}" x2="${l + pw}" y2="${yv(v).toFixed(1)}" stroke="#eef3f8"/>
      <text x="${l - 7}" y="${(yv(v) + 3.5).toFixed(1)}" text-anchor="end" fill="#a7b6c5" font-size="9.5">${tickLabel(v)}</text>`).join("");

    const xTicks = [0, 0.5, 1].map((v) => `
      <text x="${(l + v * pw).toFixed(1)}" y="${h - 9}" text-anchor="middle" fill="#a7b6c5" font-size="9.5">${v.toFixed(1)}</text>`).join("");

    $("distChart").innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="测试集预测概率分布，按真实结局分组，纵轴为对数刻度">
        ${gridLines}${bars}${xTicks}
        <line id="distThr" x1="0" y1="${t}" x2="0" y2="${t + ph}" stroke="#0c1a28" stroke-width="1.5"
              stroke-dasharray="4 4" opacity="0.55"/>
      </svg>`;
  }

  function renderDist(t) {
    const { l, t: top, w, r, h, b } = DIST;
    const x = l + t * (w - l - r);
    const line = $("distThr");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", top);
    line.setAttribute("y2", h - b);
  }

  /* ============================================================
     代价曲线
     ============================================================ */
  const THRESHOLDS = Array.from({ length: 91 }, (_, i) => (5 + i) / 100);

  function costPoints(ratio) {
    return THRESHOLDS.map((t) => {
      const s = statsAt(t);
      return [t, ((s.fn * ratio + s.fp) / N) * 100];
    });
  }

  function buildCost() {
    const { w, h, l, r, t, b } = COST;
    $("costChart").innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="不同阈值下每 100 名患者的总代价">
        <g id="costAxes"></g>
        <path id="costArea" fill="rgba(217,72,95,0.07)" d=""/>
        <path id="costLine" fill="none" stroke="#d9485f" stroke-width="2" stroke-linejoin="round"/>
        <line id="costThr" x1="${l}" y1="${t}" x2="${l}" y2="${h - b}" stroke="#0c1a28" stroke-width="1"
              stroke-dasharray="3 4" opacity="0.35"/>
        <circle id="costNowDot" r="5" fill="#fff" stroke="#d9485f" stroke-width="2.5"/>
        <circle id="costOptDot" r="6" fill="#0f8d88" stroke="#fff" stroke-width="2.5"/>
      </svg>`;
  }

  function renderCost(ratio, tNow, tOpt) {
    const { w, h, l, r, t, b } = COST;
    const pw = w - l - r;
    const ph = h - t - b;
    const pts = costPoints(ratio);
    const maxCost = Math.max(...pts.map((p) => p[1]), 1);
    const axisMax = Math.ceil(maxCost / 50) * 50 || 50;
    const x = (v) => l + ((v - 0.05) / 0.9) * pw;
    const y = (v) => t + ph - (v / axisMax) * ph;

    const line = pts.map(([th, c], i) => `${i ? "L" : "M"}${x(th).toFixed(1)} ${y(c).toFixed(1)}`).join(" ");
    $("costLine").setAttribute("d", line);
    $("costArea").setAttribute("d", `${line} L${x(0.95).toFixed(1)} ${t + ph} L${x(0.05).toFixed(1)} ${t + ph} Z`);

    const yTicks = [0, axisMax / 2, axisMax];
    $("costAxes").innerHTML =
      yTicks.map((v) => `<line x1="${l}" y1="${y(v).toFixed(1)}" x2="${l + pw}" y2="${y(v).toFixed(1)}" stroke="#eef3f8"/>
        <text x="${l - 7}" y="${(y(v) + 3.5).toFixed(1)}" text-anchor="end" fill="#a7b6c5" font-size="9.5">${v.toFixed(0)}</text>`).join("") +
      [0.05, 0.5, 0.95].map((v) => `<text x="${x(v).toFixed(1)}" y="${h - 9}" text-anchor="middle" fill="#a7b6c5" font-size="9.5">${v.toFixed(2)}</text>`).join("");

    const cx = x(tNow);
    $("costThr").setAttribute("x1", cx);
    $("costThr").setAttribute("x2", cx);

    const nowStats = statsAt(tNow);
    const nowCost = ((nowStats.fn * ratio + nowStats.fp) / N) * 100;
    const nowDot = $("costNowDot");
    nowDot.setAttribute("cx", cx);
    nowDot.setAttribute("cy", y(nowCost));

    const optStats = statsAt(tOpt);
    const optDot = $("costOptDot");
    optDot.setAttribute("cx", x(tOpt));
    optDot.setAttribute("cy", y(((optStats.fn * ratio + optStats.fp) / N) * 100));

    $("costNow").textContent = `当前阈值每 100 人总代价 ${nowCost.toFixed(1)}`;
    $("optThr").textContent = tOpt.toFixed(2);
  }

  /* ============================================================
     表单
     ============================================================ */
  const decimals = (f) => (f.step < 1 ? (String(f.step).split(".")[1] || "").length : 0);

  function fmtValue(f, v) {
    return decimals(f) ? Number(v).toFixed(decimals(f)) : String(Math.round(v));
  }

  function rangeHTML(f) {
    return `
      <div class="slider" data-key="${f.key}">
        <div class="slider__top">
          <label class="slider__label" for="in-${f.key}">${f.label}<span class="slider__unit">${f.unit || ""}</span></label>
          <span class="slider__value" id="val-${f.key}"></span>
        </div>
        <div class="slider__rail">
          <input type="range" id="in-${f.key}" min="${f.min}" max="${f.max}" step="${f.step}"
                 value="${f.default}" data-key="${f.key}">
          <output class="slider__bubble" for="in-${f.key}"></output>
        </div>
      </div>`;
  }

  function toggleHTML(f) {
    return `
      <div class="field-row">
        <div class="field-row__text">
          <span class="field-row__label">${f.label}</span>
          ${f.hint ? `<span class="field-row__hint">${f.hint}</span>` : ""}
        </div>
        <button type="button" class="switch" role="switch" aria-checked="false"
                aria-label="${f.label}" data-key="${f.key}"><span></span></button>
      </div>`;
  }

  function selectHTML(f) {
    return `
      <div class="field-row">
        <div class="field-row__text"><span class="field-row__label">${f.label}</span></div>
        <select class="select" id="in-${f.key}" data-key="${f.key}" aria-label="${f.label}">
          ${f.options.map((o) => `<option value="${o.value}">${o.label}</option>`).join("")}
        </select>
      </div>`;
  }

  function buildForm() {
    $("sliders").innerHTML = FEATURES
      .filter((f) => f.type !== "segment")
      .map((f) => (f.type === "range" ? rangeHTML(f) : f.type === "toggle" ? toggleHTML(f) : selectHTML(f)))
      .join("");

    FEATURES.forEach((f) => {
      if (f.type === "range") {
        const input = $(`in-${f.key}`);
        input.addEventListener("input", () => {
          state.values[f.key] = Number(input.value);
          state.preset = null;
          syncPresetUI();
          syncRange(f);
          render();
        });
        input.addEventListener("pointerdown", () => input.closest(".slider").classList.add("is-dragging"));
        input.addEventListener("pointerup", () => input.closest(".slider").classList.remove("is-dragging"));
        input.addEventListener("pointercancel", () => input.closest(".slider").classList.remove("is-dragging"));
        input.addEventListener("blur", () => input.closest(".slider").classList.remove("is-dragging"));
      } else if (f.type === "toggle") {
        const btn = $("sliders").querySelector(`.switch[data-key="${f.key}"]`);
        btn.addEventListener("click", () => {
          state.values[f.key] = state.values[f.key] ? 0 : 1;
          state.preset = null;
          syncPresetUI();
          syncToggle(f);
          render();
        });
      } else if (f.type === "select") {
        const sel = $(`in-${f.key}`);
        sel.addEventListener("change", () => {
          state.values[f.key] = sel.value;
          state.preset = null;
          syncPresetUI();
          render();
        });
      }
    });
  }

  function syncRange(f) {
    const input = $(`in-${f.key}`);
    const v = Number(state.values[f.key]);
    input.value = v;
    const pct = clamp(((v - f.min) / (f.max - f.min)) * 100, 0, 100);
    input.style.setProperty("--fill", `${pct}%`);
    const text = fmtValue(f, v);
    $(`val-${f.key}`).textContent = text;
    const row = input.closest(".slider");
    const bubble = row.querySelector(".slider__bubble");
    bubble.textContent = text;
    bubble.style.left = `${pct}%`;
  }

  function syncToggle(f) {
    const btn = $("sliders").querySelector(`.switch[data-key="${f.key}"]`);
    const on = !!Number(state.values[f.key]);
    btn.classList.toggle("is-on", on);
    btn.setAttribute("aria-checked", on ? "true" : "false");
  }

  function syncForm() {
    FEATURES.forEach((f) => {
      if (f.type === "range") syncRange(f);
      else if (f.type === "toggle") syncToggle(f);
      else if (f.type === "select") $(`in-${f.key}`).value = state.values[f.key];
    });

    document.querySelectorAll("#sexField .segment__opt").forEach((btn) => {
      const on = Number(btn.dataset.value) === Number(state.values.sex);
      btn.classList.toggle("is-active", on);
      btn.setAttribute("aria-checked", on ? "true" : "false");
    });

    const isMale = Number(state.values.sex) === 1;
    const s = isMale ? META.sexStats.male : META.sexStats.female;
    $("sexHint").textContent = `${isMale ? "男性" : "女性"} · 患病率 ${(s.prevalence * 100).toFixed(1)}%（${s.n.toLocaleString("zh-CN")} 人）`;
  }

  /* ============================================================
     示例病例
     ============================================================ */
  function presetKey() {
    return Number(state.values.sex) === 1 ? "male" : "female";
  }

  function applyPreset(tier) {
    const c = D.presets[presetKey()][tier];
    if (!c) return;
    Object.keys(state.values).forEach((k) => {
      if (k in c) state.values[k] = c[k];
    });
    state.preset = tier;
    syncForm();
    syncPresetUI();
    render();
  }

  function buildPresets() {
    const buttons = [
      { label: "低风险", tier: "low" },
      { label: "高风险", tier: "high" },
    ];
    $("presets").innerHTML = buttons
      .map((b) => `<button class="preset" type="button" data-tier="${b.tier}">${b.label}</button>`)
      .join("");

    $("presets").addEventListener("click", (e) => {
      const btn = e.target.closest(".preset");
      if (!btn) return;
      applyPreset(btn.dataset.tier);
    });
  }

  function syncPresetUI() {
    document.querySelectorAll(".preset").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.tier === state.preset);
    });
  }

  function buildSex() {
    $("sexField").addEventListener("click", (e) => {
      const btn = e.target.closest(".segment__opt");
      if (!btn) return;
      const next = Number(btn.dataset.value);
      if (next === Number(state.values.sex)) return;
      state.values.sex = next;
      if (state.preset) applyPreset(state.preset);
      else { syncForm(); render(); }
    });
  }

  /* ============================================================
     渲染
     ============================================================ */
  function render() {
    const { p } = predict();
    const s = statsAt(state.threshold);
    const opt = optThreshold();

    renderGauge(p);
    renderROC(s);
    renderMatrix(s);
    renderDist(state.threshold);
    renderCost(state.cost, state.threshold, opt);

    const band = p < 0.1 ? ["low", "低风险"] : p < 0.3 ? ["mid", "中等风险"] : ["high", "高风险"];
    const bandEl = $("band");
    bandEl.className = `band band--${band[0]}`;
    bandEl.textContent = band[1];

    const sex = Number(state.values.sex);
    const pool = SEX_PROBS[sex];
    const below = lowerBound(pool, p);
    $("percentile").innerHTML = `高于同性别测试人群 <b>${Math.round((below / pool.length) * 100)}%</b>`;

    const positive = p >= state.threshold;
    const verdict = $("verdict");
    verdict.className = `result__verdict ${positive ? "is-pos" : "is-neg"}`;
    verdict.textContent = `阈值 ${state.threshold.toFixed(2)}：判为${positive ? "阳性 → 建议进一步检查" : "阴性 → 暂不提示"}`;

    const truth = $("truth");
    if (state.preset) {
      const c = D.presets[presetKey()][state.preset];
      truth.hidden = false;
      truth.textContent = `示例病例的真实结局：${c._outcome === 1 ? "患糖尿病" : "未患糖尿病"}`;
    } else {
      truth.hidden = true;
    }

    renderContrib();
  }

  function renderContrib() {
    const list = FEATURES.map((f) => ({
      key: f.key,
      label: f.label,
      value: contributionOf(f, state.values[f.key]),
    }));
    const maxAbs = Math.max(...list.map((c) => Math.abs(c.value)), 0.35);
    const hasNeg = list.some((c) => c.value < -0.02);
    const hasPos = list.some((c) => c.value > 0.02);
    const base = hasNeg && hasPos ? 50 : hasPos ? 0 : 100;
    const container = $("contribChart");
    container.style.setProperty("--base", `${base}%`);

    if (!container.children.length) {
      container.innerHTML = list.map((c) => `
        <div class="contrib-row">
          <span class="contrib-row__label" title="${c.label}">${c.label}</span>
          <div class="contrib-row__track">
            <span class="contrib-row__bar" data-key="${c.key}"></span>
          </div>
          <span class="contrib-row__val" data-key="${c.key}"></span>
        </div>`).join("");
    }

    list.forEach((c) => {
      const bar = container.querySelector(`.contrib-row__bar[data-key="${c.key}"]`);
      const val = container.querySelector(`.contrib-row__val[data-key="${c.key}"]`);
      const span = c.value >= 0 ? 100 - base : base;
      const ratio = Math.min(Math.abs(c.value) / maxAbs, 1) * span;
      bar.className = `contrib-row__bar ${c.value >= 0 ? "pos" : "neg"}`;
      bar.style.width = `${ratio}%`;
      val.textContent = `${c.value >= 0 ? "+" : "−"}${Math.abs(c.value).toFixed(2)}`;
      val.className = `contrib-row__val ${c.value >= 0 ? "" : "neg"}`;
    });

    const sum = list.reduce((acc, c) => acc + c.value, 0);
    const eta = B0 + sum;
    const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
    $("logitBase").textContent = signed(B0);
    $("logitSum").textContent = signed(sum);
    $("logitTotal").textContent = signed(eta);
    $("logitProb").textContent = `${(sigmoid(eta) * 100).toFixed(1)}%`;
  }

  function optThreshold() {
    let best = THRESHOLDS[0];
    let bestCost = Infinity;
    costPoints(state.cost).forEach(([t, c]) => {
      if (c < bestCost - 1e-9) { bestCost = c; best = t; }
    });
    return best;
  }

  /* ============================================================
     阈值与代价控件
     ============================================================ */
  function buildControls() {
    const thr = $("thr");
    const cost = $("cost");

    thr.addEventListener("input", () => {
      state.threshold = Number(thr.value) / 100;
      $("thrVal").textContent = state.threshold.toFixed(2);
      thr.style.setProperty("--fill", `${((state.threshold - 0.05) / 0.9) * 100}%`);
      render();
    });

    cost.addEventListener("input", () => {
      state.cost = Number(cost.value);
      $("costVal").textContent = state.cost;
      cost.style.setProperty("--fill", `${((state.cost - 1) / 19) * 100}%`);
      render();
    });

    $("recommendBtn").addEventListener("click", () => {
      state.threshold = Math.round(optThreshold() * 100) / 100;
      thr.value = Math.round(state.threshold * 100);
      $("thrVal").textContent = state.threshold.toFixed(2);
      thr.style.setProperty("--fill", `${((state.threshold - 0.05) / 0.9) * 100}%`);
      render();
    });

    $("resetBtn").addEventListener("click", () => {
      FEATURES.forEach((f) => { state.values[f.key] = f.default; });
      state.preset = null;
      syncForm();
      syncPresetUI();
      render();
    });

    thr.style.setProperty("--fill", "50%");
    cost.style.setProperty("--fill", `${((state.cost - 1) / 19) * 100}%`);
  }

  /* ============================================================
     初始化
     ============================================================ */
  function init() {
    $("aucVal").textContent = META.auc.toFixed(3);
    buildGauge();
    buildROC();
    buildMatrix();
    buildDist();
    buildCost();
    buildForm();
    buildPresets();
    buildSex();
    buildControls();
    syncForm();
    syncPresetUI();
    render();
  }

  init();
})();
