/* ============================================================
   糖尿病风险 · 决策阈值实验室
   纯原生 JS：无框架、无依赖、无网络请求。
   ============================================================ */
(() => {
  "use strict";

  const D = window.DIABETES_MODEL;
  const FEATURES = D.features;
  const B0 = D.model.intercept;
  const COEF = D.model.coef;
  const TEST = D.testScores;
  const N_TEST = TEST.length;
  const IMPUTED = new Set(D.meta.imputedFeatures);
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);

  /* ---------------- 状态 ---------------- */
  const state = {
    values: Object.fromEntries(FEATURES.map((f) => [f.key, f.median])),
    threshold: 0.5,
    cost: 5,
    preset: -1,
  };

  /* ---------------- 数学 ---------------- */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const sigmoid = (z) => 1 / (1 + Math.exp(-clamp(z, -35, 35)));
  const medianOf = (key) => FEATURES.find((f) => f.key === key).median;

  function imputedValue(key, raw) {
    return IMPUTED.has(key) && raw === 0 ? medianOf(key) : raw;
  }

  function predict(values) {
    let eta = B0;
    FEATURES.forEach((f, i) => {
      eta += COEF[i] * ((imputedValue(f.key, values[f.key]) - f.mean) / f.sd);
    });
    return { p: sigmoid(eta), eta };
  }

  function contributions(values) {
    return FEATURES.map((f, i) => ({
      key: f.key,
      label: f.label,
      value: COEF[i] * ((imputedValue(f.key, values[f.key]) - f.mean) / f.sd),
    }));
  }

  function statsAt(t) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (const [p, y] of TEST) {
      const hit = p >= t;
      if (y === 1) { if (hit) tp++; else fn++; }
      else { if (hit) fp++; else tn++; }
    }
    return {
      tp, fp, fn, tn,
      sensitivity: tp + fn ? tp / (tp + fn) : 0,
      specificity: tn + fp ? tn / (tn + fp) : 0,
      ppv: tp + fp ? tp / (tp + fp) : 0,
      npv: tn + fn ? tn / (tn + fn) : 0,
      accuracy: (tp + tn) / N_TEST,
    };
  }

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

  function rocPath() {
    return D.roc
      .map(([f, t], i) => `${i ? "L" : "M"}${(ROCPAD.l + f * ROCW).toFixed(2)} ${(ROC_BOTTOM - t * ROCH).toFixed(2)}`)
      .join(" ");
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
    tween($("gaugeNum"), Math.round(p * 100), (v) => String(Math.round(v)));
  }

  /* ============================================================
     ROC
     ============================================================ */
  function buildROC() {
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

    const pts = D.roc.map(([f, t]) => [ROCPAD.l + f * ROCW, ROC_BOTTOM - t * ROCH]);
    const areaPath =
      `M ${ROCPAD.l} ${ROC_BOTTOM} ` +
      pts.map(([x, y]) => `L${x.toFixed(2)} ${y.toFixed(2)}`).join(" ") +
      ` L ${ROCPAD.l + ROCW} ${ROC_BOTTOM} Z`;

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
        <path d="${rocPath()}" fill="none" stroke="#0f8d88" stroke-width="2.2"
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
    const yv = (v) => t + ph - (v / distMax) * ph;

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

    const ticks = [0, Math.round(distMax / 2), distMax];
    const gridLines = ticks.map((v) => `
      <line x1="${l}" y1="${yv(v).toFixed(1)}" x2="${l + pw}" y2="${yv(v).toFixed(1)}" stroke="#eef3f8"/>
      <text x="${l - 7}" y="${(yv(v) + 3.5).toFixed(1)}" text-anchor="end" fill="#a7b6c5" font-size="9.5">${v}</text>`).join("");

    const xTicks = [0, 0.5, 1].map((v) => `
      <text x="${(l + v * pw).toFixed(1)}" y="${h - 9}" text-anchor="middle" fill="#a7b6c5" font-size="9.5">${v.toFixed(1)}</text>`).join("");

    $("distChart").innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="测试集预测概率分布，按真实结局分组">
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
      return [t, ((s.fn * ratio + s.fp) / N_TEST) * 100];
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
    const nowCost = ((nowStats.fn * ratio + nowStats.fp) / N_TEST) * 100;
    const nowDot = $("costNowDot");
    nowDot.setAttribute("cx", cx);
    nowDot.setAttribute("cy", y(nowCost));

    const optStats = statsAt(tOpt);
    const optDot = $("costOptDot");
    optDot.setAttribute("cx", x(tOpt));
    optDot.setAttribute("cy", y(((optStats.fn * ratio + optStats.fp) / N_TEST) * 100));

    $("costNow").textContent = `当前阈值每 100 人总代价 ${nowCost.toFixed(1)}`;
    $("optThr").textContent = tOpt.toFixed(2);
  }

  /* ============================================================
     指标滑杆
     ============================================================ */
  const decimals = (f) => (f.step < 1 ? (String(f.step).split(".")[1] || "").length : 0);

  function fmtValue(f, v) {
    return decimals(f) ? Number(v).toFixed(decimals(f)) : String(Math.round(v));
  }

  function buildSliders() {
    $("sliders").innerHTML = FEATURES.map((f) => `
      <div class="slider">
        <div class="slider__top">
          <label class="slider__label" for="in-${f.key}">${f.label}<span class="slider__unit">${f.unit || ""}</span></label>
          <span class="slider__value" id="val-${f.key}"></span>
        </div>
        <input type="range" id="in-${f.key}" min="${f.min}" max="${f.max}" step="${f.step}"
               value="${f.median}" data-key="${f.key}">
      </div>`).join("");

    FEATURES.forEach((f) => {
      const input = $(`in-${f.key}`);
      input.addEventListener("input", () => {
        state.values[f.key] = Number(input.value);
        state.preset = -1;
        syncPresetUI();
        render();
      });
    });
  }

  function syncSliders() {
    FEATURES.forEach((f) => {
      const input = $(`in-${f.key}`);
      const v = state.values[f.key];
      input.value = v;
      const fill = ((v - f.min) / (f.max - f.min)) * 100;
      input.style.setProperty("--fill", `${clamp(fill, 0, 100)}%`);
      const badge = $(`val-${f.key}`);
      badge.textContent = fmtValue(f, v);
      const isMissing = IMPUTED.has(f.key) && Number(v) === 0;
      badge.classList.toggle("is-zero", isMissing);
      badge.title = isMissing ? "将按中位数插补" : "";
    });
  }

  function syncPresetUI() {
    document.querySelectorAll(".preset").forEach((b) => {
      b.classList.toggle("is-active", Number(b.dataset.index) === state.preset);
    });
  }

  /* ============================================================
     渲染主流程
     ============================================================ */
  function render() {
    const { p } = predict(state.values);
    const s = statsAt(state.threshold);
    const opt = optThreshold();

    renderGauge(p);
    renderROC(s);
    renderMatrix(s);
    renderDist(state.threshold);
    renderCost(state.cost, state.threshold, opt);

    const band = p < 0.3 ? ["low", "低风险"] : p < 0.6 ? ["mid", "中等风险"] : ["high", "高风险"];
    const bandEl = $("band");
    bandEl.className = `band band--${band[0]}`;
    bandEl.textContent = band[1];

    const below = TEST.filter(([q]) => q < p).length;
    $("percentile").innerHTML = `高于测试集 <b>${Math.round((below / N_TEST) * 100)}%</b> 的样本`;

    const positive = p >= state.threshold;
    const verdict = $("verdict");
    verdict.className = `result__verdict ${positive ? "is-pos" : "is-neg"}`;
    verdict.textContent = `阈值 ${state.threshold.toFixed(2)}：判为${positive ? "阳性 → 建议进一步检查" : "阴性 → 暂不提示"}`;

    const truth = $("truth");
    if (state.preset >= 0) {
      const c = D.presets[state.preset];
      truth.hidden = false;
      truth.textContent = `示例病例的真实结局：${c._outcome === 1 ? "患糖尿病" : "未患糖尿病"}`;
    } else {
      truth.hidden = true;
    }

    renderContrib();
  }

  function renderContrib() {
    const list = contributions(state.values);
    const maxAbs = Math.max(...list.map((c) => Math.abs(c.value)), 0.35);
    const container = $("contribChart");

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
      const ratio = Math.min(Math.abs(c.value) / maxAbs, 1) * 50;
      bar.className = `contrib-row__bar ${c.value >= 0 ? "pos" : "neg"}`;
      bar.style.width = `${ratio}%`;
      val.textContent = `${c.value >= 0 ? "+" : "−"}${Math.abs(c.value).toFixed(2)}`;
      val.className = `contrib-row__val ${c.value >= 0 ? "" : "neg"}`;
    });

    const sum = list.reduce((s, c) => s + c.value, 0);
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
      FEATURES.forEach((f) => { state.values[f.key] = f.median; });
      state.preset = -1;
      syncSliders();
      syncPresetUI();
      render();
    });

    thr.style.setProperty("--fill", "50%");
    cost.style.setProperty("--fill", `${((state.cost - 1) / 19) * 100}%`);
  }

  /* ============================================================
     示例病例
     ============================================================ */
  function buildPresets() {
    const sorted = D.presets
      .map((c, i) => ({ c, i, p: c._probability }))
      .sort((a, b) => a.p - b.p);
    const low = sorted[0].i;
    const high = sorted[sorted.length - 1].i;
    const mid = sorted.reduce((best, cur) => (Math.abs(cur.p - 0.5) < Math.abs(best.p - 0.5) ? cur : best), sorted[0]).i;

    const buttons = [
      { label: "低风险", index: low },
      { label: "临界", index: mid },
      { label: "高风险", index: high },
      { label: "换一例", index: -2 },
    ];

    $("presets").innerHTML = buttons
      .map((b) => `<button class="preset" type="button" data-index="${b.index}">${b.label}</button>`)
      .join("");

    $("presets").addEventListener("click", (e) => {
      const btn = e.target.closest(".preset");
      if (!btn) return;
      let idx = Number(btn.dataset.index);
      if (idx === -2) {
        const pool = D.presets.map((_, i) => i).filter((i) => i !== state.preset);
        idx = pool[Math.floor(Math.random() * pool.length)];
      }
      if (!D.presets[idx]) return;
      state.preset = idx;
      Object.keys(state.values).forEach((k) => { state.values[k] = D.presets[idx][k]; });
      syncSliders();
      syncPresetUI();
      render();
    });
  }

  /* ============================================================
     初始化
     ============================================================ */
  function init() {
    $("aucVal").textContent = D.meta.auc.toFixed(3);
    buildGauge();
    buildROC();
    buildMatrix();
    buildDist();
    buildCost();
    buildSliders();
    buildPresets();
    buildControls();
    syncSliders();
    render();
  }

  init();
})();
