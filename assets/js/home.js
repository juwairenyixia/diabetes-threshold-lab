/* ============================================================
   首页 · 3D 球形树状图
   纯 Canvas 2D 手写透视投影 + 深度排序，零依赖。
   ============================================================ */
(() => {
  "use strict";

  const canvas = document.getElementById("orbCanvas");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const tip = document.getElementById("orbTip");
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- 因素数据 ---------------- */
  const CATEGORIES = [
    { key: "metabolic", label: "代谢指标", color: "#45e0cf" },
    { key: "lifestyle", label: "生活方式", color: "#6fa8ff" },
    { key: "genetic", label: "人群与遗传", color: "#f2c14e" },
    { key: "comorbid", label: "合并疾病", color: "#ff7a90" },
  ];

  const FACTORS = [
    // 代谢指标
    { name: "糖化血红蛋白", cat: "metabolic", note: "反映近 8–12 周平均血糖，模型中最强的单项预测因子。" },
    { name: "空腹血糖", cat: "metabolic", note: "血糖升高是糖代谢异常的直观信号，权重仅次于 HbA1c。" },
    { name: "体质指数 BMI", cat: "metabolic", note: "脂肪堆积加重胰岛素抵抗，BMI 越高风险越高。" },
    { name: "腹型肥胖", cat: "metabolic", note: "内脏脂肪比皮下脂肪更具代谢活性，与胰岛素抵抗关系密切。" },
    { name: "胰岛素抵抗", cat: "metabolic", note: "细胞对胰岛素反应减弱，是 2 型糖尿病的核心机制。" },
    // 生活方式
    { name: "吸烟", cat: "lifestyle", note: "吸烟损伤血管内皮，与胰岛素抵抗和并发症风险相关。" },
    { name: "缺乏运动", cat: "lifestyle", note: "肌肉活动减少，葡萄糖摄取下降，餐后血糖更易升高。" },
    { name: "高糖高脂饮食", cat: "lifestyle", note: "长期能量过剩推动体重增加与血糖波动。" },
    { name: "睡眠不足", cat: "lifestyle", note: "睡眠剥夺会降低胰岛素敏感性，并影响食欲调节激素。" },
    { name: "长期压力", cat: "lifestyle", note: "皮质醇长期升高会拮抗胰岛素作用，推高空腹血糖。" },
    { name: "过量饮酒", cat: "lifestyle", note: "酒精干扰肝脏糖异生，并可能带来额外热量摄入。" },
    // 人群与遗传
    { name: "年龄增长", cat: "genetic", note: "胰岛 β 细胞功能随年龄下降，风险在 45 岁后明显上升。" },
    { name: "家族史", cat: "genetic", note: "一级亲属患病会显著提高个体遗传易感性。" },
    { name: "遗传易感基因", cat: "genetic", note: "多基因累积效应影响胰岛素分泌与作用效率。" },
    { name: "妊娠糖尿病史", cat: "genetic", note: "曾患妊娠糖尿病者，产后发生 2 型糖尿病的风险明显升高。" },
    // 合并疾病
    { name: "高血压", cat: "comorbid", note: "常与胰岛素抵抗并存，共同加速血管损伤。" },
    { name: "冠心病", cat: "comorbid", note: "糖代谢异常与动脉粥样硬化互为因果。" },
    { name: "血脂异常", cat: "comorbid", note: "甘油三酯升高、HDL 降低是代谢综合征的组成部分。" },
    { name: "多囊卵巢综合征", cat: "comorbid", note: "伴随排卵障碍与高雄激素，常伴胰岛素抵抗。" },
    { name: "代谢综合征", cat: "comorbid", note: "血糖、血压、血脂、腰围多项异常聚集出现。" },
  ];

  const colorOf = Object.fromEntries(CATEGORIES.map((c) => [c.key, c.color]));

  /* ---------------- 三维结构：球面均匀分布 ---------------- */
  const R_ROOT = 0;
  const R_CAT = 76;
  const R_NODE = 168;
  const R_RING = 172;

  const GOLDEN = Math.PI * (3 - Math.sqrt(5));

  function spherePoint(i, total, radius) {
    const y = 1 - (i / Math.max(1, total - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = GOLDEN * i;
    return { x: Math.cos(theta) * r * radius, y: y * radius, z: Math.sin(theta) * r * radius };
  }

  // 类别节点：4 个方向
  const catNodes = CATEGORIES.map((c, i) => {
    const v = spherePoint(i, CATEGORIES.length, R_CAT);
    return { ...c, ...v, kind: "cat" };
  });

  // 因素节点：在球面上均匀撒开，并靠近所属类别，形成"树枝"感
  const factorNodes = [];
  {
    const n = FACTORS.length;
    const order = [];
    CATEGORIES.forEach((c) => {
      FACTORS.filter((f) => f.cat === c.key).forEach((f) => order.push(f));
    });
    order.forEach((f, i) => {
      const p = spherePoint(i, n, R_NODE);
      const cat = catNodes.find((c) => c.key === f.cat);
      const k = 0.22;
      const x = p.x + cat.x * k;
      const y = p.y + cat.y * k;
      const z = p.z + cat.z * k;
      const len = Math.hypot(x, y, z) || 1;
      factorNodes.push({
        ...f,
        x: (x / len) * R_NODE,
        y: (y / len) * R_NODE,
        z: (z / len) * R_NODE,
        kind: "factor",
      });
    });
  }

  /* ---------------- 相机与交互 ---------------- */
  const state = {
    rotX: -0.22,
    rotY: 0.6,
    spin: REDUCED ? 0 : 0.0016,
    scale: 1,
    w: 0,
    h: 0,
    dpr: 1,
    hover: null,
    dragging: false,
    lastX: 0,
    lastY: 0,
    moved: 0,
    velY: 0,
    velX: 0,
  };

  function resize() {
    const rect = canvas.getBoundingClientRect();
    state.dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.w = rect.width;
    state.h = rect.height;
    canvas.width = Math.round(rect.width * state.dpr);
    canvas.height = Math.round(rect.height * state.dpr);
    // 窄屏只显示类别标签，避免外圈文字溢出
    state.compact = rect.width < 620;
    const factor = state.compact ? 0.74 : 0.78;
    const widthFactor = state.compact ? 0.62 : 0.7;
    // 球体占高度的 factor，其余留给外圈标签与图例
    state.scale = (Math.min(rect.height * factor, rect.width * widthFactor) / 2) / R_RING;
  }

  function rotate(p) {
    const { rotX, rotY } = state;
    const cy = Math.cos(rotY), sy = Math.sin(rotY);
    const cx = Math.cos(rotX), sx = Math.sin(rotX);
    const x1 = p.x * cy - p.z * sy;
    const z1 = p.x * sy + p.z * cy;
    const y1 = p.y * cx - z1 * sx;
    const z2 = p.y * sx + z1 * cx;
    return { x: x1, y: y1, z: z2 };
  }

  // 透视：near 平面把 z 映射成缩放
  const PERSPECTIVE = 460;

  function project(p) {
    const r = rotate(p);
    const depth = PERSPECTIVE / (PERSPECTIVE + r.z);
    return {
      x: state.w / 2 + r.x * state.scale * depth,
      y: state.h / 2 - r.y * state.scale * depth * 1.02,
      z: r.z,
      s: depth,
      depth,
    };
  }
  /* ---------------- 绘制 ---------------- */
  const nodes = [...factorNodes, ...catNodes].map((n) => ({ ...n }));

  function drawBackground(t) {
    const { w, h } = state;
    // 球体内部微光
    const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.min(w, h) * 0.52);
    g.addColorStop(0, "rgba(69, 224, 207, 0.10)");
    g.addColorStop(0.45, "rgba(43, 108, 240, 0.05)");
    g.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    // 经纬网格（球笼感）
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.strokeStyle = "rgba(120, 190, 220, 0.075)";
    ctx.lineWidth = 1;
    const R = R_RING * state.scale * 1.0;
    for (let i = -2; i <= 2; i++) {
      const yy = (i / 3) * R;
      const rr = Math.sqrt(Math.max(0, R * R - yy * yy));
      ctx.beginPath();
      ctx.ellipse(0, -yy, rr, rr * 0.28, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI;
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.min(R * Math.abs(Math.cos(a)), R), R, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawEdge(a, b, color, alpha, width) {
    const pa = project(a);
    const pb = project(b);
    const depthAlpha = Math.min(1, Math.max(0.15, (pa.depth - 0.55) / 0.6));
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha * depthAlpha;
    ctx.lineWidth = width * ((pa.depth + pb.depth) / 2);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    // 轻微弯曲，形成树枝感
    const mx = (pa.x + pb.x) / 2;
    const my = (pa.y + pb.y) / 2;
    const nx = -(pb.y - pa.y);
    const ny = pb.x - pa.x;
    const len = Math.hypot(nx, ny) || 1;
    const bow = 0.12;
    ctx.quadraticCurveTo(mx + (nx / len) * len * bow * 0.3, my + (ny / len) * len * bow * 0.3, pb.x, pb.y);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawNode(n, t) {
    const p = project(n);
    const isCat = n.kind === "cat";
    const isHover = state.hover === n;
    const baseR = isCat ? 9.5 : 5.6;
    const r = baseR * p.depth * (isHover ? 1.45 : 1);
    const color = isCat ? "#8ee9dd" : colorOf[n.cat];

    // 外发光
    const glowR = r * (isCat ? 5 : 4.2);
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR);
    g.addColorStop(0, hexA(color, isHover ? 0.55 : 0.34));
    g.addColorStop(0.45, hexA(color, isHover ? 0.18 : 0.1));
    g.addColorStop(1, hexA(color, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
    ctx.fill();

    // 球体
    const body = ctx.createRadialGradient(
      p.x - r * 0.35, p.y - r * 0.4, r * 0.12,
      p.x, p.y, r
    );
    body.addColorStop(0, "#ffffff");
    body.addColorStop(0.35, color);
    body.addColorStop(1, shade(color, -0.42));
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();

    if (isCat) {
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 3.2, 0, Math.PI * 2);
      ctx.stroke();
    }
    return p;
  }

  function drawLabels(t) {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // 中心球
    const c = project({ x: 0, y: 0, z: 0 });
    const cr = 26 * c.depth;
    const g = ctx.createRadialGradient(c.x - 7, c.y - 9, 2, c.x, c.y, cr);
    g.addColorStop(0, "#ffffff");
    g.addColorStop(0.5, "#8ee9dd");
    g.addColorStop(1, "#12a8a0");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(c.x, c.y, cr, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.font = "600 12px var(--sans)";
    ctx.fillStyle = "#08202b";
    ctx.fillText("糖尿病", c.x, c.y + 0.5);

    // 类别标签：推到球体轮廓之外
    const placed = [{ x: c.x - cr, y: c.y - cr, w: cr * 2, h: cr * 2 }];
    const overlaps = (b) =>
      placed.some((p) => b.x < p.x + p.w && b.x + b.w > p.x && b.y < p.y + p.h && b.y + b.h > p.y);

    const R_PROJ = R_RING * state.scale;
    const margin = 6;
    const inBounds = (b) =>
      b.x >= margin && b.y >= margin &&
      b.x + b.w <= state.w - margin && b.y + b.h <= state.h - margin;

    // 候选标签先入框检查，再落盘
    const place = (label, anchor, ux, uy, font, color, factor, leader) => {
      ctx.font = font;
      const w = ctx.measureText(label).width;
      const lx = c.x + ux * R_PROJ * factor;
      const ly = c.y + uy * R_PROJ * factor;
      const box = { x: lx - w / 2, y: ly - 9, w, h: 18 };
      if (!inBounds(box) || overlaps(box)) return false;
      placed.push(box);
      if (leader) {
        ctx.strokeStyle = leader;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(anchor.x + ux * 8, anchor.y + uy * 8);
        ctx.lineTo(lx - ux * (w / 2 + 3), ly - uy * 5);
        ctx.stroke();
      }
      ctx.fillStyle = color;
      ctx.fillText(label, lx, ly);
      return true;
    };

    // 类别标签优先占用最外圈
    const catLabels = CATEGORIES
      .map((cat) => {
        const n = catNodes.find((x) => x.key === cat.key);
        const p = project(n);
        const dx = p.x - c.x;
        const dy = p.y - c.y;
        const len = Math.hypot(dx, dy) || 1;
        return { cat, p, ux: dx / len, uy: dy / len };
      })
      .filter((o) => o.p.depth >= 0.6)
      .sort((a, b) => b.p.depth - a.p.depth);

    catLabels.forEach(({ cat, p, ux, uy }) => {
      ctx.globalAlpha = Math.min(1, (p.depth - 0.55) * 2.6);
      if (state.compact) {
        // 窄屏：放在节点外侧，但不与节点自身重叠
        const pad = 12;
        ctx.font = "600 11.5px var(--sans)";
        const w = ctx.measureText(cat.label).width;
        const lx = p.x + ux * (pad + w / 2);
        const ly = p.y + uy * (pad + 6);
        const box = { x: lx - w / 2, y: ly - 9, w, h: 18 };
        if (inBounds(box) && !overlaps(box)) {
          placed.push(box);
          ctx.fillStyle = "#eaf3fa";
          ctx.fillText(cat.label, lx, ly);
        }
      } else {
        place(cat.label, p, ux, uy, "600 12.5px var(--sans)", "#eaf3fa",
          1.08, hexA(colorOf[cat.key], 0.38));
      }
      ctx.globalAlpha = 1;
    });

    // 因素标签：窄屏隐藏，宽屏推到更外圈，被占用则退回内一档
    if (state.compact) return;

    const factorLabels = factorNodes
      .map((n) => {
        const p = project(n);
        const dx = p.x - c.x;
        const dy = p.y - c.y;
        const len = Math.hypot(dx, dy) || 1;
        return { n, p, ux: dx / len, uy: dy / len };
      })
      .filter((o) => o.p.depth > 0.9)
      .sort((a, b) => b.p.depth - a.p.depth);

    factorLabels.forEach(({ n, p, ux, uy }) => {
      const ok = place(n.name, p, ux, uy, "500 11.5px var(--sans)", "#c3d6e6",
        1.24, hexA(colorOf[n.cat], 0.4));
      if (!ok) {
        place(n.name, p, ux, uy, "500 11.5px var(--sans)", "#a9bed0",
          1.4, hexA(colorOf[n.cat], 0.24));
      }
    });
  }

  function render(time) {
    const { w, h, dpr } = state;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drawBackground(time);

    // 边：中心→类别，类别→因素
    catNodes.forEach((c) => drawEdge({ x: 0, y: 0, z: 0 }, c, "rgba(142, 233, 221, 0.85)", 0.5, 2.6));
    factorNodes.forEach((f) => {
      const cat = catNodes.find((c) => c.key === f.cat);
      drawEdge(cat, f, hexA(colorOf[f.cat], 0.9), 0.42, 1.5);
    });

    // 节点按深度排序，远的先画
    const ordered = nodes
      .map((n) => ({ n, p: project(n) }))
      .sort((a, b) => a.p.z - b.p.z);
    ordered.forEach(({ n }) => drawNode(n, time));

    drawLabels(time);

    // 更新动画
    if (!state.dragging) {
      state.rotY += state.spin + state.velY;
      state.rotX += state.velX;
      state.velY *= 0.94;
      state.velX *= 0.94;
      state.rotX = Math.max(-0.9, Math.min(0.9, state.rotX));
    }
    requestAnimationFrame(render);
  }

  /* ---------------- 颜色工具 ---------------- */
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.max(0, Math.min(255, Math.round(v + (amt > 0 ? (255 - v) * amt : v * amt))));
    return `rgb(${f((n >> 16) & 255)}, ${f((n >> 8) & 255)}, ${f(n & 255)})`;
  }

  /* ---------------- 交互 ---------------- */
  function pickAt(cx, cy) {
    let best = null;
    nodes.forEach((n) => {
      const p = project(n);
      const rr = (n.kind === "cat" ? 13 : 10) * p.depth + 5;
      const d = Math.hypot(p.x - cx, p.y - cy);
      if (d < rr && (!best || p.z < best.z)) best = { n, p, z: p.z };
    });
    return best;
  }

  function showTip(hit, cx, cy) {
    if (!hit) {
      tip.hidden = true;
      state.hover = null;
      return;
    }
    const { n } = hit;
    if (n.kind === "cat") {
      const count = FACTORS.filter((f) => f.cat === n.key).length;
      document.getElementById("tipCat").textContent = "类别";
      document.getElementById("tipName").textContent = n.label;
      document.getElementById("tipNote").textContent = `共 ${count} 个相关因素，悬停其他节点查看细节。`;
    } else {
      document.getElementById("tipCat").textContent =
        CATEGORIES.find((c) => c.key === n.cat).label;
      document.getElementById("tipName").textContent = n.name;
      document.getElementById("tipNote").textContent = n.note;
    }
    const rect = canvas.getBoundingClientRect();
    tip.style.left = `${cx - rect.left}px`;
    tip.style.top = `${cy - rect.top}px`;
    tip.hidden = false;
    state.hover = n;
  }

  const localPoint = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  canvas.addEventListener("pointerdown", (e) => {
    state.dragging = true;
    state.moved = 0;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    canvas.classList.add("is-dragging");
    canvas.setPointerCapture?.(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (state.dragging) {
      const dx = e.clientX - state.lastX;
      const dy = e.clientY - state.lastY;
      state.moved += Math.abs(dx) + Math.abs(dy);
      state.rotY += dx * 0.006;
      state.rotX = Math.max(-0.9, Math.min(0.9, state.rotX - dy * 0.005));
      state.velY = dx * 0.0016;
      state.velX = -dy * 0.0012;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      tip.hidden = true;
      return;
    }
    const p = localPoint(e);
    const hit = pickAt(p.x, p.y);
    showTip(hit, e.clientX, e.clientY);
  });

  const endDrag = () => {
    state.dragging = false;
    canvas.classList.remove("is-dragging");
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    endDrag();
    tip.hidden = true;
    state.hover = null;
  });

  canvas.addEventListener("touchstart", () => { state.spin = 0; }, { passive: true });

  /* ---------------- 图例 ---------------- */
  const legend = document.getElementById("orbLegend");
  if (legend) {
    legend.innerHTML = CATEGORIES.map(
      (c) => `<li><i style="background:${c.color};color:${c.color}"></i>${c.label}</li>`
    ).join("");
  }

  /* ---------------- 启动 ---------------- */
  let ro = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(resize);
    ro.observe(canvas);
  }
  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(render);

  document.getElementById("orbHint").textContent =
    REDUCED ? "数据已按类别分组" : "拖动可旋转 · 悬停查看说明";
})();
