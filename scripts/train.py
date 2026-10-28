"""
训练 Pima Indians Diabetes 逻辑回归模型，导出前端可直接使用的模型资产。

不依赖 numpy / sklearn：纯标准库实现 IRLS（牛顿法）求解逻辑回归，
保证在任何只装了 Python 的机器上都能复现。

用法：python scripts/train.py
输出：assets/js/data.js
"""

import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "pima.csv"
OUT_PATH = ROOT / "assets" / "js" / "data.js"

FEATURES = [
    # key, 中文名, 单位, 滑块下限, 上限, 步长, 缺失值说明
    ("Pregnancies", "怀孕次数", "次", 0, 17, 1, None),
    ("Glucose", "血浆葡萄糖", "mg/dL", 40, 200, 1, "0 表示未测量"),
    ("BloodPressure", "舒张压", "mmHg", 30, 120, 1, "0 表示未测量"),
    ("SkinThickness", "三头肌皮褶厚度", "mm", 0, 60, 1, "0 表示未测量"),
    ("Insulin", "血清胰岛素", "μU/mL", 0, 400, 5, "0 表示未测量"),
    ("BMI", "体质指数", "kg/m²", 15, 55, 0.1, "0 表示未测量"),
    ("DiabetesPedigreeFunction", "家族史评分", "", 0.05, 2.5, 0.01, None),
    ("Age", "年龄", "岁", 18, 90, 1, None),
]
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
SEED = 42
TEST_RATIO = 0.2
L2 = 1e-4


def load_rows():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {k: float(v) for k, v in raw.items() if k != "Outcome"}
            row["Outcome"] = int(float(raw["Outcome"]))
            rows.append(row)
    return rows


def sigmoid(z):
    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def solve(a, b):
    """高斯消元（部分主元），求解 a·x = b。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("奇异矩阵")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_logistic(x, y, l2=L2, iters=60):
    """IRLS 拟合逻辑回归，返回 [b0, b1, ..., bd]。"""
    n = len(x)
    d = len(x[0])
    beta = [0.0] * (d + 1)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        hess = [[0.0] * (d + 1) for _ in range(d + 1)]
        for i in range(n):
            eta = beta[0] + sum(beta[j + 1] * x[i][j] for j in range(d))
            p = sigmoid(eta)
            w = p * (1.0 - p)
            r = y[i] - p
            grad[0] += r
            hess[0][0] += w
            for j in range(d):
                grad[j + 1] += r * x[i][j]
                hess[0][j + 1] += w * x[i][j]
                hess[j + 1][0] = hess[0][j + 1]
                for k in range(j + 1):
                    hess[j + 1][k + 1] += w * x[i][j] * x[i][k]
                    hess[k + 1][j + 1] = hess[j + 1][k + 1]
        # L2 只作用于特征系数，不惩罚截距
        for j in range(1, d + 1):
            grad[j] -= l2 * beta[j]
            hess[j][j] += l2
        delta = solve(hess, grad)
        beta = [beta[i] + delta[i] for i in range(d + 1)]
        if max(abs(v) for v in delta) < 1e-9:
            break
    return beta


def roc_curve(pairs):
    """pairs: [(p, y)] -> [(fpr, tpr)], AUC。"""
    pairs = sorted(pairs, key=lambda t: -t[0])
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    tp = fp = 0
    pts = [(0.0, 0.0)]
    auc = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for p, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        fpr, tpr = fp / neg, tp / pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        pts.append((fpr, tpr))
        prev_fpr, prev_tpr = fpr, tpr
    return pts, auc


def stats_at(pairs, thr):
    tp = sum(1 for p, y in pairs if p >= thr and y == 1)
    fp = sum(1 for p, y in pairs if p >= thr and y == 0)
    fn = sum(1 for p, y in pairs if p < thr and y == 1)
    tn = sum(1 for p, y in pairs if p < thr and y == 0)

    def div(a, b):
        return float(a) / b if b else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, len(pairs)),
    }


def main():
    rows = load_rows()
    n = len(rows)

    # 1. 分层切分
    rng = random.Random(SEED)
    pos = [i for i, r in enumerate(rows) if r["Outcome"] == 1]
    neg = [i for i, r in enumerate(rows) if r["Outcome"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = round(len(pos) * TEST_RATIO)
    n_neg_test = round(len(neg) * TEST_RATIO)
    test_idx = set(pos[:n_pos_test] + neg[:n_neg_test])
    train_idx = [i for i in range(n) if i not in test_idx]
    test_sorted = sorted(test_idx)

    # 2. 训练集上统计缺失处理与标准化参数
    medians, zero_counts = {}, {}
    for key, *_ in FEATURES:
        vals = [rows[i][key] for i in train_idx]
        if key in ZERO_AS_MISSING:
            zero_counts[key] = sum(1 for v in vals if v == 0)
            vals = [v for v in vals if v != 0] or [0.0]
        medians[key] = statistics.median(vals)

    def impute(row):
        return [
            row[key] if not (key in ZERO_AS_MISSING and row[key] == 0) else medians[key]
            for key, *_ in FEATURES
        ]

    x_train = [impute(rows[i]) for i in train_idx]
    y_train = [rows[i]["Outcome"] for i in train_idx]
    means = [statistics.fmean(col) for col in zip(*x_train)]
    sds = [statistics.pstdev(col) or 1.0 for col in zip(*x_train)]

    def standardize(xs):
        return [[(v - means[j]) / sds[j] for j, v in enumerate(row)] for row in xs]

    x_train_z = standardize(x_train)
    beta = fit_logistic(x_train_z, y_train)

    def predict(x):
        row = impute(x)
        z = [(v - means[j]) / sds[j] for j, v in enumerate(row)]
        eta = beta[0] + sum(beta[j + 1] * z[j] for j in range(len(z)))
        return sigmoid(eta)

    # 3. 测试集
    test_pairs = [(predict(rows[i]), rows[i]["Outcome"]) for i in test_sorted]
    roc, auc = roc_curve(test_pairs)
    at50 = stats_at(test_pairs, 0.5)
    brier = statistics.fmean((p - y) ** 2 for p, y in test_pairs)

    # 4. 页面示例病例（取自测试集，覆盖不同风险区间，保留真实结局用于教学）
    presets, used = [], set()
    for target in (0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.97):
        best = None
        for i, (p, _) in zip(test_sorted, test_pairs):
            if i in used:
                continue
            score = abs(p - target)
            if best is None or score < best[0]:
                best = (score, p, i)
        _, p, i = best
        used.add(i)
        case = {key: round(rows[i][key], 2) for key, *_ in FEATURES}
        case["_probability"] = round(p, 4)
        case["_outcome"] = rows[i]["Outcome"]
        presets.append(case)

    # 5. 组装导出
    payload = {
        "meta": {
            "dataset": "Pima Indians Diabetes Database (NIDDK / UCI)",
            "n": n,
            "n_train": len(train_idx),
            "n_test": len(test_sorted),
            "prevalence": round(sum(r["Outcome"] for r in rows) / n, 4),
            "seed": SEED,
            "testRatio": TEST_RATIO,
            "l2": L2,
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "at50": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at50.items()},
            "zeroCounts": zero_counts,
            "imputedFeatures": sorted(ZERO_AS_MISSING),
        },
        "features": [
            {
                "key": key,
                "label": label,
                "unit": unit,
                "min": lo,
                "max": hi,
                "step": step,
                "median": round(medians[key], 3),
                "mean": round(means[j], 4),
                "sd": round(sds[j], 4),
                "missingHint": hint,
            }
            for j, (key, label, unit, lo, hi, step, hint) in enumerate(FEATURES)
        ],
        "model": {
            "intercept": round(beta[0], 6),
            "coef": [round(b, 6) for b in beta[1:]],
        },
        "testScores": [[round(p, 4), y] for p, y in test_pairs],
        "presets": presets,
        "roc": [[round(f, 4), round(t, 4)] for f, t in roc],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    js = (
        "/* 由 scripts/train.py 自动生成，请勿手工修改。\n"
        " * 数据：Pima Indians Diabetes Database；模型：L2 正则逻辑回归（标准化特征）。\n"
        " */\n"
        "window.DIABETES_MODEL = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    OUT_PATH.write_text(js, encoding="utf-8")

    print(f"训练集 {len(train_idx)} / 测试集 {len(test_sorted)}")
    print(f"测试集 AUC = {auc:.4f}  Brier = {brier:.4f}")
    print(f"阈值 0.5：敏感度 {at50['sensitivity']:.3f}  特异度 {at50['specificity']:.3f}  准确率 {at50['accuracy']:.3f}")
    print(f"已写出 {OUT_PATH.relative_to(ROOT)}（{OUT_PATH.stat().st_size / 1024:.1f} KB）")


if __name__ == "__main__":
    main()
