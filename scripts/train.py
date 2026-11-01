"""
训练混合性别糖尿病风险模型（逻辑回归），导出前端可直接使用的资产。

数据：Diabetes Prediction Dataset（Kaggle，10 万条 EHR 衍生记录）
不依赖 numpy / sklearn：纯标准库实现 IRLS（牛顿法）。

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
CSV_PATH = ROOT / "data" / "diabetes_prediction.csv"
OUT_PATH = ROOT / "assets" / "js" / "data.js"

SEED = 42
TEST_RATIO = 0.2
L2 = 1e-4

SMOKING = [
    ("never", "从不吸烟", None),
    ("former", "已戒烟", "smoking_former"),
    ("current", "目前吸烟", "smoking_current"),
    ("not current", "目前未吸烟", "smoking_not_current"),
    ("ever", "曾经吸烟", "smoking_ever"),
    ("No Info", "不详", "smoking_noinfo"),
]
CONT = ["age", "bmi", "hba1c", "glucose"]
BINARY = ["hypertension", "heart"]


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


def fit_logistic(x, y, l2=L2, iters=40):
    n = len(x)
    d = len(x[0])
    beta = [0.0] * (d + 1)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        hess = [[0.0] * (d + 1) for _ in range(d + 1)]
        for i in range(n):
            row = x[i]
            eta = beta[0] + sum(beta[j + 1] * row[j] for j in range(d))
            p = sigmoid(eta)
            w = p * (1.0 - p)
            r = y[i] - p
            grad[0] += r
            hess[0][0] += w
            for j in range(d):
                xj = row[j]
                grad[j + 1] += r * xj
                hess[0][j + 1] += w * xj
                hess[j + 1][0] = hess[0][j + 1]
                for k in range(j + 1):
                    hess[j + 1][k + 1] += w * xj * row[k]
                    hess[k + 1][j + 1] = hess[j + 1][k + 1]
        for j in range(1, d + 1):
            grad[j] -= l2 * beta[j]
            hess[j][j] += l2
        delta = solve(hess, grad)
        beta = [beta[i] + delta[i] for i in range(d + 1)]
        if max(abs(v) for v in delta) < 1e-9:
            break
    return beta


def auc_of(pairs):
    pairs = sorted(pairs, key=lambda t: -t[0])
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    tp = fp = 0
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        fpr, tpr = fp / neg, tp / pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return auc


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


def load_rows():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if raw["gender"] not in ("Female", "Male"):
                continue
            age = float(raw["age"])
            bmi = float(raw["bmi"])
            if age < 18 or bmi < 10 or bmi > 60:
                continue
            rows.append({
                "sex": 0 if raw["gender"] == "Female" else 1,
                "age": age,
                "bmi": bmi,
                "hba1c": float(raw["HbA1c_level"]),
                "glucose": float(raw["blood_glucose_level"]),
                "hypertension": int(raw["hypertension"]),
                "heart": int(raw["heart_disease"]),
                "smoking": raw["smoking_history"],
                "y": int(raw["diabetes"]),
            })
    n_before = len(rows)
    seen = set()
    uniq = []
    for r in rows:
        key = (r["sex"], r["age"], r["bmi"], r["hba1c"], r["glucose"],
               r["hypertension"], r["heart"], r["smoking"], r["y"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq, n_before - len(uniq)


def main():
    rows, n_dups = load_rows()
    n = len(rows)

    rng = random.Random(SEED)
    pos = [i for i, r in enumerate(rows) if r["y"] == 1]
    neg = [i for i, r in enumerate(rows) if r["y"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = round(len(pos) * TEST_RATIO)
    n_neg_test = round(len(neg) * TEST_RATIO)
    test_idx = set(pos[:n_pos_test] + neg[:n_neg_test])
    train_idx = [i for i in range(n) if i not in test_idx]
    test_sorted = sorted(test_idx)

    means = {k: statistics.fmean(rows[i][k] for i in train_idx) for k in CONT}
    sds = {k: statistics.pstdev([rows[i][k] for i in train_idx]) or 1.0 for k in CONT}

    def encode(r):
        v = [r["sex"]]
        v += [(r[k] - means[k]) / sds[k] for k in CONT]
        v += [r[k] for k in BINARY]
        for key, _, term in SMOKING:
            if term:
                v.append(1.0 if r["smoking"] == key else 0.0)
        return v

    x_train = [encode(rows[i]) for i in train_idx]
    y_train = [rows[i]["y"] for i in train_idx]
    beta = fit_logistic(x_train, y_train)
    weights = dict(zip(
        ["sex"] + CONT + BINARY + [term for _, _, term in SMOKING if term],
        beta[1:],
    ))

    def predict(r):
        eta = beta[0] + sum(w * v for w, v in zip(beta[1:], encode(r)))
        return sigmoid(eta)

    test_pairs = [(predict(rows[i]), rows[i]["y"]) for i in test_sorted]
    auc = auc_of(test_pairs)
    brier = statistics.fmean((p - y) ** 2 for p, y in test_pairs)
    at50 = stats_at(test_pairs, 0.5)

    def pick(sex, target):
        best = None
        for i, (p, y) in zip(test_sorted, test_pairs):
            r = rows[i]
            if r["sex"] != sex:
                continue
            if not (18 <= r["age"] <= 80 and 15 <= r["bmi"] <= 55):
                continue
            score = abs(p - target)
            if best is None or score < best[0]:
                best = (score, p, i)
        return best

    presets = {"female": {}, "male": {}}
    for sex, label in ((0, "female"), (1, "male")):
        for tier, target in (("low", 0.01), ("high", 0.95)):
            _, p, i = pick(sex, target)
            r = rows[i]
            presets[label][tier] = {
                "sex": r["sex"], "age": r["age"], "bmi": round(r["bmi"], 1),
                "hba1c": round(r["hba1c"], 1), "glucose": r["glucose"],
                "hypertension": r["hypertension"], "heart": r["heart"],
                "smoking": r["smoking"],
                "_probability": round(p, 4), "_outcome": r["y"],
            }

    sex_stats = {}
    for s, label in ((0, "female"), (1, "male")):
        sub = [r for r in rows if r["sex"] == s]
        sex_stats[label] = {
            "n": len(sub),
            "prevalence": round(sum(r["y"] for r in sub) / len(sub), 4),
        }

    features = [
        {
            "key": "sex", "type": "segment", "label": "性别",
            "options": [{"value": 0, "label": "女性"}, {"value": 1, "label": "男性"}],
            "default": 0,
        },
        {
            "key": "age", "type": "range", "label": "年龄", "unit": "岁",
            "min": 18, "max": 80, "step": 1, "default": 60,
        },
        {
            "key": "bmi", "type": "range", "label": "体质指数", "unit": "kg/m²",
            "min": 15, "max": 55, "step": 0.1, "default": 32,
        },
        {
            "key": "hba1c", "type": "range", "label": "糖化血红蛋白", "unit": "%",
            "min": 3.5, "max": 9, "step": 0.1, "default": 6.4,
        },
        {
            "key": "glucose", "type": "range", "label": "空腹血糖", "unit": "mg/dL",
            "min": 60, "max": 300, "step": 1, "default": 160,
        },
        {
            "key": "hypertension", "type": "toggle", "label": "高血压",
            "hint": "已确诊或正在服药", "default": 0,
        },
        {
            "key": "heart", "type": "toggle", "label": "冠心病",
            "hint": "心肌梗死或冠心病史", "default": 0,
        },
        {
            "key": "smoking", "type": "select", "label": "吸烟史",
            "options": [{"value": k, "label": lab, "term": term} for k, lab, term in SMOKING],
            "default": "never",
        },
    ]

    payload = {
        "meta": {
            "dataset": "Diabetes Prediction Dataset (Kaggle, EHR-derived)",
            "source": "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset",
            "n": n,
            "nBeforeDedup": n + n_dups,
            "nDropped": n_dups,
            "nTrain": len(train_idx),
            "nTest": len(test_sorted),
            "prevalence": round(sum(r["y"] for r in rows) / n, 4),
            "seed": SEED,
            "testRatio": TEST_RATIO,
            "l2": L2,
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "at50": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at50.items()},
            "sexStats": sex_stats,
        },
        "features": features,
        "model": {
            "intercept": round(beta[0], 6),
            "weights": {k: round(v, 6) for k, v in weights.items()},
            "means": {k: round(means[k], 4) for k in CONT},
            "sds": {k: round(sds[k], 4) for k in CONT},
        },
        "testScores": [[round(p, 4), y, rows[i]["sex"]] for (p, y), i in zip(test_pairs, test_sorted)],
        "presets": presets,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    js = (
        "/* 由 scripts/train.py 自动生成，请勿手工修改。\n"
        " * 数据：Diabetes Prediction Dataset（含性别）；模型：L2 正则逻辑回归。\n"
        " */\n"
        "window.DIABETES_MODEL = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    OUT_PATH.write_text(js, encoding="utf-8")

    print(f"清洗后 {n + n_dups} 行，去重删除 {n_dups} 行，保留 {n} 行")
    print(f"训练 {len(train_idx)} / 测试 {len(test_sorted)}；患病率 {payload['meta']['prevalence']:.3f}")
    print(f"AUC = {auc:.4f}  Brier = {brier:.4f}")
    print(f"阈值 0.5：敏感度 {at50['sensitivity']:.3f}  特异度 {at50['specificity']:.3f}  准确率 {at50['accuracy']:.3f}")
    print(f"女性 {sex_stats['female']['n']}（{sex_stats['female']['prevalence']:.3f}） / 男性 {sex_stats['male']['n']}（{sex_stats['male']['prevalence']:.3f}）")
    print(f"已写出 {OUT_PATH.relative_to(ROOT)}（{OUT_PATH.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
