"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts a launch day's Twitch viewers, hours 2..24, as five quantiles an hour,
from the case's hour-1 print and the comparables' past launches.

Model: a 3-way ensemble, blended in log space with equal weights, of
  1. a pooled spline-ridge fit of log(viewers(h)/viewers(1)) on the hour and
     on log hour-1 size (tent-spline basis, smoothness-penalised);
  2. the same fit with a second size feature, the hour-1 surge over the
     category's own pre-release level (90th percentile of the last 7 days);
  3. a 50/50 blend of the pooled per-hour median curve and a per-hour
     OLS of the log-ratio on log hour-1 size.
Each component's quantiles are median * exp(z_tau * sigma); the blend's
spread is shrunk 10% toward its median. Size features are clamped to the
comparables' observed range so an unprecedented launch (GTA VI) cannot
extrapolate the fitted slopes into nonsense. Any failure falls back to the
house persistence baseline, so the output is always valid.

Deterministic: no randomness, no network, no time, no dict-order dependence
(everything sorted). Runtime is far under the 8 s/case cap.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
BASE_SIGMA = 0.40          # per-component quantile spread, in log units
SHRINK = 0.9               # blend spread shrink toward the blended median
KNOTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
LAM_SMOOTH = 10.0
LAM_GAMMA = 3.0
LAM_GAMMA2 = 6.0
CLAMP_PAD = 0.5            # size features may stray this far past the observed range


def round2(x):
    """x rounded to 2 decimals, half up (the starter's rounding)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def robust_sigma(xs):
    if len(xs) < 3:
        return 0.30
    med = median(xs)
    mad = median([abs(x - med) for x in xs])
    return max(0.08, min(1.2, 1.4826 * mad))


def solve(A, b):
    """Dense Gaussian elimination with partial pivoting (small, well-posed systems)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def tent(h):
    """Piecewise-linear hat basis over KNOTS (1.0 at a knot, 0 at the others)."""
    if h <= KNOTS[0]:
        return [1.0 if i == 0 else 0.0 for i in range(len(KNOTS))]
    if h >= KNOTS[-1]:
        return [1.0 if i == len(KNOTS) - 1 else 0.0 for i in range(len(KNOTS))]
    out = [0.0] * len(KNOTS)
    for i in range(len(KNOTS) - 1):
        if KNOTS[i] <= h <= KNOTS[i + 1]:
            f = (h - KNOTS[i]) / (KNOTS[i + 1] - KNOTS[i])
            out[i] = 1 - f
            out[i + 1] = f
            break
    return out


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hour_means(hourly, day_start):
    """hour h (1..24) -> mean viewers of points in [day_start+(h-1)h, day_start+h h)."""
    out = {}
    for pt in hourly:
        try:
            t = parse_ts(pt["from"])
        except Exception:
            continue
        v = pt.get("viewers")
        if v is None:
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out.setdefault(h, []).append(v)
    return {h: sum(v) / len(v) for h, v in out.items()}


def baseline_p90(hourly, day_start, days=7):
    """90th percentile of the category's viewers in the `days` before day_start."""
    lo = day_start - timedelta(days=days)
    vals = sorted(pt["viewers"] for pt in hourly
                  if pt.get("viewers")
                  and lo <= parse_ts(pt["from"]) < day_start)
    if not vals:
        return None
    return vals[int(0.9 * (len(vals) - 1))]


def comp_curves(case):
    """[(name, p1, {h: log-ratio})] for comparables with a valid hour-1 print."""
    curves = []
    for c in sorted(case["comparables"], key=lambda x: x["name"]):
        try:
            ds = parse_ts(c["day_start"])
            hm = hour_means(c["hourly"], ds)
        except Exception:
            continue
        if 1 not in hm or hm[1] <= 0:
            continue
        p1 = hm[1]
        lr = {h: math.log(hm[h] / p1) for h in hm if h >= 2 and hm[h] > 0}
        if len(lr) >= 10:
            curves.append((c["name"], p1, lr))
    return curves


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fit_spline(curves, features, lam_smooth=LAM_SMOOTH, lam_gamma=LAM_GAMMA,
               lam_gamma2=LAM_GAMMA2):
    """Ridge fit of log-ratio on tent(h) + sum_j feature_j * tent(h).

    features: {name: (f1, f2, ...)}; returns predict(h, feats) -> log-ratio.
    """
    nfeat = len(next(iter(features.values()))) if features else 0
    K = len(KNOTS)
    P = K * (1 + nfeat)
    A = [[0.0] * P for _ in range(P)]
    b = [0.0] * P
    for name, _p1, lr in curves:
        fs = features[name]
        for h, y in lr.items():
            B = tent(h)
            x = B + [f * v for f in fs for v in B]
            for i in range(P):
                b[i] += x[i] * y
                for j in range(P):
                    A[i][j] += x[i] * x[j]
    for block in range(1 + nfeat):
        for i in range(K - 2):
            for a_, c_ in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
                for b_, d_ in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
                    A[block * K + a_][block * K + b_] += lam_smooth * c_ * d_
        for i in range(K):
            A[block * K + i][block * K + i] += 1e-6
    lams = [lam_gamma, lam_gamma2, lam_gamma2, lam_gamma2]
    for j in range(nfeat):
        for i in range(K):
            A[(j + 1) * K + i][(j + 1) * K + i] += lams[j]
    coef = solve(A, b)
    blocks = [coef[i * K:(i + 1) * K] for i in range(1 + nfeat)]

    def predict(h, feats):
        B = tent(h)
        out = sum(blocks[0][i] * B[i] for i in range(K))
        for j, f in enumerate(feats):
            out += f * sum(blocks[j + 1][i] * B[i] for i in range(K))
        return out

    return predict


def quantiles_from_logmed(p1, logmed, sigma):
    return [p1 * math.exp(logmed + z * sigma) for z in Z]


def persistence(p1):
    """House baseline (the starter), used as a fallback, never as the forecast."""
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            for h in range(2, 25)}


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = comp_curves(case)
    if len(curves) < 5:
        return persistence(p1)

    ref = median([math.log(p) for _, p, _ in curves])
    f1_lo = min(math.log(p) - ref for _, p, _ in curves) - CLAMP_PAD
    f1_hi = max(math.log(p) - ref for _, p, _ in curves) + CLAMP_PAD
    # Winsorise the size features at their 90th percentile: the fitted
    # log-size slopes keep falling past the biggest comparables, but the
    # biggest launches' actual evening ratios flatten near 1x (ELDEN RING
    # 875k -> 0.90x at h20, Cyberpunk 852k -> 0.90x). Treating any launch
    # above the 90th percentile as the 90th percentile stops the model
    # forecasting a 0.3x evening for a GTA-VI-scale hour-1 print.
    _f1s = sorted(math.log(p) - ref for _, p, _ in curves)
    cap_f1 = _f1s[int(0.9 * (len(_f1s) - 1))]
    day_start = parse_ts(case["day_start"])

    # --- feature maps for the comparables ---
    by_name = {c["name"]: c for c in case["comparables"]}
    feat1 = {}
    feat2 = {}
    _f2raw = {}
    for name, cp1, _lr in curves:
        # Comparables keep their raw size features in the fit (only the outer
        # guard clamp); the p90 cap applies at prediction time, below.
        feat1[name] = (clamp(math.log(cp1) - ref, f1_lo, f1_hi),)
        c = by_name.get(name)
        bl = baseline_p90(c["hourly"], parse_ts(c["day_start"])) if c else None
        _f2raw[name] = (math.log(cp1) - math.log(bl)) if bl else 0.0
    _f2s = sorted(_f2raw.values())
    cap_f2 = _f2s[int(0.9 * (len(_f2s) - 1))]
    for name, cp1, _lr in curves:
        feat2[name] = (feat1[name][0], clamp(_f2raw[name],
                                             min(_f2raw.values()) - CLAMP_PAD,
                                             max(_f2raw.values()) + CLAMP_PAD))
    f2_vals = [f[1] for f in feat2.values()]
    f2_lo, f2_hi = min(f2_vals) - CLAMP_PAD, max(f2_vals) + CLAMP_PAD

    case_f1 = clamp(min(math.log(p1) - ref, cap_f1), f1_lo, f1_hi)
    cbl = baseline_p90(case.get("history_hourly") or [], day_start)
    case_f2 = clamp(min((math.log(p1) - math.log(cbl)) if cbl else 0.0, cap_f2),
                    f2_lo, f2_hi)

    preds = []
    try:
        pred1 = fit_spline(curves, feat1)
        preds.append({h: quantiles_from_logmed(p1, pred1(h, (case_f1,)), BASE_SIGMA)
                      for h in range(2, 25)})
    except Exception:
        pass
    try:
        pred2 = fit_spline(curves, feat2, lam_gamma2=LAM_GAMMA2)
        preds.append({h: quantiles_from_logmed(p1, pred2(h, (case_f1, case_f2)), BASE_SIGMA)
                      for h in range(2, 25)})
    except Exception:
        pass
    try:
        # component 3: 50/50 blend of pooled median and per-hour OLS on log size
        med_lr, sig = {}, {}
        for h in range(2, 25):
            xs = [lr[h] for _, _, lr in curves if h in lr]
            med_lr[h] = median(xs) if xs else 0.0
            sig[h] = robust_sigma(xs) if xs else 0.30
        pa = {h: quantiles_from_logmed(p1, med_lr[h], sig[h]) for h in range(2, 25)}
        lp = math.log(p1)
        lp_lo = min(math.log(cp1) for _, cp1, _ in curves) - CLAMP_PAD
        lp_hi = max(math.log(cp1) for _, cp1, _ in curves) + CLAMP_PAD
        lp_c = min(clamp(lp, lp_lo, lp_hi), ref + cap_f1)
        ols = {}
        for h in range(2, 25):
            pts = [(math.log(cp1), lr[h]) for _, cp1, lr in curves if h in lr]
            if len(pts) >= 5:
                n = len(pts)
                mx = sum(x for x, _ in pts) / n
                my = sum(y for _, y in pts) / n
                sxx = sum((x - mx) ** 2 for x, _ in pts)
                sxy = sum((x - mx) * (y - my) for x, y in pts)
                b_ = sxy / sxx if sxx > 0 else 0.0
                a_ = my - b_ * mx
                ols[h] = a_ + b_ * lp_c
            else:
                ols[h] = med_lr[h]
        pc = {h: quantiles_from_logmed(p1, ols[h], sig[h]) for h in range(2, 25)}
        preds.append({h: [math.exp(0.5 * (math.log(max(1.0, a)) + math.log(max(1.0, c))))
                          for a, c in zip(pa[h], pc[h])] for h in range(2, 25)})
    except Exception:
        pass

    if not preds:
        return persistence(p1)

    # --- blend the components' quantiles in log space, shrink spread 10% ---
    out = {}
    for h in range(2, 25):
        blended = [math.exp(sum(math.log(max(1.0, p[h][i])) for p in preds) / len(preds))
                   for i in range(5)]
        med = blended[2]
        out[h] = [max(1.0, med * (q / med) ** SHRINK) if q > 0 else 1.0
                  for q in blended]
    return out


def clean(rows):
    """Enforce the output contract: 23 rows, 5 finite numbers in (0, 1e9], non-decreasing."""
    hours = []
    for row in sorted(rows, key=lambda r: r["hour"]):
        qs = []
        prev = 0.0
        for q in row["q"]:
            v = float(q)
            if not math.isfinite(v) or v <= 0:
                v = 1.0
            v = min(v, 1e9)
            v = max(v, prev)
            qs.append(round2(v))
            prev = qs[-1]
        hours.append({"hour": row["hour"], "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            pred = forecast_case(case)
        except Exception:
            p1 = 1.0
            try:
                p1 = max(1.0, float(next(k["viewers"] for k in case["known"]
                                        if k["hour"] == 1)))
            except Exception:
                pass
            pred = persistence(p1)
        rows = [{"hour": h, "q": pred[h]} for h in range(2, 25)]
        doc = clean(rows)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
