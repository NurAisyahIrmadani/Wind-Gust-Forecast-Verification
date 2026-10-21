"""
Combined baseline skill of RAW ECMWF (Windy) forecasts vs ERA5
Wind speed (left column) and wind gust (right column)
Rows: (a,b) RMSE, (c,d) Pearson correlation  [optional extra row: bias]

Error bars = 1 SD of a bootstrap over the 48 valid-time records
(resampling with replacement, N_BOOT draws). This closely reproduces the
bars in the single-variable figures (e.g. wind 0 h: RMSE 1.17 +/- 0.11).

The correlation here is the Pearson r: anomalies are taken relative to the sample
mean of each series. If you have a true climatology, pass it in via `anomaly_ref`
in acc() to obtain a proper anomaly correlation coefficient (ACC).

Usage:  python 04_skill_wind_gust_combined.py
Requires: pandas, numpy, matplotlib, openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------------
WIND_FILE = "Wind Windy Cut 24h.xlsx"
GUST_FILE = "Wind Gust Windy Cut 24h.xlsx"
SHEET     = "Reconstructed"
LEADS     = [0, 6, 12, 18, 24]
N_BOOT    = 5000
SEED      = 42
SHOW_BIAS = False      # True adds a third row with mean bias (forecast - ERA5)
SHARE_Y   = False      # True forces the same y-range for wind and gust
OUT_PREFIX = "Fig_skill_raw_combined"
# -------------------------------------------------------------------------


def load(path):
    df = pd.read_excel(path, sheet_name=SHEET)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(subset=["Valid Date"]).copy()
    obs_col = [c for c in df.columns if "observation" in str(c).lower()][0]
    df = df.rename(columns={obs_col: "obs"})
    df = df.rename(columns={c: int(c) for c in df.columns
                            if str(c).strip().isdigit()})
    return df.reset_index(drop=True)


def rmse(f, o):
    return np.sqrt(np.mean((f - o) ** 2))


def bias(f, o):
    return np.mean(f - o)


def acc(f, o, anomaly_ref=None):
    """Anomaly correlation. Default: anomalies about each series' own mean."""
    if anomaly_ref is None:
        fa, oa = f - f.mean(), o - o.mean()
    else:
        fa, oa = f - anomaly_ref, o - anomaly_ref
    return np.sum(fa * oa) / np.sqrt(np.sum(fa ** 2) * np.sum(oa ** 2))


def skill_with_boot(df, fn):
    """Point estimate and bootstrap SD for each lead time."""
    rng = np.random.default_rng(SEED)
    o = df["obs"].to_numpy()
    n = len(o)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    est, sd = [], []
    for L in LEADS:
        f = df[L].to_numpy(dtype=float)
        est.append(fn(f, o))
        sd.append(np.std([fn(f[i], o[i]) for i in idx], ddof=1))
    return np.array(est), np.array(sd)


def panel(ax, est, sd, title, ylabel, color, ylim=None, zero_line=False):
    ax.errorbar(LEADS, est, yerr=sd, color=color, marker="o", ms=6, lw=2,
                capsize=3, elinewidth=1.2)
    for x, y in zip(LEADS, est):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color)
    if zero_line:
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(LEADS)
    ax.set_xlim(-2, 26)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", ls="--", lw=0.5, color="0.8")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = {"Wind speed": load(WIND_FILE), "Wind gust": load(GUST_FILE)}
    colors = {"Wind speed": "tab:blue", "Wind gust": "tab:red"}
    unit = r"m s$^{-1}$"

    rows = [("RMSE", rmse, f"RMSE ({unit})"),
            ("Correlation", acc, "Correlation (-)")]
    if SHOW_BIAS:
        rows.insert(1, ("Bias", bias, f"Bias, forecast - ERA5 ({unit})"))

    # compute everything first (so y-limits can be shared if requested)
    res = {}
    for var, df in data.items():
        for name, fn, _ in rows:
            res[(var, name)] = skill_with_boot(df, fn)

    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 3.3 * len(rows)),
                             sharex=True, squeeze=False)
    letters = iter("abcdefgh")
    for r, (name, fn, ylab) in enumerate(rows):
        if SHARE_Y:
            lo = min((res[(v, name)][0] - res[(v, name)][1]).min() for v in data)
            hi = max((res[(v, name)][0] + res[(v, name)][1]).max() for v in data)
            pad = 0.08 * (hi - lo)
            shared = (lo - pad, hi + pad)
        for c, var in enumerate(data):
            est, sd = res[(var, name)]
            ylim = shared if SHARE_Y else None
            if name == "Correlation" and not SHARE_Y:
                ylim = (0, 1)
            panel(axes[r, c], est, sd, f"({next(letters)}) {var} - {name}",
                  ylab if c == 0 else "", colors[var], ylim=ylim,
                  zero_line=(name in ("Correlation", "Bias")))
    for ax in axes[-1]:
        ax.set_xlabel("Lead time (h)")

    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")

    # summary table for the manuscript
    out = []
    for var in data:
        for name, _, _ in rows:
            est, sd = res[(var, name)]
            for L, e, s in zip(LEADS, est, sd):
                out.append(dict(variable=var, metric=name, lead_h=L,
                                value=round(e, 3), boot_sd=round(s, 3)))
    tab = pd.DataFrame(out)
    tab.to_csv(f"{OUT_PREFIX}_table.csv", index=False)
    print(tab.pivot_table(index=["variable", "metric"], columns="lead_h",
                          values="value").round(2))
    print(f"\nSaved {OUT_PREFIX}.png / .pdf / _table.csv")


if __name__ == "__main__":
    main()
