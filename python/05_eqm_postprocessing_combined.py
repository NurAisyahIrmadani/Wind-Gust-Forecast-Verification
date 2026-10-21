"""
Script 5 - Multi-window diurnal EQM post-processing, WIND SPEED + WIND GUST combined

What it does
  1. Re-implements the per-window (02/08/14/20 WITA) empirical quantile mapping
     (EQM) separately for each lead time, in Python.
  2. Evaluates it two ways:
       - in-sample   : calibrated and evaluated on all 12 days (as in the MATLAB figures)
       - LODO-CV     : leave-one-day-out cross-validation (out-of-sample)
  3. Saves combined wind|gust figures and a summary table:
       Fig_eqm_overall   : RMSE, MAE, R   raw vs EQM (vs EQM-CV), wind | gust
       Fig_eqm_diurnal   : RMSE by valid time, raw (solid) vs EQM (dashed)
       Fig_eqm_ts_6h     : 6-h forecast time series, ERA5 vs raw vs EQM
       Fig_eqm_ts_all    : all lead times after EQM (supplementary)
       Fig_eqm_table.csv : all numbers behind the figures

Method settings (N_QUANT, interpolation, tail handling) were inferred so that the
output reproduces the MATLAB results; check them against your MATLAB script.

Usage:  python 05_eqm_postprocessing_combined.py
Requires: pandas, numpy, matplotlib, openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ----------------------------- CONFIG -----------------------------------
WIND_FILE = "Wind Windy Cut 24h.xlsx"
GUST_FILE = "Wind Gust Windy Cut 24h.xlsx"
SHEET     = "Reconstructed"
LEADS     = [0, 6, 12, 18, 24]
WINDOWS   = [2, 8, 14, 20]          # valid hours (WITA)
N_QUANT   = 12                      # number of quantiles used in the mapping
TS_LEAD   = 6                       # lead time shown in the time-series figure
SHOW_CV   = True                    # add LODO-CV bars / line
DIURNAL_EQM = "insample"            # "insample" or "cv" for dashed lines in diurnal fig
OUT       = "Fig_eqm"
# -------------------------------------------------------------------------

UNIT = r"m s$^{-1}$"
LEAD_COL = dict(zip(LEADS, ["tab:blue", "tab:orange", "tab:green",
                            "tab:purple", "tab:red"]))


# ----------------------------- data -------------------------------------
def load(path):
    df = pd.read_excel(path, sheet_name=SHEET)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(subset=["Valid Date"]).copy()
    t = pd.to_datetime(df["Time"].astype(str), format="mixed")
    df["win"] = t.dt.hour
    df["day"] = pd.to_datetime(df["Valid Date"]).dt.normalize()
    df["valid"] = df["day"] + pd.to_timedelta(t.dt.hour, unit="h") \
        + pd.to_timedelta(t.dt.minute, unit="m")
    obs_col = [c for c in df.columns if "observation" in str(c).lower()][0]
    df = df.rename(columns={obs_col: "obs"})
    df = df.rename(columns={c: int(c) for c in df.columns
                            if str(c).strip().isdigit()})
    return df.sort_values("valid").reset_index(drop=True)


# ----------------------------- EQM --------------------------------------
def eqm_fit_apply(f_train, o_train, f_apply, nq=N_QUANT):
    """Map forecast values to the observed distribution via matched quantiles.
    Values outside the training range are clamped to the end quantiles."""
    p = np.linspace(0, 1, nq)
    qf = np.quantile(f_train, p)
    qo = np.quantile(o_train, p)
    qf_u, idx = np.unique(qf, return_index=True)      # drop ties (integer data)
    return np.interp(f_apply, qf_u, qo[idx])


def add_eqm(df):
    """Adds columns eqm_<L> (in-sample) and cv_<L> (leave-one-day-out)."""
    df = df.copy()
    days = df["day"].unique()
    for L in LEADS:
        f = df[L].to_numpy(float)
        ins = np.full(len(df), np.nan)
        cv = np.full(len(df), np.nan)
        for w in WINDOWS:
            m = (df["win"] == w).to_numpy()
            ins[m] = eqm_fit_apply(f[m], df["obs"].to_numpy()[m], f[m])
            for d in days:
                tr = m & (df["day"] != d).to_numpy()
                te = m & (df["day"] == d).to_numpy()
                if te.any():
                    cv[te] = eqm_fit_apply(f[tr], df["obs"].to_numpy()[tr], f[te])
        df[f"eqm_{L}"] = ins
        df[f"cv_{L}"] = cv
    return df


# ----------------------------- metrics ----------------------------------
def rmse(a, o): return float(np.sqrt(np.mean((a - o) ** 2)))
def mae(a, o):  return float(np.mean(np.abs(a - o)))
def corr(a, o): return float(np.corrcoef(a, o)[0, 1])
METRICS = {"RMSE": rmse, "MAE": mae, "R": corr}


def summarize(df, var):
    rows = []
    o = df["obs"].to_numpy()
    for L in LEADS:
        for name, fn in METRICS.items():
            raw = fn(df[L].to_numpy(float), o)
            ins = fn(df[f"eqm_{L}"].to_numpy(), o)
            cv = fn(df[f"cv_{L}"].to_numpy(), o)
            rows.append(dict(variable=var, lead_h=L, metric=name,
                             raw=raw, eqm_insample=ins, eqm_cv=cv))
    t = pd.DataFrame(rows)
    return t


# ----------------------------- figures ----------------------------------
def fig_overall(data, tab):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9), sharex=True)
    x = np.arange(len(LEADS))
    w = 0.26 if SHOW_CV else 0.36
    letters = iter("abcdef")
    for r, metric in enumerate(["RMSE", "MAE", "R"]):
        for c, var in enumerate(data):
            ax = axes[r, c]
            t = tab[(tab.variable == var) & (tab.metric == metric)].set_index("lead_h")
            ax.bar(x - w, t.loc[LEADS, "raw"], w, label="Raw ECMWF",
                   color="#0072BD", edgecolor="k", lw=0.6)
            ax.bar(x, t.loc[LEADS, "eqm_insample"], w, label="EQM (in-sample)",
                   color="#D95319", edgecolor="k", lw=0.6)
            if SHOW_CV:
                ax.bar(x + w, t.loc[LEADS, "eqm_cv"], w, label="EQM (LODO-CV)",
                       color="#EDB120", edgecolor="k", lw=0.6, hatch="//")
            ax.set_title(f"({next(letters)}) {var} - {metric}", loc="left",
                         fontweight="bold", fontsize=11)
            ax.set_ylabel(f"{metric} ({UNIT})" if metric != "R" else "Pearson R (-)")
            ax.set_xticks(x); ax.set_xticklabels([f"{L} h" for L in LEADS])
            if metric == "R":
                ax.set_ylim(0, 1)
            ax.grid(axis="y", ls="--", lw=0.5, color="0.85"); ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Forecast lead time")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}_overall.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}_overall.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_diurnal(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    col = "eqm" if DIURNAL_EQM == "insample" else "cv"
    for ax, (var, df), letter in zip(axes, data.items(), "ab"):
        o_by = {w: df[df.win == w] for w in WINDOWS}
        for L in LEADS:
            raw = [rmse(o_by[w][L].to_numpy(float), o_by[w]["obs"].to_numpy()) for w in WINDOWS]
            eq = [rmse(o_by[w][f"{col}_{L}"].to_numpy(), o_by[w]["obs"].to_numpy()) for w in WINDOWS]
            ax.plot(range(4), raw, "-o", color=LEAD_COL[L], lw=1.6, ms=5)
            ax.plot(range(4), eq, "--s", color=LEAD_COL[L], lw=1.6, ms=5, mfc="none")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{w:02d}:00" for w in WINDOWS])
        ax.set_xlabel("Valid time (WITA)"); ax.set_ylabel(f"RMSE ({UNIT})")
        ax.set_title(f"({letter}) {var}", loc="left", fontweight="bold", fontsize=11)
        ax.grid(ls="--", lw=0.5, color="0.85")
        ax.spines[["top", "right"]].set_visible(False)
    h = [Line2D([0], [0], color=LEAD_COL[L], lw=2, label=f"{L} h") for L in LEADS]
    h += [Line2D([0], [0], color="k", ls="-", marker="o", label="Raw"),
          Line2D([0], [0], color="k", ls="--", marker="s", mfc="none",
                 label="EQM" + (" (LODO-CV)" if col == "cv" else ""))]
    fig.legend(handles=h, ncol=7, loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}_diurnal.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}_diurnal.pdf", bbox_inches="tight")
    plt.close(fig)


def _nights(ax, df):
    for d in pd.date_range(df["valid"].min().normalize(),
                           df["valid"].max().normalize() + pd.Timedelta(days=1)):
        ax.axvspan(d - pd.Timedelta(hours=4), d + pd.Timedelta(hours=3),
                   color="0.93", lw=0, zorder=0)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=WINDOWS))


def fig_ts_6h(data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for ax, (var, df), letter in zip(axes, data.items(), "ab"):
        _nights(ax, df)
        ax.plot(df["valid"], df[TS_LEAD], color="tab:red", ls=":", lw=1.6,
                label=f"Raw {TS_LEAD}-h forecast", zorder=3)
        ax.plot(df["valid"], df[f"eqm_{TS_LEAD}"], color="b", ls="-.", lw=1.8,
                label=f"EQM {TS_LEAD}-h forecast", zorder=4)
        if SHOW_CV:
            ax.plot(df["valid"], df[f"cv_{TS_LEAD}"], color="tab:green", ls="--",
                    lw=1.3, label=f"EQM {TS_LEAD}-h (LODO-CV)", zorder=4)
        ax.plot(df["valid"], df["obs"], color="k", lw=2.4, label="ERA5", zorder=5)
        ax.set_ylim(0, max(df[TS_LEAD].max(), df["obs"].max()) * 1.2)
        ax.set_ylabel(f"{var} ({UNIT})")
        ax.set_title(f"({letter}) {var}", loc="left", fontweight="bold", fontsize=11)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
        ax.grid(axis="y", ls="--", lw=0.5, color="0.85")
        ax.spines[["top", "right"]].set_visible(False)
    _date_axis(axes[-1])
    axes[-1].set_xlabel("Valid time, 22 Aug - 2 Sep 2026 (WITA)")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT}_ts_6h.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}_ts_6h.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ts_all(data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for ax, (var, df), letter in zip(axes, data.items(), "ab"):
        _nights(ax, df)
        for L in LEADS:
            ax.plot(df["valid"], df[f"eqm_{L}"], color=LEAD_COL[L], lw=1.0,
                    label=f"EQM {L} h", zorder=3)
        ax.plot(df["valid"], df["obs"], color="k", lw=2.4, label="ERA5", zorder=5)
        ax.set_ylim(0, max(df["obs"].max(), df[[f"eqm_{L}" for L in LEADS]].to_numpy().max()) * 1.2)
        ax.set_ylabel(f"{var} ({UNIT})")
        ax.set_title(f"({letter}) {var}", loc="left", fontweight="bold", fontsize=11)
        ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper right")
        ax.grid(axis="y", ls="--", lw=0.5, color="0.85")
        ax.spines[["top", "right"]].set_visible(False)
    _date_axis(axes[-1])
    axes[-1].set_xlabel("Valid time, 22 Aug - 2 Sep 2026 (WITA)")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT}_ts_all.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}_ts_all.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------- main -------------------------------------
def main():
    data = {"Wind speed": add_eqm(load(WIND_FILE)),
            "Wind gust": add_eqm(load(GUST_FILE))}

    tab = pd.concat([summarize(df, v) for v, df in data.items()], ignore_index=True)
    tab["red_insample_%"] = 100 * (tab.raw - tab.eqm_insample) / tab.raw
    tab["red_cv_%"] = 100 * (tab.raw - tab.eqm_cv) / tab.raw
    tab.round(3).to_csv(f"{OUT}_table.csv", index=False)

    for v in data:
        for m in ["RMSE", "MAE", "R"]:
            t = tab[(tab.variable == v) & (tab.metric == m)].set_index("lead_h")
            print(f"\n{v} - {m}   (lead: {LEADS})")
            for col in ["raw", "eqm_insample", "eqm_cv"]:
                print(f"  {col:13s}", np.round(t.loc[LEADS, col].to_numpy(), 2))

    fig_overall(data, tab)
    fig_diurnal(data)
    fig_ts_6h(data)
    fig_ts_all(data)
    print(f"\nSaved {OUT}_overall / _diurnal / _ts_6h / _ts_all (.png/.pdf) and {OUT}_table.csv")


if __name__ == "__main__":
    main()
