"""
Raw time series: ECMWF (Windy) vs ERA5, 22 Aug - 2 Sep 2026
Panels: (a) wind speed, (b) wind gust  (gust panel is skipped if no file is given)

Usage:
    python 03_timeseries_wind_gust_combined.py
Edit the CONFIG block below to point to your files.
Requires: pandas, numpy, matplotlib, openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------- CONFIG -----------------------------------
WIND_FILE = "Wind Windy Cut 24h.xlsx"
GUST_FILE = "Wind Gust Windy Cut 24h.xlsx"
SHEET     = "Reconstructed"
LEADS     = [0, 6, 12, 18, 24]

# "envelope": ensemble mean of all leads + min-max shading (cleanest)
# "all"    : one line per lead time (busy)
# "ends"   : only 0 h and 24 h
MODE      = "envelope"

OUT_PREFIX = "Fig_timeseries_wind_gust_combined"   # saves .png (300 dpi) and .pdf
# -------------------------------------------------------------------------


def load(path, sheet=SHEET):
    """Read the Windy/ERA5 workbook and build a proper valid datetime (WITA)."""
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(subset=["Valid Date"]).copy()

    # 'Time' is read by Excel as 1900-01-0x hh:mm or as datetime.time: keep h:m only
    t = pd.to_datetime(df["Time"].astype(str), format="mixed")
    df["valid"] = pd.to_datetime(df["Valid Date"]).dt.normalize() \
        + pd.to_timedelta(t.dt.hour, unit="h") \
        + pd.to_timedelta(t.dt.minute, unit="m")

    obs_col = [c for c in df.columns if "observation" in str(c).lower()][0]
    df = df.rename(columns={obs_col: "obs"})
    df = df.rename(columns={c: int(c) for c in df.columns
                            if str(c).strip().isdigit()})
    return df.sort_values("valid").reset_index(drop=True)


def stats(df, name):
    """Quick RMSE / correlation per lead time (raw), printed to console."""
    print(f"\n{name}: n = {len(df)}")
    print(f"  ERA5 mean = {df['obs'].mean():.2f}, max = {df['obs'].max():.2f} m/s")
    for L in LEADS:
        e = df[L] - df["obs"]
        rmse = np.sqrt((e ** 2).mean())
        r = np.corrcoef(df[L], df["obs"])[0, 1]
        print(f"  {L:>2d} h: RMSE = {rmse:.2f}, bias = {e.mean():+.2f}, r = {r:.2f}")


def shade_nights(ax, df):
    """Light grey band for 20-02 WITA (night) to help read the diurnal cycle."""
    d0 = df["valid"].min().normalize()
    d1 = df["valid"].max().normalize() + pd.Timedelta(days=1)
    for day in pd.date_range(d0, d1, freq="D"):
        ax.axvspan(day - pd.Timedelta(hours=4), day + pd.Timedelta(hours=3),
                   color="0.92", lw=0, zorder=0)


def draw(ax, df, title, ylabel):
    shade_nights(ax, df)
    x = df["valid"]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(LEADS)))

    if MODE == "envelope":
        F = df[LEADS].to_numpy()
        ax.fill_between(x, F.min(1), F.max(1), color="tab:blue", alpha=0.25,
                        lw=0, label="ECMWF range (0-24 h)", zorder=2)
        ax.plot(x, F.mean(1), color="tab:blue", lw=1.6, marker="o", ms=3.5,
                label="ECMWF mean (0-24 h)", zorder=3)
    elif MODE == "ends":
        for L, c in ((0, "tab:blue"), (24, "tab:orange")):
            ax.plot(x, df[L], color=c, lw=1.3, marker="o", ms=3,
                    label=f"ECMWF {L} h", zorder=3)
    else:  # "all"
        for L, c in zip(LEADS, cols):
            ax.plot(x, df[L], color=c, lw=1.0, marker="o", ms=2.5,
                    label=f"ECMWF {L} h", zorder=3)

    ax.plot(x, df["obs"], color="k", lw=2.2, marker="s", ms=3.5,
            label="ERA5", zorder=4)

    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls="--", lw=0.5, color="0.8")
    ymax = max(df[LEADS].to_numpy().max(), df["obs"].max())
    ax.set_ylim(0, ymax * 1.2)   # headroom for the legend
    ax.spines[["top", "right"]].set_visible(False)


def main():
    panels = [("(a) Wind speed", load(WIND_FILE), "Wind speed (m s$^{-1}$)")]
    stats(panels[0][1], "Wind speed")
    if GUST_FILE:
        g = load(GUST_FILE)
        panels.append(("(b) Wind gust", g, "Wind gust (m s$^{-1}$)"))
        stats(g, "Wind gust")

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.6 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (title, df, ylab) in zip(axes, panels):
        draw(ax, df, title, ylab)
    axes[0].legend(ncol=3, fontsize=8.5, frameon=False, loc="upper right")

    ax = axes[-1]
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[2, 8, 14, 20]))
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlabel("Valid time, 22 Aug - 2 Sep 2026 (WITA; minor ticks = 02, 08, 14, 20)")
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")
    print(f"\nSaved {OUT_PREFIX}.png / .pdf")


if __name__ == "__main__":
    main()
