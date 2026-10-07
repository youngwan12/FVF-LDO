"""FVF LDO paper figures (IEEE single column 3.5 in; fig6 double column 7.16 in).

Usage:  python scripts/ldo_figures.py
Reads WaveView CSV exports from data/ and writes PDF + PNG to figures/.
Extracted metrics go to figures/metrics.txt (readable) and figures/summary.json.
Layouts follow the reference figures in the 2026 LDO paper figure guide
(JSSC'22 / SOVC'26 style, bold Arial).
"""
from pathlib import Path
import json
import logging
import re
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # bold-fallback chatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Simulation conditions printed on the figures. None = not drawn.
COND = {
    "psr_vin": None,      # e.g. 1.0  [V]
    "psr_vout": None,     # e.g. 0.8  [V]
    "tr_vin": None,       # e.g. (1.0, 1.1) for the two tr1.csv columns [V]
    "ldr_vin": 1.0,       # VIN of the loadR1.csv DC sweep [V]
}

# ---------------------------------------------------------------- style ----
COL_W = 3.5    # IEEE single column width [in]
DCOL_W = 7.16  # IEEE double column width [in]

# Reference figures (JSSC'22 / SOVC'26) use bold Arial. Liberation Sans is the
# metric-compatible Arial clone used only when Arial itself is not installed.
_avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
FONT = next(f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _avail)
if FONT != "Arial":
    warnings.warn(f"Arial is not installed; figures use {FONT} instead", stacklevel=1)
try:                                   # larger sub/superscripts (default 0.7 is < 6 pt here)
    import matplotlib._mathtext as _mathtext
    _mathtext.SHRINK_FACTOR = 0.8
except (ImportError, AttributeError):
    pass

ANN = 7.5      # annotation font size [pt]
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.fontset": "custom",
    "mathtext.rm": f"{FONT}:bold",
    "mathtext.it": f"{FONT}:bold",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": f"{FONT}:bold",
    "mathtext.default": "rm",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.2,
    "lines.markeredgewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.4,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#222222",
    "legend.fancybox": False,
    "legend.handlelength": 2.0,
    "patch.linewidth": 0.6,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.h_pad": 0.02,
    "savefig.dpi": 600,
    "savefig.bbox": None,        # keep the exact column width (3.5 / 7.16 in)
    "pdf.fonttype": 42,          # editable TrueType text in PDF
    "ps.fonttype": 42,
})

# Distinct categorical colours (blue, orange, teal, purple, pink, black, amber, red)
# plus per-series markers for print / B&W
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e8579a", "#222222", "#eda100", "#e34948"]
MRK = ["o", "s", "^", "D", "v", "p", "h", "<"]
VOUT_C = "#c2187a"   # scope-style VOUT trace (magenta)
ILOAD_C = "#1f9e3a"  # scope-style ILOAD trace (green)
INK = "#222222"
MUTED = "#666666"
WARN = "#d62728"
ARROW = dict(arrowstyle="<|-|>", mutation_scale=7, lw=0.8, color=INK, shrinkA=0, shrinkB=0)
EPS = 1e-9


def figure(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout="constrained", **kw)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf/.png")


def txt(ax, x, y, s, size=ANN, box=True, **kw):
    """Annotation with a white backing so gridlines do not run through it."""
    if box:
        kw.setdefault("bbox", dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    kw.setdefault("zorder", 6)
    return ax.text(x, y, s, fontsize=size, **kw)


def framed(ax, x, y, s, **kw):
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=ANN, zorder=6,
                   bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.6), **kw)


def eng_hz(x, _=None):
    for v, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= v:
            return f"{x / v:g}{s}"
    return f"{x:g}"


def log_axis(ax, axis="x", hz=False):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(10))
    a.set_minor_locator(LogLocator(10, subs=np.arange(2, 10) * 0.1))
    a.set_major_formatter(FuncFormatter(eng_hz if hz else lambda x, _: f"{x:g}"))


def ma_label(i):
    return f"{i * 1e6:.0f} µA" if i < 1e-3 else f"{i * 1e3:g} mA"


def rng(vals, fmt="{:.1f}"):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def rnd(x, n=4):
    """Round for summary.json (the CSV exports carry 4-5 significant digits)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(f"{x:.{n}g}")


# ------------------------------------------------------------ loading ----
def load(name):
    """Parse a WaveView 'format table' export -> (column names, ndarray)."""
    lines = (DATA / name).read_text().splitlines()
    header = lines[1]
    sep = "," if "," in header else None
    cols = [c.strip() for c in header.split(sep) if c.strip()]
    rows = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        rows.append([float(v) for v in ln.split(sep) if v.strip()])
    return cols, np.array(rows)


SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3}


def param(col):
    """'v(out):load2=10m' -> 0.01"""
    m = re.search(r"=([-\d.]+)([fpnumk]?)$", col)
    return float(m.group(1)) * SI.get(m.group(2), 1.0)


def at(t, y, t0):
    return y[np.argmin(np.abs(t - t0))]


def before(t, y, t0):
    """Last sample strictly before t0 (the exports repeat time stamps at the step edges)."""
    return y[np.where(t < t0 - 1e-6)[0][-1]]


def settle_time(t, v, t0, band, t1):
    """Time from t0 until v stays within +-band (inclusive) of its value at t1."""
    m = (t >= t0) & (t <= t1)
    tt, vv = t[m], v[m]
    out = np.abs(vv - vv[-1]) > band + 1e-7      # 0.1 uV tolerance: the 0.1 mV-quantised
    return (tt[np.where(out)[0][-1] + 1] - t0) if out.any() else 0.0   # data hit the band edge


metrics = []
SUMMARY = {}          # every computed value, written to figures/summary.json


def log(s):
    print(s)
    metrics.append(s)


# ------------------------------------------------- 1. Max load current ----
MAXLOAD_DV = 1e-3     # max load: largest load whose steady-state VOUT drop stays <= 1 mV


def fig_maxload():
    cols, d = load("maxload1.csv")
    t, V = d[:, 0] * 1e6, d[:, 1:]                     # [us]
    il = np.array([param(c) for c in cols[1:]])
    v0 = before(t, V[:, 0], 1.0)                       # light load (150 uA)
    vss = before(t, V, 5.0)                            # under load, just before release
    vmin = V[(t > 1) & (t < 5)].min(0)
    vmax = V[(t > 5) & (t < 10)].max(0)
    ok = il[(v0 - vss) <= MAXLOAD_DV + EPS]
    imax = ok.max()

    log("[Max load]  light-load VOUT = %.1f mV;  ILOAD,max = %.0f mA (largest load with "
        "steady-state drop <= %.0f mV)" % (v0 * 1e3, imax * 1e3, MAXLOAD_DV * 1e3))
    for i, vs, vm, vx in zip(il, vss, vmin, vmax):
        log("  ILOAD=%5.0f mA  VOUT(<5us)=%.1f mV  dV_ss=%.1f mV  undershoot=%.1f mV  overshoot=%.1f mV"
            % (i * 1e3, vs * 1e3, (v0 - vs) * 1e3, (v0 - vm) * 1e3, (vx - v0) * 1e3))
    SUMMARY["max_load"] = dict(v_light_mV=rnd(v0 * 1e3), imax_mA=rnd(imax * 1e3),
                               criterion_mV=MAXLOAD_DV * 1e3, edge_ns=500, rows=[
        dict(iload_mA=rnd(i * 1e3), vout_ss_mV=rnd(vs * 1e3), dv_ss_mV=rnd((v0 - vs) * 1e3, 2),
             undershoot_mV=rnd((v0 - vm) * 1e3, 3), overshoot_mV=rnd((vx - v0) * 1e3, 3))
        for i, vs, vm, vx in zip(il, vss, vmin, vmax)])

    sel = [0, 4, 9, 12, 13, 14]                         # 10, 50, 100, 130, 140, 150 mA
    fig, (ax1, ax2) = figure(COL_W, 4.0, nrows=2, height_ratios=[1.2, 1])
    for c, k in reversed(list(zip(SER, sel))):         # largest load first: small needles stay visible
        ax1.plot(t, V[:, k] * 1e3, color=c, label=f"{il[k]*1e3:.0f} mA")
        j = np.argmin(np.where((t > 1) & (t < 5), V[:, k], np.inf))
        ax1.plot(t[j], V[j, k] * 1e3, marker="v", ms=3.5, color=c, mec="white", mew=0.4, zorder=5)
    h, lab = ax1.get_legend_handles_labels()
    h, lab = h[::-1], lab[::-1]
    order = [0, 3, 1, 4, 2, 5]                          # rows read 10/50/100, 130/140/150 mA
    ax1.legend([h[i] for i in order], [lab[i] for i in order], ncol=3, loc="lower right",
               title=r"$I_{LOAD}$ step from 150 µA (500 ns edge)", columnspacing=0.8, handlelength=1.4)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(745, 845)
    ax1.set_xlabel("Time [µs]\n(a)")
    ax1.set_ylabel(r"$V_{OUT}$ [mV]")

    for k, (y, lab_) in enumerate([(vmax, "Maximum (overshoot)"), (vss, "Steady state"),
                                   (vmin, "Minimum (undershoot)")]):
        ax2.plot(il * 1e3, y * 1e3, marker=MRK[k], color=SER[k], mfc="white", label=lab_)
    ax2.axvline(imax * 1e3, color=MUTED, ls=":", lw=0.8)
    txt(ax2, imax * 1e3 - 2, 795, rf"$I_{{LOAD,max}}$ ≈ {imax*1e3:.0f} mA", ha="right", va="center")
    ax2.set_xlabel("Load Current [mA]\n(b)")
    ax2.set_ylabel(r"$V_{OUT}$ [mV]")
    ax2.set_xlim(0, 160)
    ax2.set_ylim(775, 852)
    ax2.legend(loc="upper left", handlelength=1.6)
    save(fig, "fig1_max_load")
    return imax


# ------------------------------------------------- 2. Line regulation ----
def line_reg_metrics(vin, v, vr):
    """Regulated window and line regulation for one VREF.

    plateau : median VOUT for VIN in [VREF+0.2, 1.1] V
    window  : longest run with |VOUT - plateau| < 2 mV
    vin_min : VIN where VOUT reaches plateau - 2 mV (interpolated) -> dropout = vin_min - plateau
    lnr     : |least-squares slope| over the window, skipping its first 30 mV
    """
    plateau = np.median(v[(vin >= vr + 0.2 - EPS) & (vin <= 1.1 + EPS)])
    idx = np.where(np.abs(v - plateau) < 2e-3)[0]
    run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    k0 = run[0]
    if k0 > 0:
        vin_min = np.interp(plateau - 2e-3, [v[k0 - 1], v[k0]], [vin[k0 - 1], vin[k0]])
    else:
        vin_min = vin[k0]
    hi = vin[run[-1]]
    w = run[vin[run] >= vin[k0] + 0.03 - EPS]
    slope = np.polyfit(vin[w], v[w], 1)[0]
    return dict(plateau=plateau, vin_min=vin_min, vin_hi=hi, lnr=abs(slope) * 1e3,
                lnr_signed=slope * 1e3, fit_lo=vin[w[0]], fit_hi=vin[w[-1]])


def fig_line_reg():
    cols, d = load("linR1.csv")
    vin, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    rows = [line_reg_metrics(vin, v, vr) for vr, v in zip(vref, V.T)]
    log("[Line regulation]  regulated: |VOUT - plateau| < 2 mV; dropout at plateau - 2 mV "
        "(interpolated); LNR = |fit slope| over the window minus its first 30 mV")
    for vr, r in zip(vref, rows):
        log("  VREF=%3.0f mV  VOUT=%.1f mV  VIN %.3f-%.2f V  dropout=%.0f mV  LNR=%.1f mV/V "
            "(slope %+.1f, fit %.2f-%.2f V)" % (vr * 1e3, r["plateau"] * 1e3, r["vin_min"], r["vin_hi"],
                                               (r["vin_min"] - r["plateau"]) * 1e3, r["lnr"],
                                               r["lnr_signed"], r["fit_lo"], r["fit_hi"]))
    SUMMARY["line_reg"] = [dict(vref_V=rnd(vr), vout_mV=rnd(r["plateau"] * 1e3), vin_min_V=rnd(r["vin_min"]),
                                vin_hi_V=rnd(r["vin_hi"]), dropout_mV=rnd((r["vin_min"] - r["plateau"]) * 1e3, 3),
                                lnr_mV_per_V=rnd(r["lnr"], 2), fit_V=[rnd(r["fit_lo"]), rnd(r["fit_hi"])])
                           for vr, r in zip(vref, rows)]

    fig, ax = figure(COL_W, 2.75)
    ax.plot([0.5, 1.2], [0.5, 1.2], ls="--", color=INK, lw=0.8)
    txt(ax, 0.60, 0.645, r"$V_{OUT}$ = $V_{IN}$", rotation=37, ha="center", va="center", box=False)
    for k, (vr, v, r) in enumerate(zip(vref, V.T, rows)):
        ax.plot(vin, v, color=SER[k], marker=MRK[k], markevery=slice(k, len(vin) - 1, 5), mfc="white",
                label=f"{vr:.2f} V  ({r['lnr']:.1f})")
    vdo = [r["vin_min"] - r["plateau"] for r in rows]
    framed(ax, 0.03, 0.96, rf"$V_{{DO}}$ = {rng(np.array(vdo) * 1e3, '{:.0f}')} mV", va="top")
    ax.set_xlim(0.5, 1.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Input Voltage [V]")
    ax.set_ylabel("Output Voltage [V]")
    ax.legend(title=r"$V_{REF}$  (LNR [mV/V])", loc="lower right", borderpad=0.4,
              labelspacing=0.25, handlelength=1.8)
    save(fig, "fig2_line_regulation")


# ------------------------------------------------- 3. Load regulation ----
def load_reg_metrics(x, v, vr, i_ref=10e-3, tol=1e-3, i_lo_min=1e-3):
    """From a DC sweep VOUT(ILOAD) for one VREF.

    regulating : VOUT(i_ref) within 1 % of VREF (otherwise the curve is in dropout)
    imin       : lowest load from which VOUT stays within +-tol of VOUT(i_ref)
    imax       : first load where VOUT < 0.99*VREF (inf: never in the sweep, nan: from the start)
    ldr        : -slope of a least-squares fit VOUT(ILOAD) over [max(imin, 1 mA), min(imax, end)]
    """
    lx = np.log10(x)
    v_ref = np.interp(np.log10(i_ref), lx, v)
    regulating = abs(v_ref - vr) < 0.01 * vr
    below = np.where(v < 0.99 * vr)[0]
    if not below.size:
        imax = np.inf
    elif below[0] == 0:
        imax = np.nan
    else:
        k = below[0]
        imax = 10 ** np.interp(0.99 * vr, [v[k], v[k - 1]], [lx[k], lx[k - 1]])
    r = dict(regulating=bool(regulating), imax=imax, v_ref=v_ref, imin=None, ldr=None,
             i_lo=None, i_hi=None)
    if regulating:
        ok = np.abs(v - v_ref) < tol
        bad = np.where(~ok[: np.searchsorted(x, i_ref)])[0]
        imin = x[bad[-1] + 1] if bad.size else x[0]
        i_lo = max(imin, i_lo_min)
        i_hi = min(imax, x[-1]) if np.isfinite(imax) else x[-1]
        m = (x >= i_lo - EPS) & (x <= i_hi + EPS)
        r.update(imin=imin, i_lo=i_lo, i_hi=i_hi, ldr=-np.polyfit(x[m], v[m], 1)[0])  # V/A = mV/mA
    return r


def fig_load_reg():
    cols, d = load("loadR1.csv")
    x, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    if np.ptp(V, axis=0).max() < 1e-5:
        log("[Load regulation]  !! VOUT does not change over the sweep -> load not swept; re-simulate")
        return
    rs = [load_reg_metrics(x, v, vr) for vr, v in zip(vref, V.T)]
    log("[Load regulation]  DC sweep, VIN = %s V;  LDR = -fit slope from max(I_min, 1 mA) to the sweep end; "
        "I_min: VOUT within 1 mV of VOUT(10 mA); 0.99*VREF crossing for ILOAD,max" % COND["ldr_vin"])
    SUMMARY["load_reg"] = dict(vin_V=COND["ldr_vin"], sweep_end_mA=rnd(x[-1] * 1e3), rows=[])
    for vr, v, r in zip(vref, V.T, rs):
        cross = ("never (sweep end)" if np.isinf(r["imax"]) else "from the start" if np.isnan(r["imax"])
                 else "%.2f mA" % (r["imax"] * 1e3))
        if r["regulating"]:
            log("  VREF=%3.0f mV  VOUT(10mA)=%.2f mV  I_min=%.0f uA  LDR(%.0f-%.0f mA)=%.2f uV/mA  "
                "VOUT<0.99*VREF: %s" % (vr * 1e3, r["v_ref"] * 1e3, r["imin"] * 1e6, r["i_lo"] * 1e3,
                                        r["i_hi"] * 1e3, r["ldr"] * 1e3, cross))
        else:
            log("  VREF=%3.0f mV  dropout at VIN = %s V: VOUT %.2f -> %.2f mV; VOUT<0.99*VREF: %s"
                % (vr * 1e3, COND["ldr_vin"], v[0] * 1e3, v[-1] * 1e3, cross))
        SUMMARY["load_reg"]["rows"].append(dict(
            vref_V=rnd(vr), regulating=r["regulating"], vout_10mA_mV=rnd(r["v_ref"] * 1e3, 5),
            vout_first_mV=rnd(v[0] * 1e3, 5), vout_last_mV=rnd(v[-1] * 1e3, 5),
            imin_uA=rnd(r["imin"] * 1e6, 3) if r["imin"] else None,
            ldr_uV_per_mA=rnd(r["ldr"] * 1e3, 2) if r["ldr"] is not None else None,
            ldr_range_mA=[rnd(r["i_lo"] * 1e3), rnd(r["i_hi"] * 1e3)] if r["i_lo"] else None,
            below_99pct_from_mA=None if not np.isfinite(r["imax"]) else rnd(r["imax"] * 1e3, 3),
            below_99pct=cross))

    fig, ax = figure(COL_W, 2.75)
    drop = []
    for k, (vr, v, r) in enumerate(zip(vref, V.T, rs)):
        ls = "-" if r["regulating"] else (":" if not drop else "--")
        ax.semilogx(x * 1e3, v, color=SER[k], ls=ls, marker=MRK[k], mfc="white",
                    markevery=slice(2 + 4 * (k % 2), len(x) - 2, 8))
        if r["regulating"]:
            txt(ax, 0.3, r["v_ref"] + 0.007, rf"$V_{{REF}}$ = {vr:.2f} V", va="bottom")
            txt(ax, 40, r["v_ref"] - 0.007, f"LDR = {r['ldr']*1e3:.1f} µV/mA", ha="right", va="top")
            if r["imin"] > x[0]:
                vy = np.interp(np.log10(r["imin"]), np.log10(x), v)
                ax.plot(r["imin"] * 1e3, vy, "o", ms=3.2, color=INK, zorder=7)
                ax.annotate(rf"$I_{{min}}$ = {r['imin']*1e6:.0f} µA", (r["imin"] * 1e3, vy),
                            (r["imin"] * 1e3 * 0.42, vy - 0.025), fontsize=ANN, ha="center",
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK), zorder=7,
                            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
        else:
            drop.append((vr, ls))
    if drop:
        keyed = " / ".join(f"{vr:.2f} V ({'···' if ls == ':' else '– –'})" for vr, ls in drop)
        txt(ax, 0.013, 0.867, rf"$V_{{REF}}$ = {keyed}: dropout", va="bottom")
    log_axis(ax)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Output Voltage [V]")
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.set_ylim(0.66, 0.90)
    framed(ax, 0.97, 0.03, rf"$V_{{IN}}$ = {COND['ldr_vin']:.1f} V", ha="right", va="bottom")
    save(fig, "fig3_load_regulation")


# ---------------------------------------------------------------- 4. PSR ----
def fig_psr():
    cols, d = load("psr1.csv")
    f, P = d[:, 0], d[:, 1:]
    il = [param(c) for c in cols[1:]]
    fig, ax = figure(COL_W, 2.5)
    mk = list(np.searchsorted(f, 10 ** np.arange(1.5, 8.99, 0.5)))
    for k, (i, p) in enumerate(zip(il, P.T)):
        ax.semilogx(f, p, color=SER[k], marker=MRK[k], markevery=mk, mfc="white", label=ma_label(i))
    kp = int(np.argmax(P.max(0)))
    p = P[:, kp]
    txt(ax, f[p.argmax()] / 1.6, p.max() + 1.5, f"+{p.max():.1f} dB @ {f[p.argmax()]/1e6:.0f} MHz",
        ha="right", va="bottom")
    log_axis(ax, hz=True)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-70, 20)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSR [dB]")
    ax.legend(title=r"$I_{LOAD}$", loc="lower right", borderaxespad=0.8)
    if COND["psr_vin"] is not None:
        framed(ax, 0.03, 0.95, rf"$V_{{IN}}$ = {COND['psr_vin']:.2f} V" "\n"
               rf"$V_{{OUT}}$ = {COND['psr_vout']:.2f} V", va="top")
    save(fig, "fig4_psr")
    log("[PSR]  (DB20 v(out) for a unit AC source on VIN; negative = rejection)")
    pts = (1e3, 1e4, 1e5, 1e6, 1e7)
    SUMMARY["psr"] = []
    for i, p in zip(il, P.T):
        log(f"  ILOAD={ma_label(i)}: DC {p[0]:.1f} dB, "
            + ", ".join(f"{eng_hz(fx)}Hz {at(f, p, fx):.1f} dB" for fx in pts)
            + f", worst {p.max():+.1f} dB @ {eng_hz(f[p.argmax()])}Hz")
        SUMMARY["psr"].append(dict(iload_mA=rnd(i * 1e3), dc_dB=rnd(p[0]),
                                   **{f"at_{eng_hz(fx)}Hz_dB": rnd(at(f, p, fx)) for fx in pts},
                                   worst_dB=rnd(p.max()), worst_f_Hz=rnd(f[p.argmax()])))


# ------------------------------------------------ 5. Transient response ----
TS_BAND = 0.5e-3    # settling band for T_S: +-0.5 mV around the final value
T_EDGE = 0.5        # load-step edge time [us]


def tr_data():
    cols, d = load("tr1.csv")
    t = d[:, 0] * 1e6                                    # [us]
    return t, (d[:, 1], d[:, 2]), d[:, 3]


def tr_events(t, v):
    """Undershoot / overshoot of one VOUT trace, T_S, and tight boxes around each event (mV rel.)."""
    base = before(t, v, 1.0)
    m1, m2 = (t > 1) & (t < 5), (t > 5) & (t < 10)
    e = dict(base=base,
             us=base - v[m1].min(), t_us=t[m1][np.argmin(v[m1])],
             os=v[m2].max() - base, t_os=t[m2][np.argmax(v[m2])],
             ts_u=settle_time(t, v, 1.0, TS_BAND, 4.95),
             ts_o=settle_time(t, v, 5.0, TS_BAND, 9.95))
    for key, t0, ts in (("box_u", 1.0, e["ts_u"]), ("box_o", 5.0, e["ts_o"])):
        x0, x1 = t0 - 0.10, t0 + ts + 0.15
        w = (t >= x0) & (t <= x1)
        dv = (v[w] - base) * 1e3
        e[key] = (x0, x1, dv.min() - 0.3, dv.max() + 0.3)
    return e


def draw_box(ax, box):
    x0, x1, y0, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls=(0, (2, 1.5)), lw=0.8,
                           ec=INK, zorder=4))


def vline_arrow(ax, x, y0, y1):
    ax.annotate("", (x, y0), (x, y1), arrowprops=ARROW, zorder=6)


def fig_transient():
    t, vouts, il = tr_data()
    lo, hi = il.min(), il.max()
    log("[Transient]  ILOAD %s <-> %s, edge %.0f ns, T_S: from the step start until VOUT stays within "
        "+-%.1f mV (inclusive) of its final value" % (ma_label(lo), ma_label(hi), T_EDGE * 1e3, TS_BAND * 1e3))
    SUMMARY["transient"] = dict(i_lo_mA=rnd(lo * 1e3), i_hi_mA=rnd(hi * 1e3), edge_ns=T_EDGE * 1e3,
                                ts_band_mV=TS_BAND * 1e3, rows=[])
    evs = [tr_events(t, v) for v in vouts]
    for e in evs:
        log("  VOUT=%.2f V: undershoot %.1f mV, T_S %.0f ns | overshoot %.1f mV, T_S %.0f ns"
            % (e["base"], e["us"] * 1e3, e["ts_u"] * 1e3, e["os"] * 1e3, e["ts_o"] * 1e3))
        SUMMARY["transient"]["rows"].append(dict(
            vout_V=rnd(e["base"]), undershoot_mV=rnd(e["us"] * 1e3, 2), ts_under_ns=rnd(e["ts_u"] * 1e3, 3),
            overshoot_mV=rnd(e["os"] * 1e3, 2), ts_over_ns=rnd(e["ts_o"] * 1e3, 3)))

    fig, axs = figure(COL_W, 3.05, nrows=2, ncols=2, sharex=True, sharey="row", height_ratios=[1.5, 1])
    for j, (v, e) in enumerate(zip(vouts, evs)):
        av, ai = axs[0, j], axs[1, j]
        for a in (av, ai):
            a.grid(False)
        dv = (v - e["base"]) * 1e3
        av.plot(t, dv, color=VOUT_C)
        ai.plot(t, il * 1e3, color=ILOAD_C)
        draw_box(av, e["box_u"])
        draw_box(av, e["box_o"])
        xu = e["box_u"][1] + 0.35
        av.hlines(-e["us"] * 1e3, e["t_us"], xu + 0.1, color=MUTED, lw=0.6, ls=":")
        vline_arrow(av, xu, 0, -e["us"] * 1e3)
        txt(av, xu + 0.2, -e["us"] * 1e3 / 2, f"{e['us']*1e3:.1f} mV", ha="left", va="center")
        xo = e["box_o"][1] + 0.35
        av.hlines(e["os"] * 1e3, e["t_os"], xo + 0.1, color=MUTED, lw=0.6, ls=":")
        vline_arrow(av, xo, 0, e["os"] * 1e3)
        txt(av, xo + 0.2, e["os"] * 1e3 / 2, f"{e['os']*1e3:.1f} mV", ha="left", va="center")
        lab = rf"$V_{{OUT}}$ = {e['base']:.2f} V"
        if COND["tr_vin"] is not None:
            lab = rf"$V_{{IN}}$ = {COND['tr_vin'][j]:.2f} V" "\n" + lab
        txt(av, 0.3, 6.6, lab, va="top")
        txt(ai, 3.25, hi * 1e3 + 0.5, ma_label(hi), ha="center", va="bottom")
        txt(ai, 7.6, lo * 1e3 + 0.5, ma_label(lo), ha="center", va="bottom")
        txt(ai, 3.25, 4.6, r"$t_r$ = $t_f$" "\n" f"= {T_EDGE*1e3:.0f} ns", ha="center", va="center",
            linespacing=1.1)
        ai.set_xlim(0, 10)
        ai.xaxis.set_major_locator(MultipleLocator(2))
        ai.set_xlabel(f"Time [µs]\n({'ab'[j]})")
    axs[0, 0].set_ylim(-4.2, 7)
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(2))
    axs[1, 0].set_ylim(-1.5, 13)
    axs[1, 0].yaxis.set_major_locator(MultipleLocator(5))
    axs[0, 0].set_ylabel(r"Δ$V_{OUT}$ [mV]")
    axs[1, 0].set_ylabel(r"$I_{LOAD}$ [mA]")
    fig.align_ylabels(axs[:, 0])
    save(fig, "fig5_transient")

    # ---- zoom: undershoot | full | overshoot (VOUT = 0.80 V), double column ----
    v, e = vouts[0], evs[0]
    dv = (v - e["base"]) * 1e3
    us, os_ = e["us"] * 1e3, e["os"] * 1e3
    fig, axs = figure(DCOL_W, 2.75, nrows=2, ncols=3, sharex="col", height_ratios=[1.5, 1],
                      width_ratios=[1, 1.3, 1])
    av, ai = axs[0], axs[1]
    for k in range(3):
        for a in (av[k], ai[k]):
            a.grid(False)
        av[k].plot(t, dv, color=VOUT_C)
        ai[k].plot(t, il * 1e3, color=ILOAD_C)
        ai[k].set_ylim(-1.5, 13.5)
        ai[k].yaxis.set_major_locator(MultipleLocator(5))
        av[k].set_ylabel(r"Δ$V_{OUT}$ [mV]")
        ai[k].set_ylabel(r"$I_{LOAD}$ [mA]")
    titles = ("(a) Undershoot", "(b) Full transient", "(c) Overshoot")
    for k in range(3):
        ai[k].set_xlabel(f"Time [µs]\n{titles[k]}")

    # (b) full view with the two event boxes
    av[1].set_xlim(0, 10)
    av[1].set_ylim(-4.2, 7)
    av[1].yaxis.set_major_locator(MultipleLocator(2))
    av[1].xaxis.set_major_locator(MultipleLocator(2))
    for box, tag in ((e["box_u"], "(a)"), (e["box_o"], "(c)")):
        draw_box(av[1], box)
        txt(av[1], box[1] + 0.12, box[3], tag, ha="left", va="top")
    txt(av[1], 0.3, 6.6, rf"$V_{{OUT}}$ = {e['base']:.2f} V", va="top")
    txt(ai[1], 3.25, hi * 1e3 + 0.5, ma_label(hi), ha="center", va="bottom")
    txt(ai[1], 7.6, lo * 1e3 + 0.5, ma_label(lo), ha="center", va="bottom")
    txt(ai[1], 3.25, 4.6, r"$t_r$ = $t_f$" "\n" f"= {T_EDGE*1e3:.0f} ns", ha="center", va="center",
        linespacing=1.1)

    # (a) undershoot zoom
    a = av[0]
    a.set_xlim(0.88, 1.72)
    a.set_ylim(-us - 0.9, 0.8)
    a.xaxis.set_major_locator(MultipleLocator(0.2))
    a.yaxis.set_major_locator(MultipleLocator(1))
    a.hlines(0, 0.88, 1.06, color=MUTED, lw=0.6, ls="--")
    a.hlines(-us, 0.92, 1.12, color=MUTED, lw=0.6, ls=":")
    vline_arrow(a, 0.94, 0, -us)
    txt(a, 1.13, -us, rf"Δ$V_{{US}}$ = {us:.1f} mV", ha="left", va="center")
    yt = -us - 0.55
    a.annotate("", (1.0, yt), (1.0 + e["ts_u"], yt), arrowprops=ARROW, zorder=6)
    for xx in (1.0, 1.0 + e["ts_u"]):
        a.vlines(xx, yt, np.interp(xx, t, dv), color=MUTED, lw=0.5, ls=":")
    txt(a, 1.0 + e["ts_u"] + 0.03, yt, rf"$T_S$ = {e['ts_u']*1e3:.0f} ns", ha="left", va="center")
    txt(ai[0], 0.95, 0.10, r"Δ$I$/Δ$t$ = " f"{(hi-lo)*1e3:.2f} mA / {T_EDGE*1e3:.0f} ns",
        transform=ai[0].transAxes, ha="right", va="bottom")

    # (c) overshoot zoom
    a = av[2]
    a.set_xlim(4.92, 6.2)
    a.set_ylim(-0.9, os_ + 1.7)
    a.xaxis.set_major_locator(MultipleLocator(0.2))
    a.yaxis.set_major_locator(MultipleLocator(1))
    a.hlines(os_, e["t_os"], 5.8, color=MUTED, lw=0.6, ls=":")
    a.hlines(0, 5.6, 5.8, color=MUTED, lw=0.6, ls="--")
    vline_arrow(a, 5.76, 0, os_)
    txt(a, 5.8, os_ / 2, rf"Δ$V_{{OS}}$" "\n" f"= {os_:.1f} mV", ha="left", va="center", linespacing=1.1)
    yt = os_ + 0.8
    a.annotate("", (5.0, yt), (5.0 + e["ts_o"], yt), arrowprops=ARROW, zorder=6)
    for xx in (5.0, 5.0 + e["ts_o"]):
        a.vlines(xx, np.interp(xx, t, dv), yt, color=MUTED, lw=0.5, ls=":")
    txt(a, 5.0 + e["ts_o"] / 2, yt + 0.15, rf"$T_S$ = {e['ts_o']*1e3:.0f} ns", ha="center", va="bottom")
    save(fig, "fig6_transient_zoom")


# ----------------------------------------------- 6. Current efficiency ----
def fig_efficiency(imax):
    if (DATA / "iq1.csv").exists():
        return fig_efficiency_dc()
    cols, d = load("eff1.csv")
    t, I = d[:, 0] * 1e6, d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    iq_light = before(t, I[:, 0], 1.0)
    iq = before(t, I, 5.0)
    x = np.r_[150e-6, il]
    q = np.r_[iq_light, iq]
    eta = x / (x + q) * 100
    reg = x <= imax + EPS
    jp = int(np.argmax(np.where(reg, eta, -1)))
    # eta = IL / (IL + IQ) with IQ interpolated between the simulated points
    xc = np.logspace(np.log10(x[0]), np.log10(x[-1]), 200)
    qc = np.interp(np.log10(xc), np.log10(x), q)
    ec = xc / (xc + qc) * 100

    fig, ax = figure(COL_W, 2.5)
    ax.semilogx(xc * 1e3, ec, color=SER[0], lw=0.9, ls="--", label=r"Calc. ($I_Q$ interpolated)")
    show = np.isin(np.round(x * 1e3, 2), [0.15, 10, 20, 50, 100, 140, 150])
    ax.semilogx(x[show] * 1e3, eta[show], ls="none", marker=MRK[0], color=SER[0], mfc="white",
                label="Simulated")
    ax.plot(x[jp] * 1e3, eta[jp], marker=MRK[0], color=SER[0], ms=4)
    log_axis(ax)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(25, 102)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Current Efficiency [%]")
    ax.legend(loc="lower right", title=f"Peak = {eta[jp]:.2f}% @ {x[jp]*1e3:.0f} mA\n"
              rf"$I_Q$ = {rng(q[reg] * 1e6, '{:.0f}')} µA", handlelength=1.6)
    save(fig, "fig7_current_efficiency")
    log("[Current efficiency]  (transient eff1.csv: IQ = i(vvss) at the last sample before each step; "
        "peak within ILOAD,max = %.0f mA)" % (imax * 1e3))
    for a_, b_, e in zip(x, q, eta):
        log("  ILOAD=%8.3f mA  IQ=%.1f uA  eta=%.2f %%" % (a_ * 1e3, b_ * 1e6, e))
    SUMMARY["efficiency"] = dict(
        source="eff1.csv (transient)", peak_pct=rnd(eta[jp], 5), peak_at_mA=rnd(x[jp] * 1e3),
        iq_range_uA=[rnd(q[reg].min() * 1e6), rnd(q[reg].max() * 1e6)],
        rows=[dict(iload_mA=rnd(a_ * 1e3), iq_uA=rnd(b_ * 1e6), eta_pct=rnd(e, 5)) for a_, b_, e in zip(x, q, eta)])


def fig_efficiency_dc():
    """Current efficiency from the DC sweep: iq1.csv = i(vvss) vs ILOAD per VREF."""
    cols, d = load("iq1.csv")
    x, Q = d[:, 0], np.abs(d[:, 1:])
    vref = [param(c) for c in cols[1:]]
    lc, ld = load("loadR1.csv")                         # same sweep: stop where regulation ends
    imaxs = {round(param(c), 4): load_reg_metrics(ld[:, 0], v, param(c))["imax"]
             for c, v in zip(lc[1:], ld[:, 1:].T)}
    fig, ax = figure(COL_W, 2.5)
    log("[Current efficiency]  (DC sweep iq1.csv, eta = IL/(IL+IQ), up to the 0.99*VREF crossing)")
    best = (0, 0, 0)
    SUMMARY["efficiency"] = dict(source="iq1.csv (DC sweep)", rows=[])
    for k, (vr, q) in enumerate(zip(vref, Q.T)):
        im = imaxs.get(round(vr, 4), np.inf)
        m = x <= (im if np.isfinite(im) else x[-1])
        eta = x / (x + q) * 100
        ax.semilogx(x[m] * 1e3, eta[m], color=SER[k], marker=MRK[k], markevery=(k, 8), mfc="white",
                    label=f"{vr:.2f} V")
        if m.any():
            j = int(np.argmax(np.where(m, eta, -1)))
            best = max(best, (eta[j], x[j], vr))
            log("  VREF=%3.0f mV  IQ(100uA)=%.1f uA  IQ(10mA)=%.1f uA  peak eta=%.2f %% @ %.1f mA"
                % (vr * 1e3, np.interp(-4, np.log10(x), q) * 1e6, np.interp(-2, np.log10(x), q) * 1e6,
                   eta[j], x[j] * 1e3))
            SUMMARY["efficiency"]["rows"].append(dict(vref_V=rnd(vr), peak_pct=rnd(eta[j], 5),
                                                      peak_at_mA=rnd(x[j] * 1e3)))
    log_axis(ax)
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Current Efficiency [%]")
    ax.legend(title=rf"$V_{{REF}}$ (peak {best[0]:.2f}% @ {best[1]*1e3:.0f} mA)", loc="lower right",
              ncol=2, columnspacing=0.8, handlelength=1.6)
    save(fig, "fig7_current_efficiency")


# ---------------------------------------------------------- 7. Stability ----
def undo_phase_db(x):
    """Recover lstb phase from a column exported as 20*log10(|phase|).

    The sign is lost in such an export. The phase is positive down to its zero crossing,
    near the minimum of |phase|; the minimum sample itself gets whichever sign gives the
    smoother curve. After the crossing each point takes whichever of -|p| or -(360-|p|)
    keeps the curve monotonically falling (unwrapped below -180 deg). Only the part
    below the zero crossing is unambiguous; re-export lstb(phase) in degrees to be exact.
    """
    m = 10 ** (x / 20)
    k0 = int(np.argmin(m))
    if 0 < k0 < len(m) - 1:
        keep_pos = abs((m[k0 - 1] - m[k0]) - (m[k0] + m[k0 + 1])) < \
            abs((m[k0 - 1] + m[k0]) - (-m[k0] + m[k0 + 1]))
        k0 += int(keep_pos)
    ph = m.copy()
    prev = ph[k0 - 1] if k0 else 0.0
    prev = 0.0 if prev > 0 else prev
    for k in range(k0, len(m)):
        cands = [-m[k], -(360 - m[k])]
        ok = [c for c in cands if c <= prev + 1.0] or cands
        ph[k] = prev = min(ok, key=lambda c: abs(c - prev))
    return ph


STB = {
    "fig8a_stability_overall_loop": dict(tag="Overall loop", gain_ticks=(-60, 61, 20), gain_lim=(-62, 66),
                                         ph_ticks=(-270, 181, 90), ph_lim=(-315, 205)),
    "fig8b_stability_fast_loop": dict(tag="Fast loop", gain_ticks=(-20, 41, 20), gain_lim=(-24, 46),
                                      ph_ticks=(0, 181, 45), ph_lim=(-12, 198)),
}


def fig_stability(fname, outname, phase_db=False):
    cfg = STB[outname]
    cols, d = load(fname)
    f = d[:, 0]
    n = (d.shape[1] - 1) // 2
    G, P = d[:, 1:1 + n], d[:, 1 + n:]
    if phase_db:
        P = np.column_stack([undo_phase_db(p) for p in P.T])
    il = [param(c) for c in cols[1:1 + n]]
    fig, (ax1, ax2) = figure(COL_W, 3.3, nrows=2, sharex=True)
    log(f"[Stability: {fname}]  (HSPICE lstb: phase starts at 180 deg, PM = phase at the 0 dB crossing)")
    rows = SUMMARY.setdefault("stability", {}).setdefault(outname, [])
    pms = []
    for k, (i, g, p) in enumerate(zip(il, G.T, P.T)):
        light = k == 0
        kw = dict(color=SER[k], ls="--" if light else "-", zorder=4 if light else 3)
        ax1.semilogx(f, g, label=ma_label(i), **kw)
        ax2.semilogx(f, p, **kw)
        c = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        if c.size:
            c = c[0]
            lf = np.interp(0, [g[c + 1], g[c]], np.log10([f[c + 1], f[c]]))
            pm = np.interp(lf, np.log10(f[c:c + 2]), p[c:c + 2])
            pms.append((pm, i, 10 ** lf))
            ax1.plot(10 ** lf, 0, marker=MRK[k], color=SER[k], mfc="white", ms=3.5, zorder=6)
            ax2.plot(10 ** lf, pm, marker=MRK[k], color=SER[k], mfc="white", ms=3.5, zorder=6)
            log("  ILOAD=%s  DC gain=%.1f dB  UGF=%.1f MHz  PM=%.1f deg"
                % (ma_label(i), g[0], 10 ** lf / 1e6, pm))
            rows.append(dict(iload_mA=rnd(i * 1e3), dc_gain_dB=rnd(g[0]), ugf_MHz=rnd(10 ** lf / 1e6),
                             pm_deg=rnd(pm, 3)))
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_ylabel("Loop Gain [dB]")
    ax2.set_ylabel("Phase [°]")
    ax2.set_xlabel("Frequency [Hz]")
    log_axis(ax2, hz=True)
    ax2.set_xlim(f[0], f[-1])
    ax1.set_yticks(range(*cfg["gain_ticks"]))
    ax1.set_ylim(*cfg["gain_lim"])
    ax2.set_yticks(range(*cfg["ph_ticks"]))
    ax2.set_ylim(*cfg["ph_lim"])
    ax1.legend(title=r"$I_{LOAD}$", ncol=3, loc="lower left", columnspacing=0.6, handlelength=1.4,
               borderpad=0.3)
    framed(ax1, 0.97, 0.92, cfg["tag"], ha="right", va="top")
    worst = min(pms)
    framed(ax2, 0.03, 0.06 if outname.endswith("fast_loop") else 0.32,
           f"PM ≥ {worst[0]:.1f}° ({ma_label(worst[1])})\nUGF = {rng([u / 1e6 for _, _, u in pms])} MHz",
           va="bottom")
    fig.align_ylabels((ax1, ax2))
    save(fig, outname)


if __name__ == "__main__":
    imax = fig_maxload()
    fig_line_reg()
    fig_load_reg()
    fig_psr()
    fig_transient()
    fig_efficiency(imax)
    # stb1.csv phase columns were exported as dB of the phase -> undo it
    fig_stability("stb1.csv", "fig8a_stability_overall_loop", phase_db=True)
    fig_stability("f_stb1.csv", "fig8b_stability_fast_loop")
    (OUT / "metrics.txt").write_text("\n".join(metrics) + "\n")
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False) + "\n")
