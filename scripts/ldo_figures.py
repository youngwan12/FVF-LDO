"""FVF LDO paper figures (IEEE single-column, 3.5 in).

Usage:  python scripts/ldo_figures.py
Reads WaveView CSV exports from data/ and writes PDF + PNG to figures/.
Extracted metrics are printed and saved to figures/metrics.txt.
Layouts follow the reference figures in the 2026 LDO paper figure guide
(JSSC'22 / SOVC'26 style).
"""
from pathlib import Path
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator

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
COL_W = 3.5  # IEEE single column width [in]
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "lines.markeredgewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.4,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#222222",
    "legend.fancybox": False,
    "legend.handlelength": 2.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # editable TrueType text in PDF
    "ps.fonttype": 42,
})

# Distinct categorical colors (fixed order) + per-series markers for print/B&W
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MRK = ["o", "s", "^", "D", "v", "p", "h", "<"]
VOUT_C = "#c2187a"   # scope-style VOUT trace (magenta)
ILOAD_C = "#1f9e3a"  # scope-style ILOAD trace (green)
INK = "#222222"
MUTED = "#666666"
WARN = "#d62728"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf/.png")


def panel_label(ax, s, y=-0.30):
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top")


def eng_hz(x, _):
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
    return rf"{i * 1e6:.0f} $\mu$A" if i < 1e-3 else f"{i * 1e3:g} mA"


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


def settle_time(t, v, t0, band, t1):
    """Time from t0 until v stays within +-band of its value at t1."""
    m = (t >= t0) & (t <= t1)
    tt, vv = t[m], v[m]
    out = np.abs(vv - vv[-1]) > band
    return (tt[np.where(out)[0][-1] + 1] - t0) if out.any() else 0.0


metrics = []


def log(s):
    print(s)
    metrics.append(s)


# ------------------------------------------------- 1. Max load current ----
def fig_maxload():
    cols, d = load("maxload1.csv")
    t, V = d[:, 0], d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    v0 = at(t, V[:, 0], 0.95e-6)                       # light-load (150 uA) VOUT
    vss = np.array([at(t, V[:, k], 4.99e-6) for k in range(V.shape[1])])
    m = (t > 1e-6) & (t < 5e-6)
    vmin = V[m].min(0)

    log("[Max load]  light-load VOUT = %.1f mV" % (v0 * 1e3))
    for i, vs, vm in zip(il, vss, vmin):
        log("  ILOAD=%5.0f mA  VOUT(4.99us)=%.1f mV  dV_ss=%.1f mV  undershoot=%.1f mV"
            % (i * 1e3, vs * 1e3, (v0 - vs) * 1e3, (v0 - vm) * 1e3))

    sel = [0, 4, 9, 12, 13, 14]                         # 10,50,100,130,140,150 mA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.7),
                                   gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55})
    for c, k in zip(SER, sel):
        ax1.plot(t * 1e6, V[:, k] * 1e3, color=c, label=f"{il[k]*1e3:.0f} mA")
    ax1.set_xlim(0, 10)
    ax1.set_xlabel(r"Time [$\mu$s]")
    ax1.set_ylabel("Output Voltage [mV]")
    ax1.legend(title=r"$I_\mathrm{LOAD}$ (from 150 $\mu$A)", ncol=3, loc="lower right",
               title_fontsize=7, columnspacing=0.8, handlelength=1.4)
    ax1.set_ylim(755, 840)
    panel_label(ax1, "(a)")

    for k, (y, lab) in enumerate([(vss, "Steady state"), (vmin, "Minimum (undershoot)")]):
        ax2.plot(il * 1e3, y * 1e3, marker=MRK[k], color=SER[k], mfc="white", label=lab)
    ax2.set_xlabel("Load Current [mA]")
    ax2.set_ylabel("Output Voltage [mV]")
    ax2.set_xlim(0, 160)
    ax2.legend(loc="lower left")
    panel_label(ax2, "(b)")
    save(fig, "fig1_max_load")


# ------------------------------------------------- 2. Line regulation ----
def line_reg_metrics(vin, v, vr):
    """Regulated VIN window (|VOUT - plateau| < 2 mV) and LNR over it."""
    plateau = np.median(v[(vin >= vr + 0.2) & (vin <= 1.1)])
    idx = np.where(np.abs(v - plateau) < 2e-3)[0]
    run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    lo, hi = vin[run[0]], vin[run[-1]]
    # slope over the settled part of the window (skip the first 30 mV above dropout)
    w = run[vin[run] >= lo + 0.03]
    lnr = abs(np.polyfit(vin[w], v[w], 1)[0]) * 1e3
    return plateau, lo, hi, lnr


def fig_line_reg():
    cols, d = load("linR1.csv")
    vin, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot([0.5, 1.2], [0.5, 1.2], ls="--", color=INK, lw=0.8)
    ax.text(0.585, 0.64, r"$V_\mathrm{OUT}=V_\mathrm{IN}$", rotation=40, fontsize=7,
            ha="center", va="center")
    log("[Line regulation]  (regulated: |VOUT - plateau| < 2 mV, LNR = dVOUT/dVIN over it)")
    rows = []
    for k, (vr, v) in enumerate(zip(vref, V.T)):
        ax.plot(vin, v, color=SER[k], marker=MRK[k], markevery=5, mfc="white",
                label=f"{vr:.2f} V")
        rows.append(line_reg_metrics(vin, v, vr))
    for k, (vr, (pl, lo, hi, lnr)) in enumerate(zip(vref, rows)):
        ax.text(1.19, pl + 0.012, f"{lnr:.2f} mV/V", ha="right", va="bottom", fontsize=6.5)
        log("  VREF=%3.0f mV  VOUT=%.1f mV  VIN range %.2f-%.2f V  dropout=%.0f mV  LNR=%.2f mV/V"
            % (vr * 1e3, pl * 1e3, lo, hi, (lo - pl) * 1e3, lnr))
    vdo = [lo - pl for pl, lo, _, _ in rows]
    ax.text(0.77, 0.94, rf"$V_\mathrm{{DO}}$ = {min(vdo)*1e3:.0f}$-${max(vdo)*1e3:.0f} mV",
            fontsize=7, ha="left")
    ax.set_xlim(0.5, 1.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Input Voltage [V]")
    ax.set_ylabel("Output Voltage [V]")
    ax.legend(title=r"$V_\mathrm{REF}$", title_fontsize=7, loc="upper left", ncol=1,
              borderpad=0.4, labelspacing=0.25)
    save(fig, "fig2_line_regulation")


# ------------------------------------------------- 3. Load regulation ----
def load_reg_metrics(x, v, vr, i_ref=10e-3, tol=1e-3, i_lo_min=1e-3):
    """From a DC sweep VOUT(ILOAD) for one VREF.

    regulating : VOUT(i_ref) within 1 % of VREF (otherwise the curve is in dropout)
    imin       : lowest load from which VOUT stays within +-tol of VOUT(i_ref)
    imax       : first load above imin where VOUT < 0.99*VREF (inf: not reached in sweep)
    ldr        : (VOUT(i_lo) - VOUT(i_hi)) / (i_hi - i_lo) [mV/mA],
                 i_lo = max(imin, 1 mA), i_hi = min(imax, sweep end)
    """
    lx = np.log10(x)
    v_ref = np.interp(np.log10(i_ref), lx, v)
    regulating = abs(v_ref - vr) < 0.01 * vr
    ok = np.abs(v - v_ref) < tol
    bad = np.where(~ok[: np.searchsorted(x, i_ref)])[0]
    imin = x[bad[-1] + 1] if bad.size else x[0]
    below = np.where((v < 0.99 * vr) & (x > imin))[0]
    if below.size:
        k = below[0]
        imax = 10 ** np.interp(0.99 * vr, [v[k], v[k - 1]], [lx[k], lx[k - 1]])
    else:
        imax = np.inf
    i_hi = min(imax, x[-1])
    i_lo = max(imin, i_lo_min)
    v_lo, v_hi = np.interp(np.log10(i_lo), lx, v), np.interp(np.log10(i_hi), lx, v)
    ldr = (v_lo - v_hi) / (i_hi - i_lo)                  # V/A == mV/mA
    return dict(regulating=regulating, imin=imin, imax=imax, i_lo=i_lo, i_hi=i_hi,
                v_lo=v_lo, v_hi=v_hi, ldr=ldr)


def fig_load_reg():
    cols, d = load("loadR1.csv")
    x, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    if np.ptp(V, axis=0).max() < 1e-5:
        log("[Load regulation]  !! VOUT does not change over the sweep -> load not swept; re-simulate")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    log("[Load regulation]  DC sweep, VIN = %s V;  LDR = dVOUT/dILOAD from I_min to "
        "min(ILOAD,max, sweep end), from >= 1 mA; I_min: VOUT within 1 mV of VOUT(10 mA); "
        "ILOAD,max: VOUT < 0.99*VREF" % COND["ldr_vin"])
    drop = []
    for k, (vr, v) in enumerate(zip(vref, V.T)):
        r = load_reg_metrics(x, v, vr)
        ls = "-" if r["regulating"] else "--"
        ax.semilogx(x * 1e3, v, color=SER[k], ls=ls, marker=MRK[k], markevery=(k, 8),
                    mfc="white")
        if r["regulating"]:
            ax.text(0.3, r["v_hi"] + 0.006,
                    rf"$V_\mathrm{{REF}}$ = {vr:.2f} V", fontsize=6.5, va="bottom")
            ax.text(x[-1] * 1e3 * 0.75, r["v_hi"] + 0.006, f"{r['ldr']*1e3:.1f} $\\mu$V/mA",
                    fontsize=6.5, ha="right", va="bottom")
            if r["imin"] > x[0]:
                ax.annotate(rf"$I_\mathrm{{min}}$ = {r['imin']*1e6:.0f} $\mu$A",
                            (r["imin"] * 1e3, np.interp(np.log10(r["imin"]), np.log10(x), v)), (r["imin"] * 1e3 * 2.5, r["v_lo"] - 0.022),
                            fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
            im = ">%.0f mA (sweep end)" % (x[-1] * 1e3) if np.isinf(r["imax"]) else "%.1f mA" % (r["imax"] * 1e3)
            log("  VREF=%3.0f mV  VOUT=%.2f mV  I_min=%.0f uA  LDR(%.0f-%.0f mA)=%.4f mV/mA  ILOAD,max=%s"
                % (vr * 1e3, r["v_hi"] * 1e3, r["imin"] * 1e6, r["i_lo"] * 1e3, r["i_hi"] * 1e3,
                   r["ldr"], im))
        else:
            drop.append(vr)
            log("  VREF=%3.0f mV  dropout at VIN = %s V: VOUT %.1f -> %.1f mV over the sweep"
                % (vr * 1e3, COND["ldr_vin"], v[0] * 1e3, v[-1] * 1e3))
    if drop:
        v_d = V[:, [vref.index(vr) for vr in drop]].mean(1)
        ax.text(x[0] * 1e3 * 1.3, v_d[0] + 0.006,
                rf"$V_\mathrm{{REF}}$ = {', '.join(f'{vr:.2f}' for vr in drop)} V (dropout)",
                fontsize=6.5, va="bottom")
    log_axis(ax)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Output Voltage [V]")
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.set_ylim(0.67, 0.89)
    ax.text(0.98, 0.03, rf"$V_\mathrm{{IN}}$ = {COND['ldr_vin']:.1f} V",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.6))
    save(fig, "fig3_load_regulation")


# ---------------------------------------------------------------- 4. PSR ----
def fig_psr():
    cols, d = load("psr1.csv")
    f, P = d[:, 0], d[:, 1:]
    il = [param(c) for c in cols[1:]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    mk = np.searchsorted(f, 10 ** np.arange(1, 9.01, 0.5))
    for k, (i, p) in enumerate(zip(il, P.T)):
        ax.semilogx(f, p, color=SER[k], marker=MRK[k], markevery=list(mk), mfc="white",
                    label=rf"$I_\mathrm{{Load}}$ = {ma_label(i)}")
    log_axis(ax, hz=True)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-60, 10)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSR [dB]")
    leg = ax.legend(loc="upper left")
    if COND["psr_vin"] is not None:
        ax.text(0.42, 0.95, rf"$V_\mathrm{{IN}}$ = {COND['psr_vin']:.2f} V" "\n"
                rf"$V_\mathrm{{OUT}}$ = {COND['psr_vout']:.2f} V",
                transform=ax.transAxes, va="top", fontsize=7)
    save(fig, "fig4_psr")
    log("[PSR]")
    for i, p in zip(il, P.T):
        pts = ", ".join(f"{eng_hz(fx, 0)}Hz: {at(f, p, fx):.1f} dB" for fx in (1e3, 1e5, 1e6, 1e7))
        log(f"  ILOAD={ma_label(i).replace('$\\mu$', 'u')}: DC {p[0]:.1f} dB, {pts}, "
            f"worst {p.max():.1f} dB @ {eng_hz(f[p.argmax()], 0)}Hz")


# ------------------------------------------------ 5. Transient response ----
def tr_data():
    cols, d = load("tr1.csv")
    t = d[:, 0] * 1e6
    return t, (d[:, 1], d[:, 2]), d[:, 3]


def scope_axes(ax):
    ax.set_xticks(np.linspace(*ax.get_xlim(), 11))
    ax.set_yticks(np.linspace(0, 1, 9))
    ax.tick_params(labelleft=False, labelbottom=False, length=0)
    ax.grid(True, which="major", ls=":", color="#bbbbbb", lw=0.4)
    ax.grid(False, which="minor")
    ax.set_ylim(0, 1)


def scale_bar(ax, x, y, dx, dy, tx, ty):
    """Scope-style scale bars (data coords): vertical dy from y up, horizontal dx below it."""
    kw = dict(arrowstyle="<->", lw=0.6, color=INK, shrinkA=0, shrinkB=0, mutation_scale=5)
    ax.annotate("", (x, y), (x, y + dy), arrowprops=kw)
    ax.text(x - 0.01 * np.ptp(ax.get_xlim()), y + dy / 2, ty, ha="right", va="center", fontsize=6.5)
    yh = y - 0.04
    ax.annotate("", (x - dx, yh), (x, yh), arrowprops=kw)
    ax.text(x - dx / 2, yh - 0.015, tx, ha="center", va="top", fontsize=6.5)


def fig_transient():
    t, vouts, il = tr_data()
    lo, hi = il.min(), il.max()
    t_edge = 0.5
    log("[Transient]  ILOAD %.0f uA <-> %.0f mA, edge %.0f ns" % (lo * 1e6, hi * 1e3, t_edge * 1e3))

    vdiv, tdiv = 2e-3, 1.0                              # 2 mV/div, 1 us/div (8 x 10 div)
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.9), gridspec_kw={"wspace": 0.05})
    for j, (ax, v) in enumerate(zip(axs, vouts)):
        base = at(t, v, 0.95)
        us = base - v[(t > 1) & (t < 5)].min()
        os_ = v[(t > 5) & (t < 10)].max() - base
        ts_u = settle_time(t, v, 1.0, 0.5e-3, 4.95)
        ts_o = settle_time(t, v, 5.0, 0.5e-3, 9.95)
        log("  VOUT=%.2f V: undershoot %.1f mV, Ts %.0f ns | overshoot %.1f mV, Ts %.0f ns  (Ts: +-0.5 mV)"
            % (base, us * 1e3, ts_u * 1e3, os_ * 1e3, ts_o * 1e3))
        ax.set_xlim(0, 10)
        yv = 0.62 + (v - base) / vdiv / 8                # VOUT: 2 mV/div around 0.62
        yi = 0.08 + (il - lo) / (hi - lo) * 0.27         # ILOAD in the lower part
        ax.plot(t, yv, color=VOUT_C, lw=0.9)
        ax.plot(t, yi, color=ILOAD_C, lw=1.1)
        scope_axes(ax)
        for x0 in (0.85, 4.85):                           # dotted boxes on the step events
            ax.add_patch(Rectangle((x0, 0.40), 0.9, 0.45, fill=False, ls=":", lw=0.6, ec=INK))
        lab = rf"$V_\mathrm{{OUT}}$ = {base:.2f} V"
        if COND["tr_vin"] is not None:
            lab = rf"$V_\mathrm{{IN}}$ = {COND['tr_vin'][j]:.2f} V" "\n" + lab
        ax.text(0.25, 0.97, lab, va="top", fontsize=7)
        ax.text(3.25, yi.max() + 0.02, ma_label(hi), fontsize=7, ha="center", va="bottom")
        ax.text(7.6, 0.10, ma_label(lo), fontsize=6.5, ha="center", va="bottom")
        ax.text(3.25, 0.21, rf"$T_\mathrm{{Edge}}$ = {t_edge*1e3:.0f} ns", fontsize=6.5,
                ha="center", va="center")
        scale_bar(ax, 9.6, 0.40, tdiv, 1 / 8, rf"{tdiv:g} $\mu$s", f"{vdiv*1e3:g} mV")
        panel_label(ax, f"({'ab'[j]})", y=-0.04)
    save(fig, "fig5_transient")

    # zoom: undershoot | full | overshoot  (first column, VOUT = 0.80 V)
    v = vouts[0]
    base = at(t, v, 0.95)
    fig = plt.figure(figsize=(COL_W, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.06)
    axl, axc, axr = (fig.add_subplot(gs[0, k]) for k in range(3))
    windows = [(axl, 0.9, 2.1, 1.0), (axr, 4.9, 6.1, 5.0)]
    axc.set_xlim(0, 10)
    axc.plot(t, 0.62 + (v - base) / vdiv / 8, color=VOUT_C, lw=0.8)
    axc.plot(t, 0.08 + (il - lo) / (hi - lo) * 0.27, color=ILOAD_C, lw=1.0)
    scope_axes(axc)
    for x0 in (0.85, 4.85):
        axc.add_patch(Rectangle((x0, 0.40), 0.9, 0.45, fill=False, ls=":", lw=0.6, ec=INK))
    axc.text(0.3, 0.97, rf"$V_\mathrm{{OUT}}$ = {base:.2f} V", va="top", fontsize=6.5)
    axc.text(3.25, 0.37, ma_label(hi), fontsize=6.5, ha="center", va="bottom")
    scale_bar(axc, 9.5, 0.40, 2.0, 1 / 8, r"2 $\mu$s", "2 mV")

    zv = 2e-3                                           # zoom: 2 mV/div, 120 ns/div
    for ax, t0, t1, ts in windows:
        m = (t >= t0) & (t <= t1)
        ax.set_xlim(t0, t1)
        ax.plot(t[m], 0.60 + (v[m] - base) / zv / 8, color=VOUT_C, lw=0.9)
        ax.plot(t[m], 0.08 + (il[m] - lo) / (hi - lo) * 0.27, color=ILOAD_C, lw=1.0)
        scope_axes(ax)
    us = base - v[(t > 1) & (t < 5)].min()
    os_ = v[(t > 5) & (t < 10)].max() - base
    ts_u = settle_time(t, v, 1.0, 0.5e-3, 4.95)
    ts_o = settle_time(t, v, 5.0, 0.5e-3, 9.95)
    tmin = t[(t > 1) & (t < 5)][np.argmin(v[(t > 1) & (t < 5)])]
    tmax = t[(t > 5) & (t < 10)][np.argmax(v[(t > 5) & (t < 10)])]
    arr = dict(arrowstyle="<->", lw=0.6, color=INK, shrinkA=0, shrinkB=0, mutation_scale=5)
    y0, yd = 0.60, 0.60 - us / zv / 8
    axl.annotate("", (tmin, y0), (tmin, yd), arrowprops=arr)
    axl.text(tmin + 0.07, yd + 0.01, rf"$V_\mathrm{{Droop}}$ = {us*1e3:.1f} mV", fontsize=6, va="top")
    axl.annotate("", (1.0, 0.85), (1.0 + ts_u, 0.85), arrowprops=arr)
    axl.text(1.0 + ts_u / 2, 0.87, rf"$T_\mathrm{{S}}$ = {ts_u*1e3:.0f} ns", fontsize=6, ha="center", va="bottom")
    axl.text(2.05, 0.08, r"$\Delta I/\Delta T$ =" "\n" rf"{(hi-lo)*1e3:.2f} mA/{t_edge*1e3:.0f} ns",
             fontsize=6, ha="right", va="bottom")
    yo = 0.60 + os_ / zv / 8
    axr.annotate("", (tmax, y0), (tmax, yo), arrowprops=arr)
    axr.text(tmax + 0.05, yo, rf"$V_\mathrm{{Overshoot}}$" "\n" rf"= {os_*1e3:.1f} mV", fontsize=6, va="top")
    axr.annotate("", (5.0, 0.50), (5.0 + ts_o, 0.50), arrowprops=arr)
    axr.text(5.0 + ts_o / 2, 0.48, rf"$T_\mathrm{{S}}$ = {ts_o*1e3:.0f} ns", fontsize=6, ha="center", va="top")
    scale_bar(axr, 6.05, 0.22, 0.2, 1 / 8, "200 ns", "2 mV")
    save(fig, "fig6_transient_zoom")


# ----------------------------------------------- 6. Current efficiency ----
def fig_efficiency():
    if (DATA / "iq1.csv").exists():
        return fig_efficiency_dc()
    cols, d = load("eff1.csv")
    t, I = d[:, 0], d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    iq_light = at(t, I[:, 0], 0.95e-6)
    iq = np.array([at(t, I[:, k], 4.99e-6) for k in range(I.shape[1])])
    x = np.r_[150e-6, il]
    q = np.r_[iq_light, iq]
    eta = x / (x + q) * 100
    # eta = IL / (IL + IQ) with IQ interpolated between the simulated points
    xc = np.logspace(np.log10(x[0]), np.log10(x[-1]), 200)
    qc = np.interp(np.log10(xc), np.log10(x), q)
    ec = xc / (xc + qc) * 100

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.semilogx(xc * 1e3, ec, color=SER[0], lw=1.0, label=r"$I_\mathrm{L}/(I_\mathrm{L}+I_\mathrm{Q})$")
    ax.semilogx(x * 1e3, eta, ls="none", marker=MRK[0], color=SER[0], mfc="white",
                label="Simulated points")
    log_axis(ax)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(30, 102)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Current Efficiency [%]")
    ax.text(0.97, 0.38, f"Peak Efficiency = {eta.max():.2f}%\n"
            rf"@ $I_\mathrm{{LOAD}}$ = {x[eta.argmax()]*1e3:.0f} mA" "\n"
            rf"$I_\mathrm{{Q}}$ = {q.min()*1e6:.0f}$-${q.max()*1e6:.0f} $\mu$A",
            transform=ax.transAxes, ha="right", va="center", fontsize=7,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.6))
    ax.legend(loc="lower right")
    save(fig, "fig7_current_efficiency")
    log("[Current efficiency]  (transient data eff1.csv)")
    for a_, b_, e in zip(x, q, eta):
        log("  ILOAD=%8.3f mA  IQ=%.1f uA  eta=%.2f %%" % (a_ * 1e3, b_ * 1e6, e))


def fig_efficiency_dc():
    """Current efficiency from the DC sweep: iq1.csv = i(vvss) vs ILOAD per VREF."""
    cols, d = load("iq1.csv")
    x, Q = d[:, 0], np.abs(d[:, 1:])
    vref = [param(c) for c in cols[1:]]
    # stop each curve where the LDO leaves regulation (from loadR1.csv, same sweep)
    lc, ld = load("loadR1.csv")
    imaxs = {}
    for c, v in zip(lc[1:], ld[:, 1:].T):
        imaxs[round(param(c), 4)] = load_reg_metrics(ld[:, 0], v, param(c))["imax"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    log("[Current efficiency]  (DC sweep iq1.csv, eta = IL/(IL+IQ), up to ILOAD,max)")
    best = (0, 0, 0)
    for k, (vr, q) in enumerate(zip(vref, Q.T)):
        im = imaxs.get(round(vr, 4), np.inf)
        m = x <= (im if np.isfinite(im) else x[-1])
        eta = x / (x + q) * 100
        ax.semilogx(x[m] * 1e3, eta[m], color=SER[k], marker=MRK[k], markevery=6, mfc="white",
                    label=f"{vr:.2f} V")
        if m.any():
            j = np.argmax(np.where(m, eta, -1))
            best = max(best, (eta[j], x[j], vr))
            log("  VREF=%3.0f mV  IQ(100uA)=%.1f uA  IQ(100mA)=%.1f uA  peak eta=%.2f %% @ %.1f mA"
                % (vr * 1e3, np.interp(-4, np.log10(x), q) * 1e6,
                   np.interp(-1, np.log10(x), q) * 1e6, eta[j], x[j] * 1e3))
    log_axis(ax)
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Current Efficiency [%]")
    ax.legend(title=r"$V_\mathrm{REF}$", title_fontsize=7, loc="lower right", ncol=2,
              columnspacing=0.8, handlelength=1.6)
    ax.text(0.03, 0.95, f"Peak Efficiency = {best[0]:.2f}%\n"
            rf"@ $I_\mathrm{{LOAD}}$ = {best[1]*1e3:.0f} mA",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.6))
    save(fig, "fig7_current_efficiency")


# ---------------------------------------------------------- 7. Stability ----
def undo_phase_db(x):
    """Recover lstb phase from a column exported as 20*log10(|phase|).

    Sign is lost in such an export: phase is positive down to the zero crossing
    (minimum of |phase|); after it, each point takes whichever of -|p| or
    -(360-|p|) keeps the curve monotonically falling (unwrapped below -180 deg).
    """
    m = 10 ** (x / 20)
    k0 = int(np.argmin(m))
    ph = m.copy()
    prev = 0.0
    for k in range(k0, len(m)):
        cands = [-m[k], -(360 - m[k])]
        ok = [c for c in cands if c <= prev + 1.0] or cands
        ph[k] = prev = min(ok, key=lambda c: abs(c - prev))
    return ph


def fig_stability(fname, outname, phase_db=False):
    cols, d = load(fname)
    f = d[:, 0]
    n = (d.shape[1] - 1) // 2
    G, P = d[:, 1:1 + n], d[:, 1 + n:]
    if phase_db:
        P = np.column_stack([undo_phase_db(p) for p in P.T])
    il = [param(c) for c in cols[1:1 + n]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True,
                                   gridspec_kw={"hspace": 0.1})
    log(f"[Stability: {fname}]  (PM read as lstb phase at 0 dB crossing)")
    pms = []
    for k, (i, g, p) in enumerate(zip(il, G.T, P.T)):
        kw = dict(color=SER[k])
        ax1.semilogx(f, g, label=ma_label(i), **kw)
        ax2.semilogx(f, p, **kw)
        c = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        if c.size:
            c = c[0]
            lf = np.interp(0, [g[c + 1], g[c]], np.log10([f[c + 1], f[c]]))
            pm = np.interp(lf, np.log10(f[c:c + 2]), p[c:c + 2])
            pms.append(pm)
            log("  ILOAD=%s  DC gain=%.1f dB  UGF=%.1f MHz  PM=%.1f deg"
                % (ma_label(i).replace("$\\mu$", "u"), g[0], 10 ** lf / 1e6, pm))
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_ylabel("Loop Gain [dB]")
    ax2.set_ylabel(r"Phase [$^\circ$]")
    ax2.set_xlabel("Frequency [Hz]")
    log_axis(ax2, hz=True)
    ax2.set_xlim(f[0], f[-1])
    ax1.legend(title=r"$I_\mathrm{LOAD}$", title_fontsize=7, ncol=2, loc="lower left",
               columnspacing=0.8, handlelength=1.6)
    ax2.text(0.03, 0.08, f"PM = {min(pms):.1f}$-${max(pms):.1f}$^\\circ$", transform=ax2.transAxes,
             fontsize=7, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.6))
    fig.align_ylabels((ax1, ax2))
    save(fig, outname)


if __name__ == "__main__":
    fig_maxload()
    fig_line_reg()
    fig_load_reg()
    fig_psr()
    fig_transient()
    fig_efficiency()
    # stb1.csv phase columns were exported as dB of the phase -> undo it
    fig_stability("stb1.csv", "fig8a_stability_overall_loop", phase_db=True)
    fig_stability("f_stb1.csv", "fig8b_stability_fast_loop")
    (OUT / "metrics.txt").write_text("\n".join(metrics) + "\n")
