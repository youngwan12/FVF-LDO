"""FVF LDO paper figures (IEEE single-column, 3.5 in).

Usage:  python scripts/ldo_figures.py
Reads WaveView CSV exports from data/ and writes PDF + PNG to figures/.
Extracted metrics are printed and saved to figures/metrics.txt.
"""
from pathlib import Path
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

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
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#bbbbbb",
    "legend.fancybox": False,
    "legend.handlelength": 1.8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # editable TrueType text in PDF
    "ps.fonttype": 42,
})

# Categorical (identity) and ordinal (ordered sweep) colors
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK = "#222222"
MUTED = "#666666"


def ordinal(n):
    """n colors light->dark from the blue ramp (step 250..700)."""
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf/.png")


def panel_label(ax, s):
    ax.text(0.5, -0.36, s, transform=ax.transAxes, ha="center", va="top")


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

    # (a) waveforms for a subset of loads
    sel = [0, 4, 9, 12, 13, 14]                         # 10,50,100,130,140,150 mA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.6),
                                   gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55})
    for c, k in zip(ordinal(len(sel)), sel):
        ax1.plot(t * 1e6, V[:, k] * 1e3, color=c, label=f"{il[k]*1e3:.0f} mA")
    ax1.set_xlim(0, 10)
    ax1.set_xlabel(r"Time ($\mu$s)")
    ax1.set_ylabel(r"$V_\mathrm{OUT}$ (mV)")
    ax1.legend(title=r"$I_\mathrm{LOAD}$ (150 $\mu$A $\rightarrow$)", ncol=3, loc="lower right",
               title_fontsize=7, columnspacing=0.8, handlelength=1.2)
    ax1.set_ylim(755, 840)
    panel_label(ax1, "(a)")

    # (b) steady-state VOUT vs ILOAD
    ax2.plot(il * 1e3, vss * 1e3, "o-", color=CAT[0], ms=3, mfc="white", mew=0.8,
             label="Steady state")
    ax2.plot(il * 1e3, vmin * 1e3, "s--", color=CAT[1], ms=3, mfc="white", mew=0.8,
             label="Minimum (undershoot)")
    ax2.set_xlabel(r"$I_\mathrm{LOAD}$ (mA)")
    ax2.set_ylabel(r"$V_\mathrm{OUT}$ (mV)")
    ax2.set_xlim(0, 160)
    ax2.legend(loc="lower left")
    panel_label(ax2, "(b)")
    save(fig, "fig1_max_load")


# ------------------------------------------------- 2. Line regulation ----
def fig_line_reg():
    cols, d = load("linR1.csv")
    vin, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.plot(vin, vin, ls=":", color=MUTED, lw=0.8)
    ax.text(0.62, 0.66, r"$V_\mathrm{OUT}=V_\mathrm{IN}$", color=MUTED, rotation=38,
            fontsize=7, ha="center", va="center")
    ys = []
    for c, vr, v in zip(ordinal(len(vref)), vref, V.T):
        ax.plot(vin, v, color=c)
        ys.append(v[-1])
    for k in range(1, len(ys)):          # keep direct labels >= 35 mV apart
        if ys[k] - ys[k - 1] < 0.035:
            ys[k - 1] = ys[k] - 0.035
    for vr, y in zip(vref, ys):
        ax.text(1.205, y, f"{vr*1e3:.0f} mV", va="center", fontsize=7, color=INK)
    ax.set_xlim(0.5, 1.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel(r"$V_\mathrm{IN}$ (V)")
    ax.set_ylabel(r"$V_\mathrm{OUT}$ (V)")
    ax.text(1.205, 0.96, r"$V_\mathrm{REF}$", fontsize=7, color=INK)
    save(fig, "fig2_line_regulation")

    log("[Line regulation]  (regulated: |VOUT - plateau| < 2 mV, LNR = fit slope)")
    for vr, v in zip(vref, V.T):
        plateau = np.median(v[(vin >= vr + 0.2) & (vin <= 1.1)])
        idx = np.where(np.abs(v - plateau) < 2e-3)[0]
        run = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        run = max(run, key=len)
        lo, hi = vin[run[0]], vin[run[-1]]
        lnr = np.polyfit(vin[run], v[run], 1)[0] * 1e3
        log("  VREF=%3.0f mV  VOUT=%.1f mV  VIN range %.2f-%.2f V  "
            "dropout=%.0f mV  LNR=%.2f mV/V"
            % (vr * 1e3, plateau * 1e3, lo, hi, (lo - plateau) * 1e3, lnr))


# ------------------------------------------------- 3. Load regulation ----
def fig_load_reg():
    cols, d = load("loadR1.csv")
    x, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for c, vr, v in zip(ordinal(len(vref)), vref, V.T):
        ax.semilogx(x * 1e3, v * 1e3, color=c, label=f"{vr*1e3:.0f} mV")
    ax.set_xlabel(r"$I_\mathrm{LOAD}$ (mA)")
    ax.set_ylabel(r"$V_\mathrm{OUT}$ (mV)")
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.legend(title=r"$V_\mathrm{REF}$", title_fontsize=7, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig3_load_regulation")
    spread = np.ptp(V, axis=0) * 1e3
    log("[Load regulation]  VOUT spread over sweep per VREF (mV): "
        + ", ".join(f"{s:.2f}" for s in spread)
        + "  <- sweep shows no load dependence; check netlist")


# ---------------------------------------------------------------- 4. PSR ----
def fig_psr():
    cols, d = load("psr1.csv")
    f, P = d[:, 0], d[:, 1:]
    labels = [r"$I_\mathrm{LOAD}$ = 100 $\mu$A", r"$I_\mathrm{LOAD}$ = 50 mA"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for c, ls, lab, p in zip(CAT, ["-", "--"], labels, P.T):
        ax.semilogx(f, p, color=c, ls=ls, label=lab)
    ax.set_xlim(f[0], f[-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSR (dB)")
    ax.legend(loc="upper left")
    save(fig, "fig4_psr")
    log("[PSR]")
    for name, p in zip(["100uA", "50mA"], P.T):
        pts = ", ".join(f"{fx:.0e} Hz: {at(f, p, fx):.1f} dB" for fx in (1e3, 1e6, 1e7))
        log(f"  {name}: DC {p[0]:.1f} dB, {pts}, worst {p.max():.1f} dB @ {f[p.argmax()]:.2e} Hz")


# ------------------------------------------------ 5. Transient response ----
def fig_transient():
    cols, d = load("tr1.csv")
    t, v8, v9, il = d[:, 0] * 1e6, d[:, 1], d[:, 2], d[:, 3]
    fig, axs = plt.subplots(3, 1, figsize=(COL_W, 3.6), sharex=True,
                            gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1, 0.8]})
    log("[Transient]  ILOAD %.0f uA <-> %.0f mA" % (il.min() * 1e6, il.max() * 1e3))
    for ax, v, c, name in zip(axs[:2], (v8, v9), CAT[:2], ("0.8", "0.9")):
        base = at(t, v, 0.95)
        us = base - v[(t > 1) & (t < 5)].min()
        os_ = v[(t > 5) & (t < 10)].max() - base
        ax.plot(t, v * 1e3, color=c)
        ax.set_ylabel(r"$V_\mathrm{OUT}$ (mV)")
        ax.text(0.98, 0.9, rf"$V_\mathrm{{OUT}}\approx${name} V", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
        ax.annotate(f"{us*1e3:.1f} mV", xy=(1.3, (base - us) * 1e3), xytext=(2.3, (base - us) * 1e3 + 0.5),
                    fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
        ax.annotate(f"{os_*1e3:.1f} mV", xy=(5.5, (base + os_) * 1e3), xytext=(6.5, (base + os_) * 1e3 - 1),
                    fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
        ax.set_ylim(base * 1e3 - 5, base * 1e3 + 7)
        log("  VOUT=%s V: undershoot %.1f mV, overshoot %.1f mV" % (name, us * 1e3, os_ * 1e3))
    axs[2].plot(t, il * 1e3, color=INK)
    axs[2].set_ylabel(r"$I_\mathrm{LOAD}$ (mA)")
    axs[2].set_ylim(-1.5, 12)
    axs[2].set_xlabel(r"Time ($\mu$s)")
    axs[2].set_xlim(0, 10)
    fig.align_ylabels(axs)
    save(fig, "fig5_transient")

    # zoomed rising / falling edges
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 2.9), sharex="col",
                            gridspec_kw={"hspace": 0.12, "wspace": 0.12, "height_ratios": [1.4, 1]})
    for j, (t0, t1, lab) in enumerate([(0.9, 2.0, "(a)"), (4.9, 6.0, "(b)")]):
        m = (t >= t0) & (t <= t1)
        axs[0, j].plot(t[m], (v8[m] - at(t, v8, 0.95)) * 1e3, color=CAT[0], label="0.8 V")
        axs[0, j].plot(t[m], (v9[m] - at(t, v9, 0.95)) * 1e3, color=CAT[1], ls="--", label="0.9 V")
        axs[1, j].plot(t[m], il[m] * 1e3, color=INK)
        axs[1, j].set_xlim(t0, t1)
        axs[1, j].set_xlabel(r"Time ($\mu$s)")
        axs[0, j].set_ylim(-4, 6.5)
        axs[1, j].set_ylim(-1.5, 12)
        panel_label(axs[1, j], lab)
        if j:
            axs[0, j].tick_params(labelleft=False)
            axs[1, j].tick_params(labelleft=False)
    axs[0, 0].set_ylabel(r"$\Delta V_\mathrm{OUT}$ (mV)")
    axs[1, 0].set_ylabel(r"$I_\mathrm{LOAD}$ (mA)")
    axs[0, 0].legend(loc="upper left", title=r"$V_\mathrm{OUT}$", title_fontsize=7)
    fig.align_ylabels(axs[:, 0])
    save(fig, "fig6_transient_zoom")


# ----------------------------------------------- 6. Current efficiency ----
def fig_efficiency():
    cols, d = load("eff1.csv")
    t, I = d[:, 0], d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    iq_light = at(t, I[:, 0], 0.95e-6)
    il_light = 150e-6
    iq = np.array([at(t, I[:, k], 4.99e-6) for k in range(I.shape[1])])
    x = np.r_[il_light, il]
    q = np.r_[iq_light, iq]
    eta = x / (x + q) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    ax1.semilogx(x * 1e3, q * 1e6, "o-", color=CAT[0], ms=3, mfc="white", mew=0.8)
    ax1.set_ylabel(r"$I_\mathrm{Q}$ ($\mu$A)")
    ax1.set_ylim(200, 290)
    ax2.semilogx(x * 1e3, eta, "s-", color=CAT[1], ms=3, mfc="white", mew=0.8)
    ax2.set_ylabel("Current efficiency (%)")
    ax2.set_xlabel(r"$I_\mathrm{LOAD}$ (mA)")
    ax2.set_ylim(30, 105)
    ax2.annotate(f"{eta[-1]:.2f}%", xy=(x[-1] * 1e3, eta[-1]), xytext=(20, 80), fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax2.annotate(f"{eta[0]:.1f}%", xy=(x[0] * 1e3, eta[0]), xytext=(0.4, 40), fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    fig.align_ylabels((ax1, ax2))
    save(fig, "fig7_current_efficiency")
    log("[Current efficiency]")
    for a, b, e in zip(x, q, eta):
        log("  ILOAD=%8.3f mA  IQ=%.1f uA  eta=%.2f %%" % (a * 1e3, b * 1e6, e))


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


def fig_stability(fname, outname, title, phase_db=False):
    cols, d = load(fname)
    f = d[:, 0]
    n = (d.shape[1] - 1) // 2
    G, P = d[:, 1:1 + n], d[:, 1 + n:]
    if phase_db:
        P = np.column_stack([undo_phase_db(p) for p in P.T])
    il = [param(c) for c in cols[1:1 + n]]
    colors = ordinal(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    log(f"[Stability: {fname}]  (PM read as lstb phase at 0 dB crossing)")
    for c, i, g, p in zip(colors, il, G.T, P.T):
        lab = f"{i*1e6:.0f} $\\mu$A" if i < 1e-3 else f"{i*1e3:.1f} mA"
        ax1.semilogx(f, g, color=c, label=lab)
        ax2.semilogx(f, p, color=c)
        k = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        if k.size:
            k = k[0]
            lf = np.interp(0, [g[k + 1], g[k]], np.log10([f[k + 1], f[k]]))
            ugf = 10 ** lf
            pm = np.interp(lf, np.log10(f[k:k + 2]), p[k:k + 2])
            log("  ILOAD=%s  DC gain=%.1f dB  UGF=%.1f MHz  PM=%.1f deg"
                % (lab.replace("$\\mu$", "u"), g[0], ugf / 1e6, pm))
    ax1.axhline(0, color=MUTED, lw=0.6)
    ax1.set_ylabel("Loop gain (dB)")
    ax2.set_ylabel(r"Phase ($^\circ$)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(f[0], f[-1])
    ax1.legend(title=r"$I_\mathrm{LOAD}$", title_fontsize=7, ncol=2, loc="lower left",
               columnspacing=0.8)
    if title:
        ax1.set_title(title)
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
    fig_stability("stb1.csv", "fig8a_stability_overall_loop", None, phase_db=True)
    fig_stability("f_stb1.csv", "fig8b_stability_fast_loop", None)
    (OUT / "metrics.txt").write_text("\n".join(metrics) + "\n")
