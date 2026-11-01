"""Presentation versions of the FVF LDO figures.

Usage:  python scripts/slide_figures.py
Writes PNGs to figures/slide/. Each figure is sized to its slot on a 1920 x 1080
slide at 144 px/in (a 13.33 in wide PowerPoint slide), so 16 pt text in the figure
shows as 16 pt on the slide. All text is >= 16 pt; subscripts are avoided for that
reason (VOUT, ILOAD written plainly). Data handling is shared with ldo_figures.py.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

import ldo_figures as F
from ldo_figures import EPS, INK, MRK, SER, WARN, before, load, param, settle_time

OUT = F.ROOT / "figures" / "slide"
OUT.mkdir(exist_ok=True)
PX_PER_IN = 144
ANN = 16

mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.6,
    "lines.linewidth": 2.4,
    "lines.markersize": 7,
    "lines.markeredgewidth": 1.6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "grid.linewidth": 0.8,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.4,
    "patch.linewidth": 1.2,
    "figure.constrained_layout.w_pad": 0.05,
    "figure.constrained_layout.h_pad": 0.05,
})


def sfig(w_px, h_px, **kw):
    return plt.subplots(figsize=(w_px / PX_PER_IN, h_px / PX_PER_IN), layout="constrained", **kw)


def ssave(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=2 * PX_PER_IN)
    plt.close(fig)
    print(f"  -> figures/slide/{name}.png")


def txt(ax, x, y, s, **kw):
    kw.setdefault("bbox", dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    kw.setdefault("zorder", 6)
    return ax.text(x, y, s, fontsize=ANN, **kw)


def hz_axis(ax, ticks=(10, 1e3, 1e5, 1e7, 1e9)):
    ax.xaxis.set_major_locator(LogLocator(10))
    ax.xaxis.set_minor_locator(LogLocator(10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: F.eng_hz(x) if any(abs(x / t - 1) < 1e-6 for t in ticks) else ""))
    ax.xaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------- transient ----
def s_load_steps():
    """VOUT for 150 uA -> ILOAD steps (10-150 mA)."""
    cols, d = load("maxload1.csv")
    t, V = d[:, 0] * 1e6, d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    fig, ax = sfig(940, 590)
    for c, k in reversed(list(zip(SER, [0, 4, 9, 12, 13, 14]))):     # largest load drawn first
        ax.plot(t, V[:, k] * 1e3, color=c, label=f"{il[k] * 1e3:.0f} mA")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], ncol=2, loc="upper left", title="ILOAD step from 150 µA",
              columnspacing=1.0, borderaxespad=0.6)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(765, 860)
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("VOUT [mV]")
    ssave(fig, "s_load_steps")


def s_load_step():
    """US/OS, settling time and steady-state drop versus the step size."""
    rows = F.SUMMARY["load_step"]["rows"]
    il = np.array([r["iload_mA"] for r in rows])
    g = lambda key: np.array([r[key] for r in rows])
    fig, (a1, a2, a3) = sfig(700, 770, nrows=3, sharex=True)
    a1.plot(il, g("os_mV"), color=SER[1], marker=MRK[1], mfc="white", label="Overshoot")
    a1.plot(il, g("us_mV"), color=SER[0], marker=MRK[0], mfc="white", label="Undershoot")
    txt(a1, 108, 27.5, "Overshoot", color=SER[1], ha="right", va="center", bbox=None)
    txt(a1, 104, 5, "Undershoot", color=SER[0], ha="left", va="center", bbox=None)
    a1.set_ylabel("ΔV [mV]")
    a1.set_ylim(0, 36)
    a1.yaxis.set_major_locator(MultipleLocator(10))
    a2.plot(il, g("ts_up_ns") / 1e3, color=SER[0], marker=MRK[0], mfc="white", label="Load ↑")
    a2.plot(il, g("ts_dn_ns") / 1e3, color=SER[1], marker=MRK[1], mfc="white", label="Load ↓")
    a2.set_ylabel("Ts [µs]")
    a2.set_ylim(0, 2.9)
    a2.yaxis.set_major_locator(MultipleLocator(1))
    a2.legend(loc="upper left", borderaxespad=0.4)
    a3.plot(il, g("dv_ss_mV"), color=SER[2], marker=MRK[2], mfc="white")
    a3.axhline(F.MAXLOAD_DV * 1e3, color=WARN, ls="--", lw=1.6)
    txt(a3, 6, F.MAXLOAD_DV * 1e3 + 0.08, "1 mV", color=WARN, va="bottom")
    a3.set_ylabel("ΔVss [mV]")
    a3.set_ylim(0, 1.6)
    a3.yaxis.set_major_locator(MultipleLocator(0.5))
    a3.set_xlabel("ILOAD [mA]")
    a3.set_xlim(0, 156)
    a3.xaxis.set_major_locator(MultipleLocator(50))
    a3.xaxis.set_minor_locator(MultipleLocator(10))
    for a in (a1, a2, a3):
        a.axvline(F.IMAX_VERIFIED * 1e3, color=INK, ls=":", lw=1.8)
    txt(a3, F.IMAX_VERIFIED * 1e3 + 3, 1.3, "ILOAD,max", va="center")
    fig.align_ylabels((a1, a2, a3))
    ssave(fig, "s_load_step")


# ------------------------------------------------------------- stability ----
def stb(fname, phase_db):
    cols, d = load(fname)
    f = d[:, 0]
    n = (d.shape[1] - 1) // 2
    G, P = d[:, 1:1 + n], d[:, 1 + n:]
    if phase_db:
        P = np.column_stack([F.undo_phase_db(p) for p in P.T])
    return f, G, P, [param(c) for c in cols[1:1 + n]]


def s_stability():
    """Overall (left) and fast (right) loop gain / phase at 0.1, 10 and 50 mA."""
    fig, ax = sfig(980, 650, nrows=2, ncols=2, sharex=True)
    pick = (0, 1, 5)
    for col, (fname, pdb, cfg, title) in enumerate((
            ("stb1.csv", True, F.STB["fig8a_stability_overall_loop"], "Overall loop"),
            ("f_stb1.csv", False, F.STB["fig8b_stability_fast_loop"], "Fast loop"))):
        f, G, P, il = stb(fname, pdb)
        a1, a2 = ax[0, col], ax[1, col]
        for n, k in enumerate(pick):
            g, p = G[:, k], P[:, k]
            kw = dict(color=SER[n], ls="--" if n == 0 else "-")
            a1.semilogx(f, g, label=F.ma_label(il[k]).replace(".1 mA", " mA"), **kw)
            a2.semilogx(f, p, **kw)
            c = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
            if c.size:
                c = c[0]
                lf = np.interp(0, [g[c + 1], g[c]], np.log10([f[c + 1], f[c]]))
                pm = np.interp(lf, np.log10(f[c:c + 2]), p[c:c + 2])
                a1.plot(10 ** lf, 0, marker=MRK[n], color=SER[n], mfc="white", zorder=6)
                a2.plot(10 ** lf, pm, marker=MRK[n], color=SER[n], mfc="white", zorder=6)
        a1.axhline(0, color=INK, lw=1.0)
        a1.set_title(title, color=SER[1] if col else SER[0])
        lo, hi, st = cfg["gain_ticks"]
        a1.set_yticks(range(lo, hi, 2 * st if col == 0 else st))
        a1.set_ylim(*cfg["gain_lim"])
        lo, hi, st = cfg["ph_ticks"]
        a2.set_yticks(range(lo, hi, 2 * st))
        a2.set_ylim(*cfg["ph_lim"])
        hz_axis(a2)
        a2.set_xlim(f[0], f[-1])
        a2.set_xlabel("Frequency [Hz]")
    ax[0, 0].set_ylabel("Gain [dB]")
    ax[1, 0].set_ylabel("Phase [°]")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, title=None, frameon=False)
    ssave(fig, "s_stability")


def s_loop_vs_load():
    st = F.SUMMARY["stability"]
    fig, (a1, a2) = sfig(600, 560, nrows=2, sharex=True)
    for k, key in enumerate(("fig8a_stability_overall_loop", "fig8b_stability_fast_loop")):
        r = st[key]
        x = np.array([q["iload_mA"] for q in r])
        for a, y in ((a1, [q["ugf_MHz"] for q in r]), (a2, [q["pm_deg"] for q in r])):
            a.semilogx(x[:2], y[:2], color=SER[k], ls=":", lw=1.8)
            a.semilogx(x[1:], y[1:], color=SER[k], marker=MRK[k], mfc="white")
            a.semilogx(x[:1], y[:1], color=SER[k], marker=MRK[k], mfc="white", ls="none")
    ratio = min(f["ugf_MHz"] / o["ugf_MHz"] for f, o in zip(st["fig8b_stability_fast_loop"],
                                                            st["fig8a_stability_overall_loop"]))
    txt(a1, 0.09, 1000, "Fast loop", color=SER[1], va="center", bbox=None)
    txt(a1, 0.09, 42, "Overall loop", color=SER[0], va="center", bbox=None)
    txt(a1, 3, 110, f"≥ {ratio:.0f}×", va="center", ha="center")
    a1.annotate("", (3, 17), (3, 330), arrowprops=dict(arrowstyle="<|-|>", mutation_scale=14, lw=1.6, color=INK))
    a1.set_yscale("log")
    a1.set_ylim(8, 2500)
    a1.yaxis.set_major_locator(LogLocator(10))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_ylabel("UGF [MHz]")
    a2.set_ylabel("PM [°]")
    a2.set_ylim(30, 100)
    a2.yaxis.set_major_locator(MultipleLocator(20))
    a2.axhline(45, color=WARN, ls="--", lw=1.6)
    a2.set_xlabel("Load Current [mA]")
    F.log_axis(a2)
    a2.set_xlim(0.07, 70)
    fig.align_ylabels((a1, a2))
    ssave(fig, "s_loop_vs_load")


# ------------------------------------------------------- efficiency / PSR ----
def s_efficiency():
    cols, d = load("eff1.csv")
    t, I = d[:, 0] * 1e6, d[:, 1:]
    il = np.array([param(c) for c in cols[1:]])
    x = np.r_[150e-6, il]
    q = np.r_[before(t, I[:, 0], 1.0), before(t, I, 5.0)]
    eta = x / (x + q) * 100
    reg = x <= F.IMAX_VERIFIED + EPS
    jp = int(np.argmax(np.where(reg, eta, -1)))
    xc = np.logspace(np.log10(x[0]), np.log10(x[-1]), 200)
    ec = xc / (xc + np.interp(np.log10(xc), np.log10(x), q)) * 100
    fig, ax = sfig(980, 700)
    ax.semilogx(xc * 1e3, ec, color=SER[0], ls="--", label="Calc. (IQ interpolated)")
    show = np.isin(np.round(x * 1e3, 2), [0.15, 10, 20, 50, 100, 140, 150])
    ax.semilogx(x[show] * 1e3, eta[show], ls="none", marker=MRK[0], color=SER[0], mfc="white", label="Simulated")
    ax.plot(x[jp] * 1e3, eta[jp], marker=MRK[0], color=SER[0], ms=10)
    F.log_axis(ax)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(25, 102)
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Current Efficiency [%]")
    ax.legend(loc="lower right", borderaxespad=0.6,
              title=f"{eta[jp]:.2f} % @ ILOAD,max = {x[jp] * 1e3:.0f} mA\n"
                    f"IQ = {F.rng(q[reg] * 1e6, '{:.0f}')} µA")
    ssave(fig, "s_efficiency")


def s_psr():
    cols, d = load("psr1.csv")
    f, P = d[:, 0], d[:, 1:]
    il = [param(c) for c in cols[1:]]
    m = f <= F.PSR_FMAX * (1 + EPS)
    fig, ax = sfig(880, 630)
    n = len(il)
    tg = 10 ** np.arange(1.5, min(np.log10(f[m][-1]), 7.4), 1 / n)
    mk = [int(j) for j in np.searchsorted(f[m], tg)]
    for k, (i, p) in enumerate(zip(il, P.T)):
        ax.semilogx(f[m], p[m], color=SER[k], marker=MRK[k], markevery=mk[k::n], mfc="white", label=F.ma_label(i))
    hz_axis(ax, ticks=(10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8))
    ax.set_xlim(f[0], F.PSR_FMAX)
    ax.set_ylim(-70, 10)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSR [dB]")
    ax.legend(title="ILOAD", loc="lower right", borderaxespad=0.6)
    ssave(fig, "s_psr")


# ------------------------------------------------------------ regulation ----
def s_line_reg():
    cols, d = load("linR1.csv")
    vin, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    rows = [F.line_reg_metrics(vin, v, vr) for vr, v in zip(vref, V.T)]
    fig, ax = sfig(820, 600)
    x0, x1 = F.LNR_X0, vin[-1]
    m = vin >= x0 - 0.02 - EPS
    ax.plot([x0, x1], [x0, x1], ls="--", color=INK, lw=1.6)
    txt(ax, x0 + 0.015, 0.947, "VOUT = VIN", ha="left", va="center", bbox=None)
    inner = np.where((vin[m] > x0 + EPS) & (vin[m] < x1 - EPS))[0]
    for k, (vr, v) in enumerate(zip(vref, V.T)):
        ax.plot(vin[m], v[m], color=SER[k], marker=MRK[k], markevery=[int(j) for j in inner[k::len(vref)]],
                mfc="white", label=f"{vr:.2f} V")
    r = rows[-2]
    y = r["plateau"]
    ax.annotate("", (y, y), (r["vin_min"], y), arrowprops=dict(arrowstyle="<|-|>", mutation_scale=14, lw=1.6,
                                                                color=INK, shrinkA=0, shrinkB=0), zorder=7)
    txt(ax, (y + r["vin_min"]) / 2, y + 0.007, f"VDO = {(r['vin_min'] - y) * 1e3:.0f} mV", ha="center", va="bottom")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0.55, 0.97)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("Input Voltage [V]")
    ax.set_ylabel("Output Voltage [V]")
    h, lab = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]                     # ncol=3 fills column-wise: rows read 0.70 0.75 0.80 / 0.85 0.90
    ax.legend([h[i] for i in order], [lab[i] for i in order], title="VREF", loc="lower right", ncol=3,
              borderaxespad=0.5, columnspacing=0.9, handletextpad=0.5)
    ssave(fig, "s_line_reg")


def s_load_reg():
    cols, d = load("loadR1.csv")
    x, V = d[:, 0], d[:, 1:]
    vref = [param(c) for c in cols[1:]]
    rs = [F.load_reg_metrics(x, v, vr) for vr, v in zip(vref, V.T)]
    fig, ax = sfig(820, 600)
    m = x >= F.LDR_I0 * 0.99
    xm = x[m] * 1e3
    drop = []
    for k, (vr, v, r) in enumerate(zip(vref, V.T, rs)):
        ls = "-" if r["regulating"] else ((0, (1, 1.2)) if not drop else (0, (4, 2)))
        tk = np.arange(2.5 + 2.5 * (k % 2), xm[-1] - 2, 5)
        ax.plot(xm, v[m], color=SER[k], ls=ls)
        ax.plot(tk, np.interp(tk, xm, v[m]), ls="none", color=SER[k], marker=MRK[k], mfc="white")
        if r["regulating"]:
            txt(ax, 1.5, r["v_ref"] + 0.008, f"VREF = {vr:.2f} V,  LDR = {r['ldr'] * 1e3:.1f} µV/mA", va="bottom")
        else:
            drop.append((k, vr, ls))
    hs = [Line2D([], [], color=SER[k], ls=ls, marker=MRK[k], mfc="white") for k, _, ls in drop]
    ax.legend(hs, [f"{vr:.2f} V" for _, vr, _ in drop], ncol=len(drop), loc="upper right",
              title=f"VREF in dropout (VIN = {F.COND['ldr_vin']:.1f} V)", borderaxespad=0.5,
              handlelength=2.2, columnspacing=1.2)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.set_xlabel("Load Current [mA]")
    ax.set_ylabel("Output Voltage [V]")
    ax.set_xlim(0, x[-1] * 1e3)
    ax.set_ylim(0.66, 0.94)
    ssave(fig, "s_load_reg")


if __name__ == "__main__":
    # the paper script fills SUMMARY (stability rows, load-step rows); its own figures are not redrawn here
    F.SUMMARY.clear()
    F.save = lambda fig, name: plt.close(fig)
    F.fig_stability("stb1.csv", "fig8a_stability_overall_loop", phase_db=True)
    F.fig_stability("f_stb1.csv", "fig8b_stability_fast_loop")
    F.fig_load_step()
    s_load_steps()
    s_load_step()
    s_stability()
    s_loop_vs_load()
    s_efficiency()
    s_psr()
    s_line_reg()
    s_load_reg()
