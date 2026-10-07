# FVF LDO – paper figures

`python scripts/ldo_figures.py` reads the WaveView CSV exports in `data/` and writes
IEEE single-column (3.5 in) figures to `figures/` as PDF (vector, embedded fonts) and
PNG (600 dpi). Extracted numbers (undershoot, dropout, PSR, Iq, UGF/PM …) go to
`figures/metrics.txt`.

Requires `numpy` and `matplotlib`. Font: Times New Roman (falls back to Liberation
Serif / DejaVu Serif when Times is not installed).

| Figure | Source | Content |
|---|---|---|
| fig1_max_load | maxload1.csv | (a) VOUT for load steps 150 µA → 10–150 mA, (b) steady-state / minimum VOUT vs ILOAD |
| fig2_line_regulation | linR1.csv | VOUT vs VIN, VREF = 0.70–0.90 V |
| fig3_load_regulation | loadR1.csv | VOUT vs ILOAD, VREF = 0.70–0.90 V |
| fig4_psr | psr1.csv | PSR at ILOAD = 100 µA and 50 mA |
| fig5_transient | tr1.csv | VOUT (0.8 V / 0.9 V) and ILOAD, 150 µA ↔ 10 mA |
| fig6_transient_zoom | tr1.csv | zoom on rising (a) and falling (b) load steps |
| fig7_current_efficiency | eff1.csv | Iq and current efficiency vs ILOAD |
| fig8a_stability_overall_loop | stb1.csv | overall loop gain & phase vs ILOAD |
| fig8b_stability_fast_loop | f_stb1.csv | fast (FVF) loop gain & phase vs ILOAD |
