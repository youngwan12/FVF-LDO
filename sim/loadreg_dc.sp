* FVF LDO - load regulation / max load / quiescent current (DC sweep)
* One run gives: VOUT vs ILOAD per VREF (load regulation, max load)
*                i(vvss) = IQ  (current efficiency)

.protect
.lib '/cad/lib/SEC/28n_rf/LNR28LPP/Platform_PROC-HSPICE_sec150930_0306/All/LNR28LPP_Hspice.lib' NN
.unprotect
.include './fvf_yw_fb_3.ckt'

.option post=1 accurate=1 runlvl=6
.option dcon=1 dccap
.option probe

*---------------------------------------------------------------- params --
.param vin=1.1          $ fixed VIN (>= VOUT,max + ~0.17 V dropout)
.param vr=0.8           $ swept by .dc below

*--------------------------------------------------------------- sources --
vvdd            VDD             0       vin
vvss            VSS             0       0
vref            VREF            0       vr
vbias           VBIAS           0       0.5
vpbias          pbias           0       0.5
vnbias          nbias           0       0.4
vnbias1         nbias1          0       0.45

* Load: tied to 0 (not VSS) so that i(vvss) carries IQ only.
* vvss = 0 V, so regulation is identical to tying it to VSS.
iload           OUT             0       dc 0

cload           OUT             VSS     1p
rlstb           OUT             OUT1    0

*-------------------------------------------------------------- analysis --
.op
.dc iload dec 20 10u 200m   sweep vr 0.7 0.9 0.05

.probe v(out) i(vvdd) i(vvss) i(iload)

*----------------------------------------------------------------- export --
* No .meas needed - scripts/ldo_figures.py computes everything from CSV.
* WaveView export (format table, CSV), X axis = ILOAD:
*   data/loadR1.csv : v(out)  for all vr
*   data/iq1.csv    : i(vvss) for all vr
* -> load regulation [mV/mA], ILOAD,max (VOUT < 0.99*VREF), IQ, current efficiency

*--------------------------------------------------------- transient (ref) --
* For the load-step transient use a separate deck with:
*   .param load1=150u load2=10m tedge=0.5u
*   iload OUT 0 pwl(0 load1 1u load1 '1u+tedge' load2 5u load2 '5u+tedge' load1)
*   .tran 0.01n 10u
* (an element can only be defined once, so do not keep both iload lines)

.end
