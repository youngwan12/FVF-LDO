* FVF LDO - load regulation / max load / quiescent current (DC sweep)
* One run gives: VOUT vs ILOAD per VREF (load regulation, max load)
*                i(vvss) = IQ  (current efficiency)

.protect
.lib '/cad/lib/SEC/28n_rf/LNR28LPP/Platform_PROC-HSPICE_sec150930_0306/All/LNR28LPP_Hspice.lib' NN
.unprotect
.include './fvf_yw_fb_3.ckt'

.option post=1 accurate=1 runlvl=6
.option measdgt=8 dcon=1 dccap
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

*----------------------------------------------------------- measurements --
* (results per VREF in the .ms0 file)
.meas dc vout_lo  find v(out)  at=100u
.meas dc vout_hi  find v(out)  at=100m
.meas dc ldreg    param='(vout_lo-vout_hi)/(100m-100u)'    $ [V/A] = [mV/mA]
.meas dc imax     when v(out)='0.99*vr' fall=1             $ max load (VOUT < 99% VREF)
.meas dc iq_lo    find i(vvss) at=100u
.meas dc iq_hi    find i(vvss) at=100m
.meas dc ivdd_hi  find i(vvdd) at=100m

*--------------------------------------------------------- transient (ref) --
* For the load-step transient use a separate deck with:
*   .param load1=150u load2=10m tedge=0.5u
*   iload OUT 0 pwl(0 load1 1u load1 '1u+tedge' load2 5u load2 '5u+tedge' load1)
*   .tran 0.01n 10u
* (an element can only be defined once, so do not keep both iload lines)

.end
