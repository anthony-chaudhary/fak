# H1 recurrence sensitivity -- Qwen3.6-27B token-3 drift (random vs systematic)

- Params: L=48 kHd=128 vHd=128 prefill=22 decode=3 eps=6.0e-08 seed=20260628
- Flip-plausible bar (relative logit-proxy): 0.0875; f32 rounding eps ~ 6.0e-08
- **Systematic eps-flip threshold:** ~5.78e-03 per-reduction relative bias reaches the flip bar by token 3 (f32-rounding too small to flip: True)
- **Reading:** H1 SHARPENED: pure reduction-ORDER rounding (random per-step) stays BOUNDED under the near-1 decay -- so rounding alone is an unlikely sole cause of a token-3 flip. A SYSTEMATIC per-step bias (correlated FMA/order/algorithmic difference) INTEGRATES and dominates by ~Nx at token 3. The systematic per-reduction relative bias that reaches the flip bar by token 3 is ~5.8e-03 -- roughly 96265x larger than f32 rounding, i.e. an ALGORITHMIC difference, not float noise. DIRECTIVE for the Mac probe: hunt for a SYSTEMATIC, consistently-signed kernel difference in the recurrent scan, and check the state-FEEDING ops (H2 q/k L2-norm, H3 gated RMSNorm) which inject a steady, correlated bias into st.

| decay g | rand relΔ t1→t3 | rand grow | rand integ | sys relΔ t1→t3 | sys grow | sys integ | sys/rand @t3 | sys flip? |
|---:|---|---:|:--:|---|---:|:--:|---:|:--:|
| 0.9000 | 3.15e-07→2.48e-07 | 0.79x | no | 8.47e-07→9.02e-07 | 1.07x | yes | 3.6x | no |
| 0.9900 | 3.19e-07→2.48e-07 | 0.78x | no | 8.48e-07→9.08e-07 | 1.07x | yes | 3.7x | no |
| 0.9990 | 3.19e-07→2.48e-07 | 0.78x | no | 8.48e-07→9.09e-07 | 1.07x | yes | 3.7x | no |
| 0.9999 | 3.19e-07→2.48e-07 | 0.78x | no | 8.48e-07→9.09e-07 | 1.07x | yes | 3.7x | no |
