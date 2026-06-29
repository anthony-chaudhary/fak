#!/usr/bin/env python3
"""H1 sensitivity stressor for the Qwen3.6-27B token-3 drift (host-independent).

The investigation note (`token3-drift-investigation-2026-06-28.md` §2-H1, §4, §5
step 2) ranks the **GDN delta-rule recurrent scan** as the prime suspect for the
fak<->llama.cpp token-3 argmax flip, and flags THIS experiment -- "measure how a
per-step state error compounds with depth x tokens ... predicts whether the 27B's
48-layer depth is enough to explain the observed token-3 flip" -- as "the
highest-value host-independent experiment short of the Mac capture." It had not
been run. This is it.

It does NOT touch the 27B weights or llama.cpp. It models ONE representative GDN
head's delta-rule recurrence (faithful to `internal/model/qwen35.go:481-517`):

    st[i,d] *= g                       # decay the persistent state (g near 1)
    kvmem[d]  = SUM_i st[i,d]*k[i]      # <-- reduction #1 (order/dtype-sensitive)
    delta[d]  = (v[d] - kvmem[d]) * b
    st[i,d]  += k[i]*delta[d]           # rank-1 state update
    out[d]    = SUM_i st[i,d]*q[i]      # <-- reduction #2 (order/dtype-sensitive)

stacked over L layers with a residual stream, run over P prefill + D decode
positions. We compare a clean float64 REFERENCE against two perturbed arms that
inject a controlled relative error into BOTH reductions -- the exact place fak's
serial-f32 scan and llama.cpp's threadgroup-reduction kernel can differ:

    RANDOM  - each reduction scaled by (1 + eps*N(0,1)) -> models reduction-ORDER
              rounding (a different but equally-valid summation order). Uncorrelated
              step to step.
    SYSTEMATIC - each reduction scaled by (1 + eps) constant -> models a genuine
              algorithmic / ordering / FMA-fusion BIAS that pushes the same way
              every step. Correlated step to step.

The question H1 makes falsifiable, sharpened: with the decay g near 1, a RANDOM
per-step error does a decaying random walk and reaches a small BOUNDED stationary
magnitude; a SYSTEMATIC per-step error INTEGRATES (the near-1 decay barely damps a
correlated push). So the experiment settles *which kind* of kernel difference can
reach argmax-flipping magnitude by decode token 3 -- telling the Mac probe whether
to hunt for mere rounding (unlikely sole cause) or a systematic bias (the real
suspect).

Honesty: this is a SENSITIVITY / PLAUSIBILITY model with assumed magnitudes, swept
over decay and perturbation class; it is not a quantitative prediction of the real
27B logits. A faithful reproduction of the fak<->llama.cpp drift needs llama.cpp in
the loop (Mac-gated, doc §4). What this CAN settle is the random-vs-systematic
integrate-or-bound dynamics, which is exactly H1's open question.
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in the fixture toolchain
    print("SKIP: numpy not available (needed for the H1 sensitivity stressor)", file=sys.stderr)
    raise SystemExit(3)

SCHEMA = "qwen36-h1-recurrence-sensitivity/v2"

# 27B-representative GDN dims (doc §4: "48 GDN layers and vHd=128"). The exact kHd of
# Qwen3.6-27B is not asserted; 128 is representative and the conclusion is swept. The
# fixed 22-token ChatML prompt -> P=22 prefill positions, then decode the 2 agreed
# tokens + the divergent 3rd -> token-3 is decode step 3.
DEFAULT_LAYERS = 48
DEFAULT_KHD = 128
DEFAULT_VHD = 128
DEFAULT_PREFILL = 22
DEFAULT_DECODE = 3
# f32 ULP-class relative error for the reduction (a different summation order of ~128
# f32 terms differs by a few ULP ~ 1e-7..1e-6 relative). Used as the per-step eps.
EPS_F32 = 6e-8
# The observed near-tie that flips at token 3: fak's top-1 vs top-2 gap is ~1.75 logits
# on a logit scale of ~20. A logit-proxy perturbation of relative magnitude ~1.75/20 is
# the "plausibly argmax-flipping" bar.
FLIP_REL_BAR = 1.75 / 20.0


def run_stack(layers, khd, vhd, prefill, decode, decay, seed, mode, eps):
    """Run the L-layer residual GDN stack in one arm; return the logit-proxy (final
    residual at the current position) at each decode step. `mode` in
    {'ref','random','systematic'} controls the reduction perturbation."""
    rng = np.random.default_rng(seed)
    perturb_rng = np.random.default_rng(seed + 1)  # independent stream for the noise
    st = [np.zeros((khd, vhd), dtype=np.float64) for _ in range(layers)]
    total = prefill + decode
    qs = rng.standard_normal((total, layers, khd)) * 0.5
    ks = rng.standard_normal((total, layers, khd)) * 0.5
    vs = rng.standard_normal((total, layers, vhd)) * 0.5
    betas = rng.uniform(0.2, 0.9, size=(total, layers))
    embeds = rng.standard_normal((total, vhd)) * 0.5

    def reduce(mat_i_d, vec_i):
        base = (mat_i_d * vec_i[:, None]).sum(axis=0)  # [vhd], f64
        if mode == "ref":
            return base
        if mode == "systematic":
            return base * (1.0 + eps)                  # same push every step
        # random: a fresh relative perturbation per component per call
        return base * (1.0 + eps * perturb_rng.standard_normal(base.shape))

    decode_proxies = []
    for t in range(total):
        x = embeds[t].copy()
        for l in range(layers):
            q, k, v, b = qs[t, l], ks[t, l], vs[t, l], betas[t, l]
            st[l] *= decay
            kvmem = reduce(st[l], k)        # reduction #1
            delta = (v - kvmem) * b
            st[l] += np.outer(k, delta)     # rank-1 update
            out = reduce(st[l], q)          # reduction #2
            x = x + out                     # residual add (logit proxy)
        if t >= prefill:
            decode_proxies.append(x.copy())
    return decode_proxies


def rel_div(a, b):
    denom = np.linalg.norm(b) or 1.0
    return float(np.linalg.norm(a - b) / denom)


def sweep(layers, khd, vhd, prefill, decode, seed, decays, eps):
    rows = []
    for g in decays:
        ref = run_stack(layers, khd, vhd, prefill, decode, g, seed, "ref", eps)
        rnd = run_stack(layers, khd, vhd, prefill, decode, g, seed, "random", eps)
        sysm = run_stack(layers, khd, vhd, prefill, decode, g, seed, "systematic", eps)
        steps = []
        for i in range(decode):
            steps.append({
                "decode_token": i + 1,
                "rel_div_random": rel_div(rnd[i], ref[i]),
                "rel_div_systematic": rel_div(sysm[i], ref[i]),
            })
        r0, rT = steps[0], steps[-1]
        rand_growth = rT["rel_div_random"] / (r0["rel_div_random"] or 1e-30)
        sys_growth = rT["rel_div_systematic"] / (r0["rel_div_systematic"] or 1e-30)
        # the systematic arm is linear in eps for small eps, so amplification A =
        # sys_relDiv_t3 / eps is eps-independent; the per-step systematic bias that
        # reaches the flip bar by token 3 is eps_flip = FLIP_REL_BAR / A. This converts
        # the result into the actionable number: "a systematic per-reduction relative
        # bias of >= eps_flip explains the token-3 flip; pure f32 rounding (~1e-7) does not."
        amplification = rT["rel_div_systematic"] / (eps or 1e-30)
        eps_flip = FLIP_REL_BAR / amplification if amplification else float("inf")
        rows.append({
            "decay": g,
            "steps": steps,
            "random_tok_growth": rand_growth,
            "systematic_tok_growth": sys_growth,
            "random_integrates": rand_growth > 1.05,
            "systematic_integrates": sys_growth > 1.05,
            "random_flip_plausible": rT["rel_div_random"] >= FLIP_REL_BAR,
            "systematic_flip_plausible": rT["rel_div_systematic"] >= FLIP_REL_BAR,
            # how much larger the systematic effect is than the random one at token 3
            "systematic_over_random_t3": (rT["rel_div_systematic"] / (rT["rel_div_random"] or 1e-30)),
            "depth_position_amplification": amplification,
            "systematic_eps_flip_threshold": eps_flip,
        })
    return rows


def build_report(args):
    decays = [float(x) for x in args.decays.split(",")]
    rows = sweep(args.layers, args.khd, args.vhd, args.prefill, args.decode,
                 args.seed, decays, args.eps)
    near1 = [r for r in rows if r["decay"] >= 0.99]
    rand_bounded = all(not r["random_integrates"] for r in near1) if near1 else False
    sys_integrates = any(r["systematic_integrates"] for r in near1)
    sys_dominates = all(r["systematic_over_random_t3"] > 3 for r in near1) if near1 else False
    eps_flips = [r["systematic_eps_flip_threshold"] for r in (near1 or rows)]
    eps_flip = sorted(eps_flips)[len(eps_flips) // 2] if eps_flips else float("inf")
    return {
        "schema": SCHEMA,
        "params": {
            "layers": args.layers, "khd": args.khd, "vhd": args.vhd,
            "prefill": args.prefill, "decode": args.decode, "seed": args.seed,
            "eps": args.eps, "flip_rel_bar": FLIP_REL_BAR, "f32_rounding_eps": EPS_F32,
        },
        "decays": decays,
        "rows": rows,
        "verdict": {
            "random_bounded_at_near1_decay": rand_bounded,
            "systematic_integrates_at_near1_decay": sys_integrates,
            "systematic_dominates_random_at_t3": sys_dominates,
            "systematic_eps_flip_threshold": eps_flip,
            "f32_rounding_too_small_to_flip": eps_flip > EPS_F32 * 100,
            "reading": _reading(rand_bounded, sys_integrates, sys_dominates, eps_flip),
        },
    }


def _reading(rand_bounded, sys_integrates, sys_dominates, eps_flip):
    if rand_bounded and sys_integrates and sys_dominates:
        return ("H1 SHARPENED: pure reduction-ORDER rounding (random per-step) stays "
                "BOUNDED under the near-1 decay -- so rounding alone is an unlikely sole "
                "cause of a token-3 flip. A SYSTEMATIC per-step bias (correlated FMA/order/"
                "algorithmic difference) INTEGRATES and dominates by ~Nx at token 3. The "
                f"systematic per-reduction relative bias that reaches the flip bar by token 3 "
                f"is ~{eps_flip:.1e} -- roughly {eps_flip / EPS_F32:.0f}x larger than f32 "
                "rounding, i.e. an ALGORITHMIC difference, not float noise. DIRECTIVE for the "
                "Mac probe: hunt for a SYSTEMATIC, consistently-signed kernel difference in "
                "the recurrent scan, and check the state-FEEDING ops (H2 q/k L2-norm, H3 "
                "gated RMSNorm) which inject a steady, correlated bias into st.")
    if rand_bounded and sys_integrates:
        return ("Reduction rounding bounds; a systematic bias integrates -- consistent with "
                "H1-systematic over H1-random, but it does not dominate strongly at these "
                "magnitudes. The real 27B logit scale decides.")
    if not rand_bounded:
        return ("Even random reduction error grows token-over-token here -- the dynamics "
                "amplify; H1 (any reduction difference) is a live integrating cause.")
    return "inconclusive across the swept decays."


def render_md(report):
    v = report["verdict"]
    p = report["params"]
    lines = [
        "# H1 recurrence sensitivity -- Qwen3.6-27B token-3 drift (random vs systematic)",
        "",
        f"- Params: L={p['layers']} kHd={p['khd']} vHd={p['vhd']} prefill={p['prefill']} "
        f"decode={p['decode']} eps={p['eps']:.1e} seed={p['seed']}",
        f"- Flip-plausible bar (relative logit-proxy): {p['flip_rel_bar']:.4f}; "
        f"f32 rounding eps ~ {p['f32_rounding_eps']:.1e}",
        f"- **Systematic eps-flip threshold:** ~{v['systematic_eps_flip_threshold']:.2e} "
        f"per-reduction relative bias reaches the flip bar by token 3 "
        f"(f32-rounding too small to flip: {v['f32_rounding_too_small_to_flip']})",
        f"- **Reading:** {v['reading']}",
        "",
        "| decay g | rand relΔ t1→t3 | rand grow | rand integ | sys relΔ t1→t3 | sys grow | "
        "sys integ | sys/rand @t3 | sys flip? |",
        "|---:|---|---:|:--:|---|---:|:--:|---:|:--:|",
    ]
    for r in report["rows"]:
        s0, sT = r["steps"][0], r["steps"][-1]
        lines.append(
            f"| {r['decay']:.4f} | {s0['rel_div_random']:.2e}→{sT['rel_div_random']:.2e} | "
            f"{r['random_tok_growth']:.2f}x | {'yes' if r['random_integrates'] else 'no'} | "
            f"{s0['rel_div_systematic']:.2e}→{sT['rel_div_systematic']:.2e} | "
            f"{r['systematic_tok_growth']:.2f}x | {'yes' if r['systematic_integrates'] else 'no'} | "
            f"{r['systematic_over_random_t3']:.1f}x | "
            f"{'YES' if r['systematic_flip_plausible'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--khd", type=int, default=DEFAULT_KHD)
    ap.add_argument("--vhd", type=int, default=DEFAULT_VHD)
    ap.add_argument("--prefill", type=int, default=DEFAULT_PREFILL)
    ap.add_argument("--decode", type=int, default=DEFAULT_DECODE)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--eps", type=float, default=EPS_F32,
                    help="per-step relative reduction error (f32 ULP-class)")
    ap.add_argument("--decays", default="0.9,0.99,0.999,0.9999",
                    help="comma list of decay g values to sweep")
    ap.add_argument("--out", help="write the witness JSON here")
    ap.add_argument("--markdown", help="write the markdown summary here")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    report = build_report(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(render_md(report))
    print(render_md(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
