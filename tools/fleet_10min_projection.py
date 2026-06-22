#!/usr/bin/env python3
"""Project the wall-clock of a 5-agent x 200-turn fleet session onto a MacBook-Pro 7B,
from EXACT per-arm token/step counts and a MEASURED 7B rate card.

This is the agent-count + wall-clock companion to `cmd/sessionbench`: sessionbench runs the
three arms LIVE on the local kernel (any model shape, ratios faithful, absolute time = this
box); this script converts the *exact, timing-free* per-arm work (the same prefillTokens
arithmetic sessionbench reports) into a wall-clock PROJECTION at a target host's measured
single-stream rates. Nothing here is modeled throughput -- the rates are committed M3 Pro
measurements (BENCHMARK-AUTHORITY) and the batching factor is read from a real sessionbench run.

THE THREE ARMS (identical to sessionbench):
  A naive-stateless   re-prefill the whole growing context every turn       prefill = C*sum_t(P + t*(D+R))
  B per-agent-KV      warm per-agent cache, serial decode, prefix x C       prefill = C*(P + (T-1)*R)
  C fak fused         prefix prefilled ONCE + cloned + BATCHED decode        prefill = P + C*(T-1)*R

WALL-CLOCK MODEL (per arm) = prefill_tokens / prefill_tok_s + decode_token_latencies / decode_tok_s
  decode token-latencies: A,B are SERIAL  -> C*T*D ;  C is BATCHED -> C*T*D / batch_eff
  (batch_eff = how many agents one weight-stream step serves; bandwidth-bound ideal ~ C. We CAP
   the projection at batch_eff <= C so small-model per-call-overhead amortization -- which inflates
   the measured ratio and will NOT transfer to a Metal forward -- cannot over-claim.)

HONESTY FENCES
  * arm A prefill is a LOWER BOUND: flat tok/s ignores the O(L^2) growth of prefill self-attention,
    which only makes A *worse*. Reported as ">=".
  * the per-token rates are SINGLE-STREAM measurements at short context; late-turn prefill/decode
    cost rises as KV grows. For the terse regime here (ctx_max = P + T*(D+R)) the rise is modest;
    the live reproduce command below is the falsifiable ground truth.
  * absolute wall-clock is a PROJECTION onto the target host's rates; only the work counts and the
    cross-arm ratios are exact. Reproduce live on the target with the printed sessionbench command.
"""
import argparse, json, sys

# Committed M3 Pro Qwen2.5-7B-Instruct Q8 rate cards (single-stream, short context).
# Source: docs/benchmarks/QWEN25-7B-RESULTS.md, BENCHMARK-AUTHORITY row "Qwen2.5-7B" (commit 34c74f4),
# artifact experiments/model-ladder/modelbench-qwen25-7b-q8.json.
RATE_CARDS = {
    # MEASURED 2026-06-22 on node-macos-a (Apple M3 Pro) via llama-batched-bench on the 7B Q8 GGUF
    # (Metal -ngl 99, flash-attn). prefill 392 t/s; decode 17.4 t/s single-stream; AND the directly
    # measured 5-way batched aggregate decode 47.2 t/s (2.71x single, NOT the ideal 5x). When
    # decode_batch_agg_tok_s is set the projector uses it for arm C's batched decode directly,
    # rather than single/batch_eff -- the honest, measured fleet-decode floor.
    "llamacpp-metal-m3-measured": {"prefill_tok_s": 392.1, "decode_tok_s": 17.41,
                          "decode_batch_agg_tok_s": 44.0,  # context-averaged over the fleet's 2k->8.4k span:
                          # measured 5-way batched decode 47.2 (ctx 2k) / 44.4 (5k) / 41.9 (8k) t/s
                          "note": "MEASURED M3 Pro llama-batched-bench 2026-06-22, Qwen2.5-7B Q8 (macbook-m3pro-7b-batched-ctx.log)"},
    # the older committed single-stream card (kept for provenance / comparison).
    "llamacpp-metal-m3": {"prefill_tok_s": 192.92, "decode_tok_s": 17.27,
                          "note": "llama.cpp b9707 Metal -ngl 99, M3 Pro, Q8_0 (QWEN25-7B-RESULTS.md)"},
    # fak's own in-kernel forward today (pure-Go CPU Q8/NEON). The honest single-stream gap:
    # the Metal GEMM decode lane is open (internal/metalgemm), so this is the floor fak's kernel
    # orchestrates around, not its ceiling.
    "fak-purego-m3":     {"prefill_tok_s": 16.1,  "decode_tok_s": 8.7,
                          "note": "fak in-kernel CPU Q8 NEON, M3 Pro (QWEN25-7B-RESULTS.md)"},
}

def prefill_tokens(P, T, C, D, R):
    a = C * sum(P + t * (D + R) for t in range(T))   # naive: re-prefill whole context each turn
    b = C * (P + (T - 1) * R)                         # per-agent: prefix x C + incremental
    c = P + C * (T - 1) * R                           # fak: prefix ONCE total + incremental
    return a, b, c

def project(P, T, C, D, R, rate, batch_eff):
    pre, dec = rate["prefill_tok_s"], rate["decode_tok_s"]
    a_tok, b_tok, c_tok = prefill_tokens(P, T, C, D, R)
    serial_dec_s = (C * T * D) / dec   # A, B decode serially at the single-stream rate
    # C batches all agents into one weight stream. Prefer a DIRECTLY MEASURED aggregate batched
    # throughput when the rate card carries one (the honest floor); else fall back to single/batch_eff.
    agg = rate.get("decode_batch_agg_tok_s")
    batched_dec_s = (C * T * D) / agg if agg else (C * T * D / batch_eff) / dec
    arms = {
        "A_naive_stateless": {"prefill_s": a_tok / pre, "decode_s": serial_dec_s,
                              "prefill_tokens": a_tok, "lower_bound": True},
        "B_per_agent_kv":    {"prefill_s": b_tok / pre, "decode_s": serial_dec_s,
                              "prefill_tokens": b_tok, "lower_bound": False},
        "C_fak_fused":       {"prefill_s": c_tok / pre, "decode_s": batched_dec_s,
                              "prefill_tokens": c_tok, "lower_bound": False},
    }
    for k, v in arms.items():
        v["total_s"] = v["prefill_s"] + v["decode_s"]
    return arms

def measured_batch_eff(artifact_path, C):
    """Read the batching speedup actually measured by sessionbench: arm B (serial) decode_ms /
    arm C (batched) decode_ms. Both arms decode the SAME C*T*D token-decodes, so the ratio IS the
    effective batch efficiency. Capped at C for the projection (see module docstring)."""
    with open(artifact_path) as f:
        rep = json.load(f)
    cell = rep["cells"][0]
    raw = cell["arm_B_per_agent_kv"]["decode_ms"] / cell["arm_C_fak_fused"]["decode_ms"]
    return raw, min(raw, float(C))

def fmt_dur(s):
    if s >= 3600: return f"{s/3600:.1f} h"
    if s >= 60:   return f"{s/60:.1f} min"
    return f"{s:.1f} s"

def selftest():
    """Assert the exact prefill-token arithmetic (the contention-free floor under every projected
    minute) against hand-checked values. Mirrors `go test ./cmd/sessionbench`'s count gate."""
    # 5 agents x 200 turns, P=2048 D=20 R=40 (the headline regime)
    a, b, c = prefill_tokens(2048, 200, 5, 20, 40)
    assert c == 2048 + 5 * 199 * 40 == 41848, c
    assert b == 5 * (2048 + 199 * 40) == 50040, b
    assert a == 5 * sum(2048 + t * 60 for t in range(200)) == 8018000, a
    assert round(a / c, 1) == 191.6, a / c
    assert round(b / c, 2) == 1.20, b / c
    # 5 agents x 50 turns, P=2048 D=32 R=64 (the committed headline-qwen-50x5 regime)
    a2, b2, c2 = prefill_tokens(2048, 50, 5, 32, 64)
    assert c2 == 2048 + 5 * 49 * 64 == 17728, c2
    assert b2 == 5 * (2048 + 49 * 64) == 25920, b2  # matches headline-qwen-50x5.json prefill_tokens.b
    print("selftest OK: prefill-token arithmetic exact (5x200 and 5x50 regimes)")

def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--turns",  type=int, default=200)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--decode", type=int, default=20)
    ap.add_argument("--result", type=int, default=40)
    ap.add_argument("--budget-min", type=float, default=10.0, help="the <N min target to test against")
    ap.add_argument("--artifact", default="", help="sessionbench JSON to read the measured batch_eff from")
    ap.add_argument("--batch-eff", type=float, default=0.0, help="override batch_eff (default: measured, capped at agents)")
    ap.add_argument("--out", default="", help="write the projection JSON here")
    args = ap.parse_args()

    P, T, C, D, R = args.prefix, args.turns, args.agents, args.decode, args.result
    budget_s = args.budget_min * 60.0

    raw_be, be = None, args.batch_eff
    if args.batch_eff <= 0:
        if not args.artifact:
            be = float(C)  # ideal bandwidth-bound assumption if no measurement supplied
        else:
            raw_be, be = measured_batch_eff(args.artifact, C)

    print(f"# fleet 5x200 -> 7B MacBook wall-clock projection")
    print(f"workload: agents C={C}  turns T={T}  prefix P={P}  decode D={D}/turn  result R={R}/turn")
    print(f"context grows P -> P+T*(D+R) = {P} -> {P + T*(D+R)} tokens/agent")
    if raw_be is not None:
        print(f"batch_eff: measured {raw_be:.2f}x (arm B/C decode) -> using {be:.2f}x (capped at agents={C})")
    else:
        print(f"batch_eff: {be:.2f}x")
    print(f"target: arm C (fak fused) < {args.budget_min:.0f} min ({budget_s:.0f}s)\n")

    out = {"schema": "fak.fleet-10min-projection.v1",
           "workload": {"prefix": P, "turns": T, "agents": C, "decode": D, "result": R},
           "budget_s": budget_s, "batch_eff": be, "batch_eff_raw": raw_be,
           "rate_cards": {}, "verdict": {}}

    for name, rate in RATE_CARDS.items():
        arms = project(P, T, C, D, R, rate, be)
        c_total = arms["C_fak_fused"]["total_s"]
        passed = c_total < budget_s
        out["rate_cards"][name] = {"rate": rate, "arms": arms,
                                   "fak_under_budget": passed, "fak_total_s": c_total}
        out["verdict"][name] = passed
        print(f"## rate card: {name}  ({rate['note']})")
        print(f"   {'arm':<20} {'prefill':>10} {'decode':>10} {'total':>10}")
        for k in ("A_naive_stateless", "B_per_agent_kv", "C_fak_fused"):
            v = arms[k]
            lb = ">=" if v["lower_bound"] else "  "
            print(f"   {k:<20} {fmt_dur(v['prefill_s']):>10} {fmt_dur(v['decode_s']):>10} {lb}{fmt_dur(v['total_s']):>9}")
        bc = arms["B_per_agent_kv"]["total_s"] / c_total
        ac = arms["A_naive_stateless"]["total_s"] / c_total
        print(f"   fak fused: {fmt_dur(c_total)}  ->  {'UNDER' if passed else 'OVER'} the {args.budget_min:.0f}-min budget")
        print(f"   vs tuned per-agent KV: {bc:.1f}x   vs naive (>=): {ac:.0f}x\n")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
