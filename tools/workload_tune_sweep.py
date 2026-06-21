#!/usr/bin/env python3
"""
workload_tune_sweep.py — sweep a benchmark-tuning knob over a transcript-derived workload
and record how fak's prefix-reuse win responds.

This is the "tuning over time / what-if" layer the goal asks for: take the real workload
profile (tools/transcript_workload.py), then vary ONE knob — by default the tool-call
fraction, i.e. "simulate what % of turns are tool calls" — and watch the throughput and
the reuse-vs-no-reuse speedup move. Because the result-ingest between turns IS the per-agent
KV growth, raising the tool-call fraction raises the share of work that prefix-reuse cannot
amortise, so the knob directly stresses the thing fak is built to win.

It shells out to a prebuilt `fleetserve` binary in -workload mode once per knob value and
folds the per-concurrency points into one sweep artifact + table. This is an
internal ablation, not the tuned external reference; use
tools/fak_llama_turn_agent_compare.py for the fair head-to-head row.

Usage:
  python tools/workload_tune_sweep.py \
      --bin fak/.benchbin/fleetserve.exe \
      --model-dir fak/internal/model/.cache/smollm2-135m \
      --profile fak/experiments/agent-live/realistic-workload/profile.json \
      --knob toolfrac --grid 0.25,0.5,1,1.5,2,3 \
      --concurrency 16 --track-pct 50 --turn-cap 24 --tune-decode 0.2 --tune-prefix 0.1 --reps 2 \
      --out fak/experiments/agent-live/realistic-workload/tune-sweep.json \
      --md  fak/experiments/agent-live/realistic-workload/TUNE-SWEEP.md
"""
import argparse, json, os, subprocess, sys, tempfile

import fleet_version

KNOB_FLAG = {
    "toolfrac": "-tune-toolfrac",
    "result":   "-tune-result",
    "decode":   "-tune-decode",
    "prefix":   "-tune-prefix",
}


def run_point(binp, profile, knob, value, concurrency, track_pct, turn_cap,
              tune_decode, tune_result, reps, quant, model_dir=None, tune_prefix=1.0,
              timeout_s=240):
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        outp = tf.name
    cmd = [binp]
    if quant:
        cmd.append("-quant")
    if model_dir:
        cmd += ["-dir", model_dir]
    cmd += ["-workload", profile, "-track-pct", str(track_pct),
            "-concurrency", concurrency, "-reps", str(reps), "-out", outp]
    if turn_cap:
        cmd += ["-turn-cap", str(turn_cap)]
    # base scales (applied to every point); the swept knob overrides its own flag
    scales = {"-tune-decode": str(tune_decode), "-tune-result": str(tune_result),
              "-tune-prefix": str(tune_prefix)}
    scales[KNOB_FLAG[knob]] = str(value)
    for k, v in scales.items():
        cmd += [k, v]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"\n[warn] fleetserve timed out (> {timeout_s}s) for {knob}={value}; skipping point\n")
        try:
            os.unlink(outp)
        except OSError:
            pass
        return None
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-1000:])
        # Skip this point rather than abandoning the whole sweep — a single transient
        # process kill (seen under heavy fleet CPU contention) shouldn't lose the rest.
        sys.stderr.write(f"\n[warn] fleetserve failed for {knob}={value} (rc={proc.returncode}); skipping point\n")
        try:
            os.unlink(outp)
        except OSError:
            pass
        return None
    with open(outp, encoding="utf-8") as f:
        rep = json.load(f)
    os.unlink(outp)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="prebuilt fleetserve binary")
    ap.add_argument("--model-dir", default=None, help="fleetserve -dir model export (else its default)")
    ap.add_argument("--profile", required=True, help="fak.workload.v1 profile JSON")
    ap.add_argument("--knob", choices=list(KNOB_FLAG), default="toolfrac")
    ap.add_argument("--grid", default="0.25,0.5,1,1.5,2,3", help="comma-separated knob multipliers")
    ap.add_argument("--concurrency", default="16", help="comma-separated agent counts")
    ap.add_argument("--track-pct", type=int, default=50)
    ap.add_argument("--turn-cap", type=int, default=24)
    ap.add_argument("--tune-decode", type=float, default=0.2,
                    help="base decode scale held fixed across the sweep (keeps wall-clock tractable)")
    ap.add_argument("--tune-result", type=float, default=1.0)
    ap.add_argument("--tune-prefix", type=float, default=1.0,
                    help="base prefix scale held fixed across the sweep (shrink for tractable no-reuse prefill)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=240, help="per-point fleetserve timeout (s)")
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    grid = [float(x) for x in a.grid.split(",") if x.strip()]
    app_ver = fleet_version.app_version()
    quant = not a.no_quant
    sweep = []
    for v in grid:
        rep = run_point(a.bin, a.profile, a.knob, v, a.concurrency, a.track_pct,
                        a.turn_cap, a.tune_decode, a.tune_result, a.reps, quant,
                        a.model_dir, a.tune_prefix, a.timeout)
        if rep is None:
            continue
        for p in rep.get("points", []):
            sweep.append({
                "version": app_ver,
                "knob": a.knob, "knob_value": v,
                "comparison_role": "internal_reuse_vs_noreuse_ablation",
                "effective_tool_call_fraction": rep.get("effective_toolfrac"),
                "result_tokens_total_per_agent": rep.get("result_tokens_total_per_agent"),
                "decode_tokens_total_per_agent": rep.get("decode_tokens_total_per_agent"),
                "turns": rep.get("turns"),
                "concurrency": p["concurrency"],
                "reuse_agent_turns_per_sec": p["reuse_agent_turns_per_sec"],
                "noreuse_agent_turns_per_sec": p["noreuse_agent_turns_per_sec"],
                "reuse_total_ms": p["reuse_total_ms"],
                "noreuse_total_ms": p["noreuse_total_ms"],
                "reuse_result_prefill_ms": p["reuse_result_prefill_ms"],
                "reuse_decode_ms": p["reuse_decode_ms"],
                "reuse_speedup_vs_noreuse": p["reuse_speedup_vs_noreuse"],
            })

    artifact = {
        "schema": "fak.workload-tune-sweep.v1",
        "app_version": app_ver,
        "profile": a.profile,
        "knob": a.knob,
        "grid": grid,
        "comparison_role": "internal_reuse_vs_noreuse_ablation",
        "headline_policy": {
            "use_for": "candidate_tuning_sensitivity",
            "not_for": "head-to-head claim against tuned serving engines",
            "external_reference_tool": "tools/fak_llama_turn_agent_compare.py",
        },
        "base": {"track_pct": a.track_pct, "turn_cap": a.turn_cap,
                 "tune_decode": a.tune_decode, "tune_result": a.tune_result,
                 "tune_prefix": a.tune_prefix, "concurrency": a.concurrency,
                 "reps": a.reps, "quant": quant},
        "points": sweep,
    }

    # table
    lines = []
    lines.append(f"# Workload tuning sweep — knob = {a.knob}\n")
    lines.append(f"Profile: `{a.profile}`  ·  track p{a.track_pct}, turn-cap {a.turn_cap}, "
                 f"decode×{a.tune_decode}, prefix×{a.tune_prefix}, reps {a.reps}, "
                 f"quant {quant}\n")
    lines.append("Knob scales the swept parameter over the real transcript track. "
                 "This is an internal reuse-vs-no-reuse ablation for candidate tuning "
                 "sensitivity, not a head-to-head claim against a tuned serving engine. "
                 "Throughput is fak with prefix reuse; ablation speedup is reuse / no-reuse "
                 "(same kernels, same decode; only the P-token preamble is shared vs re-prefilled).\n")
    hdr = f"| {a.knob}× | eff tool-frac | result tok/agent | C | reuse turns/s | no-reuse turns/s | ablation speedup |"
    lines.append(hdr)
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for s in sweep:
        eff = s["effective_tool_call_fraction"]
        lines.append(
            f"| {s['knob_value']:g} | {eff:.3f} | {s['result_tokens_total_per_agent']:,} | "
            f"{s['concurrency']} | {s['reuse_agent_turns_per_sec']:.2f} | "
            f"{s['noreuse_agent_turns_per_sec']:.2f} | {s['reuse_speedup_vs_noreuse']:.2f}× |")
    table = "\n".join(lines)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(artifact, open(a.out, "w", encoding="utf-8"), indent=2)
        sys.stderr.write(f"wrote {a.out}\n")
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w", encoding="utf-8").write(table + "\n")
        sys.stderr.write(f"wrote {a.md}\n")
    print(table)


if __name__ == "__main__":
    main()
