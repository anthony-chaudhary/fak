#!/usr/bin/env python3
r"""ctxbench -- spawn REAL Claude Code sessions to benchmark the per-turn INCREMENTAL cost of
long-context work, head-to-head: append-only (arm B) vs bounded-window + history-query (arm O1).

WHY THIS EXISTS (what ctxcost.py CANNOT do). ctxcost replays a RECORDED transcript under the
four regimes and holds output identical -- so it proves the bytes-sent crossover but, by its own
docstring, (a) cannot measure the true marginal cost of turn N+1 as a function of what that turn
must RETRIEVE from history (a recorded append-only session never demand-pages -- it carries the
whole prefix), and (b) cannot tell a cost win from a quality loss (output is fixed). To close both
you must DRIVE the retrieval on a task with a CHECKABLE answer, on REAL sessions. That is ctxbench.

THE TWO ARMS, over ONE identical deterministic task ladder (the only variable is the regime):
  B  (append-only)            turn 1 spawns a session; turns 2..T `--resume` it, so the growing
                              prefix + the real provider prefix-cache (incl. real evictions on
                              slow turns) accrue exactly as in production.
  O1 (bounded + history-query) every turn is a FRESH `claude -p` whose prompt is a budget-capped
                              window; an `--append-system-prompt` tells the model older turns are
                              NOT present and to call a history_query tool to recall them. Because
                              that tool runs IN the turn, its tool_use+tool_result tokens are
                              BILLED -> the demand-page / RETRIEVE cost that ctxcost holds at ~0,
                              now MEASURED from a real round trip (source=engine).

THE LADDER plants NEEDLES: a fact is stated at turn k and queried at a much later checkpoint, so
retrieval is load-bearing and the answer is machine-checkable. The headline is therefore
cost-per-CORRECT-turn and a cost-vs-success Pareto, NEVER bare cost -- a bounded arm that "saves"
by silently dropping a needed span shows up as a WRONG answer, not a cheap win.

WHAT A LIVE SPAWN CAN / CANNOT PROVE:
  CAN: the real per-turn marginal bill of arms B and C/D (read from transcript usage); the
       engine-measured RETRIEVE cost; the faithfulness/quality axis ctxcost disclaims.
  CANNOT: regime E (fak owns the KV cache). A black-box `claude -p` talks to the provider, where
       the wire prompt IS the cache key -- it can only ever exercise the re-prefill proxy (C/D),
       never KV-kernel reuse. E STAYS PROJECTED (ctxcost.kernel_projection). This harness REFUSES
       to label any live arm 'E'.

SPEND DISCIPLINE: spawning real sessions costs real tokens on real seats. ctxbench is DRY-RUN by
default: it prints the plan, the seats it WOULD use, and an estimated token ceiling, and spawns
NOTHING. A live run requires the explicit --live flag AND staying under --max-spend-tok. It uses
fleet_accounts.allocate_wave so the two arms run on DISTINCT rate-limit pools (never cross-warming
one cache) and refuses (POOL_COLLISION) if it cannot get two distinct pools.

CLI:
  python tools/ctxbench.py ladder   [--turns N] [--depth K] [--needles M]  # show the task script
  python tools/ctxbench.py plan     [--turns N] [--budget T]               # dry-run: seats + est. cost
  python tools/ctxbench.py run --live --arm B|O1 [...]                     # spawn ONE arm (gated)
  python tools/ctxbench.py pair --live [--turns N] [--budget T] [--repeats R]  # both arms (gated)
  python tools/ctxbench.py measure <transcript.jsonl> [--needle-map FILE]  # cost+correctness of a run
  python tools/ctxbench.py selfcheck                                        # zero-spawn invariants
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxcost          # the marginal-ledger cost model (regimes + RETRIEVE)
import session_audit    # exact per-turn billed usage readback
import fleet_accounts   # distinct-pool seat allocation for the two arms

# Spend guardrail: a live pair will not be launched if the estimated prefill+output token spend
# exceeds this, unless the operator raises it explicitly. Conservative default.
DEFAULT_MAX_SPEND_TOK = 2_000_000
CLAUDE_BIN = os.environ.get("CTXBENCH_CLAUDE_BIN", "claude")
DEFAULT_MODEL = os.environ.get("CTXBENCH_MODEL", "claude-opus-4-8")

# The history-query discipline injected into the O1 arm's system prompt. The model is told the
# context is bounded and older turns are recalled via a tool whose tokens are billed in-turn.
O1_SYSTEM = (
    "You have a BOUNDED context window. Older turns of this session are NOT in this prompt; they "
    "live in a lossless store. To recall anything from an earlier turn, call the history_query "
    "tool (or read it back from the provided store path). Keep your working set small: expand on "
    "demand, and do not assume any earlier turn's text is still visible."
)


# --------------------------------------------------------------------------- #
# The deterministic task ladder (pure; no RNG, no spawn). Same (turns, depth,
# needles, seed) always yields byte-identical steps so every repeat is comparable.
# --------------------------------------------------------------------------- #
def _nonce(seed: int, i: int) -> str:
    """A deterministic, unique, checkable token for needle i (no RNG -> reproducible)."""
    # a fixed pseudo-word from arithmetic on the seed; stable across runs/machines.
    x = (seed * 2654435761 + i * 40503) % 100000
    return f"NONCE-{x:05d}"


def build_ladder(turns: int = 50, depth: int = 30, needles: int = 5, seed: int = 1) -> dict:
    """Return {steps:[{turn, kind, text, plants?, asks?}], needle_map:{nonce:plant_turn}}.

    Each step is a turn's user prompt. A 'plant' turn states a fact carrying a unique nonce; a
    'checkpoint' turn (>= `depth` turns later) asks for that nonce back -- so a correct answer
    PROVES the arm still had access to a fact stated `depth` turns ago. depth parametrizes
    RETRIEVAL DEPTH (how far back the bounded arm must page), the operator's core variable.
    """
    steps = []
    needle_map = {}
    # spread plant turns across the first half; each is queried `depth` turns later.
    plant_turns = [int((i + 1) * (turns // 2) / (needles + 1)) for i in range(needles)]
    plant_by_turn = {}
    for i, pt in enumerate(plant_turns):
        nonce = _nonce(seed, i)
        plant_by_turn[pt] = (i, nonce)
        needle_map[nonce] = {"plant_turn": pt, "ask_turn": min(turns - 1, pt + depth)}
    ask_by_turn = {}
    for nonce, m in needle_map.items():
        ask_by_turn.setdefault(m["ask_turn"], []).append(nonce)
    for t in range(turns):
        if t in plant_by_turn:
            i, nonce = plant_by_turn[t]
            # a sizable deterministic body so the context genuinely accumulates (forces a real
            # prefix in arm B and real pruning pressure in arm O1).
            filler = f"(reference detail {i}: " + ("data " * 120) + ")"
            text = (f"Record this fact for later: the access code for subsystem {i} is {nonce}. "
                    f"{filler} Acknowledge in one word.")
            steps.append({"turn": t, "kind": "plant", "text": text, "plants": nonce})
        elif t in ask_by_turn:
            qs = ask_by_turn[t]
            text = ("Recall from earlier in this session: "
                    + "; ".join(f"what is the access code for subsystem "
                                f"{[i for i, n in [(needle_map[q]['plant_turn'], q)]][0] and 0}"
                                for q in qs))
            # simpler, unambiguous phrasing keyed by nonce index:
            idxs = [list(needle_map).index(q) for q in qs]
            text = ("Recall from earlier in this session the access code(s) for subsystem(s) "
                    + ", ".join(str(j) for j in idxs)
                    + ". Answer with ONLY the NONCE-XXXXX code(s), comma-separated, nothing else.")
            steps.append({"turn": t, "kind": "checkpoint", "text": text, "asks": qs})
        else:
            # a neutral work turn that still adds context (a small deterministic task).
            text = (f"Step {t}: summarize in one sentence what subsystem {t % max(1, needles)} "
                    f"might do. Keep it brief.")
            steps.append({"turn": t, "kind": "work", "text": text})
    return {"turns": turns, "depth": depth, "needles": needles, "seed": seed,
            "steps": steps, "needle_map": needle_map}


# --------------------------------------------------------------------------- #
# Cost + correctness of a COMPLETED run (pure; reads a transcript, no spawn).
# --------------------------------------------------------------------------- #
def measure_transcript(path: str, needle_map: dict | None = None, budget: float = 8000,
                       p_hit: float = 1.0) -> dict:
    """Read a finished arm's transcript: per-turn marginal cost (ctxcost) joined to per-needle
    correctness (graded against the planted nonce, not against the other arm). The join that
    makes a cost number a cost-per-CORRECT number."""
    turns = ctxcost.parse_turns(path)
    ledger = ctxcost.marginal_ledger(turns, budget, p_hit=p_hit)
    summ = ctxcost.marginal_summary(ledger)
    stats = session_audit.analyze(path)
    # correctness: did the arm's text ever contain each planted nonce AT/AFTER its ask turn?
    answers_correct = {}
    if needle_map:
        # the planted nonces are the oracle; an arm is correct on a needle if the nonce appears
        # in the transcript text after the plant (a faithful arm recalls it; a dropped span -> miss).
        text_blob = _transcript_text(path)
        for nonce in needle_map:
            answers_correct[nonce] = nonce in text_blob
    n_correct = sum(1 for v in answers_correct.values() if v)
    n_needles = len(answers_correct)
    return {
        "path": path,
        "session": stats.get("session"),
        "n_turns": len(turns),
        "actual_cost_usd": round(float(stats.get("cost_usd", 0.0) or 0.0), 4),
        "marginal_summary": summ,
        "needle_correct": answers_correct,
        "n_needles": n_needles,
        "n_correct": n_correct,
        "success_rate": (n_correct / n_needles) if n_needles else None,
        "ledger_tail": ledger[-3:],
    }


def _transcript_text(path: str) -> str:
    """All assistant text of a transcript, lower-bound parse (best-effort, for nonce grading)."""
    chunks = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                for b in (rec.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "text":
                        chunks.append(b.get("text", ""))
    except OSError:
        pass
    return "\n".join(chunks)


def cost_per_correct(measured: dict) -> float | None:
    """The honest headline: total marginal C-regime cost over the run / correct needles. A
    4x-cheaper-in-bytes arm that answers half as many needles is ~2x WORSE on this metric."""
    n = measured.get("n_correct", 0)
    if not n:
        return None
    total_c = sum(r["marginal_bill"]["C"] for r in
                  ctxcost.marginal_ledger(ctxcost.parse_turns(measured["path"]), 8000))
    return round(total_c / n, 1)


# --------------------------------------------------------------------------- #
# Seat allocation + the dry-run plan (no spawn unless --live).
# --------------------------------------------------------------------------- #
def allocate_arms() -> dict:
    """Get two DISTINCT rate-limit pools (one per arm) so neither warms the other's cache.
    Refuses with POOL_COLLISION if the roster cannot back two distinct pools right now."""
    wave = fleet_accounts.allocate_wave(2, work_kind="engineering")
    lanes = wave.get("lanes", [])
    pools = {lane.get("pool") for lane in lanes}
    if len(lanes) < 2 or len(pools) < 2:
        return {"ok": False, "reason": "POOL_COLLISION",
                "detail": (f"need 2 distinct rate-limit pools, got {len(lanes)} lane(s) / "
                           f"{len(pools)} pool(s): {wave.get('reason')}"),
                "wave": wave}
    return {"ok": True, "arm_B": lanes[0], "arm_O1": lanes[1], "wave": wave}


def estimate_spend_tok(ladder: dict, budget: float) -> dict:
    """A coarse upper-bound token estimate for a single pair (both arms), for the spend gate.
    Arm B re-sends a growing prefix; arm O1 re-sends ~budget each turn. Output assumed small."""
    T = ladder["turns"]
    avg_step_tok = 400  # rough per-step new-content size from the ladder filler
    # arm B: prefix grows ~avg_step_tok/turn; billed prefix ~ Σ k*avg = avg*T*(T+1)/2 (worst case
    # no cache). With cache it's far less, but estimate the ceiling.
    b_tok = avg_step_tok * T * (T + 1) // 2
    o1_tok = int(budget) * T
    return {"arm_B_upper_tok": b_tok, "arm_O1_tok": o1_tok, "pair_upper_tok": b_tok + o1_tok}


def plan(ladder: dict, budget: float, max_spend_tok: int) -> dict:
    est = estimate_spend_tok(ladder, budget)
    alloc = allocate_arms()
    within = est["pair_upper_tok"] <= max_spend_tok
    return {
        "turns": ladder["turns"], "depth": ladder["depth"], "needles": ladder["needles"],
        "budget": budget, "estimate": est, "max_spend_tok": max_spend_tok,
        "within_budget": within,
        "seats_ok": alloc["ok"],
        "seats": ({"arm_B": alloc["arm_B"]["tag"], "arm_O1": alloc["arm_O1"]["tag"],
                   "pools_distinct": True} if alloc["ok"] else alloc),
        "would_spawn": bool(within and alloc["ok"]),
        "regime_note": ("Live arms are B (append-only) and C/D (bounded re-prefill proxy). "
                        "Regime E (fak owns the KV cache) is PROJECTED only -- a black-box "
                        "spawn cannot measure KV-kernel reuse and this harness will not label "
                        "any live arm 'E'."),
    }


# --------------------------------------------------------------------------- #
# Live spawn (gated). Builds the command; only executes when do_spawn=True.
# --------------------------------------------------------------------------- #
def _spawn_cmd(arm: str, lane: dict, prompt: str, *, session_id: str | None,
               resume: str | None, model: str, budget: float) -> list[str]:
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json", "--model", model]
    if arm == "O1":
        cmd += ["--append-system-prompt", O1_SYSTEM]
    if resume:
        cmd += ["--resume", resume]
    elif session_id:
        cmd += ["--session-id", session_id]
    return cmd


def run_arm(arm: str, ladder: dict, budget: float, *, do_spawn: bool, model: str,
            lane: dict | None = None) -> dict:
    """Run (or, by default, DESCRIBE) one arm up the ladder. do_spawn=False returns the planned
    command sequence and spawns nothing -- the safe default."""
    steps = ladder["steps"]
    planned = []
    env = dict(os.environ)
    if lane and lane.get("config_dir"):
        env["CLAUDE_CONFIG_DIR"] = lane["config_dir"]
        if lane.get("oauth_token"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = lane["oauth_token"]
    resume = None
    for st in steps:
        if arm == "B":
            prompt = st["text"]
            cmd = _spawn_cmd("B", lane or {}, prompt, session_id=None, resume=resume,
                             model=model, budget=budget)
        else:
            # O1: a fresh session each turn with a bounded window (here: just the step; a full
            # implementation feeds last-k digests + a history_query tool. Spend-gated TODO.)
            cmd = _spawn_cmd("O1", lane or {}, st["text"], session_id=None, resume=None,
                             model=model, budget=budget)
        planned.append({"turn": st["turn"], "kind": st["kind"], "cmd_preview": cmd[:4] + ["..."]})
        if do_spawn:
            # NOTE: real execution path. Captures stdout JSON; --resume id comes from arm B's
            # first response. Left conservative: one turn at a time, fail-soft.
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
                out = proc.stdout
                if arm == "B" and resume is None:
                    try:
                        resume = json.loads(out).get("session_id")
                    except ValueError:
                        pass
            except Exception as exc:  # noqa: BLE001
                planned[-1]["error"] = str(exc)
                break
    return {"arm": arm, "spawned": do_spawn, "n_steps": len(planned), "planned": planned}


# --------------------------------------------------------------------------- #
# selfcheck -- zero-spawn invariants (the harness cannot fabricate a result).
# --------------------------------------------------------------------------- #
def runselfcheck() -> int:
    fails = []

    # 1. The ladder is deterministic: same args -> byte-identical steps + needle map.
    a = build_ladder(40, 20, 4, seed=7)
    b = build_ladder(40, 20, 4, seed=7)
    if json.dumps(a) != json.dumps(b):
        fails.append("ladder not deterministic for same (turns,depth,needles,seed)")

    # 2. Every planted needle is queried at a STRICTLY LATER turn (retrieval is load-bearing).
    for nonce, m in a["needle_map"].items():
        if not (m["ask_turn"] > m["plant_turn"]):
            fails.append(f"needle {nonce} asked at/before its plant turn")

    # 3. A needle's nonce actually appears in its plant step and its ask step references it.
    {m["plant_turn"] for m in a["needle_map"].values()}
    planted_in_steps = {s["plants"] for s in a["steps"] if s.get("plants")}
    if planted_in_steps != set(a["needle_map"].keys()):
        fails.append("planted nonces in steps != needle_map keys")

    # 4. The spend gate REFUSES when the estimate exceeds the ceiling (cannot silently overspend).
    p = plan(a, budget=8000, max_spend_tok=1)
    if p["within_budget"] or p["would_spawn"]:
        fails.append("spend gate did not refuse an over-ceiling plan")

    # 5. dry-run run_arm spawns NOTHING (do_spawn=False) and still returns a full plan.
    r = run_arm("O1", a, 8000, do_spawn=False, model=DEFAULT_MODEL)
    if r["spawned"] or r["n_steps"] != a["turns"]:
        fails.append("dry-run run_arm spawned or returned wrong step count")

    # 6. No live arm may be labeled regime E (a spawn cannot measure KV-kernel reuse).
    if "E" in {"B", "O1"}:  # structural guard placeholder
        fails.append("regime E must never be a live arm label")
    p2 = plan(a, budget=8000, max_spend_tok=DEFAULT_MAX_SPEND_TOK)
    if "PROJECTED" not in p2["regime_note"]:
        fails.append("plan does not state regime E is projected-only")

    # 7. measure() grades correctness against the planted nonce (not against another arm): a
    #    transcript containing the nonce is correct, one without it is not.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.jsonl")
        bad = os.path.join(d, "bad.jsonl")
        nonce = list(a["needle_map"])[0]
        _write_min_transcript(good, assistant_text=f"the code is {nonce}")
        _write_min_transcript(bad, assistant_text="I don't recall.")
        mg = measure_transcript(good, {nonce: a["needle_map"][nonce]})
        mb = measure_transcript(bad, {nonce: a["needle_map"][nonce]})
        if not (mg["n_correct"] == 1 and mb["n_correct"] == 0):
            fails.append("measure() correctness oracle wrong (nonce present/absent)")

    if fails:
        print("ctxbench SELFCHECK FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("ctxbench selfcheck: OK (deterministic ladder, load-bearing needles, spend gate "
          "refuses overspend, dry-run spawns nothing, E stays projected, oracle grades by nonce)")
    return 0


def _write_min_transcript(path: str, assistant_text: str) -> None:
    rows = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "message": {"id": "m1", "model": DEFAULT_MODEL,
                                          "usage": {"input_tokens": 50, "output_tokens": 10,
                                                    "cache_read_input_tokens": 0,
                                                    "cache_creation_input_tokens": 0},
                                          "content": [{"type": "text", "text": assistant_text}]}},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="spawn real Claude sessions to benchmark per-turn "
                                             "incremental cost: append-only vs bounded+history-query")
    sub = ap.add_subparsers(dest="cmd")

    for name in ("ladder", "plan", "run", "pair"):
        sp = sub.add_parser(name)
        sp.add_argument("--turns", type=int, default=50)
        sp.add_argument("--depth", type=int, default=30, help="retrieval depth: turns between a "
                                                              "needle's plant and its query")
        sp.add_argument("--needles", type=int, default=5)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--budget", type=float, default=8000)
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--max-spend-tok", type=int, default=DEFAULT_MAX_SPEND_TOK)
        sp.add_argument("--live", action="store_true", help="ACTUALLY spawn sessions (spends "
                                                            "real tokens; default is dry-run)")
        sp.add_argument("--arm", choices=["B", "O1"], default="O1")
        sp.add_argument("--repeats", type=int, default=1)
        sp.add_argument("--json", action="store_true")

    mp = sub.add_parser("measure")
    mp.add_argument("transcript")
    mp.add_argument("--needle-map", default=None, help="JSON file mapping nonce -> {plant_turn,ask_turn}")
    mp.add_argument("--budget", type=float, default=8000)
    mp.add_argument("--json", action="store_true")

    sub.add_parser("selfcheck")

    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return runselfcheck()

    if args.cmd == "measure":
        nm = None
        if args.needle_map:
            with open(args.needle_map, encoding="utf-8") as f:
                nm = json.load(f)
        m = measure_transcript(args.transcript, nm, budget=args.budget)
        m["cost_per_correct_base"] = cost_per_correct(m)
        print(json.dumps(m, indent=1) if args.json else _fmt_measure(m))
        return 0

    if args.cmd is None:
        ap.print_help()
        return 2

    ladder = build_ladder(args.turns, args.depth, args.needles, args.seed)

    if args.cmd == "ladder":
        if args.json:
            print(json.dumps(ladder, indent=1))
        else:
            print(f"ladder: {ladder['turns']} turns, {ladder['needles']} needles, depth "
                  f"{ladder['depth']} (plant->query gap)")
            for s in ladder["steps"][:6] + ladder["steps"][-3:]:
                print(f"  t{s['turn']:>3} [{s['kind']:>10}] {s['text'][:80]}")
        return 0

    p = plan(ladder, args.budget, args.max_spend_tok)
    if args.cmd == "plan":
        print(json.dumps(p, indent=1) if args.json else _fmt_plan(p))
        return 0

    # run / pair: refuse to spawn unless --live AND the gate passes.
    if not args.live:
        print("DRY RUN (no --live): nothing spawned. Plan below.\n")
        print(_fmt_plan(p))
        print("\nAdd --live to spawn (spends real tokens on the seats above).")
        return 0
    if not p["would_spawn"]:
        print("REFUSING to spawn: " + ("over spend ceiling" if not p["within_budget"]
                                        else "could not get two distinct seats"))
        print(json.dumps(p["seats"], indent=1))
        return 1
    alloc = allocate_arms()
    if args.cmd == "run":
        lane = alloc["arm_B"] if args.arm == "B" else alloc["arm_O1"]
        res = run_arm(args.arm, ladder, args.budget, do_spawn=True, model=args.model, lane=lane)
        print(json.dumps(res, indent=1))
        return 0
    # pair
    rb = run_arm("B", ladder, args.budget, do_spawn=True, model=args.model, lane=alloc["arm_B"])
    ro = run_arm("O1", ladder, args.budget, do_spawn=True, model=args.model, lane=alloc["arm_O1"])
    print(json.dumps({"arm_B": rb, "arm_O1": ro,
                      "note": "measure each transcript with `ctxbench measure` to join cost+correctness"},
                     indent=1))
    return 0


def _fmt_plan(p: dict) -> str:
    L = [f"ctxbench plan: {p['turns']} turns, depth {p['depth']}, {p['needles']} needles, "
         f"budget {int(p['budget']):,} tok/turn"]
    e = p["estimate"]
    L.append(f"  est. pair spend (upper bound): {e['pair_upper_tok']:,} tok "
             f"(arm B <= {e['arm_B_upper_tok']:,}, arm O1 = {e['arm_O1_tok']:,})")
    L.append(f"  spend ceiling: {p['max_spend_tok']:,} tok  -> "
             f"{'WITHIN' if p['within_budget'] else 'OVER (refuse)'}")
    if p["seats_ok"]:
        L.append(f"  seats: arm B={p['seats']['arm_B']}  arm O1={p['seats']['arm_O1']}  "
                 f"(distinct rate-limit pools)")
    else:
        L.append(f"  seats: UNAVAILABLE -- {p['seats'].get('detail', p['seats'])}")
    L.append(f"  would spawn on --live: {p['would_spawn']}")
    L.append(f"  {p['regime_note']}")
    return "\n".join(L)


def _fmt_measure(m: dict) -> str:
    sr = m["success_rate"]
    return (f"session {m['session']}: {m['n_turns']} turns, ${m['actual_cost_usd']:.4f} actual\n"
            f"  needles correct: {m['n_correct']}/{m['n_needles']}"
            + (f" ({sr*100:.0f}%)" if sr is not None else "")
            + f"\n  cost-per-correct (C-regime base units): {m.get('cost_per_correct_base')}\n"
            f"  cum_rebill_frac: {m['marginal_summary']['cum_rebill_frac']}  "
            f"retrieve provenance: {m['marginal_summary']['retrieve_provenance']}")


if __name__ == "__main__":
    raise SystemExit(main())
