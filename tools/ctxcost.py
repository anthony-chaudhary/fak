#!/usr/bin/env python3
"""
ctxcost.py — per-turn COST replay: append-only-with-cache vs O(1)-context-plus-history.

The question (the contrarian peer of ctxwin.py): ctxwin proves you can HALVE the window
prefix-stably so the ~90% prompt cache survives. This tool asks the opposite — what if you
DON'T preserve the cache? If every turn reconstructs a fresh, BOUNDED (O(1)) context from a
lossless history store, you lose provider-side prefix-cache hits, but you send dramatically
less in the first place. Does that net out cheaper, and faster, than the cached append-only
loop? On a cache-friendly provider (warm Anthropic prompt cache) and on a "random API" with
no/poor caching?

The honest lever this exploits: a Claude Code transcript RECORDS the real billed accounting
of the append-only-with-cache regime, per turn:
    usage = {input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens}
so the incumbent's bill is MEASURED, not modelled. Total context sent that turn =
input + cache_creation + cache_read. We only have to MODEL the O(1)-reconstruct regime
against that ground truth and read off the crossover.

THE FOUR REGIMES (cost in base-input units; fresh input = 1.0×, the constant cancels in any
ratio and converts to $ by × input_$/Mtok):
  A  naive / cache-hostile         every turn re-sends the FULL prompt at full price (1.0×).
                                   The "random API with no prefix cache" world. Σ ~ Θ(n²).
  B  append-only + prefix cache    the MEASURED real bill: fresh·1.0 + write·1.25 + read·0.1.
                                   The status quo we are trying to beat (warm Anthropic).
  C  O(1) reconstruct, no cache    send a bounded window W_t = min(prompt_t, budget) at full
                                   price every turn (reconstruction mutates the prefix → no
                                   provider cache hit). Σ ~ Θ(n·budget) — LINEAR.
  D  O(1) reconstruct + stable     keep a byte-stable head S (system+tools) cached (read 0.1×
     cached prefix                 after a one-time 1.25× write); reconstruct only the tail
                                   (W_t − S) fresh each turn. The hybrid.
  E  fak OWNS the cache (PROJECTED, a COMPUTE axis, reported separately by kernel_projection()):
     when fak runs the engine the KV cache is an addressable kernel object, so the bounded view
     is reconstructed by REUSING + EVICTING cached spans (bit-exact, a single Kraw re-rotation,
     no re-prefill) instead of re-sending a prompt. You get bounded AND cached at once. Not live
     yet (the KV kernel is dormant on the proxy loop) — see kernel_projection() for the honest model.

DEFAULT MULTIPLIERS (Anthropic, base-input units; verified via the claude-api skill):
  fresh 1.0 · cache-read 0.1 · cache-write-5m 1.25 · cache-write-1h 2.0 · output 5.0
  (output is 5× input for Opus $5/$25, Sonnet $3/$15, Haiku $1/$5 alike → the crossover in
   base-units is model-INDEPENDENT; only the workload shape matters.)

WHAT THIS PROVES — AND WHAT IT DOES NOT:
  - A/B token counts are the provider's OWN billed usage (exact). The reconstruct budget W is
    a MODEL of the bounded planner (internal/ctxplan), not a measurement.
  - The per-turn "billed prompt" (fresh+write+read) is ~98% cache-READ of the same growing
    prefix, re-counted every turn. The genuinely-NEW context per turn is the uncached fresh+write
    delta (a few thousand tokens). The win is not re-billing that prefix. A cross-turn SUM of
    prompt is a token×turns area under the growing prefix, NEVER a distinct-context size.
  - The eviction sweep (--scenario evict:F) degrades regime B ONLY; C and D hold no provider
    cache, so they are immune to eviction BY CONSTRUCTION. So "the crossover widens as the cache
    degrades" is DEFINITIONAL (it raises B's bill against an invariant C), not an observed cache
    measurement. F is an ILLUSTRATIVE sensitivity dial, not a measured tool-latency-vs-5min-TTL
    distribution on this corpus.
  - Output tokens are held IDENTICAL across regimes. This assumes a faithful bounded context
    yields the same generation — that is the QUALITY axis, which this COST harness does NOT
    establish; it is the job of ctxplan's faithfulness witness + a task-success eval. State
    it loudly; never let a cost win masquerade as a quality claim.
  - Anti-overclaim (selfcheck): at budget ≥ max prompt, C must equal A exactly (nothing is
    pruned, so "reconstruct" is a no-op) — the tool cannot fabricate a saving.

OBSERVABILITY — the deeper reason for O(1)+history (the `trace` verb):
  Cost is one column. The point of keeping a lossless history store and reconstructing each
  turn is that the bounded view is a DETERMINISTIC function of (history, forecast, budget), so
  every step is REPLAYABLE and fully inspectable: what was billed, what was genuinely new, what
  the reconstruction would send vs PRUNE, and which turns are events — a cache-write spike, or a
  turn whose new content does not fit the budget (the faithfulness risk surfaced as a VISIBLE
  event, not a silent drop). This is the opposite trade from append-only+cache, which REQUIRES a
  byte-immutable head (no per-turn injection, no reorder, deterministic serialization) to keep
  the cache warm — a requirement that directly fights observability. O(1)+history gives up the
  cache to gain total observability + deterministic replay. `trace` emits that per-turn ledger.

Usage:
  python tools/ctxcost.py replay    [--budget T ...] [--scenario warm|cold|evict:F]
                                     [--since-days N] [--max N] [--all] [--json OUT] [--md OUT]
  python tools/ctxcost.py crossover [--scenario ...] [--stable-prefix S] [--max N] [--json OUT]
  python tools/ctxcost.py trace     [<session.jsonl> | --session SUBSTR] [--budget T]
                                     [--scenario ...] [--jsonl OUT]
  python tools/ctxcost.py selfcheck
"""
import sys
import os
import json
import glob
import argparse
import datetime

# ---- pricing, in base-input units (fresh input := 1.0). See module docstring. ----------
M_FRESH = 1.0
M_READ = 0.1
M_WRITE_5M = 1.25
M_WRITE_1H = 2.0
M_OUTPUT = 5.0
# Latency proxy: TTFT scales with PREFILL tokens. A cache read avoids recompute but is NOT
# free in time — reading KV from HBM/host costs bandwidth — so we charge it the SAME 0.1x the
# cost model uses, not 0. (Charging it 0x while charging 0.1x for dollars is self-inconsistent
# and prints a misleadingly low warm-cache TTFT.) Decode time is excluded (identical across
# regimes). This coefficient is the load-bearing latency assumption; set it to 0 to recover the
# "cache-read is free in time" view. A genuine cache MISS re-prefills the whole prompt at 1.0.
READ_TIME_MULT = 0.1
# Illustrative $ conversion (Opus 4.8 input price). Ratios are price-independent; this only
# turns base-units into a dollar figure for the headline.
USD_PER_MTOK_INPUT_OPUS = 5.0

EXCLUDE_NS_SUBSTR = ["pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"]
NS_INCLUDE_PREFIX = "C--work"

# --------------------------------------------------------------------------------------
# transcript discovery (same scope rule as ctxwin.py: the ACTIVE account's C--work* sessions)
# --------------------------------------------------------------------------------------
def discover_roots():
    home = os.path.expanduser("~")
    roots = []
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        roots.append(os.path.join(cfg, "projects"))
    for d in sorted(glob.glob(os.path.join(home, ".claude*"))):
        if ".DELETED" in d:
            continue
        p = os.path.join(d, "projects")
        if os.path.isdir(p) and p not in roots:
            roots.append(p)
    return roots


def discover(roots, since_days=None, ns_prefix=NS_INCLUDE_PREFIX, max_n=None):
    cutoff = None
    if since_days is not None:
        cutoff = datetime.datetime.now().timestamp() - since_days * 86400
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for ns in os.listdir(root):
            if any(s in ns for s in EXCLUDE_NS_SUBSTR):
                continue
            if ns_prefix and not ns.startswith(ns_prefix):
                continue
            nsdir = os.path.join(root, ns)
            if not os.path.isdir(nsdir):
                continue
            # Top-level session transcripts only: a subagent/workflow transcript is a
            # SEPARATE context that does not re-send the parent's growing prefix, so folding
            # them in would dilute the very effect we measure. Skip the nested dirs.
            for path in glob.glob(os.path.join(nsdir, "*.jsonl")):
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if cutoff and st.st_mtime < cutoff:
                    continue
                out.append({"path": path, "ns": ns, "size": st.st_size, "mtime": st.st_mtime})
    seen = set()
    uniq = []
    for r in sorted(out, key=lambda r: r["size"], reverse=True):
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        uniq.append(r)
    return uniq[:max_n] if max_n else uniq


# --------------------------------------------------------------------------------------
# parse a transcript into ORDERED TURNS, each carrying the provider's billed usage.
# A "turn" = one assistant API response. Streaming snapshots repeat the same message.id;
# we keep one record per id (the one with the most output tokens = the completed turn).
# --------------------------------------------------------------------------------------
class Turn:
    __slots__ = ("fresh", "write", "read", "out")

    def __init__(self, fresh, write, read, out):
        self.fresh = fresh      # input_tokens (uncached, full price)
        self.write = write      # cache_creation_input_tokens (cache write)
        self.read = read        # cache_read_input_tokens (cache hit)
        self.out = out          # output_tokens

    @property
    def prompt(self):
        # The FULL context the model saw this turn (cached + uncached). prompt-caching.md:
        # total prompt = input + cache_creation + cache_read.
        return self.fresh + self.write + self.read


def parse_turns(path):
    """Return [Turn, ...] in transcript order, deduped by assistant message.id."""
    by_id = {}       # message.id -> (order_first_seen, best Turn)
    order = 0
    seen_order = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "assistant":
                continue
            m = rec.get("message")
            if not isinstance(m, dict):
                continue
            u = m.get("usage")
            if not isinstance(u, dict):
                continue
            mid = m.get("id") or rec.get("uuid")
            if mid is None:
                mid = f"_anon{order}"
            fresh = int(u.get("input_tokens", 0) or 0)
            write = int(u.get("cache_creation_input_tokens", 0) or 0)
            read = int(u.get("cache_read_input_tokens", 0) or 0)
            out = int(u.get("output_tokens", 0) or 0)
            t = Turn(fresh, write, read, out)
            prev = by_id.get(mid)
            if prev is None:
                seen_order[mid] = order
                order += 1
                by_id[mid] = t
            else:
                # keep the most-complete snapshot of this message (max output tokens)
                if out > prev.out:
                    by_id[mid] = t
    turns = sorted(by_id.items(), key=lambda kv: seen_order[kv[0]])
    return [t for _, t in turns]


# --------------------------------------------------------------------------------------
# the cost model — the heart. Given a session's turns + parameters, return the four bills
# and a latency proxy. Everything is deterministic (no RNG): eviction is evenly SPACED.
# --------------------------------------------------------------------------------------
def _is_miss(i, evict_frac):
    """Deterministic, evenly-spaced cache miss schedule. evict_frac in [0,1].
    0 -> never; 1 -> always; 0.25 -> every 4th turn (i = 0,4,8,...) is a cold re-prefill."""
    if evict_frac <= 0:
        return False
    if evict_frac >= 1:
        return True
    period = max(1, round(1.0 / evict_frac))
    return (i % period) == 0


def replay_session(turns, budget, stable_prefix=0, evict_frac=0.0, write_mult=M_WRITE_5M):
    """Return a dict of {regime: {cost, ttft_sum, ttft_max}} for one session.

    Latency proxy: TTFT scales with the UNCACHED prefill tokens of a turn (a cache READ is
    near-free in prefill time; fresh + write tokens must be computed). We report it in the
    same token unit so it is comparable across regimes — not milliseconds.
    """
    A = B = C = D = 0.0
    ttftA = ttftB = ttftC = ttftD = 0.0
    ttftA_mx = ttftB_mx = ttftC_mx = ttftD_mx = 0.0
    d_prefix_written = False

    for i, t in enumerate(turns):
        out_cost = t.out * M_OUTPUT
        miss = _is_miss(i, evict_frac)

        # --- A: naive / no prefix cache. Full prompt at full price every turn. ---
        a = t.prompt * M_FRESH + out_cost
        a_ttft = t.prompt
        A += a
        ttftA += a_ttft
        ttftA_mx = max(ttftA_mx, a_ttft)

        # --- B: append-only WITH cache (measured). On an evicted turn the whole prefix
        #        re-prefills at full price (cold cache == A for that turn). ---
        if miss:
            b = t.prompt * M_FRESH + out_cost
            b_ttft = t.prompt
        else:
            b = t.fresh * M_FRESH + t.write * write_mult + t.read * M_READ + out_cost
            # cache reads still cost prefill time (KV bandwidth), charged at READ_TIME_MULT —
            # a large warm b_ttft is usually a big cache-WRITE delta, not a miss.
            b_ttft = t.fresh + t.write + READ_TIME_MULT * t.read
        B += b
        ttftB += b_ttft
        ttftB_mx = max(ttftB_mx, b_ttft)

        # --- C: O(1) reconstruct, no cache. Bounded window at full price every turn. ---
        # budget None or <0 means "unbounded" (no cap); budget==0 is a real empty window.
        w = t.prompt if (budget is None or budget < 0) else min(t.prompt, budget)
        c = w * M_FRESH + out_cost
        C += c
        ttftC += w
        ttftC_mx = max(ttftC_mx, w)

        # --- D: O(1) reconstruct + stable cached head S. ---
        s = min(stable_prefix, w)
        tail = w - s
        if s > 0 and not miss and d_prefix_written:
            head_cost = s * M_READ
            head_ttft = s * READ_TIME_MULT      # cached head still costs read-bandwidth time
        else:
            head_cost = s * write_mult           # first turn, or an evicted turn, writes S
            head_ttft = s
            if s > 0:
                d_prefix_written = True
        d = head_cost + tail * M_FRESH + out_cost
        d_ttft = head_ttft + tail
        D += d
        ttftD += d_ttft
        ttftD_mx = max(ttftD_mx, d_ttft)

    return {
        "n_turns": len(turns),
        "total_prompt_tok": sum(t.prompt for t in turns),
        # genuinely-NEW context per turn (the uncached fresh+write delta). Distinct from
        # total_prompt, which re-counts the cached prefix every turn (see render_md note).
        "total_new_tok": sum(t.fresh + t.write for t in turns),
        # the largest single full prompt EVER held this session (a real context size, unlike
        # the cross-turn sum of prompt which double-counts the growing prefix).
        "peak_prompt_tok": max((t.prompt for t in turns), default=0),
        "total_out_tok": sum(t.out for t in turns),
        "max_prompt_tok": max((t.prompt for t in turns), default=0),
        "A": {"cost": A, "ttft_sum": ttftA, "ttft_max": ttftA_mx},
        "B": {"cost": B, "ttft_sum": ttftB, "ttft_max": ttftB_mx},
        "C": {"cost": C, "ttft_sum": ttftC, "ttft_max": ttftC_mx},
        "D": {"cost": D, "ttft_sum": ttftD, "ttft_max": ttftD_mx},
    }


def kernel_projection(all_turns, budgets):
    """The PROJECTED kernel-owned regime (E): what changes when fak runs the engine itself and
    the KV cache is an addressable kernel object, not the provider's opaque prefix.

    On a black-box API the wire prompt IS the cache key, so to bound the context you re-send a
    smaller prompt and the provider RE-PREFILLS it (regime C). You must choose: bounded-but-
    re-prefilled (C) or cached-but-unbounded (B). Owning the cache decouples them — reconstruction
    becomes a cache operation (reuse survivor KV, evict pruned spans bit-exactly via a single
    Kraw re-rotation, internal/model/kvcache.go Evict), NOT a re-prefill. So:

      * PREFILL floor: the kernel prefills only the genuinely-NEW content per turn (fresh+write);
        survivor KV is reused, eviction is ~free (O(survivors) RoPE rotations, no forward pass).
        Σ(fresh+write) is the irreducible new-information floor. (A warm provider cache, B, also
        prefills ~this floor — the kernel does NOT beat B on prefill; it beats C, which re-prefills
        the whole window every turn.)
      * DECODE/resident: decode attends over the RESIDENT set. B carries the full growing prefix
        (resident ∝ prompt_t, unbounded); the kernel bounds resident to W (∝ min(prompt_t, W)).
        The decode-attention proxy is Σ resident·out; the kernel's is a small fraction of B's.

    UNITS/AXIS: this is GPU COMPUTE (prefill tokens + attention FLOPs), NOT the API-dollar bill of
    regimes A–D. The two are not directly comparable; reported separately for exactly that reason.
    STATUS: PROJECTED, not live. The KV kernel is dormant on the live proxy loop (gateway
    PoisonEvictor is a no-op on the proxy planner); bit-exact eviction is proven on a synthetic
    model (internal/kvmmu) but the bounded-reconstruct serve loop is unbuilt, and the general
    mid-stream (already-attended) bit-exact reselect is the audited non-prefix-reuse research item
    (exact+cheap today only for write-time evict / append-after-evict). Linear-attention layers
    (Gated-DeltaNet) cannot evict a span and fail closed. Treat E as the CEILING owning the cache
    buys, honestly labeled, not a measured win.
    """
    flat = [t for ts in all_turns for t in ts]
    new = sum(t.fresh + t.write for t in flat)          # kernel prefill FLOOR (new info only)
    prompt = sum(t.prompt for t in flat)                # A re-prefills the full prompt every turn
    b_decode = sum(t.prompt * t.out for t in flat)      # B: decode over the unbounded prefix
    out = {
        "new_information_floor_tok": new,
        "a_reprefill_tok": prompt,
        "floor_frac_of_a": round(new / prompt, 4) if prompt else None,
        "c_reprefill_tok": {},                          # proxy re-prefills the window each turn
        "decode_attention_ratio_E_over_B": {},          # how much the kernel bounds decode vs B
    }
    for bud in budgets:
        c_re = sum(min(t.prompt, bud) for t in flat)
        e_decode = sum(min(t.prompt, bud) * t.out for t in flat)
        out["c_reprefill_tok"][str(int(bud))] = c_re
        out["decode_attention_ratio_E_over_B"][str(int(bud))] = round(e_decode / b_decode, 4) if b_decode else None
    return out


def trace_session(turns, budget, stable_prefix=0, evict_frac=0.0, write_mult=M_WRITE_5M):
    """The OBSERVABILITY ledger: one fully-inspectable, REPLAYABLE record per turn.

    This is the point of O(1)+history that the cost crossover is only one column of: because
    the bounded view is a deterministic function of (turns, budget, forecast), every step can
    be replayed exactly and seen in full — what was billed, what was genuinely new, what the
    reconstruction would send vs PRUNE, and which turns are observability events (a cache-write
    spike, or a turn whose new content does not fit the budget — the faithfulness risk surfaced
    as a VISIBLE event, never a silent drop). Same (turns, budget, evict) always yields the same
    ledger, so an operator or the agent itself can replay a session and learn from every step.
    """
    recs = []
    d_prefix_written = False
    big_write = 0
    for i, t in enumerate(turns):
        miss = _is_miss(i, evict_frac)
        out_cost = t.out * M_OUTPUT
        a = t.prompt * M_FRESH + out_cost
        if miss:
            b = t.prompt * M_FRESH + out_cost
            b_ttft = t.prompt
        else:
            b = t.fresh * M_FRESH + t.write * write_mult + t.read * M_READ + out_cost
            b_ttft = t.fresh + t.write + READ_TIME_MULT * t.read
        w = t.prompt if (budget is None or budget < 0) else min(t.prompt, budget)
        c = w * M_FRESH + out_cost
        s = min(stable_prefix, w)
        tail = w - s
        if s > 0 and not miss and d_prefix_written:
            d = s * M_READ + tail * M_FRESH + out_cost
            d_ttft = s * READ_TIME_MULT + tail
        else:
            d = s * write_mult + tail * M_FRESH + out_cost
            d_ttft = s + tail
            if s > 0:
                d_prefix_written = True
        new_ctx = t.fresh + t.write          # genuinely-new (uncached) context this turn
        pruned = max(0, t.prompt - w)
        # observability event classification — what an operator/agent would want flagged
        if not miss and t.write > max(0, big_write):
            big_write = t.write
        event = "miss" if miss else ("cache_write_spike" if t.write >= 50000 else "warm")
        holds_new = w >= new_ctx             # could the window hold at least the new content?
        recs.append({
            "turn": i,
            "fresh": t.fresh, "write": t.write, "read": t.read, "prompt": t.prompt, "out": t.out,
            "new_context_tok": new_ctx,
            "cache_event": event,
            "cost_base_units": {"A": round(a, 1), "B": round(b, 1), "C": round(c, 1), "D": round(d, 1)},
            "ttft_prefill_tok": {"A": t.prompt, "B": round(b_ttft, 1), "C": w, "D": round(d_ttft, 1)},
            "reconstruct": {
                "budget": budget, "window_tok": w, "pruned_tok": pruned,
                "holds_new_context": holds_new,     # FALSE => the budget would truncate this turn's new content
            },
        })
    return recs


def trace_summary(recs):
    """Fold the per-turn ledger into the observability headline: how many steps are events."""
    n = len(recs)
    truncated = [r for r in recs if not r["reconstruct"]["holds_new_context"]]
    spikes = [r for r in recs if r["cache_event"] == "cache_write_spike"]
    misses = [r for r in recs if r["cache_event"] == "miss"]
    return {
        "n_turns": n,
        "turns_window_cannot_hold_new_context": len(truncated),
        "turns_window_cannot_hold_new_context_idx": [r["turn"] for r in truncated][:20],
        "cache_write_spikes": len(spikes),
        "cache_misses": len(misses),
        "max_new_context_tok": max((r["new_context_tok"] for r in recs), default=0),
        "max_pruned_tok": max((r["reconstruct"]["pruned_tok"] for r in recs), default=0),
    }


def fold(sessions_stats):
    """Sum per-session replay dicts into a corpus total."""
    tot = {k: {"cost": 0.0, "ttft_sum": 0.0, "ttft_max": 0.0} for k in "ABCD"}
    n_turns = total_prompt = total_new = total_out = 0
    sum_peak = 0
    for s in sessions_stats:
        n_turns += s["n_turns"]
        total_prompt += s["total_prompt_tok"]
        total_new += s.get("total_new_tok", 0)
        sum_peak += s.get("peak_prompt_tok", 0)
        total_out += s["total_out_tok"]
        for k in "ABCD":
            tot[k]["cost"] += s[k]["cost"]
            tot[k]["ttft_sum"] += s[k]["ttft_sum"]
            tot[k]["ttft_max"] = max(tot[k]["ttft_max"], s[k]["ttft_max"])
    return {"n_turns": n_turns, "total_prompt_tok": total_prompt, "total_new_tok": total_new,
            "sum_peak_prompt_tok": sum_peak, "total_out_tok": total_out, "regimes": tot}


# --------------------------------------------------------------------------------------
# crossover search — the headline number. Find the LARGEST reconstruct budget B* at which
# regime C (or D) still ties or beats regime B. "You can send up to B* tokens/turn fresh
# and still break even with the warm cache." Reported absolute AND as a fraction of the
# average full prompt (how much smaller the O(1) view must be).
# --------------------------------------------------------------------------------------
def crossover(all_turns, target_regime="B", recon="C", stable_prefix=0, evict_frac=0.0):
    """Bisect the budget at which `recon` cost == `target_regime` cost over the whole corpus.
    Returns (budget_star, frac_of_avg_prompt) or (None, None) if C never beats target even
    at budget 0 (impossible) / always beats it (budget unbounded)."""
    def corpus_cost(budget):
        stats = [replay_session(ts, budget, stable_prefix, evict_frac) for ts in all_turns]
        f = fold(stats)["regimes"]
        return f[recon]["cost"], f[target_regime]["cost"]

    # target cost is budget-independent; compute once at a large budget
    hi = max((t.prompt for ts in all_turns for t in ts), default=1)
    _, target = corpus_cost(hi)
    # at budget 0: C sends nothing of history (w=min(prompt,0)=0 → only output). cheapest.
    lo_cost, _ = corpus_cost(0)
    hi_cost, _ = corpus_cost(hi)
    if lo_cost > target:
        return 0.0, 0.0      # even the empty window can't beat target (degenerate)
    if hi_cost <= target:
        return None, None    # full reconstruct already <= target (target is that bad)
    lo, hib = 0.0, float(hi)
    for _ in range(40):
        mid = (lo + hib) / 2.0
        c, _ = corpus_cost(mid)
        if c <= target:
            lo = mid
        else:
            hib = mid
    n_turns = sum(len(ts) for ts in all_turns)
    avg_prompt = (sum(t.prompt for ts in all_turns for t in ts) / n_turns) if n_turns else 0
    return lo, (lo / avg_prompt if avg_prompt else None)


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------
def load_corpus(args):
    roots = args.root or discover_roots()
    nsp = "" if args.all else NS_INCLUDE_PREFIX
    sess = discover(roots, since_days=args.since_days, ns_prefix=nsp, max_n=args.max)
    corpus = []
    for s in sess:
        turns = parse_turns(s["path"])
        if turns:
            corpus.append((os.path.basename(s["path"]), turns))
    return corpus


def parse_scenario(scn):
    """warm -> evict 0; cold -> compare against A (handled by caller); evict:F -> frac."""
    if scn == "warm":
        return 0.0
    if scn.startswith("evict:"):
        return float(scn.split(":", 1)[1])
    if scn == "cold":
        return 1.0
    raise SystemExit(f"unknown scenario {scn!r}; use warm | cold | evict:0.25")


def to_usd(base_units):
    return base_units * USD_PER_MTOK_INPUT_OPUS / 1_000_000.0


# --------------------------------------------------------------------------------------
# CLI: replay
# --------------------------------------------------------------------------------------
def cmd_replay(args):
    corpus = load_corpus(args)
    if not corpus:
        print("no transcripts found; try --all or --root", file=sys.stderr)
        return 2
    evict = parse_scenario(args.scenario)
    budgets = args.budget or [4000, 8000, 16000, 32000]
    all_turns = [ts for _, ts in corpus]

    # baseline fold (budget irrelevant for A/B; use first budget)
    base = fold([replay_session(ts, budgets[0], args.stable_prefix, evict) for ts in all_turns])
    out = {
        "generated": args.now or "(stamp at commit)",
        "scenario": args.scenario,
        "stable_prefix_tok": args.stable_prefix,
        "n_sessions": len(corpus),
        "n_turns": base["n_turns"],
        "total_prompt_tok": base["total_prompt_tok"],
        "total_new_tok": base["total_new_tok"],
        "sum_peak_prompt_tok": base["sum_peak_prompt_tok"],
        "total_out_tok": base["total_out_tok"],
        # the average PROMPT BILLED per turn (~98% of it is cache-read of the same growing
        # prefix, re-counted each turn) vs the genuinely-NEW context per turn (fresh+write).
        "avg_prompt_per_turn": round(base["total_prompt_tok"] / max(1, base["n_turns"]), 1),
        "avg_new_per_turn": round(base["total_new_tok"] / max(1, base["n_turns"]), 1),
        "regimes": {},
        "budgets": {},
    }
    for k in ("A", "B"):
        r = base["regimes"][k]
        out["regimes"][k] = {
            "cost_base_units": round(r["cost"], 1),
            "cost_usd_opus": round(to_usd(r["cost"]), 4),
            "ttft_mean_tok": round(r["ttft_sum"] / max(1, base["n_turns"]), 1),
            "ttft_max_tok": int(r["ttft_max"]),
        }
    bcost = base["regimes"]["B"]["cost"]
    acost = base["regimes"]["A"]["cost"]
    for bud in budgets:
        f = fold([replay_session(ts, bud, args.stable_prefix, evict) for ts in all_turns])
        rec = {}
        for k in ("C", "D"):
            r = f["regimes"][k]
            rec[k] = {
                "cost_base_units": round(r["cost"], 1),
                "cost_usd_opus": round(to_usd(r["cost"]), 4),
                "vs_B": round(r["cost"] / bcost, 3) if bcost else None,
                "vs_A": round(r["cost"] / acost, 3) if acost else None,
                "ttft_mean_tok": round(r["ttft_sum"] / max(1, f["n_turns"]), 1),
                "ttft_max_tok": int(r["ttft_max"]),
            }
        out["budgets"][str(bud)] = rec

    # crossover vs B and vs A
    bx, bxf = crossover(all_turns, "B", "C", args.stable_prefix, evict)
    ax, axf = crossover(all_turns, "A", "C", args.stable_prefix, evict)
    out["crossover"] = {
        "C_beats_B_below_budget_tok": round(bx, 1) if bx is not None else None,
        "C_beats_B_as_frac_of_avg_prompt": round(bxf, 4) if bxf is not None else None,
        "C_beats_A_below_budget_tok": round(ax, 1) if ax is not None else None,
        "C_beats_A_as_frac_of_avg_prompt": round(axf, 4) if axf is not None else None,
    }
    out["kernel_owned_projected"] = kernel_projection(all_turns, budgets)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(render_md(out))
        print(f"wrote {args.md}")
    if not args.json and not args.md:
        print(json.dumps(out, indent=2))
    return 0


def render_md(o):
    L = []
    L.append("# O(1) context vs append-only-with-cache — per-turn cost replay\n")
    L.append(f"_Generated: {o['generated']} · {o['n_sessions']} sessions · {o['n_turns']} turns · "
             f"scenario `{o['scenario']}` · stable prefix {o['stable_prefix_tok']} tok._\n")
    L.append(f"Average **prompt billed per turn: {o['avg_prompt_per_turn']:,.0f} tok** — but "
             f"~98% of that is cache-read of the same growing prefix, re-counted each turn. "
             f"The genuinely-NEW context per turn (uncached fresh+write) is only "
             f"**{o['avg_new_per_turn']:,.0f} tok**. (The cross-turn sum of billed prompt, "
             f"{o['total_prompt_tok']:,} tok, is a token×turns area under the growing prefix, "
             f"NOT a distinct-context size; the largest single context ever held across all "
             f"sessions sums to {o['sum_peak_prompt_tok']:,} tok.)\n")
    L.append("## Incumbent regimes (token counts are the provider's OWN billed usage)\n")
    L.append("| regime | cost (base-units) | $ @Opus input | mean TTFT (tok) | max TTFT (tok) |\n|---|--:|--:|--:|--:|")
    for k, name in (("A", "A · naive / no cache"), ("B", "B · append-only + cache (measured)")):
        r = o["regimes"][k]
        L.append(f"| {name} | {r['cost_base_units']:,.0f} | ${r['cost_usd_opus']:,.2f} | "
                 f"{r['ttft_mean_tok']:,.0f} | {r['ttft_max_tok']:,} |")
    L.append("\n## O(1) reconstruct, swept by per-turn budget\n")
    L.append("`C` = fresh bounded window (no cache). `D` = bounded window + stable cached head. "
             "`vs B` < 1.0 means **cheaper than the warm cache**; `vs A` < 1.0 means cheaper than no cache.\n")
    L.append("| budget (tok) | C cost | C vs B | C vs A | C maxTTFT | D cost | D vs B | D maxTTFT |\n"
             "|--:|--:|--:|--:|--:|--:|--:|--:|")
    for bud, rec in o["budgets"].items():
        c, d = rec["C"], rec["D"]
        L.append(f"| {int(bud):,} | {c['cost_base_units']:,.0f} | {c['vs_B']} | {c['vs_A']} | "
                 f"{c['ttft_max_tok']:,} | {d['cost_base_units']:,.0f} | {d['vs_B']} | {d['ttft_max_tok']:,} |")
    cx = o["crossover"]
    L.append("\n## Crossover — how small must the O(1) window be?\n")
    if cx["C_beats_B_below_budget_tok"] is not None:
        L.append(f"- **vs the warm cache (B):** the fresh O(1) window wins while it stays under "
                 f"**{cx['C_beats_B_below_budget_tok']:,.0f} tok/turn** — "
                 f"**{cx['C_beats_B_as_frac_of_avg_prompt']*100:.1f}%** of the average billed prompt "
                 f"(and roughly {cx['C_beats_B_below_budget_tok']/max(1,o['avg_new_per_turn']):.0f}× "
                 f"the genuinely-new context per turn, so the planner has real room for relevant history).")
    else:
        L.append("- **vs the warm cache (B):** even a full reconstruct is already ≤ B "
                 "(the cache is not paying off on this corpus).")
    if cx["C_beats_A_below_budget_tok"] is not None:
        L.append(f"- **vs no cache (A, the \"random API\"):** the O(1) window wins while under "
                 f"**{cx['C_beats_A_below_budget_tok']:,.0f} tok/turn** "
                 f"(**{cx['C_beats_A_as_frac_of_avg_prompt']*100:.1f}%** of the average full context).")
    k = o.get("kernel_owned_projected")
    if k:
        L.append("\n## When fak owns the cache (regime E — projected, compute axis)\n")
        L.append("On a black-box API the wire prompt *is* the cache key, so bounding the context means "
                 "re-sending a smaller prompt and the provider RE-PREFILLS it (regime C). You must choose: "
                 "bounded-but-re-prefilled (C) or cached-but-unbounded (B). When fak runs the engine, the KV "
                 "cache is an addressable kernel object: reconstruction reuses survivor KV and evicts pruned "
                 "spans bit-exactly (a single `Kraw` re-rotation, no forward pass), so you get **bounded AND "
                 "cached at once**.\n")
        floor = k["new_information_floor_tok"]
        L.append(f"- **Prefill floor (lossless).** The kernel prefills only the genuinely-new content per turn; "
                 f"survivor KV is reused, eviction is ~free. The session's irreducible new-information floor is "
                 f"**{floor:,} tok** = {k['floor_frac_of_a']*100:.1f}% of what regime A re-prefills "
                 f"({k['a_reprefill_tok']:,} tok every turn, caching nothing).")
        re8 = k["c_reprefill_tok"].get("8000")
        if re8:
            L.append(f"  Regime C (proxy, cannot address the cache) re-prefills its window every turn — "
                     f"{re8:,} tok at an 8K budget, **{re8/max(1,floor):.1f}× the floor**, because it re-sends "
                     f"history it cannot reuse. The kernel removes that penalty by reusing cached survivors. "
                     f"(At very small budgets C re-prefills *below* the floor only because it TRUNCATES new "
                     f"content to fit — the lossy-bounded case the lossless kernel avoids. A warm provider cache "
                     f"B also prefills ~the floor, so the kernel matches B on prefill and wins on decode + eviction.)")
        dr4 = k["decode_attention_ratio_E_over_B"].get("4000")
        if dr4 is not None:
            L.append(f"- **Bounded decode.** B's decode attends over the unbounded growing prefix; the kernel "
                     f"bounds the resident set, cutting the decode-attention proxy to **{dr4*100:.1f}% of B** at a "
                     f"4K window — the part a warm cache cannot bound because it cannot evict.")
        L.append("\n> Regime E is a **compute** axis (prefill + attention FLOPs), not the API-dollar bill of A–D; "
                 "they are reported separately because they are not directly comparable. It is **projected, not "
                 "live**: the KV kernel is dormant on the proxy loop, bit-exact eviction is proven on a synthetic "
                 "model (`internal/kvmmu`) but the bounded-reconstruct serve loop is unbuilt, and the general "
                 "mid-stream bit-exact reselect is the audited non-prefix-reuse research item. See "
                 "[addressable-kv-cache.md](addressable-kv-cache.md).\n")
    L.append("\n> **TTFT** is a prefill-token proxy (decode time, identical across regimes, is "
             f"excluded); cache reads are charged READ_TIME_MULT={READ_TIME_MULT} of fresh time, the "
             "same 0.1× the cost model uses (set it to 0 for the cache-read-is-free view). A large "
             "warm `max TTFT` is usually a big cache-WRITE delta, not a miss. The robust latency "
             "result is C's BOUNDED tail vs B's unbounded one.")
    L.append("> Output tokens are held identical across regimes — this prices the *bytes sent*, "
             "not generation quality. Whether a bounded view preserves task success is the "
             "separate faithfulness axis (`internal/ctxplan`). A/B counts are exact billed usage; "
             "the reconstruct budget is a model of the bounded planner.\n")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
# CLI: crossover (just the headline, across scenarios)
# --------------------------------------------------------------------------------------
def cmd_crossover(args):
    corpus = load_corpus(args)
    if not corpus:
        print("no transcripts found; try --all or --root", file=sys.stderr)
        return 2
    all_turns = [ts for _, ts in corpus]
    scenarios = args.scenario_list or ["warm", "evict:0.25", "evict:0.5", "cold"]
    out = {"n_sessions": len(corpus), "stable_prefix_tok": args.stable_prefix, "scenarios": {}}
    for scn in scenarios:
        evict = parse_scenario(scn)
        bx, bxf = crossover(all_turns, "B", "C", args.stable_prefix, evict)
        dx, dxf = crossover(all_turns, "B", "D", args.stable_prefix, evict)
        out["scenarios"][scn] = {
            "C_beats_B_below_budget_tok": round(bx, 1) if bx is not None else None,
            "C_beats_B_frac": round(bxf, 4) if bxf is not None else None,
            "D_beats_B_below_budget_tok": round(dx, 1) if dx is not None else None,
            "D_beats_B_frac": round(dxf, 4) if dxf is not None else None,
        }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")
    else:
        print(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------------------
# CLI: trace — the per-turn OBSERVABILITY ledger for ONE session (replay every step).
# --------------------------------------------------------------------------------------
def cmd_trace(args):
    if args.transcript:
        turns = parse_turns(args.transcript)
        label = os.path.basename(args.transcript)
    else:
        corpus = load_corpus(args)
        if not corpus:
            print("no transcripts found; pass a transcript path or try --all", file=sys.stderr)
            return 2
        if args.session:
            hits = [(n, ts) for n, ts in corpus if args.session in n]
            if not hits:
                print(f"no session matching {args.session!r}", file=sys.stderr)
                return 2
            label, turns = hits[0]
        else:
            # default: the heaviest session by summed prompt — the most instructive to replay
            label, turns = max(corpus, key=lambda c: sum(t.prompt for t in c[1]))
    if not turns:
        print("session has no usable turns", file=sys.stderr)
        return 2
    evict = parse_scenario(args.scenario)
    recs = trace_session(turns, args.budget, args.stable_prefix, evict)
    summ = trace_summary(recs)
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(recs)} per-turn records to {args.jsonl}")
        print(json.dumps({"session": label, "budget": args.budget, **summ}, indent=2))
        return 0
    # human view: the summary + the first/last few turns + every truncation event
    print(f"\n  ctxcost · replay trace — {label}")
    print(f"  budget {args.budget} tok/turn · scenario {args.scenario} · {summ['n_turns']} turns\n")
    print(f"  observability events:")
    print(f"    • {summ['turns_window_cannot_hold_new_context']} turns whose NEW content "
          f"({summ['max_new_context_tok']:,} tok max) would not fit the {int(args.budget):,}-tok window"
          f"  (the faithfulness risk, made VISIBLE)")
    print(f"    • {summ['cache_write_spikes']} cache-write spikes · {summ['cache_misses']} misses · "
          f"max pruned {summ['max_pruned_tok']:,} tok\n")
    show = recs[:4] + recs[-3:] if len(recs) > 7 else recs
    print(f"  {'turn':>5} {'prompt':>9} {'new':>8} {'window':>8} {'pruned':>9} {'C$base':>9} {'B$base':>10} {'event':>17} fit")
    seen = set()
    for r in show:
        if r["turn"] in seen:
            continue
        seen.add(r["turn"])
        fit = "OK" if r["reconstruct"]["holds_new_context"] else "TRUNC!"
        print(f"  {r['turn']:>5} {r['prompt']:>9,} {r['new_context_tok']:>8,} "
              f"{r['reconstruct']['window_tok']:>8,} {r['reconstruct']['pruned_tok']:>9,} "
              f"{r['cost_base_units']['C']:>9,.0f} {r['cost_base_units']['B']:>10,.0f} "
              f"{r['cache_event']:>17} {fit}")
    print(f"\n  Every record is a deterministic function of (turns, budget) — re-run to replay "
          f"the exact ledger. --jsonl OUT emits all {summ['n_turns']} turns for offline learning.\n")
    return 0


# --------------------------------------------------------------------------------------
# CLI: selfcheck — synthetic turns with KNOWN structure. Anti-overclaim CI gate.
# --------------------------------------------------------------------------------------
def runselfcheck():
    print("== ctxcost selfcheck: synthetic turns (no real data) ==\n")
    fails = 0

    # A growing append-only session: each turn the prefix grows by 1000 tok, fully cached
    # after the first. fresh small, read = prior prompt, write = the 1000 delta.
    turns = []
    prefix = 0
    for i in range(50):
        delta = 1000
        read = prefix
        write = delta
        fresh = 2
        prefix += delta
        turns.append(Turn(fresh, write, read, out=100))

    # 1) Anti-overclaim: budget >= max prompt -> C == A exactly (reconstruct is a no-op).
    big = max(t.prompt for t in turns) + 1
    r = replay_session(turns, big)
    ok = abs(r["C"]["cost"] - r["A"]["cost"]) < 1e-6
    print(f"  no-overclaim (C==A at big budget)   {'PASS' if ok else 'FAIL'}  "
          f"C={r['C']['cost']:.0f} A={r['A']['cost']:.0f}")
    fails += not ok

    # 2) B <= A always (a cache never costs MORE than no cache on the same traffic).
    ok = r["B"]["cost"] <= r["A"]["cost"] + 1e-6
    print(f"  cache never harmful (B<=A)          {'PASS' if ok else 'FAIL'}  "
          f"B={r['B']['cost']:.0f} A={r['A']['cost']:.0f}")
    fails += not ok

    # 3) The thesis is a CROSSOVER, not "small always wins". Below the crossover a fresh
    #    O(1) window beats even a perfectly warm cache; above it the 0.1x cache-read wins.
    #    Assert BOTH halves so we never overclaim that O(1) is unconditionally cheaper.
    small = replay_session(turns, 2000)["C"]["cost"]      # below crossover
    large = replay_session(turns, 8000)["C"]["cost"]      # above crossover
    bcost = replay_session(turns, 2000)["B"]["cost"]
    ok = small < bcost < large
    print(f"  thesis is a CROSSOVER                {'PASS' if ok else 'FAIL'}  "
          f"C@2k={small:.0f} < B={bcost:.0f} < C@8k={large:.0f}")
    fails += not ok

    # 4) Crossover is found and lies strictly inside (0, max_prompt).
    bx, bxf = crossover([turns], "B", "C")
    mx = max(t.prompt for t in turns)
    ok = bx is not None and 0 < bx < mx
    print(f"  crossover in range                  {'PASS' if ok else 'FAIL'}  "
          f"B*={bx} (max_prompt={mx})")
    fails += not ok

    # 5) Monotonicity: a bigger budget never makes C cheaper.
    c_small = replay_session(turns, 2000)["C"]["cost"]
    c_big = replay_session(turns, 20000)["C"]["cost"]
    ok = c_big >= c_small - 1e-6
    print(f"  monotonic C in budget               {'PASS' if ok else 'FAIL'}  "
          f"C@2k={c_small:.0f} C@20k={c_big:.0f}")
    fails += not ok

    # 6) Eviction degrades B toward A (cold == A).
    warm = replay_session(turns, 4000, evict_frac=0.0)["B"]["cost"]
    cold = replay_session(turns, 4000, evict_frac=1.0)["B"]["cost"]
    a = replay_session(turns, 4000)["A"]["cost"]
    ok = abs(cold - a) < 1e-6 and cold > warm
    print(f"  eviction: cold B == A > warm B      {'PASS' if ok else 'FAIL'}  "
          f"warm={warm:.0f} cold={cold:.0f} A={a:.0f}")
    fails += not ok

    # 7) kernel projection: the new-information floor <= C re-prefill <= A re-prefill, and the
    #    bounded decode proxy is a fraction of B's unbounded one. (E gets bounded AND floor-prefill.)
    kp = kernel_projection([turns], [4000])
    floor = kp["new_information_floor_tok"]
    c_re = kp["c_reprefill_tok"]["4000"]
    a_re = kp["a_reprefill_tok"]
    dr = kp["decode_attention_ratio_E_over_B"]["4000"]
    # robust invariants only: the floor and C's re-prefill are both bounded by A's full re-prefill,
    # and the bounded decode proxy never exceeds B's unbounded one. (floor vs C is budget-dependent:
    # below a turn's new-content size, C re-prefills LESS than the floor because it TRUNCATES new
    # content — the lossy-bounded case the kernel avoids by reusing survivors instead of re-sending.)
    ok = floor <= a_re and c_re <= a_re and 0 < dr <= 1.0
    print(f"  kernel: floor<=A, C<=A, decode bounded          {'PASS' if ok else 'FAIL'}  "
          f"floor={floor:.0f} C={c_re:.0f} A={a_re:.0f} decodeE/B={dr}")
    fails += not ok

    print()
    if fails:
        print(f"SELFCHECK FAILED — {fails} check(s) failed")
        return 1
    print("OK — model is anti-overclaim (C==A at full budget), cache-sane (B<=A), and the "
          "thesis + crossover hold on a synthetic growing session")
    return 0


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="per-turn cost replay: O(1) context vs append-only-with-cache")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("replay", help="full per-regime cost + latency table over discovered sessions")
    r.add_argument("--budget", type=float, action="append", default=None,
                   help="reconstruct budget(s) in tokens (repeatable; default 4k/8k/16k/32k)")
    r.add_argument("--scenario", default="warm", help="warm | cold | evict:F (e.g. evict:0.25)")
    r.add_argument("--stable-prefix", type=float, default=2000,
                   help="bytes kept byte-stable + cached in regime D (system+tools), tok (default 2000)")
    r.add_argument("--since-days", type=float, default=None)
    r.add_argument("--root", action="append", default=None)
    r.add_argument("--max", type=int, default=20, help="cap at the N heaviest sessions (default 20)")
    r.add_argument("--all", action="store_true", help="include all namespaces (default C--work* only)")
    r.add_argument("--json", default=None)
    r.add_argument("--md", default=None)
    r.add_argument("--now", default=None)

    c = sub.add_parser("crossover", help="just the crossover budget across cache scenarios")
    c.add_argument("--scenario-list", nargs="*", default=None,
                   help="scenarios to sweep (default: warm evict:0.25 evict:0.5 cold)")
    c.add_argument("--stable-prefix", type=float, default=2000)
    c.add_argument("--since-days", type=float, default=None)
    c.add_argument("--root", action="append", default=None)
    c.add_argument("--max", type=int, default=20)
    c.add_argument("--all", action="store_true")
    c.add_argument("--json", default=None)

    t = sub.add_parser("trace", help="per-turn replayable OBSERVABILITY ledger for one session")
    t.add_argument("transcript", nargs="?", default=None, help="a single .jsonl, or omit to pick a discovered session")
    t.add_argument("--session", default=None, help="substring of a discovered session basename to trace")
    t.add_argument("--budget", type=float, default=8000, help="reconstruct budget tok/turn (default 8000)")
    t.add_argument("--scenario", default="warm", help="warm | cold | evict:F")
    t.add_argument("--stable-prefix", type=float, default=2000)
    t.add_argument("--since-days", type=float, default=None)
    t.add_argument("--root", action="append", default=None)
    t.add_argument("--max", type=int, default=20)
    t.add_argument("--all", action="store_true")
    t.add_argument("--jsonl", default=None, help="write all per-turn records as JSONL for offline replay/learning")

    sub.add_parser("selfcheck", help="synthetic turns: anti-overclaim + thesis + crossover (CI gate)")

    args = ap.parse_args(argv)
    if args.cmd == "replay":
        return cmd_replay(args)
    if args.cmd == "crossover":
        return cmd_crossover(args)
    if args.cmd == "trace":
        return cmd_trace(args)
    if args.cmd == "selfcheck":
        return runselfcheck()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
