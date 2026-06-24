#!/usr/bin/env python3
r"""switcher_shadow -- observe what the account switcher WOULD have done, and report
the savings, WITHOUT ever changing the live account ("shadow mode").

The fleet already has two of three pieces live on this host:
  * the trunk/leak/claim guards (git hooks under tools/githooks -- installed by
    tools/install_trunk_guard.py; an off-trunk branch is refused with OFF_TRUNK);
  * the tier-aware account switcher (tools/fleet_accounts.py: classify_task ->
    light/gardening=tier2(GLM) vs hard/engineering=tier1(Opus); resolve_account picks
    a real available seat).

What was missing is the OBSERVABILITY layer this module is: it watches FINISHED Claude
Code sessions, replays the switcher's classify_task over each session's real prompts to
ask "which tier would the switcher have routed this to?", and reports the savings -- as a
running, host-local ledger and an on-demand report. It is SHADOW: a decision is recorded,
never executed; no account is ever switched, no live work is touched. Pure read-over-
transcripts.

TWO SAVINGS AXES, kept SEPARATE on purpose (never summed -- they price different mistakes):

  B) CACHE-REUSE (the headline -- applies to EVERY session).
     REALIZED, not projected: a transcript's cache_read tokens were billed at the cheap
     cache-read rate instead of the full input rate. That delta is money ALREADY SAVED by
     prompt-cache reuse, read straight from the usage records (session_audit pricing). We
     also surface what reuse is still on the table (cache_hit_frac).

  A) TIER-DOWNGRADE (a conservatively-gated FOOTNOTE -- rare by design).
     PROJECTED, NOT realized: counts the tier-1 (Opus) tokens of sessions the switcher
     would call TRIVIAL -- and only those. A session qualifies ONLY when EVERY typed prompt
     classifies "light" at confidence >= light_threshold AND the session made ZERO
     side-effecting tool calls (any Bash/Edit/Write/... => real work => tier-1 => $0 claimed)
     AND a tier-2 seat was actually available. The unit is "tier-1 token-hours movable to a
     flat-rate tier-2 seat", NOT dollars: tier-2/3 are flat-rate/local seats with ~$0
     marginal token cost, so a dollar "saving" is doubly assumed (Opus price is an
     assumption; the tier-2 marginal price is a second one). A dollar figure is offered ONLY
     as a derived, double-flagged line, and the counterfactual ignores GLM's different
     tokenizer + a cache regime that never ran on GLM. It is an UPPER BOUND under an
     UNVERIFIED equal-quality assumption.

CLI:
    python tools/switcher_shadow.py report          # human card (cross-session rollup)
    python tools/switcher_shadow.py report --json    # machine-readable rollup
    python tools/switcher_shadow.py --hook            # Stop-hook mode: append ONE finished
                                                       # session (reads transcript_path from
                                                       # the Stop event JSON on stdin)
    python tools/switcher_shadow.py ledger           # dump the host-local ledger
    python tools/switcher_shadow.py selfcheck         # anti-overclaim invariants

The ledger is host-local and OFF-REPO (~/.claude/fak-shadow-ledger.jsonl), append-only,
idempotent (keyed on session id + content fingerprint), size-capped (rotates at LEDGER_CAP).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Reuse the fleet's switcher decision + the exact session token accounting. This tool
# adds NO new economics: classify_task is the routing brain, session_audit is the meter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet_accounts  # routing decision ONLY
import session_audit   # transcript discovery + EXACT token accounting + pricing

# The verbatim honesty caveat stamped on EVERY surface (report header, JSON top field,
# every ledger row). Mirrors session_audit.PRICING_IS_ASSUMPTION discipline + the o1 doc's
# "this is not a quality claim" stance. Do not soften.
CAVEAT = (
    "SAVINGS ARE PROJECTED, NOT REALIZED (tier-downgrade column) -- counts tier-1 tokens on "
    "sessions classified trivial (every prompt light, no write-tools, tier-2 seat available); "
    "assumes EQUAL QUALITY on tier-2 GLM (UNVERIFIED, no GLM run); unit = tier-1 token-hours "
    "movable to a flat-rate seat; dollar figures doubly assumed (Opus price per "
    "session_audit.PRICING_IS_ASSUMPTION; tier-2 marginal price assumed ~$0 on a flat-rate/"
    "local seat). Cache-reuse column is REALIZED (read from usage records). The two columns "
    "are DIFFERENT axes and are never summed."
)

HOME = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
LEDGER_PATH = os.environ.get(
    "FAK_SHADOW_LEDGER", os.path.join(HOME, ".claude", "fak-shadow-ledger.jsonl"))
LEDGER_CAP_BYTES = int(os.environ.get("FAK_SHADOW_LEDGER_CAP", str(8 * 1024 * 1024)))  # 8 MiB

# Opus 4.8 USD/1e6 tokens (the SAME assumed table session_audit flags). Full-input vs
# cache-read rate is what makes prompt-cache reuse a realized saving.
_OPUS_INPUT, _OPUS_CWRITE, _OPUS_CREAD, _OPUS_OUTPUT = session_audit.PRICING["opus"]


# --------------------------------------------------------------------------- #
# The shadow decision for ONE finished session (pure; no I/O, no mutation).
# --------------------------------------------------------------------------- #
def _is_write_tool(name: str) -> bool:
    """A tool call that mutates state or spawns work -> the session did REAL work, so it
    can never qualify as trivial/tier-2 regardless of how its prompts read."""
    return name not in session_audit.READ_ONLY_TOOLS


def classify_session(stats: dict, *, tier2_available: bool,
                     policy: dict | None = None) -> dict:
    """Decide the tier the switcher WOULD have routed this whole session to, and why.

    The gate is WHOLE-SESSION and conservative on purpose (a real Opus engineering session
    also contains trivial turns like "thanks" -- gating on ANY light prompt would claim
    that engineering work could have run on GLM). A session is "trivial" (=> shadow tier-2)
    ONLY when:
      * it has at least one typed prompt, AND
      * EVERY typed prompt classifies "light" at confidence >= light_threshold, AND
      * it made ZERO side-effecting (non-read-only) tool calls, AND
      * a tier-2 seat was actually available (else route_account would up-shift to tier-1).
    Anything else => tier-1 (no savings claimed).

    Returns {would_tier, qualifies_tier2, reason, n_prompts, all_light, no_write_tools,
    tier2_available, light_threshold}.
    """
    pol = policy or fleet_accounts.load_policy()
    prompts = [text for (_ts, text) in stats.get("prompts", []) if text]
    n = len(prompts)
    threshold = float(pol.get("routing", {}).get("light_confidence", 0.999))

    if n == 0:
        return {
            "would_tier": 1, "qualifies_tier2": False,
            "reason": "no typed prompts -> cannot judge trivial; tier-1",
            "n_prompts": 0, "all_light": False, "no_write_tools": None,
            "tier2_available": tier2_available, "light_threshold": threshold,
        }

    all_light = True
    for text in prompts:
        verdict = fleet_accounts.classify_task(text, "auto", pol)
        is_light = (verdict.get("class") == "light"
                    and float(verdict.get("confidence", 0.0)) >= threshold)
        if not is_light:
            all_light = False
            break

    write_calls = sum(c for name, c in (stats.get("tools") or {}).items()
                      if _is_write_tool(name))
    no_write_tools = write_calls == 0

    qualifies = bool(all_light and no_write_tools and tier2_available)
    if qualifies:
        reason = "every prompt light + no write-tools + tier-2 seat available -> tier-2"
    elif not all_light:
        reason = "a prompt is non-trivial (hard/dev) -> tier-1"
    elif not no_write_tools:
        reason = f"{write_calls} side-effecting tool call(s) -> real work -> tier-1"
    else:
        reason = "no tier-2 seat available -> switcher would up-shift to tier-1"

    return {
        "would_tier": 2 if qualifies else 1,
        "qualifies_tier2": qualifies, "reason": reason,
        "n_prompts": n, "all_light": all_light, "no_write_tools": no_write_tools,
        "tier2_available": tier2_available, "light_threshold": threshold,
    }


def cache_reuse_savings(tok: dict) -> dict:
    """REALIZED prompt-cache savings for one session, read straight from usage records.

    cache_read tokens were billed at the cache-read rate ($1.50/Mtok) instead of full input
    ($15/Mtok); that delta is money the prompt cache ALREADY saved this session. We also
    report the headroom still on the table: a fresh-input token NOT served from cache.
    """
    cread = int(tok.get("cache_read", 0) or 0)
    fresh = int(tok.get("input", 0) or 0)
    cwrite = int(tok.get("cache_create", 0) or 0)
    # realized: what cache_read tokens WOULD have cost at full input rate, minus what they
    # actually cost at the cache-read rate.
    realized_usd = cread * (_OPUS_INPUT - _OPUS_CREAD) / 1e6
    served = cread + cwrite + fresh
    cache_hit_frac = (cread / served) if served else None
    return {
        "cache_read_tokens": cread,
        "fresh_input_tokens": fresh,
        "cache_write_tokens": cwrite,
        "cache_hit_frac": cache_hit_frac,
        "realized_usd_opus": round(realized_usd, 4),
        "realized_is_assumption": session_audit.PRICING_IS_ASSUMPTION,
    }


def tier_downgrade_savings(stats: dict, decision: dict) -> dict:
    """PROJECTED tier-downgrade savings for one session -- $0 unless it qualified as trivial.

    Unit is tier-1 token-hours movable to a flat-rate seat (the tokens broken out), NOT
    dollars. The derived dollar line is a DOUBLE-flagged upper bound (see CAVEAT) and
    deliberately ZEROES the cache columns: replaying Claude-produced cache tokens at GLM
    rates would model a cache regime that never ran on GLM. So the dollar figure prices only
    the FRESH input + output the trivial session spent on tier-1.
    """
    tok = stats.get("tokens", {}) or {}
    if not decision.get("qualifies_tier2"):
        return {
            "qualifies": False,
            "movable_input_tokens": 0, "movable_output_tokens": 0,
            "movable_total_tokens": 0,
            "hypothetical_usd_opus_upper_bound": 0.0,
            "unit": "tier-1 token-hours movable to a flat-rate tier-2 seat",
        }
    fresh = int(tok.get("input", 0) or 0)
    out = int(tok.get("output", 0) or 0)
    # Upper-bound $ the trivial session spent on tier-1 (Opus), CACHE COLUMNS ZEROED on
    # purpose (the GLM counterfactual has no comparable cache regime). This is the most that
    # could move to a cheaper seat -- not a realized saving.
    hyp_usd = (fresh * _OPUS_INPUT + out * _OPUS_OUTPUT) / 1e6
    return {
        "qualifies": True,
        "movable_input_tokens": fresh, "movable_output_tokens": out,
        "movable_total_tokens": fresh + out,
        "hypothetical_usd_opus_upper_bound": round(hyp_usd, 4),
        "unit": "tier-1 token-hours movable to a flat-rate tier-2 seat",
    }


def _content_fingerprint(stats: dict) -> str:
    """Idempotency key payload: which billed turns this transcript had. A re-fire (same
    session, no new turns) reproduces this exactly so the ledger upserts instead of
    double-counting."""
    return f"{stats.get('assistant_turns', 0)}:{stats.get('n_records', 0)}"


def shadow_one(path: str, *, tier2_available: bool | None = None,
               policy: dict | None = None) -> dict:
    """The full shadow verdict + both savings axes for ONE session transcript path."""
    pol = policy or fleet_accounts.load_policy()
    if tier2_available is None:
        tier2_available = _tier2_available(pol)
    stats = session_audit.analyze(path)
    if "error" in stats:
        return {"session": os.path.splitext(os.path.basename(path))[0],
                "path": path, "error": stats["error"]}
    decision = classify_session(stats, tier2_available=bool(tier2_available), policy=pol)
    return {
        "session": stats.get("session"),
        "path": path,
        "fingerprint": _content_fingerprint(stats),
        "ts_max": stats.get("ts_max"),
        "models": stats.get("models", {}),          # plain dict (not a Counter)
        "assistant_turns": stats.get("assistant_turns", 0),
        "n_prompts": stats.get("n_prompts", 0),
        "actual_cost_usd": round(float(stats.get("cost_usd", 0.0) or 0.0), 4),
        "decision": decision,
        "cache_reuse": cache_reuse_savings(stats.get("tokens", {}) or {}),
        "tier_downgrade": tier_downgrade_savings(stats, decision),
        "caveat": CAVEAT,
    }


# --------------------------------------------------------------------------- #
# tier-2 availability (uses fleet_accounts for the ROUTING decision only)
# --------------------------------------------------------------------------- #
def _tier2_available(policy: dict | None = None) -> bool:
    """Is a tier-2 (GLM/opencode) seat available right now? Mirrors route_account's
    tier-2->tier-1 up-shift: a trivial session only 'saves' if the switcher could actually
    have placed it on a cheaper seat. Best-effort; defaults False (conservative) on error."""
    try:
        avail = fleet_accounts.available_accounts(policy=policy)
        return any(fleet_accounts._as_int(r.get("model_tier"), 3) == 2 for r in avail)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Host-local ledger (off-repo, append-only, idempotent, size-capped, atomic writes)
# --------------------------------------------------------------------------- #
def _ledger_path(path: str | None = None) -> str:
    """Resolve the ledger path at CALL time. Reading the module global dynamically (rather
    than binding it as a default arg) is what lets tests monkeypatch ss.LEDGER_PATH and lets
    FAK_SHADOW_LEDGER override per-invocation."""
    return path if path is not None else LEDGER_PATH


def _read_ledger(path: str | None = None) -> list[dict]:
    path = _ledger_path(path)
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def _atomic_rewrite(path: str, rows: list[dict]) -> None:
    """Rewrite the whole ledger atomically (temp-file + os.replace). O_APPEND is not safe
    across concurrent Stop hooks on Windows; a full atomic rewrite is. Cheap at ledger sizes
    we cap to."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _rotate_if_needed(path: str | None = None, cap: int | None = None) -> None:
    path = _ledger_path(path)
    cap = LEDGER_CAP_BYTES if cap is None else cap
    try:
        if os.path.getsize(path) > cap:
            os.replace(path, path + ".1")  # keep one prior generation; bounded
    except OSError:
        pass


def upsert_ledger(row: dict, path: str | None = None) -> dict:
    """Append a shadow row, idempotent on (session, fingerprint). A re-fire of the same
    finished session (Stop hooks can fire more than once) UPSERTS in place instead of
    double-counting. Returns {action: 'inserted'|'updated'|'skipped'}."""
    path = _ledger_path(path)
    _rotate_if_needed(path)
    rows = _read_ledger(path)
    key = (row.get("session"), row.get("fingerprint"))
    for i, existing in enumerate(rows):
        if (existing.get("session"), existing.get("fingerprint")) == key:
            if existing == row:
                return {"action": "skipped", "session": row.get("session")}
            rows[i] = row
            _atomic_rewrite(path, rows)
            return {"action": "updated", "session": row.get("session")}
    rows.append(row)
    _atomic_rewrite(path, rows)
    return {"action": "inserted", "session": row.get("session")}


# --------------------------------------------------------------------------- #
# Stop-hook mode -- process ONLY the finished session; fail-OPEN, never block.
# --------------------------------------------------------------------------- #
def run_hook(stdin_text: str) -> int:
    """Stop-hook entry. Read the finished session's transcript_path from the Stop event JSON
    on stdin, shadow-score JUST that session, upsert the ledger. ALWAYS returns 0 and NEVER
    prints a decision object: a Stop hook that exits 2 or prints {decision:block} would
    BLOCK the user's session from ending. Fail-open like repo_guard.run_hook."""
    try:
        payload = json.loads(stdin_text or "{}")
        # A re-fire after a block carries stop_hook_active=true; do nothing (don't recurse,
        # don't double-count).
        if payload.get("stop_hook_active"):
            return 0
        transcript = payload.get("transcript_path") or ""
        if not transcript or not os.path.isfile(transcript):
            print("switcher_shadow: no transcript_path on stop event; skipping",
                  file=sys.stderr)
            return 0
        row = shadow_one(transcript)
        if "error" in row:
            print(f"switcher_shadow: analyze error ({row['error']}); skipping",
                  file=sys.stderr)
            return 0
        result = upsert_ledger(row)
        print(f"switcher_shadow: ledger {result['action']} {row.get('session')}",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 -- fail-open is deliberate; never wedge a session
        print(f"switcher_shadow: internal error, allowing ({exc})", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# Cross-session rollup report (on-demand; NOT the hook path)
# --------------------------------------------------------------------------- #
def rollup(rows: list[dict]) -> dict:
    """Fold ledger rows (or freshly-shadowed rows) into the headline numbers."""
    scored = [r for r in rows if "error" not in r]
    n = len(scored)
    qualifying = [r for r in scored if r.get("tier_downgrade", {}).get("qualifies")]
    realized_cache_usd = sum(
        float(r.get("cache_reuse", {}).get("realized_usd_opus", 0.0) or 0.0) for r in scored)
    movable_tokens = sum(
        int(r.get("tier_downgrade", {}).get("movable_total_tokens", 0) or 0) for r in scored)
    hyp_usd = sum(
        float(r.get("tier_downgrade", {}).get("hypothetical_usd_opus_upper_bound", 0.0) or 0.0)
        for r in scored)
    actual_usd = sum(float(r.get("actual_cost_usd", 0.0) or 0.0) for r in scored)
    return {
        "caveat": CAVEAT,
        "n_sessions": n,
        "actual_cost_usd_opus": round(actual_usd, 2),
        # B) realized, headline:
        "cache_reuse_realized_usd_opus": round(realized_cache_usd, 2),
        # A) projected, footnote:
        "tier_downgrade_qualifying_sessions": len(qualifying),
        "tier_downgrade_qualifying_frac": (len(qualifying) / n) if n else None,
        "tier_downgrade_movable_tokens": movable_tokens,
        "tier_downgrade_hypothetical_usd_opus_upper_bound": round(hyp_usd, 2),
        "pricing_is_assumption": session_audit.PRICING_IS_ASSUMPTION,
    }


def _discover_and_shadow(policy: dict | None = None, since_days=None) -> list[dict]:
    pol = policy or fleet_accounts.load_policy()
    tier2 = _tier2_available(pol)
    roots = session_audit.DEFAULT_ROOTS
    found = session_audit.discover(roots, since_days=since_days)
    rows = []
    for item in found:
        # discover() yields dicts {root, ns, path, kind, size, mtime}; tolerate a bare
        # str/tuple too in case a caller passes a hand-built list.
        if isinstance(item, dict):
            path = item.get("path")
        elif isinstance(item, (list, tuple)):
            path = item[0]
        else:
            path = item
        if not path:
            continue
        rows.append(shadow_one(path, tier2_available=tier2, policy=pol))
    return rows


def report_md(roll: dict) -> str:
    L = []
    L.append("# Account-switcher shadow-mode savings\n")
    L.append(f"> {roll['caveat']}\n")
    L.append(f"- **Sessions observed:** {roll['n_sessions']}")
    L.append(f"- **Actual spend (Opus-priced, assumed):** ${roll['actual_cost_usd_opus']:,.2f}")
    L.append("")
    L.append("## B) Cache-reuse savings -- REALIZED (headline; every session)")
    L.append(f"- Already saved by prompt-cache reuse: "
             f"**${roll['cache_reuse_realized_usd_opus']:,.2f}** "
             f"(cache_read billed at the cache rate, not full input).")
    L.append("")
    L.append("## A) Tier-downgrade savings -- PROJECTED (footnote; trivial sessions only)")
    frac = roll["tier_downgrade_qualifying_frac"]
    frac_s = f"{frac*100:.1f}%" if frac is not None else "n/a"
    L.append(f"- Qualifying (trivial) sessions: "
             f"{roll['tier_downgrade_qualifying_sessions']} of {roll['n_sessions']} ({frac_s}).")
    L.append(f"- Tier-1 tokens movable to a flat-rate tier-2 seat: "
             f"**{roll['tier_downgrade_movable_tokens']:,}**.")
    L.append(f"- Hypothetical $ upper bound (Opus-priced, equal-quality ASSUMED, "
             f"cache zeroed): ${roll['tier_downgrade_hypothetical_usd_opus_upper_bound']:,.2f}.")
    L.append("")
    L.append("_Columns A and B price different mistakes (wrong-tier vs wrong-context-regime) "
             "and are never summed._")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# selfcheck -- anti-overclaim invariants (the tool provably cannot fabricate a saving)
# --------------------------------------------------------------------------- #
def runselfcheck() -> int:
    pol = fleet_accounts.load_policy()
    fails = []

    # 1. A session with a hard/dev prompt NEVER qualifies, even with no write tools.
    hard = {"prompts": [(None, "implement the new gateway adjudicator")],
            "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = classify_session(hard, tier2_available=True, policy=pol)
    if d["qualifies_tier2"] or d["would_tier"] != 1:
        fails.append("hard prompt qualified for tier-2 (must be tier-1)")
    if tier_downgrade_savings(hard, d)["hypothetical_usd_opus_upper_bound"] != 0.0:
        fails.append("hard prompt claimed nonzero tier-downgrade savings")

    # 2. A trivial-prompt session with a WRITE tool never qualifies.
    wrote = {"prompts": [(None, "hi")], "tools": {"Write": 1},
             "tokens": {"input": 100, "output": 100}}
    d = classify_session(wrote, tier2_available=True, policy=pol)
    if d["qualifies_tier2"]:
        fails.append("session with a Write tool qualified for tier-2")

    # 3. A trivial-prompt session with NO tier-2 seat never qualifies.
    triv = {"prompts": [(None, "hi")], "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = classify_session(triv, tier2_available=False, policy=pol)
    if d["qualifies_tier2"]:
        fails.append("trivial session qualified with no tier-2 seat available")

    # 4. A genuinely trivial session WITH a tier-2 seat DOES qualify (the tool must still
    #    surface a real saving when one exists).
    d = classify_session(triv, tier2_available=True, policy=pol)
    if not d["qualifies_tier2"]:
        fails.append("genuinely trivial session failed to qualify")

    # 5. Cache-reuse realized $ is >= 0 and 0 when no cache_read.
    if cache_reuse_savings({"cache_read": 0, "input": 50})["realized_usd_opus"] != 0.0:
        fails.append("zero cache_read produced nonzero realized cache savings")
    if cache_reuse_savings({"cache_read": 1_000_000})["realized_usd_opus"] <= 0.0:
        fails.append("nonzero cache_read produced zero realized cache savings")

    # 6. The caveat is non-empty and present.
    if not CAVEAT.strip():
        fails.append("caveat is empty")

    if fails:
        print("SELFCHECK FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("switcher_shadow selfcheck: OK (cannot fabricate a saving from real work)")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    if _has(argv, ("--hook",)):
        return run_hook(sys.stdin.read())
    mode = next((a for a in argv if not a.startswith("-")), "report")
    if mode == "selfcheck":
        return runselfcheck()
    if mode == "ledger":
        rows = _read_ledger()
        if _has(argv, ("--json",)):
            print(json.dumps(rows, indent=1))
        else:
            print(f"{len(rows)} ledger row(s) at {LEDGER_PATH}")
            for r in rows:
                td = r.get("tier_downgrade", {})
                print(f"  {r.get('session','?'):<40} would-tier="
                      f"{r.get('decision',{}).get('would_tier','?')}  "
                      f"movable={td.get('movable_total_tokens',0)}  "
                      f"cache_realized=${r.get('cache_reuse',{}).get('realized_usd_opus',0)}")
        return 0
    # report (default): shadow real sessions OR fold the ledger when --ledger is passed.
    since = _arg(argv, ("--since-days",))
    if _has(argv, ("--from-ledger",)):
        rows = _read_ledger()
    else:
        rows = _discover_and_shadow(since_days=int(since) if since else None)
    roll = rollup(rows)
    if _has(argv, ("--json",)):
        print(json.dumps({"rollup": roll, "sessions": rows}, indent=1))
    else:
        print(report_md(roll))
    return 0


def _has(argv: list[str], names: tuple[str, ...]) -> bool:
    return any(a in names for a in argv)


def _arg(argv: list[str], names: tuple[str, ...], default: str = "") -> str:
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a[len(n) + 1:]
    return default


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
