#!/usr/bin/env python3
"""
ctxwin.py — context-window baseline + self-reducer for Claude Code sessions.

An agentic coding session re-sends its whole transcript every turn, so context grows
monotonically and the prompt-cache + opus quota bill scale with it. This tool answers
two questions on YOUR OWN sessions, no model in the loop:

  1) BASELINE — what is the context window made of, and how much is reducible at each
     RISK TIER? (`ctxwin baseline`)
  2) SELF-REDUCE — apply a risk-ranked reducer and PROVE it halves the window, with an
     honest, tiered loss profile. (`ctxwin reduce --target 2.0`)

TUNING — the reducer is a DIAL, not a switch. The same risk-ranked machinery serves the
operator who wants nothing touched and the operator who wants a 30k stack trace gone:
  - `--profile off`          disabled — every result kept verbatim (ratio 1.0).
  - `--profile conservative` lossless + recoverable only (noise + exact-dedup + stale reads);
                             NO windowing, so no unique current content is ever truncated.
  - `--profile balanced`     the default — low-risk tiers + bounded windowing auto-tuned to 2x.
  - `--profile aggressive`   push to 3x with a lower floor + head-only windows, and collapse
                             error-shaped results to their first line (drop the stack/stderr).
  - `--profile hyper`        push to 4x with the lowest floor; collapse a 30k error result to a
                             one-line marker ("[…tool error — ~7500 tok elided]"). Lossy BY CHOICE.
  The presets are a MONOTONIC dial — each reduces at least as hard as the one before it (higher
  target + lower floor + the lossy error tier switching on), never less.
  - Fine-grained overrides (compose with any profile): `--error-collapse off|head|oneline`,
    `--per-tool Bash=200 --per-tool TodoWrite=off` (a per-tool token budget, or `off`/`none`
    to EXEMPT a tool from all reduction), `--no-stale`, `--no-noise`.
  The lossy levers (error-collapse, sub-floor per-tool budgets) are OPT-IN and OFF in the
  default profile; an error body collapsed to a marker is NOT counted recoverable (no file
  handle), so `recoverable_frac` drops honestly when you turn the aggression up.

WHAT THE DATA SAYS (measured over the heaviest real sessions on this box; live transcripts,
so exact numbers drift run to run):
  - The window is ~56% tool_result + ~34% tool_use INPUTS (the agent's Write/Edit/Workflow
    payloads) + ~10% text. Read results + Write/Edit/Bash inputs dominate. The reducer
    touches both tool_result and tool_use inputs (text/thinking are left alone).
  - The reducible mass is RISK-RANKED, not one lever:
      • character/line noise — ANSI codes, trailing whitespace, blank-line runs, runs of
        identical lines WITHIN a result (~0.3%), plus cat-n line-number prefixes (~1.6%,
        but those are Edit-targeting SIGNAL) and cross-result repeated lines (~2.5%) — is
        ~4% of mass combined. So "strip the obvious noise" is small but not nothing — and
        still far below the stale-read share. (`baseline` reports each separately.)
      • STALE READS — a file Read that the agent LATER Edits/Writes — are ~18-22% of mass.
        This is the best LOW-RISK lever: the old read is stale BY CONSTRUCTION (the agent
        changed the file), it is recoverable (re-read), and it needs zero relevance
        guessing. ~1.28x on its own.
      • exact duplicates ~1% (lossless). Near-dup folding adds ~1%.
      • Cumulative LOW-RISK (noise + stale + dedup, no guessing) ≈ 1.3x.
      • PER-RESULT WINDOWING — cap each surviving result to a budget B, keep head+tail,
        elide the middle to a recoverable pointer — carries the rest to 2x but is
        BOUNDED-LOSS (it truncates current, unique content): cap@500→2.46x, cap@700→~2x.
  - Beware a tempting artifact: pairing tool_results to tool_use by `message.id` dedup
    UNDER-counts and fakes an 88% "superseded" number. Key results by their unique
    `tool_use_id` instead (this tool does; verified 123/123 pair without dedup).

HONESTY (same discipline as cmd/tokendemo's two-meter ledger):
  - Token counts are a tokenizer-free ~4-chars-per-token ESTIMATE (the same estimate
    `agent.EstimateAnthropicTokens` and tools/transcript_workload.py use). Ratios are
    estimate-over-estimate, so the constant cancels; magnitudes are honest, not a bill.
  - The reducer MATERIALIZES the reduced view (it rewrites bytes, it does not just count a
    budget): a stale/duplicate result becomes a real pointer; an over-budget result becomes
    a real head+tail with the middle spliced out to a pointer. So NO unique result is ever
    dropped to zero. `recoverable_frac` is MEASURED, not assumed: removed bytes count as
    recoverable only when a concrete re-fetch handle exists (a file path) or the bytes were
    redundant; the elided middle of a NON-file result (e.g. a Bash dump) is counted as
    recoverable ONLY via the production CAS page-back, so the fraction is honestly < 1.0.
  - It will NOT window a result below MIN_BUDGET — gutting an already-reasonable result
    is amputation, not noise-filtering. So a window of small results reduces ~1.0x and
    auto-tune CANNOT fabricate 2x out of incompressible context (asserted by selfcheck).

PRODUCTION PATH: this is the calibration + proof harness. The LIVE reducer is the
kernel's existing `internal/ctxmmu` Transform verdict (oversize benign result → a <2KB
recoverable pointer via the content store) tuned to a token budget + a stale-read class,
applied at the gateway seam (`internal/gateway/messages.go` completeAnthropicTurn). The
live path must stay PREFIX-STABLE (same input → same elision) or it busts the ~90%
prompt-cache that makes long sessions cheap.

Usage:
  python tools/ctxwin.py baseline [--since-days N] [--root DIR ...] [--max N]
                                  [--json OUT] [--md OUT] [--all] [--now STR]
  python tools/ctxwin.py reduce  [<session.jsonl> | --since-days N]
                                  [--profile off|conservative|balanced|aggressive|hyper]
                                  [--target 2.0] [--budget T] [--error-collapse off|head|oneline]
                                  [--per-tool TOOL=BUDGET ...] [--no-stale] [--no-noise] [--json OUT]
  python tools/ctxwin.py selfcheck      # synthetic fixtures; assert ratio + no overclaim
"""
import sys
import os
import json
import glob
import argparse
import hashlib
import re
import collections
import datetime

CHARS_PER_TOK = 4.0          # tokenizer-free estimate; cancels in ratios (documented above)
POINTER_TOK = 18             # size of an elision pointer we leave behind ("[ctxwin: …]")
DEDUP_POINTER_TOK = 14       # smaller pointer for a fully-redundant duplicate
STALE_POINTER_TOK = 16       # pointer for a stale read superseded by a later write
MIN_BUDGET = 256             # never window a result below this (amputation floor)
ERROR_HEAD_TOK = 80          # tokens kept from an error's head in --error-collapse head
EXCLUDE_NS_SUBSTR = ["pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"]
NS_INCLUDE_PREFIX = "C--work"
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Control-plane tools whose result is small + load-bearing (a todo list, a user choice);
# the aggressive/hyper presets EXEMPT these by default so aggression never guts them.
DEFAULT_EXEMPT_TOOLS = ("TodoWrite", "AskUserQuestion", "ExitPlanMode")
ERROR_COLLAPSE_MODES = ("off", "head", "oneline")

def toks(s):
    return len(s) / CHARS_PER_TOK

# --------------------------------------------------------------------------------------
# transcript discovery — scan ALL ~/.claude*/projects so we find the ACTIVE account's
# sessions (the fleet runs many accounts: ~/.claude, ~/.claude-<account>, ...).
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
# noise filter — lossless character-level cleanups applied to a result body.
# --------------------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TRAIL = re.compile(r"[ \t]+(?=\n)")
_BLANKRUN = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"[0-9]+")
_WS = re.compile(r"\s+")
_CATN = re.compile(r"(?m)^\s*\d+\t")   # the Read tool's "   123\t" line-number prefix

def filter_noise(s):
    """Strip ANSI codes, trailing whitespace, blank-line runs, and collapse runs of >=3
    identical consecutive lines to one line + a (xN) marker. Lossless for the model."""
    s = _ANSI.sub("", s)
    s = _TRAIL.sub("", s)
    s = _BLANKRUN.sub("\n\n", s)
    lines = s.split("\n")
    out = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        line = lines[i]
        marker = f"   …({run}× identical line above)"
        # Collapse a run to one line + marker ONLY when that actually saves bytes — a
        # noise filter must never INFLATE (short runs of short lines cost more in marker).
        if run >= 3 and (run - 1) * (len(line) + 1) > (len(marker) + 1):
            out.append(line)
            out.append(marker)
        else:
            out.extend([line] * run)
        i = j + 1
    return "\n".join(out)

def _h(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()

def _normh(s):
    return _h(_WS.sub(" ", _DIGITS.sub("#", s)).strip())

# --------------------------------------------------------------------------------------
# parse a transcript into ordered context-window ITEMS.
# tool_results are keyed by their UNIQUE tool_use_id (NOT message.id). A Read result is
# flagged `stale` when a LATER tool_use writes the same file path.
# --------------------------------------------------------------------------------------
class Item:
    __slots__ = ("kind", "tool", "path", "content", "toks", "h", "nh", "order", "stale", "is_error")
    def __init__(self, kind, tool, content, order, path="", is_error=False):
        self.kind = kind          # text | thinking | tool_use | result
        self.tool = tool
        self.path = path
        self.content = content
        self.toks = toks(content)
        # results AND tool_use inputs are reducible (exact-dup + windowing), so hash both.
        self.h = _h(content) if kind in ("result", "tool_use") else None
        self.nh = _normh(content) if kind == "result" else None
        self.order = order
        self.stale = False
        # `is_error` rides the tool_result block's structural flag (Claude Code sets it on a
        # failed tool call) — the honest signal the error-collapse tier keys on.
        self.is_error = is_error

def _norm_path(p):
    return os.path.normcase(str(p)) if p else ""

def _tooluse_path(inp):
    if not isinstance(inp, dict):
        return ""
    return _norm_path(inp.get("file_path") or inp.get("path") or inp.get("notebook_path") or "")

def rc_to_str(rc):
    if isinstance(rc, list):
        return "\n".join((x.get("text", "") if isinstance(x, dict) else str(x)) for x in rc)
    if isinstance(rc, str):
        return rc
    return json.dumps(rc)

def parse_window(path):
    use = {}                  # tool_use_id -> (tool name, normalized path)
    use_input = {}            # tool_use_id -> full "name {input-json}" string (last/full wins)
    use_order = {}            # tool_use_id -> first event order (for supersession)
    results = {}              # tool_use_id -> content (last/most-complete wins)
    result_err = {}           # tool_use_id -> structural is_error flag (sticky once true)
    result_order = []
    events = []               # (order, name, path) for supersession analysis
    nonresult = []
    seen_mid = set()
    ev_order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") not in ("user", "assistant"):
                continue
            m = rec.get("message")
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "tool_use":
                        tid = b.get("id")
                        tp = _tooluse_path(b.get("input"))
                        nm = b.get("name", "")
                        use[tid] = (nm, tp)
                        # tool_use INPUT is real window mass (a Write/Edit carries the full
                        # file body). Streamed snapshots share message.id and ACCUMULATE
                        # tool_use blocks, so key by tool_use id (last/full wins) — NOT the
                        # first message.id snapshot, which is often empty.
                        use_input[tid] = nm + " " + json.dumps(b.get("input", {}), separators=(",", ":"))
                        if tid not in use_order:
                            use_order[tid] = ev_order
                        events.append((ev_order, nm, tp))
                        ev_order += 1
                    elif bt == "tool_result":
                        tu = b.get("tool_use_id")
                        s = rc_to_str(b.get("content", ""))
                        if tu not in results:
                            result_order.append(tu)
                        results[tu] = s
                        # the structural error flag is sticky — once any snapshot marks this
                        # result an error, keep it (a later snapshot may drop the flag).
                        if b.get("is_error"):
                            result_err[tu] = True
            mid = m.get("id")
            if mid and mid in seen_mid:
                continue
            if mid:
                seen_mid.add(mid)
            # text/thinking are deduped by message.id (stream snapshots). tool_use mass is
            # NOT taken here — it is accounted once per tool_use id from use_input below, so
            # an empty first snapshot can't undercount a later snapshot's full input.
            if isinstance(c, str):
                nonresult.append(("text", "", c))
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        nonresult.append(("text", "", b.get("text", "") or ""))
                    elif bt == "thinking":
                        nonresult.append(("thinking", "", b.get("thinking", "") or ""))

    # supersession: for a path, the order of the LAST write event to it.
    last_write_order = {}
    for o, name, p in events:
        if name in WRITE_TOOLS and p:
            last_write_order[p] = max(last_write_order.get(p, -1), o)
    items = []
    order = 0
    for kind, tool, content in nonresult:
        items.append(Item(kind, tool, content, order))
        order += 1
    # tool_use INPUT mass — one Item per tool_use id (full input, last wins). Carry the
    # file path so a windowed Write/Edit payload is recoverable (the bytes land in the file).
    for tid in use_input:
        nm, p = use.get(tid, ("", ""))
        items.append(Item("tool_use", nm, use_input[tid], order, path=p))
        order += 1
    for tu in result_order:
        name, p = use.get(tu, ("?", ""))
        it = Item("result", name, results[tu], order, path=p,
                  is_error=result_err.get(tu, False))
        order += 1
        # stale = a Read whose path is written by a LATER tool_use. The result's own event
        # order is its issuing tool_use's order (bound by tool_use_id, NOT a name+path
        # cursor — so a user message carrying several parallel results can't misorder them).
        my_ev = use_order.get(tu, -1)
        if name == "Read" and p and p in last_write_order and last_write_order[p] > my_ev:
            it.stale = True
        items.append(it)
    return items, {"n_results": len(result_order)}

# --------------------------------------------------------------------------------------
# materialization — the reducer ACTUALLY rewrites a result's bytes (it does not merely
# count a budget). Every tier leaves a real, smaller string, so "never dropped to zero"
# and "head+tail+pointer" are properties of the emitted bytes, not assertions.
# --------------------------------------------------------------------------------------
def window_content(s, budget_tok, path="", head_frac=0.6):
    """Keep a head+tail summing to ~budget_tok and splice a real elision pointer in the
    middle. Returns the rewritten string (always shorter than s when s exceeds budget).
    head_frac is the head's share of the budget; 1.0 => head-only (no tail), which aggressive
    profiles use to keep the start of a result (the most load-bearing part) and drop the rest."""
    budget_chars = max(4 * (POINTER_TOK + 1), int(budget_tok * CHARS_PER_TOK))
    if len(s) <= budget_chars:
        return s
    handle = f", re-fetch {os.path.basename(path)}" if path else ""
    elided_chars = len(s) - budget_chars  # provisional; pointer eats a little more
    ptr = f"\n…[ctxwin: elided ~{int(elided_chars / CHARS_PER_TOK)} tok{handle}]…\n"
    avail = max(2, budget_chars - len(ptr))
    head_frac = min(1.0, max(0.0, head_frac))
    head = int(avail * head_frac)
    tail = avail - head
    return s[:head] + ptr + (s[-tail:] if tail > 0 else "")

def _stale_pointer(it):
    return f"[ctxwin: stale read of {os.path.basename(it.path)} — superseded by a later edit; re-fetch to view]"

def _dedup_pointer(it):
    return f"[ctxwin: duplicate of an earlier {it.tool} payload — elided]"

# --------------------------------------------------------------------------------------
# error-collapse — the LOSSY, opt-in tier the operator dials up. A failed tool call's body
# (a 30k stack trace, a stderr flood) is the example the user names; the structural is_error
# flag on the tool_result is the honest signal. OFF by default; only the aggressive/hyper
# presets (or `--error-collapse`) enable it. The regex is a TIGHT secondary for errors that
# arrive without the flag — deliberately conservative, since a false positive guts real text.
# --------------------------------------------------------------------------------------
_ERR_HEAD = re.compile(
    r"(?im)^\s{0,4}("
    r"error[:\s]|traceback \(most recent call last\)|fatal:|panic:|exception[:\s]|"
    r"[A-Za-z_]+(?:Error|Exception)[:\s]|command not found|no such file or directory|"
    r"permission denied|segmentation fault|exit (?:code|status) [1-9])")

def looks_like_error(it):
    """True when a RESULT is error-shaped. Primary signal: the transcript's structural
    is_error flag. Secondary: a tight error marker anchored at the head of the body."""
    if it.kind != "result":
        return False
    if getattr(it, "is_error", False):
        return True
    return bool(_ERR_HEAD.search(it.content[:300]))

def _error_oneline(content, tool, path):
    n = int(toks(content))
    handle = f"; re-fetch {os.path.basename(path)}" if path else ""
    return f"[ctxwin: {tool or 'tool'} error — ~{n} tok elided{handle}]"

def _error_head(content, tool, path, head_tok=ERROR_HEAD_TOK):
    """Keep the first line (the error MESSAGE) up to head_tok, drop the stack/stderr body."""
    head_chars = int(head_tok * CHARS_PER_TOK)
    first = content.split("\n", 1)[0]
    head = first if len(first) <= head_chars else content[:head_chars]
    n = max(0, int(toks(content) - toks(head)))
    handle = f"; re-fetch {os.path.basename(path)}" if path else ""
    return head + f"\n…[ctxwin: error body elided ~{n} tok{handle}]…"

def collapse_error(it, mode, content=None):
    """Collapse an error-shaped result. Returns (new_content, recoverable_bool). 'head' keeps
    the first line + a pointer; 'oneline' drops to a tiny marker (the '30k error -> "error"'
    case). Recoverable ONLY when the result is file-backed; an error body is otherwise gone
    from context here (the production CAS page-back is not counted, same rule as windowing).
    `content` overrides it.content (so the reducer can collapse the noise-cleaned body)."""
    body = it.content if content is None else content
    new = (_error_oneline(body, it.tool, it.path) if mode == "oneline"
           else _error_head(body, it.tool, it.path))
    return new, bool(it.path)

# --------------------------------------------------------------------------------------
# Profile — the TUNING DIAL. One object carries which tiers run, the default per-item budget
# (None => the caller auto-tunes to `target`), the head/tail split, the lossy error-collapse
# mode, and per-tool budget overrides (value None => EXEMPT that tool from all reduction).
# Named presets span `off` (ratio 1.0) -> `hyper` (collapse a 30k error to one line). The
# default profile reproduces today's reducer byte-for-byte, so honesty invariants are intact.
# --------------------------------------------------------------------------------------
class Profile:
    __slots__ = ("name", "noise", "stale", "dedup", "window", "budget", "target",
                 "min_budget", "head_frac", "error_collapse", "per_tool")

    def __init__(self, name, noise=True, stale=True, dedup=True, window=True, budget=None,
                 target=2.0, min_budget=MIN_BUDGET, head_frac=0.6, error_collapse="off",
                 per_tool=None):
        self.name = name
        self.noise = noise
        self.stale = stale
        self.dedup = dedup
        self.window = window
        self.budget = budget                 # fixed per-item budget; None => autotune to target
        self.target = target                 # autotune target when budget is None
        self.min_budget = min_budget          # amputation floor for the WINDOW tier (default path)
        self.head_frac = head_frac            # head share of a windowed item (1.0 => head-only)
        self.error_collapse = error_collapse  # off | head | oneline (lossy, opt-in)
        self.per_tool = dict(per_tool or {})  # tool -> budget override; value None => exempt

    def tool_budget(self, tool, default):
        """The window budget for a tool: a per-tool override (which may be None = exempt) or
        the default. A per-tool override is an EXPLICIT choice, so it bypasses the floor."""
        return self.per_tool[tool] if tool in self.per_tool else default

    def is_exempt(self, tool):
        return tool in self.per_tool and self.per_tool[tool] is None


def _exempt_map():
    return {t: None for t in DEFAULT_EXEMPT_TOOLS}

# The presets are a MONOTONIC dial: each one reduces at least as hard as the one before it.
# That ordering is by construction — every step raises the auto-tune target AND lowers the
# amputation floor, so a higher tier can always window below where a lower one stops, and the
# lossy error tier only switches ON as you climb. (A fixed per-item budget would break the
# monotonicity — a corpus can need a budget below the fixed one to even hit 2x — so the
# aggressive tiers auto-tune to a higher target instead, honoring the no-fabrication floor.)
PROFILES = {
    # off: nothing runs -> every result verbatim, ratio 1.0 (the "feature disabled" choice).
    "off": lambda: Profile("off", noise=False, stale=False, dedup=False, window=False),
    # conservative: lossless + recoverable only; NO windowing of unique current content.
    "conservative": lambda: Profile("conservative", window=False),
    # balanced: the default — low-risk tiers + bounded windowing auto-tuned to 2x.
    "balanced": lambda: Profile("balanced", target=2.0),
    # aggressive: push to 3x with a lower floor + head-only windows, and collapse errors to
    # their first line (the message). Strictly >= balanced (higher target, lower floor).
    "aggressive": lambda: Profile("aggressive", target=3.0, min_budget=128,
                                  head_frac=1.0, error_collapse="head", per_tool=_exempt_map()),
    # hyper: push to 4x with the lowest floor; collapse a 30k error to a one-line marker.
    # Lossy by choice. Strictly >= aggressive. (Add per-tool caps with --per-tool if wanted.)
    "hyper": lambda: Profile("hyper", target=4.0, min_budget=48, head_frac=1.0,
                             error_collapse="oneline", per_tool=_exempt_map()),
}

def make_profile(name):
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; use one of: {', '.join(PROFILES)}")
    return PROFILES[name]()

# --------------------------------------------------------------------------------------
# the risk-ranked reducer — the heart of "self-reduce 2x". The reducible surface is BOTH
# tool_results AND tool_use INPUTS (a Write/Edit/Workflow payload is large context mass too
# and is file-recoverable); text/thinking are untouched. Each item is rewritten by at most
# one collapsing tier (mutually exclusive), in risk order:
#   tier 0 NOISE   (lossless)        : strip ANSI/whitespace/repeated lines in a RESULT body.
#   tier 1 STALE   (low-risk, recov.): a Read superseded by the agent's later write to the
#                                       same path -> pointer (re-fetchable from the file).
#   tier 2 DEDUP   (lossless)        : an exact-duplicate result/input -> pointer (no new info).
#   tier 3 WINDOW  (bounded)         : an item over `budget` -> head+tail+pointer
#                                       (re-fetchable when file-backed; CAS-paged otherwise).
# NEVER windows below MIN_BUDGET; NEVER drops a unique item to zero (a head+tail or a
# pointer always survives). `recoverable` is MEASURED: removed bytes count as recoverable
# only when a concrete re-fetch handle exists (a file path) or the bytes were redundant.
# --------------------------------------------------------------------------------------
def reduce_window(items, budget, min_budget=MIN_BUDGET, noise=True, stale=True, dedup=True,
                  window=True, profile=None):
    # The reducer is profile-driven; the legacy scalar kwargs build an ad-hoc profile so every
    # existing call site (and the default path) is byte-for-byte unchanged. A `profile` object
    # carries the same knobs plus the opt-in lossy levers (error-collapse, per-tool budgets).
    if profile is None:
        profile = Profile("custom", noise=noise, stale=stale, dedup=dedup, window=window,
                          min_budget=min_budget)
    min_budget = profile.min_budget
    # Reducible surface = tool_results AND tool_use INPUTS (the agent-authored Write/Edit/
    # Workflow payloads are large context mass too, and are file-recoverable). text/thinking
    # are left untouched.
    reducible = [it for it in items if it.kind in ("result", "tool_use")]
    nonresult_tok = sum(it.toks for it in items if it.kind not in ("result", "tool_use"))
    baseline = nonresult_tok + sum(it.toks for it in reducible)
    eff_budget = max(min_budget, budget) if profile.window else 10 ** 12

    seen_h = set()
    kept = nonresult_tok
    rm = collections.Counter()        # removed token mass by tier
    cnt = collections.Counter()       # items touched by tier
    removed_recoverable = 0.0         # removed bytes with a concrete re-fetch handle
    for it in reducible:
        # EXEMPT: a per-tool override of None keeps this tool's payload verbatim (no tier
        # touches it) — for small load-bearing control-plane results (a todo list, a choice).
        if profile.is_exempt(it.tool):
            kept += it.toks
            continue
        content = it.content
        t0 = toks(content)
        # tier 0: noise — only result bodies (tool_use inputs are structured JSON args)
        if profile.noise and it.kind == "result":
            content = filter_noise(content)
            d = t0 - toks(content)
            if d > 0:
                rm["noise"] += d
                cnt["noise"] += 1
                removed_recoverable += d            # ANSI/whitespace are not information
        t1 = toks(content)
        # tier 1: stale read superseded by a later write -> pointer (results only)
        if profile.stale and it.stale:
            new = _stale_pointer(it)
            d = max(0.0, t1 - toks(new))
            kept += toks(new)
            rm["stale"] += d
            cnt["stale"] += 1
            removed_recoverable += d                # the file is re-readable
            continue
        # tier 2: exact duplicate -> pointer (hash the ORIGINAL content for determinism)
        if profile.dedup and it.h in seen_h:
            new = _dedup_pointer(it)
            d = max(0.0, t1 - toks(new))
            kept += toks(new)
            rm["dedup"] += d
            cnt["dedup"] += 1
            removed_recoverable += d                # redundant: the bytes added no new info
            continue
        seen_h.add(it.h)
        # tier 2.5: ERROR collapse (LOSSY, opt-in) -> first line or a one-line marker. Runs
        # only when the profile turns it on AND the result is error-shaped. The dropped body
        # is recoverable only if the result is file-backed (rare for an error) — so this
        # honestly lowers recoverable_frac, the signal that aggression destroys information.
        if profile.error_collapse != "off" and looks_like_error(it):
            new, recov = collapse_error(it, profile.error_collapse, content=content)
            d = max(0.0, t1 - toks(new))
            if d > 0:                               # never INFLATE a small error
                kept += toks(new)
                rm["error"] += d
                cnt["error"] += 1
                if recov:
                    removed_recoverable += d
                continue
        # tier 3: window -> real head+tail+pointer. The budget is per-tool (an explicit
        # override bypasses the floor; the default budget is floored at min_budget).
        item_budget = profile.tool_budget(it.tool, eff_budget)
        if profile.window and item_budget is not None and t1 > item_budget:
            new = window_content(content, item_budget, it.path, head_frac=profile.head_frac)
            d = max(0.0, t1 - toks(new))
            kept += toks(new)
            rm["window"] += d
            cnt["window"] += 1
            if it.path:                             # file-backed -> middle re-readable
                removed_recoverable += d
            # else: head+tail survive, but the elided middle of a non-file result is
            # recoverable only via the production CAS page-back (not by this tool) — NOT
            # counted as recoverable here, on purpose.
        else:
            kept += t1
    removed = sum(rm.values())
    d = baseline - removed
    ratio = round(baseline / d, 4) if d > 0 else None
    low = rm["noise"] + rm["stale"] + rm["dedup"]
    return {
        "profile": profile.name,
        "baseline_tok": int(baseline),
        "reduced_tok": int(baseline - removed),
        "removed_tok": int(removed),
        "ratio": ratio,
        "budget": eff_budget if profile.window else None,
        "error_collapse": profile.error_collapse,
        "per_tool": dict(profile.per_tool),
        "n_reducible": len(reducible),
        "tiers": {
            "noise":  {"removed_tok": int(rm["noise"]),  "n": cnt["noise"],  "risk": "none (lossless)"},
            "stale":  {"removed_tok": int(rm["stale"]),  "n": cnt["stale"],  "risk": "low (evidence-based, recoverable)"},
            "dedup":  {"removed_tok": int(rm["dedup"]),  "n": cnt["dedup"],  "risk": "none (lossless)"},
            "error":  {"removed_tok": int(rm["error"]),  "n": cnt["error"],  "risk": "LOSSY (opt-in; head/marker only, not file-recoverable)"},
            "window": {"removed_tok": int(rm["window"]), "n": cnt["window"], "risk": "bounded (recoverable)"},
        },
        "low_risk_removed_tok": int(low),
        "low_risk_ratio": round(baseline / (baseline - low), 4) if baseline - low > 0 else None,
        # MEASURED: fraction of removed bytes with a concrete re-fetch handle (file path or
        # redundant). < 1.0 means some windowed non-file middles / collapsed error bodies are
        # CAS-recoverable only — the honest cost of turning aggression up.
        "recoverable_frac": round(removed_recoverable / removed, 4) if removed > 0 else 1.0,
        # how much of the reduction came from the LOSSY error tier (0 in the default profile).
        "lossy_removed_tok": int(rm["error"]),
    }

def autotune(items, target, min_budget=MIN_BUDGET, profile=None):
    """Binary-search the LARGEST uniform window budget that still meets `target` (least
    lossy). The profile's tiers (noise+stale+dedup, plus any opt-in error-collapse) always
    run; only the default window budget is searched. Bounded at the profile's min_budget — if
    even the floor can't reach target, returns the honest best (never fabricates the ratio)."""
    mb = profile.min_budget if profile is not None else min_budget
    lo, hi = float(mb), 64000.0
    # first: can we even reach target windowing at the floor?
    floor = reduce_window(items, mb, min_budget=mb, profile=profile)
    if floor["ratio"] is None or floor["ratio"] < target:
        return floor, mb            # honest: target unreachable without amputation
    best = floor
    for _ in range(30):
        mid = (lo + hi) / 2.0
        r = reduce_window(items, mid, min_budget=mb, profile=profile)
        if r["ratio"] is not None and r["ratio"] >= target:
            best = r
            lo = mid       # afford a larger (less lossy) budget
        else:
            hi = mid
    return best, round(lo, 1)

# --------------------------------------------------------------------------------------
# CLI: baseline
# --------------------------------------------------------------------------------------
def cmd_baseline(args):
    roots = args.root or discover_roots()
    nsp = "" if args.all else NS_INCLUDE_PREFIX
    sess = discover(roots, since_days=args.since_days, ns_prefix=nsp, max_n=args.max)
    if not sess:
        print(f"no transcripts found under {roots} (ns_prefix={nsp!r}); try --all or --root", file=sys.stderr)
        return 2
    all_items = []
    agg_kind = collections.Counter()
    agg_tool = collections.Counter()
    agg_tool_excess = collections.Counter()   # per-tool windowing headroom over a 700-tok cap
    total = 0.0
    excess = collections.Counter()
    dup = near = stale = noise = 0.0
    catn = xrepeat = 0.0          # line-level noise: cat-n prefixes; cross-result repeats
    errmass = 0.0
    errn = 0       # error-shaped result mass (the error-collapse tier's prize)
    rq = [0.0, 0.0, 0.0, 0.0]
    for s in sess:
        items, _ = parse_window(s["path"])
        all_items.extend(items)
        results = [it for it in items if it.kind == "result"]
        reducible = [it for it in items if it.kind in ("result", "tool_use")]
        for it in items:
            agg_kind[it.kind] += it.toks
            total += it.toks
            if it.kind == "result":
                agg_tool[it.tool] += it.toks
        sh, snh, seen_lines = set(), set(), set()
        for it in results:
            if it.h in sh:
                dup += it.toks
            else:
                sh.add(it.h)
            if it.nh in snh:
                near += it.toks
            else:
                snh.add(it.nh)
            if it.stale:
                stale += it.toks
            if looks_like_error(it):
                errmass += it.toks
                errn += 1
            noise += it.toks - toks(filter_noise(it.content))
            # cat -n line-number prefixes the Read tool prepends ("   123\t"): formatting,
            # not file content (but they ARE Edit-targeting signal — informational only).
            catn += sum(len(m.group(0)) for m in _CATN.finditer(it.content)) / CHARS_PER_TOK
            # cross-result repeated lines (beyond within-result dedup the noise tier does):
            # a substantial line already seen in an EARLIER result this session.
            for ln in it.content.split("\n"):
                if len(ln) >= 20 and ln.strip():
                    hh = _h(ln)
                    if hh in seen_lines:
                        xrepeat += len(ln) / CHARS_PER_TOK
                    else:
                        seen_lines.add(hh)
        # windowing excess + recency span the FULL reducible surface (results + tool_use),
        # matching what reduce_window actually touches.
        for it in reducible:
            for b in (2000, 1000, 700, 500):
                if it.toks > b:
                    excess[b] += it.toks - b
            if it.toks > 700:                       # per-tool windowing headroom over a 700 cap
                agg_tool_excess[it.tool or "?"] += it.toks - 700
        n = len(reducible) or 1
        for i, it in enumerate(sorted(reducible, key=lambda x: x.order)):
            rq[min(3, int(4 * i / n))] += it.toks
    total = total or 1.0

    def ratio(removed):
        d = total - removed
        return round(total / d, 3) if d > 0 else None

    tuned, B = autotune(all_items, 2.0)

    # what each named profile achieves on THIS corpus — the tuning menu, measured. off=1.0,
    # conservative=lossless+recoverable (no windowing), balanced=2x, aggressive/hyper add the
    # lossy error tier. recoverable_frac drops as aggression rises — the honest cost.
    def profile_report(name):
        p = make_profile(name)
        if p.budget is None and p.window:
            rep, _ = autotune(all_items, p.target, profile=p)
        else:
            rep = reduce_window(all_items, p.budget if p.budget is not None else 0, profile=p)
        return {"ratio": rep["ratio"], "recoverable_frac": rep["recoverable_frac"],
                "lossy_removed_tok": rep["lossy_removed_tok"], "error_collapse": rep["error_collapse"]}
    profiles_out = {nm: profile_report(nm) for nm in ("off", "conservative", "balanced", "aggressive", "hyper")}

    low = noise + stale + dup
    out = {
        "generated": args.now or "(stamp at commit)",
        "n_sessions": len(sess),
        "total_tok": int(total),
        "composition_pct": {k: round(100 * v / total, 1) for k, v in agg_kind.most_common()},
        "tool_result_pct_of_total": round(100 * agg_kind.get("result", 0) / total, 1),
        "result_mass_by_tool_pct": {k: round(100 * v / max(1, agg_kind.get("result", 1)), 1)
                                    for k, v in agg_tool.most_common(8)},
        # per-tool windowing headroom: who carries the reducible mass over a 700-tok cap. This
        # is the evidence for `--per-tool` tuning (cap the heavy tools, exempt the light ones).
        "windowing_headroom_by_tool_pct": {k: round(100 * v / total, 1)
                                           for k, v in agg_tool_excess.most_common(8)},
        # error-shaped results (structural is_error flag or a tight error marker): the prize
        # the `--error-collapse` lever targets — turning hyper on reclaims most of this.
        "error_results": {"pct_of_total": round(100 * errmass / total, 2), "n": errn,
                          "note": "is_error flag or a head error marker; --error-collapse hyper "
                                  "reclaims most of this (LOSSY — not file-recoverable)"},
        "tiers": {
            "noise_lossless": {"pct": round(100 * noise / total, 2), "ratio": ratio(noise), "risk": "none"},
            "stale_reads":    {"pct": round(100 * stale / total, 2), "ratio": ratio(stale), "risk": "low (recoverable)"},
            "exact_dedup":    {"pct": round(100 * dup / total, 2),   "ratio": ratio(dup),   "risk": "none"},
            "near_dedup":     {"pct": round(100 * near / total, 2),  "ratio": ratio(near),  "risk": "none"},
        },
        # line-level noise the within-result filter does NOT catch — reported separately
        # (overlaps other tiers, so NOT folded into low_risk_cumulative).
        "line_noise": {
            "catn_prefix_pct": round(100 * catn / total, 2),
            "cross_result_repeat_pct": round(100 * xrepeat / total, 2),
            "char_noise_pct": round(100 * noise / total, 2),
            "combined_pct": round(100 * (catn + xrepeat + noise) / total, 2),
            "note": "cat-n prefixes are Edit-targeting signal (informational); cross-result "
                    "repeats are near-lossless; combined is still far below stale's share",
        },
        "low_risk_cumulative": {"pct": round(100 * low / total, 1), "ratio": ratio(low),
                                "note": "noise+stale+exact-dedup, zero relevance guessing"},
        "windowing_ratio": {f"cap@{b}": ratio(excess[b]) for b in (2000, 1000, 700, 500)},
        "windowing_excess_pct": {f"cap@{b}": round(100 * excess[b] / total, 1) for b in (2000, 1000, 700, 500)},
        "recency_quartile_pct": [round(100 * x / (sum(rq) or 1), 1) for x in rq],
        "oldest_half_pct": round(100 * (rq[0] + rq[1]) / (sum(rq) or 1), 1),
        "self_reduce_2x": {"target": 2.0, "found_budget_tok": B, "achieved_ratio": tuned["ratio"],
                           "low_risk_share_ratio": tuned["low_risk_ratio"],
                           "recoverable_frac": tuned["recoverable_frac"]},
        # the TUNING MENU, measured on this corpus: ratio + the honest recoverable_frac cost.
        "profiles": profiles_out,
    }
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
    L.append("# Context-window baseline — Claude Code sessions\n")
    L.append(f"_Generated: {o['generated']} · {o['n_sessions']} heaviest sessions · "
             f"{o['total_tok']:,} estimated context tokens (≈4 chars/tok)._\n")
    L.append("> Auto-generated snapshot — regenerate with "
             "`python tools/ctxwin.py baseline --md docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-<date>.md`. "
             "Don't hand-edit. Numbers are from this box's own sessions; aggregate-only (no transcript "
             "content or paths).\n")
    L.append("## What the window is made of\n")
    L.append("| block kind | % of context |\n|---|---|")
    for k, v in o["composition_pct"].items():
        L.append(f"| {k} | {v}% |")
    L.append(f"\n**Tool results are {o['tool_result_pct_of_total']}% of the window.** Result mass by tool:\n")
    L.append("| tool | % of result mass |\n|---|---|")
    for k, v in o["result_mass_by_tool_pct"].items():
        L.append(f"| {k} | {v}% |")
    L.append("\n## How much is reducible — by RISK tier\n")
    L.append("| tier | risk | % of context | reduction |\n|---|---|---|---|")
    t = o["tiers"]
    L.append(f"| character noise (ANSI/whitespace/repeated lines) | {t['noise_lossless']['risk']} | "
             f"{t['noise_lossless']['pct']}% | {t['noise_lossless']['ratio']}× |")
    L.append(f"| **stale reads** (a Read the agent later Edited/Wrote) | {t['stale_reads']['risk']} | "
             f"**{t['stale_reads']['pct']}%** | **{t['stale_reads']['ratio']}×** |")
    L.append(f"| exact duplicates | {t['exact_dedup']['risk']} | {t['exact_dedup']['pct']}% | {t['exact_dedup']['ratio']}× |")
    lr = o["low_risk_cumulative"]
    L.append(f"| **LOW-RISK cumulative** ({lr['note']}) | low | **{lr['pct']}%** | **{lr['ratio']}×** |")
    ln = o["line_noise"]
    L.append(f"\n- **Line-level noise** (informational, overlaps the tiers above): cat-n line-number "
             f"prefixes {ln['catn_prefix_pct']}% (Edit-targeting *signal*, not free to drop) + cross-result "
             f"repeated lines {ln['cross_result_repeat_pct']}% + character noise {ln['char_noise_pct']}% "
             f"≈ **{ln['combined_pct']}%** combined. So \"strip the obvious noise\" is ~{ln['combined_pct']}%, "
             "not nothing — but still **far below the stale-read share**. The dominant low-risk lever is "
             "**whole stale results** (files the agent read, then changed): superseded by construction, "
             "recoverable, zero relevance guessing.")
    L.append("- **Per-item windowing (bounded-loss)** carries the rest of the way — cap each surviving "
             "tool_result OR tool_use input to budget B, keep head+tail, elide the middle to a recoverable "
             "pointer:\n")
    L.append("| budget B (tok) | windowing saves | reduction |\n|---|---|---|")
    for b in (2000, 1000, 700, 500):
        L.append(f"| {b} | {o['windowing_excess_pct'][f'cap@{b}']}% | {o['windowing_ratio'][f'cap@{b}']}× |")
    L.append(f"\n- **Recency:** reducible mass by quartile (oldest→newest) = {o['recency_quartile_pct']}%; "
             f"the oldest half holds **{o['oldest_half_pct']}%** — old items are the safest to collapse.")
    if o.get("windowing_headroom_by_tool_pct"):
        L.append("\n- **Windowing headroom by tool** (% of context held over a 700-tok cap — who "
                 "to target with `--per-tool`): "
                 + ", ".join(f"{k} {v}%" for k, v in o["windowing_headroom_by_tool_pct"].items()) + ".")
    er = o.get("error_results")
    if er:
        L.append(f"- **Error-shaped results:** {er['pct_of_total']}% of the window across {er['n']} "
                 "results carry a structural `is_error` flag or a head error marker. `--error-collapse "
                 "hyper` reclaims most of this by collapsing a failed-call body to one line — **lossy, "
                 "not file-recoverable**, so it is OFF in every profile but `aggressive`/`hyper`.")
    pf = o.get("profiles")
    if pf:
        L.append("\n## Tuning menu — what each profile does to THIS corpus\n")
        L.append("The reducer is a dial. `off` keeps everything; `hyper` collapses a 30k error to one "
                 "line. `recoverable` is the honest cost — the fraction of removed bytes with a direct "
                 "re-fetch handle, which falls as the lossy error tier kicks in.\n")
        L.append("| profile | reduction | recoverable | lossy tok removed | error-collapse |\n|---|---|---|---|---|")
        for nm in ("off", "conservative", "balanced", "aggressive", "hyper"):
            p = pf.get(nm)
            if not p:
                continue
            L.append(f"| `{nm}` | {p['ratio']}× | {round(100*p['recoverable_frac'])}% | "
                     f"{p['lossy_removed_tok']:,} | {p['error_collapse']} |")
        L.append("\n> Compose finer with `--error-collapse off|head|oneline` and `--per-tool Bash=200 "
                 "--per-tool TodoWrite=off`. The lossy levers stay OFF unless you ask; `recoverable` is "
                 "MEASURED per profile, never asserted.")
    sr = o["self_reduce_2x"]
    L.append("\n## Self-reduce to 2×\n")
    L.append(f"Low-risk tiers first (noise + stale + exact-dedup), then a uniform per-result budget of "
             f"**{sr['found_budget_tok']} tok** → **{sr['achieved_ratio']}× reduction**. The low-risk tiers "
             f"alone reach {sr['low_risk_share_ratio']}×; the rest is bounded windowing. Every result keeps a "
             f"real head+tail or a pointer (**nothing is dropped to zero**), and **{round(100*sr['recoverable_frac'])}% "
             "of removed bytes have a direct re-fetch handle** (file-backed or redundant); the remainder is the "
             "elided middle of non-file results, recoverable via the production CAS page-back.\n")
    L.append("> Estimate-over-estimate ratios (≈4 chars/tok). This is the empirical pass over REAL "
             "Claude Code `.jsonl` transcripts that the `ctxplan` planned-view note "
             "(`docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md`) names as its missing measurement: the "
             "stale-read and recency tiers are the resident-redundancy signals `ctxplan`'s benefit model "
             "exploits, and \"recoverable\" here is `ctxplan`'s **Faithful** (every elided span keeps a "
             "page-back handle). The result-level windowing maps to `internal/ctxmmu` Transform "
             "(oversize→recoverable CAS pointer). Both must be applied prefix-stably at the gateway seam "
             "(`fak guard`, residual #555) so the ~90% prompt cache survives.\n")
    return "\n".join(L) + "\n"

# --------------------------------------------------------------------------------------
# CLI: reduce
# --------------------------------------------------------------------------------------
def cmd_reduce(args):
    if args.transcript:
        items, _ = parse_window(args.transcript)
        label = os.path.basename(args.transcript)
    else:
        roots = args.root or discover_roots()
        nsp = "" if args.all else NS_INCLUDE_PREFIX
        sess = discover(roots, since_days=args.since_days, ns_prefix=nsp, max_n=args.max)
        if not sess:
            print("no transcripts found", file=sys.stderr)
            return 2
        items = []
        for s in sess:
            items.extend(parse_window(s["path"])[0])
        label = f"{len(sess)} sessions"

    prof = resolve_profile(args)
    # budget/target resolution: an explicit --budget wins; else a profile with a fixed budget
    # (aggressive/hyper) uses it unless --target was given; else auto-tune to the profile target.
    if not prof.window:
        rep = reduce_window(items, 0, profile=prof)
        budget = None
    elif args.budget:
        rep = reduce_window(items, args.budget, profile=prof)
        budget = rep["budget"]
    elif prof.budget is not None and args.target is None:
        rep = reduce_window(items, prof.budget, profile=prof)
        budget = rep["budget"]
    else:
        rep, budget = autotune(items, args.target if args.target is not None else prof.target, profile=prof)
    rep["scope"] = label
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"wrote {args.json}")
        return 0
    print(f"\n  ctxwin · self-reduce — {label}")
    exempt = sorted(t for t, v in prof.per_tool.items() if v is None)
    caps = sorted(f"{t}={v}" for t, v in prof.per_tool.items() if v is not None)
    print(f"  profile {prof.name}  ·  window budget {budget} tok/result  ·  error-collapse {prof.error_collapse}")
    if caps:
        print(f"  per-tool caps: {', '.join(caps)}")
    if exempt:
        print(f"  exempt tools: {', '.join(exempt)}")
    print(f"  ── baseline {rep['baseline_tok']:,} tok  →  reduced {rep['reduced_tok']:,} tok")
    if rep["ratio"]:
        print(f"  ── REDUCTION {rep['ratio']}×  (removed {rep['removed_tok']:,} tok; "
              f"low-risk tiers alone {rep['low_risk_ratio']}×)")
    tl = rep["tiers"]
    for name in ("noise", "stale", "dedup", "error", "window"):
        ti = tl[name]
        if ti["n"]:
            print(f"     • {name:7s} {ti['n']:>4} results  −{ti['removed_tok']:>8,} tok   [{ti['risk']}]")
    rf = round(100 * rep["recoverable_frac"])
    if rep["lossy_removed_tok"] > 0:
        print(f"     • LOSSY: the error tier dropped {rep['lossy_removed_tok']:,} tok with no file handle; "
              f"only {rf}% of all removed bytes are directly re-fetchable (CAS page-back for the rest)\n")
    else:
        print(f"     • every result keeps a head+tail or pointer (nothing dropped to zero); "
              f"{rf}% of removed bytes have a direct re-fetch handle (file-backed/redundant)\n")
    return 0


def resolve_profile(args):
    """Build the reducer profile from a named preset (default `balanced` = today's behavior),
    then apply fine-grained overrides (`--no-noise/--no-stale/--error-collapse/--per-tool`)."""
    prof = make_profile(getattr(args, "profile", None) or "balanced")
    if getattr(args, "no_noise", False):
        prof.noise = False
    if getattr(args, "no_stale", False):
        prof.stale = False
    ec = getattr(args, "error_collapse", None)
    if ec:
        if ec not in ERROR_COLLAPSE_MODES:
            raise SystemExit(f"--error-collapse must be one of {', '.join(ERROR_COLLAPSE_MODES)}")
        prof.error_collapse = ec
    for kv in (getattr(args, "per_tool", None) or []):
        tool, _, raw = kv.partition("=")
        tool = tool.strip()
        raw = raw.strip().lower()
        if not tool:
            raise SystemExit(f"--per-tool expects TOOL=BUDGET (got {kv!r})")
        if raw in ("off", "none", "exempt", ""):
            prof.per_tool[tool] = None        # exempt this tool from all reduction
        else:
            try:
                prof.per_tool[tool] = max(1, int(float(raw)))
            except ValueError:
                raise SystemExit(f"--per-tool budget must be a number or off (got {raw!r} for {tool})")
    return prof

# --------------------------------------------------------------------------------------
# CLI: selfcheck — synthetic fixtures with KNOWN reducible structure. CI gate.
# --------------------------------------------------------------------------------------
def _mk(kind, tool, n_chars, salt="", path="", stale=False, is_error=False):
    body = (salt + "x" * max(0, n_chars - len(salt)))[:n_chars]
    it = Item(kind, tool, body, 0, path=path, is_error=is_error)
    it.stale = stale
    return it

def _seq(items):
    for i, it in enumerate(items):
        it.order = i
    return items

def runselfcheck():
    print("== ctxwin selfcheck: synthetic fixtures (no real data, no model) ==\n")
    fails = 0

    # 0: MATERIALIZATION witness. window_content must keep a real head+tail and splice a
    # pointer — proven on the emitted bytes, not asserted.
    big = "HEAD_MARKER " + "m" * 8000 + " TAIL_MARKER"
    win = window_content(big, 500, path="/some/file.go")
    ok = (len(win) < len(big) and win.startswith("HEAD_MARKER") and win.endswith("TAIL_MARKER")
          and "ctxwin: elided" in win and "re-fetch file.go" in win)
    print(f"  materialize        {'PASS' if ok else 'FAIL'}  shorter={len(win)<len(big)} head&tail_kept={win.startswith('HEAD_MARKER') and win.endswith('TAIL_MARKER')} pointer={'ctxwin: elided' in win}")
    fails += not ok

    # 1: HALVABLE via windowing. Ten 2000-tok FILE-backed Reads + chatter. Cap@1000 → ~2x.
    f1 = _seq([_mk("text", "", 40)] +
              [_mk("result", "Read", 8000, salt=f"r{i}", path=f"/r{i}") for i in range(10)] +
              [_mk("text", "", 40) for _ in range(10)])
    r1 = reduce_window(f1, 1000)
    ok = r1["ratio"] and r1["ratio"] >= 1.8 and r1["recoverable_frac"] == 1.0 and r1["tiers"]["window"]["n"] == 10
    print(f"  halvable           {'PASS' if ok else 'FAIL'}  ratio={r1['ratio']} windowed={r1['tiers']['window']['n']} recoverable={r1['recoverable_frac']}")
    fails += not ok
    t1, B1 = autotune(f1, 2.0)
    ok = t1["ratio"] and t1["ratio"] >= 2.0 and B1 >= MIN_BUDGET
    print(f"  autotune->2x       {'PASS' if ok else 'FAIL'}  budget={B1} achieved={t1['ratio']}")
    fails += not ok

    # 1b: recoverable_frac is MEASURED, not a constant. A windowed NON-file result (no path)
    # leaves an unrecoverable-by-this-tool middle -> frac < 1.0; a file-backed one -> 1.0.
    rb = reduce_window(_seq([_mk("result", "Bash", 8000, salt="b")]), 1000)
    rf = reduce_window(_seq([_mk("result", "Read", 8000, salt="r", path="/x")]), 1000)
    ok = rb["recoverable_frac"] < 1.0 and rf["recoverable_frac"] == 1.0
    print(f"  recoverable-meas   {'PASS' if ok else 'FAIL'}  bash(no path)={rb['recoverable_frac']} < read(file)={rf['recoverable_frac']}")
    fails += not ok

    # 2: ANTI-INFLATION / NO-FABRICATION. Small unique results, no stale → cannot reduce
    # without windowing below MIN_BUDGET (amputation), so autotune must NOT reach 2x.
    f2 = _seq([_mk("result", "Bash", 400, salt=f"u{i}") for i in range(20)])
    r2 = reduce_window(f2, 1000)
    ok = r2["ratio"] == 1.0 and r2["removed_tok"] == 0
    print(f"  anti-inflation     {'PASS' if ok else 'FAIL'}  ratio={r2['ratio']} removed={r2['removed_tok']} (must be 1.0/0)")
    fails += not ok
    t2, B2 = autotune(f2, 2.0)
    ok = t2["ratio"] is not None and t2["ratio"] < 1.5
    print(f"  no-fabrication     {'PASS' if ok else 'FAIL'}  best={t2['ratio']} @budget>={MIN_BUDGET} (cannot fabricate 2x)")
    fails += not ok

    # 3: STALE READS (low-risk tier). Eight 4000-tok Reads of /f{i}, each later Edited.
    f3 = _seq([_mk("result", "Read", 16000, salt=f"s{i}", path=f"/f{i}", stale=True) for i in range(8)] +
              [_mk("result", "Edit", 200, salt=f"e{i}", path=f"/f{i}") for i in range(8)])
    r3 = reduce_window(f3, 100000)   # huge budget so windowing never fires; ONLY stale tier
    ok = (r3["tiers"]["stale"]["n"] == 8 and r3["tiers"]["window"]["n"] == 0
          and r3["ratio"] and r3["ratio"] > 3.0 and r3["low_risk_ratio"] == r3["ratio"])
    print(f"  stale-low-risk     {'PASS' if ok else 'FAIL'}  stale={r3['tiers']['stale']['n']} window={r3['tiers']['window']['n']} ratio={r3['ratio']}")
    fails += not ok

    # 4: NOISE (lossless tier). A result with ANSI + a 5x repeated line; filter must shrink
    # it but KEEP the signal line.
    ansi_noise = "\x1b[31mERROR\x1b[0m head\n" + ("dup line\n" * 5) + "unique tail content here"
    f4 = _seq([_mk("text", "", 10), Item("result", "Bash", ansi_noise, 1)])
    r4 = reduce_window(f4, 100000, stale=False, dedup=False)
    cleaned = filter_noise(ansi_noise)
    ansi_gone = "\x1b[" not in cleaned
    signal_kept = "unique tail content here" in cleaned and "ERROR" in cleaned
    run_marked = "(5× identical line above)" in cleaned
    ok = r4["tiers"]["noise"]["removed_tok"] >= 0 and ansi_gone and signal_kept and run_marked
    print(f"  noise-lossless     {'PASS' if ok else 'FAIL'}  removed_noise={r4['tiers']['noise']['removed_tok']} ansi_stripped={ansi_gone} signal_kept={signal_kept} run_marked={run_marked}")
    fails += not ok

    # 5: ALL-DUP (lossless). Identical results → dedup tier collapses them.
    f5 = _seq([_mk("result", "Grep", 4000, salt="same") for _ in range(8)])
    r5 = reduce_window(f5, 100000)
    ok = r5["tiers"]["dedup"]["n"] == 7 and r5["tiers"]["window"]["removed_tok"] == 0 and r5["ratio"] and r5["ratio"] > 3.0
    print(f"  all-dup-lossless   {'PASS' if ok else 'FAIL'}  dedup={r5['tiers']['dedup']['n']} ratio={r5['ratio']}")
    fails += not ok

    # 6: TOOL_USE INPUTS are reducible too. Five 4000-tok file-backed Write payloads → the
    # window tier collapses the agent's bulk authored content (recoverable from the file).
    f6 = _seq([_mk("tool_use", "Write", 16000, salt=f"w{i}", path=f"/w{i}") for i in range(5)])
    r6 = reduce_window(f6, 1000)
    ok = r6["tiers"]["window"]["n"] == 5 and r6["ratio"] and r6["ratio"] >= 1.8 and r6["recoverable_frac"] == 1.0
    print(f"  tool_use-window    {'PASS' if ok else 'FAIL'}  windowed={r6['tiers']['window']['n']} ratio={r6['ratio']} recoverable={r6['recoverable_frac']}")
    fails += not ok

    # 7: DEFAULT PROFILE == legacy path, byte-for-byte. The `balanced` preset must reproduce
    # the pre-profile reducer exactly, so the honesty invariants carry over unchanged.
    r_leg = reduce_window(f1, 1000)
    r_bal = reduce_window(f1, 1000, profile=make_profile("balanced"))
    ok = (r_leg["ratio"] == r_bal["ratio"] and r_leg["removed_tok"] == r_bal["removed_tok"]
          and r_leg["recoverable_frac"] == r_bal["recoverable_frac"])
    print(f"  default==legacy    {'PASS' if ok else 'FAIL'}  legacy={r_leg['ratio']} balanced={r_bal['ratio']}")
    fails += not ok

    # 8: PROFILE off — the "disabled" choice. Even a windowable corpus reduces 1.0x / removes 0.
    r_off = reduce_window(f1, 1000, profile=make_profile("off"))
    ok = r_off["ratio"] == 1.0 and r_off["removed_tok"] == 0
    print(f"  profile-off        {'PASS' if ok else 'FAIL'}  ratio={r_off['ratio']} removed={r_off['removed_tok']}")
    fails += not ok

    # 9: PROFILE conservative — lossless + recoverable only, NO windowing. Stale reads still
    # collapse (recoverable), but a big non-file result is kept verbatim (window tier silent).
    fc = _seq([_mk("result", "Read", 16000, salt=f"s{i}", path=f"/f{i}", stale=True) for i in range(3)] +
              [_mk("result", "Edit", 200, salt=f"e{i}", path=f"/f{i}") for i in range(3)] +
              [_mk("result", "Bash", 16000, salt="bigbash")])
    rc = reduce_window(fc, 1000, profile=make_profile("conservative"))
    ok = (rc["tiers"]["window"]["n"] == 0 and rc["tiers"]["stale"]["n"] == 3
          and rc["recoverable_frac"] == 1.0 and rc["ratio"] and rc["ratio"] > 1.0
          and rc["lossy_removed_tok"] == 0)
    print(f"  profile-conserv    {'PASS' if ok else 'FAIL'}  window={rc['tiers']['window']['n']} stale={rc['tiers']['stale']['n']} recoverable={rc['recoverable_frac']}")
    fails += not ok

    # 10: ERROR COLLAPSE materialization — the '30k error -> "error"' case, proven on bytes.
    #     'head' keeps the first line (the message); 'oneline' drops to a tiny marker.
    err = _mk("result", "Bash", 30000, salt="Error: connection refused\n", is_error=True)
    head, recov_h = collapse_error(err, "head")
    one, recov_o = collapse_error(err, "oneline")
    ok = (head.startswith("Error: connection refused") and "error body elided" in head
          and len(head) < len(err.content) and recov_h is False
          and "error" in one.lower() and "elided" in one and len(one) < 64 and recov_o is False)
    print(f"  error-materialize  {'PASS' if ok else 'FAIL'}  head_msg_kept={head.startswith('Error')} oneline_len={len(one)} recoverable={recov_o}")
    fails += not ok

    # 11: HYPER on a 30k error result — big reduction, but HONESTLY lossy (recoverable_frac low,
    #     lossy_removed_tok > 0). The error tier fires once; nothing is counted re-fetchable.
    f11 = _seq([_mk("result", "Bash", 30000, salt="Error: boom\n", is_error=True)])
    r11 = reduce_window(f11, 200, profile=make_profile("hyper"))
    ok = (r11["tiers"]["error"]["n"] == 1 and r11["lossy_removed_tok"] > 0
          and r11["ratio"] and r11["ratio"] > 10.0 and r11["recoverable_frac"] < 0.5)
    print(f"  hyper-error-lossy  {'PASS' if ok else 'FAIL'}  error_n={r11['tiers']['error']['n']} ratio={r11['ratio']} recoverable={r11['recoverable_frac']} lossy={r11['lossy_removed_tok']}")
    fails += not ok

    # 12: ERROR collapse NEVER fires on a non-error result, and NEVER inflates a small error.
    f12a = _seq([_mk("result", "Bash", 8000, salt="normal output, no error here", path="/n")])
    r12a = reduce_window(f12a, 100000, profile=make_profile("hyper"))   # huge budget: only error tier could fire
    f12b = _seq([_mk("result", "Bash", 30, salt="Error: x\n", is_error=True)])   # tiny error < marker
    r12b = reduce_window(f12b, 1000, profile=make_profile("hyper"))
    ok = (r12a["tiers"]["error"]["n"] == 0 and r12b["reduced_tok"] <= r12b["baseline_tok"])
    print(f"  error-no-falsepos  {'PASS' if ok else 'FAIL'}  nonerror_collapsed={r12a['tiers']['error']['n']} small_err_inflated={r12b['reduced_tok']>r12b['baseline_tok']}")
    fails += not ok

    # 13: PER-TOOL — an exempt tool is untouched; a tight per-tool cap windows that tool harder.
    fe = _seq([_mk("result", "Bash", 8000, salt="bigexempt")])
    r_ex = reduce_window(fe, 1000, profile=Profile("ex", per_tool={"Bash": None}))
    fr = _seq([_mk("result", "Read", 8000, salt="rr", path="/x")])
    r_def = reduce_window(fr, 1000, profile=Profile("d"))               # default 1000 budget
    r_tight = reduce_window(fr, 1000, profile=Profile("t", per_tool={"Read": 100}))   # Read capped @100
    ok = (r_ex["ratio"] == 1.0 and r_ex["removed_tok"] == 0
          and r_tight["removed_tok"] > r_def["removed_tok"])
    print(f"  per-tool           {'PASS' if ok else 'FAIL'}  exempt_removed={r_ex['removed_tok']} tight>{r_def['removed_tok']}={r_tight['removed_tok']}")
    fails += not ok

    # 14: MONOTONIC DIAL — each profile must reduce at least as hard as the one before it, on a
    #     reducible corpus (windowable reads + errors + stale + dup). off <= conservative <=
    #     balanced <= aggressive <= hyper. This guards against a future preset edit (e.g. a fixed
    #     budget looser than balanced's tuned one) silently inverting the "how aggressive" dial.
    fm = _seq(
        [_mk("result", "Read", 8000, salt=f"rd{i}", path=f"/rd{i}") for i in range(8)] +
        [_mk("result", "Bash", 8000, salt=f"e{i} Error: boom", is_error=True) for i in range(3)] +
        [_mk("result", "Read", 8000, salt="st", path="/st", stale=True),
         _mk("result", "Edit", 100, salt="ed", path="/st")] +
        [_mk("result", "Grep", 8000, salt="dup"), _mk("result", "Grep", 8000, salt="dup")] +
        [_mk("text", "", 60)])
    def _prof_ratio(name):
        p = make_profile(name)
        if not p.window:
            return reduce_window(fm, 0, profile=p)["ratio"] or 1.0
        if p.budget is None:
            return autotune(fm, p.target, profile=p)[0]["ratio"] or 1.0
        return reduce_window(fm, p.budget, profile=p)["ratio"] or 1.0
    ratios = [_prof_ratio(n) for n in ("off", "conservative", "balanced", "aggressive", "hyper")]
    ok = all(ratios[i] <= ratios[i + 1] + 1e-9 for i in range(4))
    print(f"  monotonic-dial     {'PASS' if ok else 'FAIL'}  off..hyper ratios = {[round(x,3) for x in ratios]}")
    fails += not ok

    # invariants
    for nm, r in (("f1", r1), ("f2", r2), ("f3", r3), ("f5", r5), ("f6", r6),
                  ("rc", rc), ("r11", r11), ("r12a", r12a), ("r12b", r12b)):
        if not (r["reduced_tok"] <= r["baseline_tok"] and r["ratio"] and r["ratio"] >= 1.0):
            print(f"  INVARIANT FAIL on {nm}: {r}")
            fails += 1

    print()
    if fails:
        print(f"SELFCHECK FAILED — {fails} check(s) failed")
        return 1
    print("OK — risk-tiered reducer hits the documented ratios, never fabricates a reduction, "
          "never guts a result below the floor; profiles span off-to-hyper, error-collapse is opt-in "
          "and HONESTLY lossy (recoverable_frac drops), per-tool exempt/cap behaves, default==legacy")
    return 0

# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="context-window baseline + self-reducer for Claude Code sessions")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("baseline", help="decompose the context window + show reduction headroom by risk tier")
    b.add_argument("--since-days", type=float, default=None)
    b.add_argument("--root", action="append", default=None, help="transcript root(s); default = all ~/.claude*/projects")
    b.add_argument("--max", type=int, default=20, help="cap at the N heaviest sessions (default 20)")
    b.add_argument("--all", action="store_true", help="include all namespaces (default: C--work* only)")
    b.add_argument("--json", default=None)
    b.add_argument("--md", default=None)
    b.add_argument("--now", default=None, help="timestamp string to stamp into the report")

    r = sub.add_parser("reduce", help="apply the risk-tiered reducer and report the achieved ratio + loss profile")
    r.add_argument("transcript", nargs="?", default=None, help="a single .jsonl, or omit to use discovered sessions")
    r.add_argument("--profile", default=None, choices=list(PROFILES),
                   help="aggressiveness preset: off | conservative | balanced (default) | aggressive | hyper")
    r.add_argument("--target", type=float, default=None, help="reduction target for auto-tune (default: the profile's)")
    r.add_argument("--budget", type=float, default=None, help="fixed per-result token budget (skips auto-tune)")
    r.add_argument("--error-collapse", dest="error_collapse", default=None, choices=ERROR_COLLAPSE_MODES,
                   help="collapse error-shaped results: off | head (keep 1st line) | oneline (tiny marker)")
    r.add_argument("--per-tool", dest="per_tool", action="append", default=None, metavar="TOOL=BUDGET",
                   help="per-tool window budget, or TOOL=off to EXEMPT it (repeatable; e.g. Bash=200 TodoWrite=off)")
    r.add_argument("--no-stale", action="store_true", help="disable the stale-read tier")
    r.add_argument("--no-noise", action="store_true", help="disable the character-noise tier")
    r.add_argument("--since-days", type=float, default=None)
    r.add_argument("--root", action="append", default=None)
    r.add_argument("--max", type=int, default=20)
    r.add_argument("--all", action="store_true")
    r.add_argument("--json", default=None)

    sub.add_parser("selfcheck", help="synthetic fixtures: assert ratio + no overclaim (CI gate)")

    # Windows consoles default to cp1252, which can't encode the box-drawing glyphs we
    # print. Force UTF-8 so the tool renders identically on every platform.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = ap.parse_args(argv)
    if args.cmd == "baseline":
        return cmd_baseline(args)
    if args.cmd == "reduce":
        return cmd_reduce(args)
    if args.cmd == "selfcheck":
        return runselfcheck()
    ap.print_help()
    return 2

if __name__ == "__main__":
    sys.exit(main())
