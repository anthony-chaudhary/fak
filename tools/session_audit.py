#!/usr/bin/env python3
"""
session_audit.py — reusable auditor for Claude Code session-transcript JSONL files.

The transcripts live under <claude-home>/projects/<namespace>/<session-uuid>.jsonl and
carry EXACT token accounting (the `.dos/` tool-stream telemetry only has content digests,
no token counts — so this is the tool that can answer the token-weighted questions:
real input:output ratio, real prompt-cache / KV reuse, real cost, real tool mix).

Schema (per JSONL line, one record):
  type=assistant : message.usage = {input_tokens, output_tokens,
                     cache_read_input_tokens, cache_creation_input_tokens,
                     server_tool_use{web_search_requests,web_fetch_requests}, iterations[...]}
                   message.content = [ {type: thinking|text|tool_use}, ... ]
  type=user      : message.content = str (typed prompt / slash-command wrapper / hook text)
                                   or [ {type: tool_result, content: ...}, ... ]
  plus meta records: last-prompt, mode, permission-mode, attachment, file-history-snapshot,
                     system, queue-operation, ai-title, summary

Usage:
  python session_audit.py discover [--since-days N] [--root DIR ...]
  python session_audit.py audit  [--since-days N] [--root DIR ...] [--json OUT] [--md OUT]
  python session_audit.py deep   <session.jsonl>          # follow ONE trajectory top-to-bottom

All numbers from token usage are EXACT. Cost uses the PRICING table below (an ASSUMPTION,
clearly flagged) — edit it to match current rates; token counts are the ground truth.

Besides the token/cost lens, every subcommand also runs the BEHAVIORAL lens (#2365):
per-tool call/error counts and error rate, shell timeout kills (exit 143 / "timed
out"), foreground sleep-polls, Edit/Write read-discipline churn, repeated identical
failure signatures, and per-file mutation churn — the stuck/churn half a token-only
audit can't see.

Cost is split by BILLING BUCKET (provider): claude-* is the Anthropic invoice; a gemini-*
/ gpt-* / local model is a different invoice. The auditor never sums cost across buckets
and never prices a non-Claude model at Claude rates — an unpriced model is reported with
exact tokens but no fabricated cost. <synthetic> (harness-injected) is non-billed ($0).
"""
import sys
import os
import re
import json
import glob
import argparse
import statistics
import collections
import datetime

# --- pricing assumption (USD per 1e6 tokens). EDIT to match real card; flagged in output. ---
# Rates are per-model-family and ANTHROPIC-ONLY. A model that matches none of these is
# NOT silently priced as Opus — it is an UNPRICED, SEPARATE billing bucket. Claude and
# Gemini are different vendors = different invoices; the audit never sums cost across
# buckets and never invents a Claude price for a non-Claude model. Add a vendor's card to
# PRICING (and its substrings to PROVIDER_BUCKETS) to price it; until then its tokens are
# reported and its cost is left blank.
PRICING = {
    # model_substring: (input, cache_write_5m, cache_read, output)
    "opus":   (15.0, 18.75, 1.50, 75.0),
    "sonnet": ( 3.0,  3.75, 0.30, 15.0),
    "haiku":  ( 0.80, 1.00, 0.08,  4.0),
    "fable":  ( 3.0,  3.75, 0.30, 15.0),
}
PRICING_IS_ASSUMPTION = True

# Harness-injected pseudo-models that never reach any vendor — never billed, never a bucket.
NONBILLED_MODELS = {"<synthetic>", "?", ""}

# Provider / billing-bucket classification. Each provider is a DISTINCT invoice; the report
# breaks cost out per bucket and refuses to sum across them. Substring-matched, first hit wins;
# Anthropic is first so a claude-* tier never falls through to another vendor's bucket.
PROVIDER_BUCKETS = [
    ("Anthropic (Claude)",  ("claude", "opus", "sonnet", "haiku", "fable")),
    ("Google (Gemini)",     ("gemini", "gemma")),
    ("OpenAI",              ("gpt", "o1-", "o3-", "o4-", "davinci")),
    ("local / self-hosted", ("qwen", "llama", "mistral", "mixtral", "phi-", "deepseek")),
]

def provider_bucket(model):
    """Which vendor invoice a model lands on — its billing bucket, not its rate."""
    if (model or "") in NONBILLED_MODELS:
        return "non-billed (harness)"
    m = (model or "").lower()
    if not m:
        return "non-billed (harness)"
    for name, subs in PROVIDER_BUCKETS:
        if any(s in m for s in subs):
            return name
    return "UNKNOWN (unpriced bucket)"

DEFAULT_ROOTS = [
    os.path.join(os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")), "projects"),
]
# transient test-fixture / temp-workspace namespaces — never "our own sessions"
EXCLUDE_NS_SUBSTR = ["pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"]
NS_INCLUDE_PREFIX = ""   # all non-excluded namespaces by default; narrow with --ns-prefix PREFIX

READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch", "WebSearch",
                   "TodoRead", "ToolSearch",
                   # observation-only harness tools: poll/query state, never mutate it.
                   "Monitor", "TaskGet", "TaskList", "TaskOutput",
                   "ReadMcpResourceTool", "ListMcpResourcesTool", "ReadMcpResourceDirTool"}
# Bash/Edit/Write/NotebookEdit/TaskCreate/TaskUpdate/TaskStop/Workflow/etc. are
# side-effecting or spawn.

# --- behavioral lens (#2365): stuck / churn detectors -------------------------
# All detectors read ONLY what the transcript already carries (tool_use inputs +
# errored tool_results); none of them re-run anything or read the process table.
SHELL_TOOLS = {"Bash", "PowerShell"}
# A shell result that was killed by the harness deadline: SIGTERM exit 143 or the
# harness "timed out" phrasing. \W{0,3} absorbs ':'/': ' variants between
# "code"/"status" and the number.
TIMEOUT_KILL_RE = re.compile(r"exit (?:code|status)\W{0,3}143\b|timed out",
                             re.IGNORECASE)
# A foreground shell call whose command *starts* with a sleep is a poll — the
# turn is blocked doing nothing (background sleeps are fine, they don't block).
SLEEP_POLL_RE = re.compile(r"^\s*(?:sleep|start-sleep)\b", re.IGNORECASE)
# Edit/Write churn: mutation calls wasted on read-before-write discipline.
EDIT_CHURN_SIGNATURES = {
    "not_read":   "File has not been read yet",
    "stale_read": "has been modified since read",
}
MUTATION_TOOLS = {"Edit", "Write", "NotebookEdit"}
# The same failure repeated this often in one session is a stuck loop. Verbatim
# repeats key on (tool, args, error) — a true retry loop; the error-mass view
# keys on (tool, error text) alone — a recurring failure CLASS whose args may
# vary (e.g. rotating through wedged bridge sessions). Conflating the two
# false-alarmed on five distinct commands sharing one timeout string.
REPEAT_FAILURE_MIN = 3
# A file mutated this often is only a rewrite/flip-flop loop when the edits
# revisit the same regions or undo each other; distinct-region build-out
# (verb-per-edit-triple, helper extraction) is healthy iteration.
FILE_CHURN_MIN = 5
# A gap this long with ZERO transcript records is a harness/API stall — the
# kind of dead time the sleep-poll counter can't see.
STALL_GAP_S = 300

def _txt_str(content, cap=4000):
    """Flatten a content field (str or list of blocks) to text, capped at `cap`."""
    if isinstance(content, str):
        return content[:cap]
    if isinstance(content, list):
        parts, n = [], 0
        for b in content:
            if n >= cap:
                break
            if isinstance(b, dict):
                s = _txt_str(b.get("content", b.get("text", "")), cap - n)
            elif isinstance(b, str):
                s = b[:cap - n]
            else:
                continue
            parts.append(s)
            n += len(s)
        return "".join(parts)
    return ""

def _norm_head(s, cap=200):
    """whitespace-collapsed head of a string, for region/signature identity."""
    return " ".join((s or "").split())[:cap]

class BehaviorLens:
    """Per-transcript stuck/churn detectors (#2365). Fed one tool_use /
    tool_result at a time (post de-dup); `summary()` emits one plain dict:
    per-tool error counts, timeout kills, foreground sleep-polls, Edit/Write
    read-discipline churn, verbatim repeat failures (same tool+args+error, a
    true retry loop) vs failure-class mass (same tool+error, args vary), and
    per-file mutation churn discriminated into rewrite-loop vs distinct-region
    build-out via edit-region identity + revert pairs."""

    def __init__(self):
        self.errors = collections.Counter()        # tool -> errored results
        self.timeout_kills = 0
        self.sleep_polls = 0
        self.edit_churn = collections.Counter()    # not_read / stale_read
        self.verbatim_sigs = collections.Counter() # (tool, args_key, sig) -> n
        self.mass_sigs = collections.Counter()     # (tool, sig) -> n
        self.file_writes = collections.Counter()   # file_path -> mutation calls
        self.file_regions = collections.defaultdict(list)  # path -> [(old_h, new_h)]

    def see_tool_use(self, name, tool_input):
        ti = tool_input if isinstance(tool_input, dict) else {}
        if name in SHELL_TOOLS and not ti.get("run_in_background") \
                and SLEEP_POLL_RE.match(ti.get("command") or ""):
            self.sleep_polls += 1
        if name in MUTATION_TOOLS:
            path = ti.get("file_path") or ti.get("notebook_path")
            if path:
                self.file_writes[path] += 1
                # Region identity: Edit carries old/new strings; a Write is a
                # whole-file rewrite (one region), so N Writes of one file still
                # read as revisiting the same region.
                old_h = hash(_norm_head(ti.get("old_string", "")))
                new_h = hash(_norm_head(ti.get("new_string",
                                               ti.get("content", ""))))
                self.file_regions[path].append((old_h, new_h))

    def see_tool_result(self, tool, is_error, text, args_key=None):
        if not is_error:
            return
        self.errors[tool] += 1
        if tool in SHELL_TOOLS and TIMEOUT_KILL_RE.search(text):
            self.timeout_kills += 1
        for key, sig in EDIT_CHURN_SIGNATURES.items():
            if sig in text:
                self.edit_churn[key] += 1
        sig = _norm_head(text, 160)
        self.verbatim_sigs[(tool, args_key, sig)] += 1
        self.mass_sigs[(tool, sig)] += 1

    def _churn_rows(self):
        rows = []
        for f, n in self.file_writes.items():
            if n < FILE_CHURN_MIN:
                continue
            regions = self.file_regions.get(f, [])
            distinct = len({o for o, _ in regions}) or 1
            seen_old = set()
            reverts = 0
            for old_h, new_h in regions:
                if new_h in seen_old and new_h != old_h:
                    reverts += 1          # this edit restores an earlier state
                seen_old.add(old_h)
            # Rewrite loop = edits undo each other or keep revisiting the same
            # few regions; distinct-region build-out is filtered out here.
            if reverts >= 1 or distinct * 2 <= n:
                rows.append({"file": f, "count": n,
                             "distinct_regions": distinct, "reverts": reverts})
        return sorted(rows, key=lambda r: -r["count"])

    def summary(self):
        repeats = sorted(({"tool": t, "sig": sig, "count": n}
                          for (t, _a, sig), n in self.verbatim_sigs.items()
                          if n >= REPEAT_FAILURE_MIN),
                         key=lambda r: -r["count"])
        mass = sorted(({"tool": t, "sig": sig, "count": n}
                       for (t, sig), n in self.mass_sigs.items()
                       if n >= REPEAT_FAILURE_MIN),
                      key=lambda r: -r["count"])
        return {
            "tool_errors": dict(self.errors),
            "timeout_kills": self.timeout_kills,
            "sleep_polls": self.sleep_polls,
            "edit_churn": dict(self.edit_churn),
            "repeat_failures": repeats[:10],
            "max_repeat_failure": max(self.verbatim_sigs.values(), default=0),
            "failure_mass": mass[:10],
            "max_failure_mass": max(self.mass_sigs.values(), default=0),
            "file_churn": self._churn_rows()[:10],
            "max_file_churn": max(self.file_writes.values(), default=0),
        }

def price_for(model):
    """Rate card for a model, or None when we hold no card for its billing bucket.
    None means: report the tokens, never invent a cost (and never default to Opus)."""
    if (model or "") in NONBILLED_MODELS:
        return None
    m = (model or "").lower()
    if not m:
        return None
    for key, rates in PRICING.items():
        if key in m:
            return rates
    return None

def cost_usd(model, inp, cwrite, cread, out):
    rates = price_for(model)
    if rates is None:
        return 0.0          # unpriced / non-Anthropic / non-billed — kept out of the total
    pi, pcw, pcr, po = rates
    return (inp*pi + cwrite*pcw + cread*pcr + out*po) / 1e6

def model_cost(model, c):
    """Cost for one model's rolled-up token Counter (input/cache_create/cache_read/output)."""
    return cost_usd(model, c.get("input", 0), c.get("cache_create", 0),
                    c.get("cache_read", 0), c.get("output", 0))

def model_tier(model):
    """Stable model tier for mix KPIs (opus/sonnet/haiku/etc.), not a full model id."""
    if (model or "") in NONBILLED_MODELS:
        return "<synthetic>"
    m = (model or "").lower()
    for key in PRICING:
        if key in m:
            return key
    return "unpriced"

def discover(roots, since_days=None, ns_prefix=NS_INCLUDE_PREFIX, include_subagents=False):
    cutoff = None
    if since_days is not None:
        cutoff = datetime.datetime.now().timestamp() - since_days*86400
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
            # top-level session transcripts (one per conversation)
            top = set(glob.glob(os.path.join(nsdir, "*.jsonl")))
            paths = [(p, "session") for p in top]
            if include_subagents:
                # subagent / workflow transcripts live in <session-id>/**/*.jsonl —
                # SEPARATE files, so top-level session usage UNDERCOUNTS true spend.
                for p in glob.glob(os.path.join(nsdir, "**", "*.jsonl"), recursive=True):
                    if p not in top:
                        paths.append((p, "subagent"))
            for path, kind in paths:
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if cutoff and st.st_mtime < cutoff:
                    continue
                out.append({"root": root, "ns": ns, "path": path, "kind": kind,
                            "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out

def _txt_len(content):
    """char length of a content field that may be str or list of blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for b in content:
            if isinstance(b, dict):
                c = b.get("content", b.get("text", ""))
                n += _txt_len(c)
            elif isinstance(b, str):
                n += len(b)
        return n
    return 0

def _looks_like_typed_prompt(s):
    """user string that is an actual prompt (typed or slash-command), not pure hook/reminder."""
    if not isinstance(s, str):
        return False
    st = s.strip()
    if not st:
        return False
    if st.startswith("<system-reminder>") or st.startswith("Caveat:"):
        return False
    return True

def analyze(path):
    rec_types = collections.Counter()
    models = collections.Counter()
    tools = collections.Counter()
    tool_input_chars = 0
    tool_result_chars = 0
    n_tool_use = 0
    n_tool_result = 0
    n_thinking = 0
    n_text = 0
    tok = dict(input=0, output=0, cache_read=0, cache_create=0,
               web_search=0, web_fetch=0, iterations=0)
    per_model = collections.defaultdict(collections.Counter)   # model -> token Counter (per billing bucket)
    cost = 0.0
    ts_min = ts_max = None
    prompts = []          # (ts, text) — the trajectory's user asks
    assistant_turns = 0
    interrupted = 0
    dup_assistant_lines = 0
    # Claude Code writes MULTIPLE transcript lines per billed assistant turn
    # (streaming events / retries / sidechain re-serialization). Each carries the
    # SAME message.usage, so folding every line double-counts tokens/cost/turns
    # (measured ~2x, session-dependent). The model's own response id (message.id)
    # is the identity of a billed turn — verified to collapse exactly and to never
    # disagree on usage among its duplicate lines. De-dup on it; only id-less
    # lines (defensive — none seen in practice) are counted individually.
    seen_msg_ids = set()
    # Content blocks are deduped by BLOCK identity, not line identity: newer
    # transcripts stream ONE block per line under a shared message.id (skipping
    # dup lines wholesale undercounted tool calls ~6x there), while older ones
    # repeat the FULL content array on every duplicate line.
    seen_blocks = set()
    lens = BehaviorLens()
    tooluse_names = {}   # tool_use id -> tool name, to attribute tool_results
    tooluse_args = {}    # tool_use id -> args digest, for verbatim-retry keying
    prev_dt = None
    stall_gaps = 0
    max_gap_s = 0.0

    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        return {"path": path, "error": str(e)}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type")
        rec_types[t] += 1
        ts = r.get("timestamp")
        if ts:
            ts_min = ts if ts_min is None or ts < ts_min else ts_min
            ts_max = ts if ts_max is None or ts > ts_max else ts_max
            try:
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt is not None:
                if prev_dt is not None:
                    gap = (dt - prev_dt).total_seconds()
                    if gap > max_gap_s:
                        max_gap_s = gap
                    if gap >= STALL_GAP_S:
                        stall_gaps += 1   # zero-record dead time (harness stall)
                prev_dt = dt

        if t == "assistant":
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            new_turn = True
            if mid is not None:
                if mid in seen_msg_ids:
                    dup_assistant_lines += 1
                    new_turn = False   # usage already folded; blocks may be new
                else:
                    seen_msg_ids.add(mid)
            if new_turn:
                assistant_turns += 1
                models[msg.get("model", "?")] += 1
                u = msg.get("usage", {}) or {}
                inp = u.get("input_tokens", 0) or 0
                out = u.get("output_tokens", 0) or 0
                cr  = u.get("cache_read_input_tokens", 0) or 0
                cc  = u.get("cache_creation_input_tokens", 0) or 0
                stu = u.get("server_tool_use", {}) or {}
                tok["input"] += inp
                tok["output"] += out
                tok["cache_read"] += cr
                tok["cache_create"] += cc
                tok["web_search"] += stu.get("web_search_requests", 0) or 0
                tok["web_fetch"]  += stu.get("web_fetch_requests", 0) or 0
                tok["iterations"] += len(u.get("iterations", []) or [])
                cost += cost_usd(msg.get("model"), inp, cc, cr, out)
                pm = per_model[msg.get("model", "?")]
                pm["turns"] += 1
                pm["input"] += inp
                pm["output"] += out
                pm["cache_read"] += cr
                pm["cache_create"] += cc
            dedup_blocks = mid is not None
            for b in (msg.get("content") or []):
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "tool_use":
                    key = b.get("id") or (mid, "tool_use", b.get("name"),
                                          json.dumps(b.get("input", {}),
                                                     sort_keys=True, default=str))
                    if dedup_blocks:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    n_tool_use += 1
                    name = b.get("name", "?")
                    tools[name] += 1
                    tool_input_chars += _txt_len(b.get("input", {}))
                    if b.get("id"):
                        tooluse_names[b["id"]] = name
                        tooluse_args[b["id"]] = hash(json.dumps(
                            b.get("input", {}), sort_keys=True, default=str))
                    lens.see_tool_use(name, b.get("input"))
                elif bt in ("thinking", "text"):
                    body = b.get("thinking") if bt == "thinking" else b.get("text")
                    key = (mid, bt, hash(body if isinstance(body, str) else str(body)))
                    if dedup_blocks:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    if bt == "thinking":
                        n_thinking += 1
                    else:
                        n_text += 1
            if r.get("interruptedMessageId") or msg.get("stop_reason") == "interrupted":
                key = (mid, "interrupted")
                if not dedup_blocks or key not in seen_blocks:
                    seen_blocks.add(key)
                    interrupted += 1
        elif t == "user":
            msg = r.get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        n_tool_result += 1
                        tool_result_chars += _txt_len(b.get("content", ""))
                        lens.see_tool_result(
                            tooluse_names.get(b.get("tool_use_id"), "?"),
                            bool(b.get("is_error")),
                            _txt_str(b.get("content", "")),
                            args_key=tooluse_args.get(b.get("tool_use_id")))
            elif _looks_like_typed_prompt(content) and not r.get("isMeta"):
                prompts.append((ts, content.strip()[:400]))

    total_in = tok["input"] + tok["cache_read"] + tok["cache_create"]
    io_ratio = total_in / tok["output"] if tok["output"] else None
    cache_hit = tok["cache_read"] / (tok["cache_read"] + tok["cache_create"] + tok["input"]) \
                if (tok["cache_read"] + tok["cache_create"] + tok["input"]) else None
    wall = None
    if ts_min and ts_max:
        try:
            a = datetime.datetime.fromisoformat(ts_min.replace("Z", "+00:00"))
            b = datetime.datetime.fromisoformat(ts_max.replace("Z", "+00:00"))
            wall = (b - a).total_seconds()
        except Exception:
            pass
    ro = sum(v for k, v in tools.items() if k in READ_ONLY_TOOLS)
    behavior = lens.summary()
    behavior["stall_gaps"] = stall_gaps
    behavior["max_gap_s"] = round(max_gap_s, 1)
    return {
        "path": path, "session": os.path.splitext(os.path.basename(path))[0],
        "n_records": sum(rec_types.values()), "rec_types": dict(rec_types),
        "models": dict(models), "per_model": {m: dict(c) for m, c in per_model.items()},
        "assistant_turns": assistant_turns,
        "dup_assistant_lines": dup_assistant_lines,
        "n_prompts": len(prompts), "prompts": prompts,
        "n_tool_use": n_tool_use, "n_tool_result": n_tool_result,
        "tools": dict(tools), "read_only_tool_calls": ro,
        "read_only_frac": (ro / n_tool_use) if n_tool_use else None,
        "tool_input_chars": tool_input_chars, "tool_result_chars": tool_result_chars,
        "n_thinking": n_thinking, "n_text": n_text, "interrupted": interrupted,
        "tokens": tok, "total_input_tokens": total_in,
        "io_ratio": io_ratio, "cache_hit_frac": cache_hit,
        "cost_usd": cost, "ts_min": ts_min, "ts_max": ts_max, "wall_s": wall,
        "behavior": behavior,
    }

def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs)-1, int(round((p/100)*(len(xs)-1)))))
    return xs[k]

def aggregate(sessions):
    S = [s for s in sessions if "error" not in s]
    tot = collections.Counter()
    for s in S:
        for k, v in s["tokens"].items():
            tot[k] += v
    tot_cost = sum(s["cost_usd"] for s in S)
    tot_tools = collections.Counter()
    for s in S:
        tot_tools.update(s["tools"])
    ns_roll = collections.defaultdict(lambda: collections.Counter())
    ns_cost = collections.Counter()
    ns_models = collections.defaultdict(collections.Counter)   # ns -> {model: output tok}
    for s in S:
        ns = os.path.basename(os.path.dirname(s["path"]))
        ns_roll[ns]["sessions"] += 1
        ns_roll[ns]["output"] += s["tokens"]["output"]
        ns_roll[ns]["cache_read"] += s["tokens"]["cache_read"]
        ns_roll[ns]["tool_use"] += s["n_tool_use"]
        ns_cost[ns] += s["cost_usd"]
        for model, c in s.get("per_model", {}).items():
            ns_models[ns][model] += c.get("output", 0)
    # per-model and per-billing-bucket rollups (token-exact; cost added at render)
    pm_roll = collections.defaultdict(collections.Counter)
    for s in S:
        for model, c in s.get("per_model", {}).items():
            pm_roll[model].update(c)
    bucket_roll = collections.defaultdict(collections.Counter)
    for model, c in pm_roll.items():
        bucket_roll[provider_bucket(model)].update(c)
    tier_roll = collections.defaultdict(collections.Counter)
    for model, c in pm_roll.items():
        tier_roll[model_tier(model)].update(c)
    ns_opus_share = {}
    for ns, models in ns_models.items():
        out = sum(models.values())
        opus = sum(v for m, v in models.items() if model_tier(m) == "opus")
        ns_opus_share[ns] = (opus / out) if out else None
    calls = [s["n_tool_use"] for s in S]
    outs  = [s["tokens"]["output"] for s in S]
    ios   = [s["io_ratio"] for s in S if s["io_ratio"]]
    chf   = [s["cache_hit_frac"] for s in S if s["cache_hit_frac"] is not None]
    rof   = [s["read_only_frac"] for s in S if s["read_only_frac"] is not None]
    # behavioral rollup (#2365) — tolerate sessions replayed from pre-lens JSON
    beh_errors = collections.Counter()
    beh_churn = collections.Counter()
    beh_timeouts = beh_sleeps = 0
    stall_sessions = 0
    max_gap_s = 0.0
    repeat_rows, filechurn_rows, mass_rows = [], [], []
    for s in S:
        b = s.get("behavior") or {}
        beh_errors.update(b.get("tool_errors", {}))
        beh_churn.update(b.get("edit_churn", {}))
        beh_timeouts += b.get("timeout_kills", 0)
        beh_sleeps += b.get("sleep_polls", 0)
        if b.get("stall_gaps", 0):
            stall_sessions += 1
        max_gap_s = max(max_gap_s, b.get("max_gap_s", 0) or 0)
        ns = os.path.basename(os.path.dirname(s["path"]))
        for r in (b.get("repeat_failures") or [])[:1]:
            repeat_rows.append({"session": s["session"], "ns": ns, **r})
        for r in (b.get("failure_mass") or [])[:1]:
            mass_rows.append({"session": s["session"], "ns": ns, **r})
        for r in (b.get("file_churn") or [])[:1]:
            filechurn_rows.append({"session": s["session"], "ns": ns, **r})
    per_tool_beh = {t: {"calls": tot_tools.get(t, 0),
                        "errors": beh_errors.get(t, 0),
                        "error_rate": (beh_errors.get(t, 0) / tot_tools[t])
                                      if tot_tools.get(t) else None}
                    for t in set(tot_tools) | set(beh_errors)}
    behavior = {
        "per_tool": per_tool_beh,
        "timeout_kills": beh_timeouts,
        "sleep_polls": beh_sleeps,
        "edit_churn": dict(beh_churn),
        "wasted_mutation_calls": sum(beh_churn.values()),
        "stall_sessions": stall_sessions,
        "max_gap_s": round(max_gap_s, 1),
        "repeat_failure_sessions": sorted(repeat_rows, key=lambda r: -r["count"])[:10],
        "failure_mass_sessions": sorted(mass_rows, key=lambda r: -r["count"])[:10],
        "file_churn_sessions": sorted(filechurn_rows, key=lambda r: -r["count"])[:10],
    }
    return {
        "n_sessions": len(S), "totals": dict(tot), "total_cost_usd": tot_cost,
        "behavior": behavior,
        "tool_mix": dict(tot_tools.most_common()),
        "per_namespace": {k: dict(v) for k, v in ns_roll.items()},
        "per_namespace_cost": dict(ns_cost),
        "per_namespace_top_model": {k: (v.most_common(1)[0][0] if v else "?")
                                    for k, v in ns_models.items()},
        "per_namespace_opus_share": ns_opus_share,
        "per_model": {m: dict(c) for m, c in pm_roll.items()},
        "per_bucket": {b: dict(c) for b, c in bucket_roll.items()},
        "per_tier": {t: dict(c) for t, c in tier_roll.items()},
        "dist": {
            "calls_per_session": {"median": statistics.median(calls) if calls else None,
                                  "mean": round(statistics.mean(calls),1) if calls else None,
                                  "p90": _pct(calls,90), "max": max(calls) if calls else None},
            "output_tokens_per_session": {"median": statistics.median(outs) if outs else None,
                                  "p90": _pct(outs,90), "max": max(outs) if outs else None},
            "io_ratio": {"median": round(statistics.median(ios),1) if ios else None,
                         "p90": round(_pct(ios,90),1) if ios else None},
            "cache_hit_frac": {"median": round(statistics.median(chf),3) if chf else None,
                               "p10": round(_pct(chf,10),3) if chf else None,
                               "p90": round(_pct(chf,90),3) if chf else None},
            "read_only_frac": {"median": round(statistics.median(rof),3) if rof else None},
        },
    }

def fmt_int(n):
    return f"{n:,}"

def fmt_pct(frac):
    return "—" if frac is None else f"{frac*100:.1f}%"

def _namespace_name(path):
    return os.path.basename(os.path.dirname(path))

def _scope_line(sessions, ns_prefix, since_days, include_subagents, max_sessions):
    namespaces = sorted({_namespace_name(s["path"]) for s in sessions if "error" not in s})
    if len(namespaces) > 8:
        ns_desc = ", ".join(namespaces[:8]) + f", ... (+{len(namespaces)-8} more)"
    else:
        ns_desc = ", ".join(namespaces) if namespaces else "none"
    ns_filter = ns_prefix or "all non-excluded namespaces"
    window = "all-time" if since_days is None else f"last {since_days:g} days"
    kinds = "top-level session transcripts"
    if include_subagents:
        kinds += " (subagents reported separately below)"
    cap = f"; max top-level sessions: {max_sessions}" if max_sessions else ""
    return (f"{len(namespaces)} namespaces folded ({ns_desc}); "
            f"namespace filter: {ns_filter}; time window: {window}; {kinds}{cap}")

def _subagent_note(summary):
    if not summary or not summary.get("count"):
        return None
    tokens = summary.get("tokens", {})
    return (f"NOTE: +{summary['count']} subagent transcripts uncounted; "
            f"re-run with `--include-subagents` "
            f"(about +${summary.get('cost_usd', 0.0):,.2f} / "
            f"+{fmt_int(tokens.get('output', 0))} output tok).")

def report_md(sessions, agg, ns_prefix=NS_INCLUDE_PREFIX, since_days=None,
              include_subagents=False, max_sessions=None, excluded_subagents=None):
    S = [s for s in sessions if "error" not in s]
    L = []
    L.append("# Session-Transcript Audit — active scope\n")
    L.append(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ")
    L.append(f"**Top-level sessions audited:** {agg['n_sessions']}  ·  **Tool:** `tools/session_audit.py` (re-runnable)  ")
    L.append(f"**Scope:** {_scope_line(S, ns_prefix, since_days, include_subagents, max_sessions)}")
    note = _subagent_note(excluded_subagents)
    if note:
        L.append(note)
    t = agg["totals"]
    L.append("## Scope totals (EXACT token counts)\n")
    L.append(f"- **Output tokens (the actual work generated):** {fmt_int(t['output'])}")
    L.append(f"- **Fresh input tokens (billed, non-cached):** {fmt_int(t['input'])}")
    L.append(f"- **Cache-read tokens (prompt-cache / KV reuse):** {fmt_int(t['cache_read'])}")
    L.append(f"- **Cache-creation tokens:** {fmt_int(t['cache_create'])}")
    tot_in = t['input']+t['cache_read']+t['cache_create']
    L.append(f"- **Total context ingested:** {fmt_int(tot_in)}  →  **machine-wide I:O ratio = {tot_in/max(t['output'],1):.1f} : 1**")
    chf = t['cache_read']/max(tot_in,1)
    L.append(f"- **Cache-read share of all ingested context = {chf*100:.1f}%**  (this is the prompt-cache/KV reuse the harness ALREADY captures)")
    # Two DIFFERENT mechanisms reach the web — report BOTH so the line can never
    # appear to contradict the tool-mix table below (which lists the CLIENT tools):
    #   - server_tool_use: the model's built-in web_search/web_fetch (billed server-side)
    #   - the client WebSearch/WebFetch tools (tool_use blocks — these are what show
    #     up in the tool mix). Counting only the former printed "0 / 0" even when a
    #     session used the client WebFetch tool, which read as "no web activity".
    ws_c = agg["tool_mix"].get("WebSearch", 0)
    wf_c = agg["tool_mix"].get("WebFetch", 0)
    L.append(f"- **Web requests — server-tool (`server_tool_use`, billed):** "
             f"search {fmt_int(t['web_search'])} / fetch {fmt_int(t['web_fetch'])}  "
             f"·  **client tool:** WebSearch {fmt_int(ws_c)} / WebFetch {fmt_int(wf_c)}")
    L.append(f"- **Multi-iteration count:** {fmt_int(t['iterations'])}")
    flag = "  _(⚠ cost uses an ASSUMED price table — edit PRICING; token counts above are exact)_" if PRICING_IS_ASSUMPTION else ""
    L.append(f"- **Estimated Anthropic-billed cost:** ${agg['total_cost_usd']:,.2f}{flag}")
    # Surface other billing buckets so the Anthropic total is never read as "the whole bill".
    buckets = agg.get("per_bucket", {})
    other = {b: c for b, c in buckets.items()
             if b not in ("Anthropic (Claude)", "non-billed (harness)") and c.get("output", 0)}
    if other:
        parts = [f"{b} ({fmt_int(c['output'])} output tok, unpriced — add its card)"
                 for b, c in sorted(other.items(), key=lambda kv: -kv[1].get("output", 0))]
        L.append("- **⚠ Other billing buckets present (NOT in the total above — different invoices):** "
                 + "; ".join(parts))
    nb = buckets.get("non-billed (harness)", {})
    if nb.get("turns"):
        L.append(f"- **Non-billed `<synthetic>` turns (harness-injected, $0):** {fmt_int(nb.get('turns',0))} "
                 f"({fmt_int(nb.get('output',0))} output tok)")
    L.append("")

    # Per-tier share — the headline mix KPI that makes "opus-heavy vs haiku-heavy" explicit.
    L.append("## Model-mix KPI (tier shares)\n")
    L.append("| Tier | Output tok | Output share | Est. cost | Cost share |")
    L.append("|---|---:|---:|---:|---:|")
    total_output = sum(c.get("output", 0) for c in agg.get("per_tier", {}).values())
    total_priced_cost = sum(model_cost(m, c) for m, c in agg.get("per_model", {}).items())
    for tier, c in sorted(agg.get("per_tier", {}).items(),
                          key=lambda kv: -kv[1].get("output", 0)):
        tier_cost = sum(model_cost(m, mc) for m, mc in agg.get("per_model", {}).items()
                        if model_tier(m) == tier)
        out_share = (c.get("output", 0) / total_output) if total_output else None
        cost_share = (tier_cost / total_priced_cost) if total_priced_cost else None
        L.append(f"| {tier} | {fmt_int(c.get('output',0))} | {fmt_pct(out_share)} | "
                 f"${tier_cost:,.2f} | {fmt_pct(cost_share)} |")
    L.append("")

    # Per billing bucket — the answer to "is this Claude or Gemini money?". NEVER summed.
    L.append("## Cost by billing bucket (provider) — never sum across these\n")
    L.append("| Billing bucket | Turns | Output tok | Cache-read tok | Est. cost | Priced? |")
    L.append("|---|---:|---:|---:|---:|:--:|")
    for b, c in sorted(buckets.items(), key=lambda kv: -kv[1].get("output", 0)):
        bcost = sum(model_cost(m, mc) for m, mc in agg.get("per_model", {}).items()
                    if provider_bucket(m) == b)
        priced = b == "Anthropic (Claude)"
        cost_cell = f"${bcost:,.2f}" if priced else ("$0.00" if b == "non-billed (harness)" else "— (no card)")
        L.append(f"| {b} | {fmt_int(c.get('turns',0))} | {fmt_int(c.get('output',0))} | "
                 f"{fmt_int(c.get('cache_read',0))} | {cost_cell} | {'✓' if priced else ''} |")
    L.append("")

    # Per-model tiers — so a blended cost can be read as opus-heavy vs haiku-heavy.
    L.append("## Per-model breakdown (token-exact; cost Anthropic-assumed)\n")
    L.append("| Model | Bucket | Turns | Output tok | Cache-read tok | Est. cost |")
    L.append("|---|---|---:|---:|---:|---:|")
    for m, c in sorted(agg.get("per_model", {}).items(), key=lambda kv: -kv[1].get("output", 0)):
        mc = model_cost(m, c)
        cost_cell = f"${mc:,.2f}" if price_for(m) is not None else ("$0.00" if m in NONBILLED_MODELS else "— (no card)")
        L.append(f"| {m} | {provider_bucket(m)} | {fmt_int(c.get('turns',0))} | {fmt_int(c.get('output',0))} | "
                 f"{fmt_int(c.get('cache_read',0))} | {cost_cell} |")
    L.append("")

    L.append("## Per-namespace rollup\n")
    L.append("| Namespace | Sessions | Output tok | Opus output share | Cache-read tok | Tool calls | Top model (by output) | Est. cost |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|")
    top_model = agg.get("per_namespace_top_model", {})
    opus_share = agg.get("per_namespace_opus_share", {})
    for ns, v in sorted(agg["per_namespace"].items(), key=lambda kv: -kv[1]["output"]):
        L.append(f"| {ns} | {v['sessions']} | {fmt_int(v['output'])} | {fmt_pct(opus_share.get(ns))} | "
                 f"{fmt_int(v['cache_read'])} | "
                 f"{fmt_int(v['tool_use'])} | {top_model.get(ns, '?')} | ${agg['per_namespace_cost'][ns]:,.2f} |")
    L.append("")

    d = agg["dist"]
    L.append("## Distributions (per session)\n")
    L.append(f"- **Tool calls/session:** median {d['calls_per_session']['median']}, "
             f"mean {d['calls_per_session']['mean']}, p90 {d['calls_per_session']['p90']}, max {d['calls_per_session']['max']}")
    L.append(f"- **Output tokens/session:** median {fmt_int(d['output_tokens_per_session']['median'] or 0)}, "
             f"p90 {fmt_int(d['output_tokens_per_session']['p90'] or 0)}, max {fmt_int(d['output_tokens_per_session']['max'] or 0)}")
    L.append(f"- **I:O ratio/session:** median {d['io_ratio']['median']}, p90 {d['io_ratio']['p90']}")
    L.append(f"- **Cache-hit fraction/session:** median {d['cache_hit_frac']['median']}, "
             f"p10 {d['cache_hit_frac']['p10']}, p90 {d['cache_hit_frac']['p90']}")
    L.append(f"- **Read-only tool fraction/session:** median {d['read_only_frac']['median']}\n")

    L.append("## Global tool mix\n")
    L.append("| Tool | Calls | Read-only? |")
    L.append("|---|---:|:--:|")
    for name, n in list(agg["tool_mix"].items())[:25]:
        L.append(f"| {name} | {fmt_int(n)} | {'✓' if name in READ_ONLY_TOOLS else ''} |")
    L.append("")

    # Behavioral lens (#2365) — the stuck/churn half the token lens can't see.
    beh = agg.get("behavior") or {}
    if beh:
        L.append("## Behavioral lens — stuck/churn detectors\n")
        pt = beh.get("per_tool", {})
        with_calls = sorted((t for t in pt if pt[t]["calls"]),
                            key=lambda t: -pt[t]["calls"])
        show = with_calls[:12] + [t for t in pt
                                  if pt[t]["errors"] and t not in with_calls[:12]]
        L.append("| Tool | Calls | Errors | Error rate |")
        L.append("|---|---:|---:|---:|")
        for t in show:
            v = pt[t]
            L.append(f"| {t} | {fmt_int(v['calls'])} | {fmt_int(v['errors'])} | "
                     f"{fmt_pct(v['error_rate'])} |")
        churn = beh.get("edit_churn", {})
        L.append("")
        L.append(f"- **Timeout kills (shell result matched exit-143 / \"timed out\"):** "
                 f"{fmt_int(beh.get('timeout_kills', 0))}")
        L.append(f"- **Foreground sleep-polls (`sleep`/`Start-Sleep` command prefix):** "
                 f"{fmt_int(beh.get('sleep_polls', 0))}")
        L.append(f"- **Edit/Write churn (wasted mutation calls):** "
                 f"{fmt_int(beh.get('wasted_mutation_calls', 0))}  "
                 f"(not-read {fmt_int(churn.get('not_read', 0))} · "
                 f"stale-read {fmt_int(churn.get('stale_read', 0))})")
        L.append(f"- **Sessions with a ≥{STALL_GAP_S//60}-min zero-record stall "
                 f"(harness/API dead time):** {fmt_int(beh.get('stall_sessions', 0))}"
                 + (f"  (longest gap {beh.get('max_gap_s', 0)/60:.0f} min)"
                    if beh.get("stall_sessions") else ""))
        rep = beh.get("repeat_failure_sessions") or []
        L.append(f"- **Sessions with a VERBATIM retry loop "
                 f"(≥{REPEAT_FAILURE_MIN}× same tool+args+error):** {len(rep)}"
                 + (" — worst below" if rep else ""))
        if rep:
            L.append("")
            L.append("| Session | NS | Tool | × | Failure signature |")
            L.append("|---|---|---|---:|---|")
            for r in rep:
                sig = r["sig"][:80].replace("|", "\\|")
                L.append(f"| {r['session'][:8]} | {r['ns']} | {r['tool']} | "
                         f"{r['count']} | {sig} |")
        mass = beh.get("failure_mass_sessions") or []
        L.append(f"- **Sessions with a recurring failure CLASS "
                 f"(≥{REPEAT_FAILURE_MIN}× same tool+error, args vary):** {len(mass)}"
                 + (" — worst below" if mass else ""))
        if mass:
            L.append("")
            L.append("| Session | NS | Tool | × | Failure class |")
            L.append("|---|---|---|---:|---|")
            for r in mass:
                sig = r["sig"][:80].replace("|", "\\|")
                L.append(f"| {r['session'][:8]} | {r['ns']} | {r['tool']} | "
                         f"{r['count']} | {sig} |")
        fc = beh.get("file_churn_sessions") or []
        L.append(f"- **Sessions with a REWRITE loop (≥{FILE_CHURN_MIN} mutations "
                 f"of one file, same-region or reverting):** {len(fc)}"
                 + (" — worst below" if fc else ""))
        if fc:
            L.append("")
            L.append("| Session | NS | × | Regions | Reverts | File |")
            L.append("|---|---|---:|---:|---:|---|")
            for r in fc:
                fp = r["file"].replace("|", "\\|")
                L.append(f"| {r['session'][:8]} | {r['ns']} | {r['count']} | "
                         f"{r.get('distinct_regions', '—')} | {r.get('reverts', '—')} | {fp} |")
        L.append("")

    L.append("## Top 15 sessions by output tokens\n")
    L.append("| Session | NS | Turns | Tool calls | Output tok | I:O | Cache-hit | Est.$ |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in sorted(S, key=lambda x: -x["tokens"]["output"])[:15]:
        ns = os.path.basename(os.path.dirname(s["path"]))
        io = f"{s['io_ratio']:.0f}" if s["io_ratio"] else "—"
        ch = f"{s['cache_hit_frac']*100:.0f}%" if s["cache_hit_frac"] is not None else "—"
        L.append(f"| {s['session'][:8]} | {ns} | {s['assistant_turns']} | {s['n_tool_use']} | "
                 f"{fmt_int(s['tokens']['output'])} | {io} | {ch} | ${s['cost_usd']:.2f} |")
    L.append("")
    return "\n".join(L)

def cmd_discover(a):
    ss = discover(a.root or DEFAULT_ROOTS, a.since_days, "" if a.all else a.ns_prefix)
    print(f"{len(ss)} sessions")
    for s in ss[:a.limit]:
        mt = datetime.datetime.fromtimestamp(s["mtime"]).isoformat(timespec="seconds")
        print(f"  {mt}  {s['size']//1024:6d}KB  {s['ns']}/{os.path.basename(s['path'])}")

def summarize_analyses(results):
    totals = collections.Counter()
    cost = 0.0
    ok = 0
    for r in results:
        if "error" in r:
            continue
        ok += 1
        for k, v in r["tokens"].items():
            totals[k] += v
        cost += r["cost_usd"]
    return {"count": ok, "tokens": dict(totals), "cost_usd": cost}

def summarize_transcripts(records):
    results = []
    for rec in records:
        r = analyze(rec["path"])
        results.append(r)
    return summarize_analyses(results)

def cmd_audit(a):
    roots = a.root or DEFAULT_ROOTS
    ns_prefix = "" if a.all else a.ns_prefix
    ss = discover(roots, a.since_days, ns_prefix,
                  include_subagents=a.include_subagents)
    if a.max:
        ss = ss[:a.max]
    kind = {s["path"]: s.get("kind", "session") for s in ss}
    print(f"analyzing {len(ss)} transcripts ...", file=sys.stderr)
    out = []
    for s in ss:
        r = analyze(s["path"])
        r["kind"] = kind.get(s["path"], "session")
        out.append(r)
    sess = [r for r in out if r.get("kind") == "session"]
    subs = [r for r in out if r.get("kind") == "subagent"]
    agg = aggregate(sess)
    excluded_subagents = None
    if not a.include_subagents:
        subagent_records = [s for s in discover(roots, a.since_days, ns_prefix,
                                                include_subagents=True)
                            if s.get("kind") == "subagent"]
        if subagent_records:
            excluded_subagents = summarize_transcripts(subagent_records)
    md = report_md(sess, agg, ns_prefix=ns_prefix, since_days=a.since_days,
                   include_subagents=a.include_subagents, max_sessions=a.max,
                   excluded_subagents=excluded_subagents)
    if subs:
        sub_summary = summarize_analyses(subs)
        st = collections.Counter(sub_summary["tokens"])
        scost = sub_summary["cost_usd"]
        md += ("\n## Subagent / workflow spend (SEPARATE transcripts, usually uncounted)\n\n"
               f"- **{len(subs)} subagent transcripts** (workflow/fan-out agents under `<session>/subagents/`)\n"
               f"- Output tokens: {fmt_int(st['output'])}  ·  Cache-read: {fmt_int(st['cache_read'])}  "
               f"·  Cache-creation: {fmt_int(st['cache_create'])}  ·  Fresh input: {fmt_int(st['input'])}\n"
               f"- Est. cost: ${scost:,.2f}  _(assumed pricing)_\n"
               f"- **True spend = top-level + this.** Subagent output here is "
               f"{(st['output']/max(sum(r['tokens']['output'] for r in sess if 'error' not in r),1)*100):.0f}% "
               f"of the top-level session output — the orchestrator sessions undercount real work by that much.\n")
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md)
        print(f"wrote {a.md}", file=sys.stderr)
    if a.json:
        slim = {"aggregate": agg,
                "excluded_subagents": excluded_subagents,
                "sessions": [{k: v for k, v in s.items() if k != "prompts"} for s in out]}
        json.dump(slim, open(a.json, "w", encoding="utf-8"), indent=2)
        print(f"wrote {a.json}", file=sys.stderr)
    print(md)

def _iso_bucket(ts, mode):
    # ts like 2026-06-16T21:19:39.123Z
    try:
        d = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None
    if mode == "day":
        return d.strftime("%Y-%m-%d")
    return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"   # ISO week

def trend_scan(roots, ns_prefix, bucket, include_subagents, exclude_substr=None):
    """Stream every transcript, fold usage/tools into time buckets.
    Cheap substring pre-filter: only json.loads lines that can carry usage or a tool_use,
    so multi-MB tool_result lines (browser snapshots, big reads) are skipped without parsing."""
    files = discover(roots, since_days=None, ns_prefix=ns_prefix,
                     include_subagents=include_subagents)
    if exclude_substr:
        files = [f for f in files if not any(s in f["path"] for s in exclude_substr)]
    buckets = collections.defaultdict(lambda: {
        "files": 0, "assist_turns": 0,
        "tok": collections.Counter(), "tools": collections.Counter(),
        "models": collections.Counter(), "cost": 0.0,
        "tool_errors": collections.Counter(), "beh": collections.Counter()})
    # Errored tool_results are the only user lines the behavioral lens needs;
    # gating on the serialized is_error:true keeps the huge SUCCESSFUL
    # tool_result lines (browser snapshots, big reads) unparsed as before.
    err_markers = ('"is_error": true', '"is_error":true')
    n = 0
    for f in files:
        n += 1
        first_ts = None
        try:
            fh = open(f["path"], encoding="utf-8")
        except OSError:
            continue
        # find this file's bucket from its first timestamped line
        rows_assist = []
        rows_err = []
        with fh:
            for line in fh:
                if '"timestamp"' in line and first_ts is None:
                    try:
                        first_ts = json.loads(line).get("timestamp")
                    except Exception:
                        pass
                if '"usage"' in line or '"tool_use"' in line:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("type") == "assistant":
                        rows_assist.append(r)
                    if first_ts is None and r.get("timestamp"):
                        first_ts = r.get("timestamp")
                elif any(m in line for m in err_markers):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("type") == "user":
                        rows_err.append(r)
        b = _iso_bucket(first_ts or "", bucket)
        if b is None:
            continue
        B = buckets[b]
        B["files"] += 1
        seen_msg_ids = set()   # de-dup billed turns per file (see analyze())
        seen_blocks = set()    # de-dup tool_use by BLOCK identity (see analyze())
        lens = BehaviorLens()
        tooluse_names = {}
        tooluse_args = {}
        for r in rows_assist:
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            new_turn = True
            if mid is not None:
                if mid in seen_msg_ids:
                    new_turn = False
                else:
                    seen_msg_ids.add(mid)
            if new_turn:
                u = msg.get("usage", {}) or {}
                B["assist_turns"] += 1
                B["models"][msg.get("model", "?")] += 1
                inp = u.get("input_tokens", 0) or 0
                out = u.get("output_tokens", 0) or 0
                cr = u.get("cache_read_input_tokens", 0) or 0
                cc = u.get("cache_creation_input_tokens", 0) or 0
                B["tok"]["input"] += inp
                B["tok"]["output"] += out
                B["tok"]["cache_read"] += cr
                B["tok"]["cache_create"] += cc
                B["cost"] += cost_usd(msg.get("model"), inp, cc, cr, out)
            for blk in (msg.get("content") or []):
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    key = blk.get("id") or (mid, blk.get("name"),
                                            json.dumps(blk.get("input", {}),
                                                       sort_keys=True, default=str))
                    if mid is not None:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    name = blk.get("name", "?")
                    B["tools"][name] += 1
                    if blk.get("id"):
                        tooluse_names[blk["id"]] = name
                        tooluse_args[blk["id"]] = hash(json.dumps(
                            blk.get("input", {}), sort_keys=True, default=str))
                    lens.see_tool_use(name, blk.get("input"))
        for r in rows_err:
            content = (r.get("message", {}) or {}).get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result" \
                        and blk.get("is_error"):
                    lens.see_tool_result(
                        tooluse_names.get(blk.get("tool_use_id"), "?"),
                        True, _txt_str(blk.get("content", "")),
                        args_key=tooluse_args.get(blk.get("tool_use_id")))
        s = lens.summary()
        B["tool_errors"].update(s["tool_errors"])
        B["beh"]["timeout_kills"] += s["timeout_kills"]
        B["beh"]["sleep_polls"] += s["sleep_polls"]
        B["beh"]["edit_churn"] += sum(s["edit_churn"].values())
        if s["max_repeat_failure"] >= REPEAT_FAILURE_MIN:
            B["beh"]["repeat_failure_files"] += 1
        if s["failure_mass"]:
            B["beh"]["failure_mass_files"] += 1
        if s["file_churn"]:   # flagged rewrite-loops only, not raw edit counts
            B["beh"]["file_churn_files"] += 1
    return buckets, n

def cmd_trend(a):
    roots = a.root or DEFAULT_ROOTS
    nsp = "" if a.all else a.ns_prefix
    excl = ["Temp-agf", "Temp-bench"] if a.exclude_bench else None
    buckets, n = trend_scan(roots, nsp, a.bucket, a.include_subagents, excl)
    print(f"# Trend — {n} transcripts, bucket={a.bucket}, ns_prefix={nsp or '(all)'}\n")
    print(f"{'bucket':10} {'files':>6} {'turns':>7} {'out_tok':>12} {'cacheRead':>14} "
          f"{'cacheHit%':>9} {'I:O':>7} {'cost$':>10} {'err%':>5} {'t/o':>4} "
          f"{'slp':>4} {'chrn':>5}  top_model / top_tool")
    rows = []
    for b in sorted(buckets):
        B = buckets[b]
        t = B["tok"]
        tot_in = t["input"] + t["cache_read"] + t["cache_create"]
        io = tot_in / t["output"] if t["output"] else 0
        chf = t["cache_read"] / tot_in * 100 if tot_in else 0
        tm = B["models"].most_common(1)
        tt = B["tools"].most_common(1)
        tmn = (tm[0][0].replace("claude-", "")[:14]) if tm else "—"
        ttn = (tt[0][0].replace("mcp__playwright__", "pw:")[:18]) if tt else "—"
        n_calls = sum(B["tools"].values())
        n_errs = sum(B["tool_errors"].values())
        errp = n_errs / n_calls * 100 if n_calls else 0
        beh = B["beh"]
        print(f"{b:10} {B['files']:>6} {B['assist_turns']:>7} {t['output']:>12,} "
              f"{t['cache_read']:>14,} {chf:>8.1f}% {io:>7.0f} {B['cost']:>10,.0f} "
              f"{errp:>4.0f}% {beh['timeout_kills']:>4} {beh['sleep_polls']:>4} "
              f"{beh['edit_churn']:>5}  {tmn} / {ttn}")
        rows.append({"bucket": b, "files": B["files"], "turns": B["assist_turns"],
                     "tok": dict(t), "io_ratio": round(io, 1), "cache_hit_pct": round(chf, 1),
                     "cost_usd": round(B["cost"], 2),
                     "top_models": B["models"].most_common(5),
                     "top_tools": B["tools"].most_common(12),
                     "behavior": {"tool_errors": dict(B["tool_errors"].most_common()),
                                  "tool_error_pct": round(errp, 1),
                                  **{k: beh[k] for k in
                                     ("timeout_kills", "sleep_polls", "edit_churn",
                                      "repeat_failure_files", "file_churn_files")}}})
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.json}", file=sys.stderr)

def cmd_deep(a):
    s = analyze(a.session)
    print(f"# Trajectory: {s['session']}")
    print(f"records={s['n_records']} turns={s['assistant_turns']} tool_calls={s['n_tool_use']} "
          f"output_tok={fmt_int(s['tokens']['output'])} io={s['io_ratio'] and round(s['io_ratio'],1)} "
          f"cache_hit={s['cache_hit_frac'] and round(s['cache_hit_frac'],3)} cost=${s['cost_usd']:.2f}")
    print(f"tools={s['tools']}")
    b = s["behavior"]
    print(f"behavior: tool_errors={sum(b['tool_errors'].values())} {b['tool_errors']} "
          f"timeout_kills={b['timeout_kills']} sleep_polls={b['sleep_polls']} "
          f"edit_churn={b['edit_churn']} max_repeat_failure={b['max_repeat_failure']} "
          f"max_file_churn={b['max_file_churn']}")
    for r in b["repeat_failures"][:5]:
        print(f"  repeat ×{r['count']} [{r['tool']}] {r['sig'][:120]}")
    for r in b["file_churn"][:5]:
        print(f"  churn  ×{r['count']} {r['file']}")
    print("\n## User asks (the trajectory), in order:")
    for i, (ts, txt) in enumerate(s["prompts"]):
        one = " ".join(txt.split())
        print(f"  [{i:2d}] {ts}  {one[:200]}")

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Audit Claude Code session transcripts.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "audit"):
        q = sub.add_parser(name)
        q.add_argument("--root", action="append")
        q.add_argument("--since-days", type=float, default=None)
        q.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
        q.add_argument("--all", action="store_true", help="include ALL namespaces (no prefix filter)")
        if name == "discover":
            q.add_argument("--limit", type=int, default=40)
        else:
            q.add_argument("--max", type=int, default=None)
            q.add_argument("--json", default=None)
            q.add_argument("--md", default=None)
            q.add_argument("--include-subagents", action="store_true",
                           help="also fold in subagent/workflow transcripts (separate files)")
    qt = sub.add_parser("trend")
    qt.add_argument("--root", action="append")
    qt.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
    qt.add_argument("--all", action="store_true")
    qt.add_argument("--bucket", choices=["day", "week"], default="week")
    qt.add_argument("--include-subagents", action="store_true")
    qt.add_argument("--exclude-bench", action="store_true",
                    help="drop Temp-agf*/Temp-bench* eval namespaces")
    qt.add_argument("--json", default=None)
    q = sub.add_parser("deep")
    q.add_argument("session")
    a = p.parse_args()
    {"discover": cmd_discover, "audit": cmd_audit, "trend": cmd_trend,
     "deep": cmd_deep}[a.cmd](a)

if __name__ == "__main__":
    main()
