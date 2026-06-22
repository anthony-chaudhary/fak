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
"""
import sys
import os
import json
import glob
import argparse
import statistics
import collections
import datetime

# --- pricing assumption (USD per 1e6 tokens). EDIT to match real card; flagged in output. ---
PRICING = {
    # model_substring: (input, cache_write_5m, cache_read, output)
    "opus":   (15.0, 18.75, 1.50, 75.0),
    "sonnet": ( 3.0,  3.75, 0.30, 15.0),
    "haiku":  ( 0.80, 1.00, 0.08,  4.0),
    "fable":  ( 3.0,  3.75, 0.30, 15.0),
    "_default": (15.0, 18.75, 1.50, 75.0),
}
PRICING_IS_ASSUMPTION = True

DEFAULT_ROOTS = [
    os.path.join(os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")), "projects"),
]
# transient test-fixture / temp-workspace namespaces — never "our own sessions"
EXCLUDE_NS_SUBSTR = ["pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"]
NS_INCLUDE_PREFIX = "C--work"   # real project namespaces; override with --ns-prefix '' / --all

READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch", "WebSearch",
                   "TodoRead", "ToolSearch"}
# Bash/Edit/Write/NotebookEdit/Task/Workflow/etc. are side-effecting or spawn.

def price_for(model):
    m = (model or "").lower()
    for key, rates in PRICING.items():
        if key != "_default" and key in m:
            return rates
    return PRICING["_default"]

def cost_usd(model, inp, cwrite, cread, out):
    pi, pcw, pcr, po = price_for(model)
    return (inp*pi + cwrite*pcw + cread*pcr + out*po) / 1e6

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

    try:
        lines = open(path, encoding="utf-8").read().splitlines()
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

        if t == "assistant":
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            if mid is not None:
                if mid in seen_msg_ids:
                    dup_assistant_lines += 1
                    continue          # already folded this billed turn
                seen_msg_ids.add(mid)
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
            for b in (msg.get("content") or []):
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "tool_use":
                    n_tool_use += 1
                    tools[b.get("name", "?")] += 1
                    tool_input_chars += _txt_len(b.get("input", {}))
                elif bt == "thinking":
                    n_thinking += 1
                elif bt == "text":
                    n_text += 1
            if r.get("interruptedMessageId") or msg.get("stop_reason") == "interrupted":
                interrupted += 1
        elif t == "user":
            msg = r.get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        n_tool_result += 1
                        tool_result_chars += _txt_len(b.get("content", ""))
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
    return {
        "path": path, "session": os.path.splitext(os.path.basename(path))[0],
        "n_records": sum(rec_types.values()), "rec_types": dict(rec_types),
        "models": dict(models), "assistant_turns": assistant_turns,
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
    for s in S:
        ns = os.path.basename(os.path.dirname(s["path"]))
        ns_roll[ns]["sessions"] += 1
        ns_roll[ns]["output"] += s["tokens"]["output"]
        ns_roll[ns]["cache_read"] += s["tokens"]["cache_read"]
        ns_roll[ns]["tool_use"] += s["n_tool_use"]
        ns_cost[ns] += s["cost_usd"]
    calls = [s["n_tool_use"] for s in S]
    outs  = [s["tokens"]["output"] for s in S]
    ios   = [s["io_ratio"] for s in S if s["io_ratio"]]
    chf   = [s["cache_hit_frac"] for s in S if s["cache_hit_frac"] is not None]
    rof   = [s["read_only_frac"] for s in S if s["read_only_frac"] is not None]
    return {
        "n_sessions": len(S), "totals": dict(tot), "total_cost_usd": tot_cost,
        "tool_mix": dict(tot_tools.most_common()),
        "per_namespace": {k: dict(v) for k, v in ns_roll.items()},
        "per_namespace_cost": dict(ns_cost),
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

def report_md(sessions, agg):
    S = [s for s in sessions if "error" not in s]
    L = []
    L.append("# Session-Transcript Audit — recent sessions, this machine\n")
    L.append(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ")
    L.append(f"**Sessions audited:** {agg['n_sessions']}  ·  **Tool:** `tools/session_audit.py` (re-runnable)\n")
    t = agg["totals"]
    L.append("## Machine-wide totals (EXACT token counts)\n")
    L.append(f"- **Output tokens (the actual work generated):** {fmt_int(t['output'])}")
    L.append(f"- **Fresh input tokens (billed, non-cached):** {fmt_int(t['input'])}")
    L.append(f"- **Cache-read tokens (prompt-cache / KV reuse):** {fmt_int(t['cache_read'])}")
    L.append(f"- **Cache-creation tokens:** {fmt_int(t['cache_create'])}")
    tot_in = t['input']+t['cache_read']+t['cache_create']
    L.append(f"- **Total context ingested:** {fmt_int(tot_in)}  →  **machine-wide I:O ratio = {tot_in/max(t['output'],1):.1f} : 1**")
    chf = t['cache_read']/max(tot_in,1)
    L.append(f"- **Cache-read share of all ingested context = {chf*100:.1f}%**  (this is the prompt-cache/KV reuse the harness ALREADY captures)")
    L.append(f"- **Web search / fetch requests:** {fmt_int(t['web_search'])} / {fmt_int(t['web_fetch'])}")
    L.append(f"- **Multi-iteration count:** {fmt_int(t['iterations'])}")
    flag = "  _(⚠ cost uses an ASSUMED price table — edit PRICING; token counts above are exact)_" if PRICING_IS_ASSUMPTION else ""
    L.append(f"- **Estimated cost:** ${agg['total_cost_usd']:,.2f}{flag}\n")

    L.append("## Per-namespace rollup\n")
    L.append("| Namespace | Sessions | Output tok | Cache-read tok | Tool calls | Est. cost |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for ns, v in sorted(agg["per_namespace"].items(), key=lambda kv: -kv[1]["output"]):
        L.append(f"| {ns} | {v['sessions']} | {fmt_int(v['output'])} | {fmt_int(v['cache_read'])} | "
                 f"{fmt_int(v['tool_use'])} | ${agg['per_namespace_cost'][ns]:,.2f} |")
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

def cmd_audit(a):
    ss = discover(a.root or DEFAULT_ROOTS, a.since_days, "" if a.all else a.ns_prefix,
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
    md = report_md(sess, agg)
    if subs:
        st = collections.Counter()
        for r in subs:
            if "error" in r:
                continue
            for k, v in r["tokens"].items():
                st[k] += v
        scost = sum(r["cost_usd"] for r in subs if "error" not in r)
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
        "models": collections.Counter(), "cost": 0.0})
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
        b = _iso_bucket(first_ts or "", bucket)
        if b is None:
            continue
        B = buckets[b]
        B["files"] += 1
        seen_msg_ids = set()   # de-dup billed turns per file (see analyze())
        for r in rows_assist:
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            if mid is not None:
                if mid in seen_msg_ids:
                    continue
                seen_msg_ids.add(mid)
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
                    B["tools"][blk.get("name", "?")] += 1
    return buckets, n

def cmd_trend(a):
    roots = a.root or DEFAULT_ROOTS
    nsp = "" if a.all else a.ns_prefix
    excl = ["Temp-agf", "Temp-bench"] if a.exclude_bench else None
    buckets, n = trend_scan(roots, nsp, a.bucket, a.include_subagents, excl)
    print(f"# Trend — {n} transcripts, bucket={a.bucket}, ns_prefix={nsp or '(all)'}\n")
    print(f"{'bucket':10} {'files':>6} {'turns':>7} {'out_tok':>12} {'cacheRead':>14} "
          f"{'cacheHit%':>9} {'I:O':>7} {'cost$':>10}  top_model / top_tool")
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
        print(f"{b:10} {B['files']:>6} {B['assist_turns']:>7} {t['output']:>12,} "
              f"{t['cache_read']:>14,} {chf:>8.1f}% {io:>7.0f} {B['cost']:>10,.0f}  {tmn} / {ttn}")
        rows.append({"bucket": b, "files": B["files"], "turns": B["assist_turns"],
                     "tok": dict(t), "io_ratio": round(io, 1), "cache_hit_pct": round(chf, 1),
                     "cost_usd": round(B["cost"], 2),
                     "top_models": B["models"].most_common(5),
                     "top_tools": B["tools"].most_common(12)})
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
