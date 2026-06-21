#!/usr/bin/env python3
"""leak_scan.py — static leak-candidate scanner for Go hot paths (the DISCOVER phase).

A memory/goroutine-leak sweep has three phases: DISCOVER candidate sites, AUDIT each
with a reasoning agent, PROVE the fix with a regression test. This tool is the first
phase made repeatable and read-only: it greps the Go tree for the *shapes* that leaks
hide in, classifies them into a small taxonomy, ranks by signal, and emits a balanced
package PARTITION so the audit fan-out (one agent per lane) is deterministic instead of
improvised. It never decides that something IS a leak — the audit does — it decides
where a human/agent must LOOK, the same posture as readme_freshness_audit / issue_triage.

It is deliberately a heuristic, not a verifier: a clean report does not prove leak-freedom
(only the AUDIT + PROVE phases do), and a flagged site is a candidate, not a verdict. The
value is coverage + repeatability: the same shapes that hid the ctxmmu unbounded-ledger
leak and the O(n^2) decode-scores churn are exactly what it surfaces, every run, with no
one having to remember the grep list.

Taxonomy (klass -> what it catches -> default signal):
  ticker_timer_no_stop  time.NewTicker/NewTimer whose var has no nearby .Stop()   high
  ctx_cancel_unreleased context.WithCancel/Timeout/Deadline whose cancel() is      high
                        not deferred/called within the look-ahead window
  unbounded_map         a struct-field/package map written (m[k]=) but whose       high
                        package has NO delete(m,...) and no cap/LRU tell — the
                        ctxmmu shape (a ledger that only ever grows)
  goroutine_spawn       `go func(`/`go ident(` — a worker that must join or         med
                        exit on cancel; flagged for the audit to trace its exit
  body_no_close         a function reads resp.Body / .Body but the file has no      med
                        matching .Body.Close() — an unclosed-body / fd-leak tell
  loop_alloc_scaling    make([]T, <non-const>) inside a for-body — a per-iteration  low
                        buffer whose size may scale with the loop (the O(n^2) tell)

Usage:
  python leak_scan.py [ROOT ...]            # default roots: fak/internal fak/cmd
        [--include-tests]                   # also scan *_test.go (default: skip)
        [--lanes N]                         # partition packages into N audit lanes (default 4)
        [--min-signal low|med|high]         # drop findings below this signal (default low)
        [--json OUT] [--md OUT]             # write machine / human reports (default: md to stdout)

Exit status: 0 always (read-only advisory). The JSON carries `ok` = (no high-signal
candidates) so a control pane can gate on it; high-signal != confirmed leak, it means
"audit this before you ship".
"""
import sys
import os
import re
import json
import argparse
import collections

DEFAULT_ROOTS = ["fak/internal", "fak/cmd"]

SIGNAL_RANK = {"low": 0, "med": 1, "high": 2}

# --- line-level pattern classes ------------------------------------------------------

RE_TICKER = re.compile(r"\b(\w+)\s*:?=\s*time\.New(?:Ticker|Timer)\(")
RE_TICKER_BARE = re.compile(r"\btime\.New(?:Ticker|Timer)\(")
RE_CTX = re.compile(r"\b(?:(\w+)\s*,\s*)?(\w+)\s*:?=\s*context\.With(?:Cancel|Timeout|Deadline)\(")
RE_GO_FUNC = re.compile(r"\bgo\s+func\b")
RE_GO_CALL = re.compile(r"\bgo\s+[A-Za-z_]\w*(?:\.\w+)*\(")
RE_MAKE_SLICE_LOOP = re.compile(r"\bmake\(\[\][\w\.\*\[\]]+,\s*([^,)\]]+)\)")
RE_MAP_DECL = re.compile(r"\b(\w+)\s+map\[[^\]]+\][\w\.\*\[\]{}]+")
RE_MAP_WRITE = re.compile(r"\b([A-Za-z_]\w*)\[[^\]]+\]\s*(?:=|\+\+|--|\+=)")
# capture the FINAL identifier of a (possibly qualified) map expr: delete(g.held, k) -> held,
# matching how RE_MAP_WRITE/RE_MAP_DECL key on the unqualified field/var name.
RE_DELETE = re.compile(r"\bdelete\(\s*(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)\b")
RE_BODY_READ = re.compile(r"\b(?:resp|res|r|response|reply)\.Body\b")
RE_BODY_CLOSE = re.compile(r"\.Body\.Close\(\)")

# cap/LRU tells: if a package mentions any of these near its maps, assume it is bounded
# by design and DON'T raise unbounded_map (keeps SGLang-style budgeted caches quiet).
CAP_TELLS = re.compile(r"(?i)\b(maxheld|maxtokens|maxkeys|maxentries|max[_]?\w*len|"
                       r"evict|lru|trim|budget|capacity|ringbuf|ring buffer|cap on|bounded)\b")


def iter_go_files(roots, include_tests):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "testdata", "vendor")]
            for fn in filenames:
                if not fn.endswith(".go"):
                    continue
                if not include_tests and fn.endswith("_test.go"):
                    continue
                yield os.path.join(dirpath, fn)


def strip_comment(line):
    # crude // comment strip (good enough to avoid flagging commented-out code);
    # leaves string-literal "//" rare-case noise, acceptable for a heuristic scanner.
    i = line.find("//")
    return line[:i] if i >= 0 else line


def lookahead_has(lines, start, window, *needles):
    end = min(len(lines), start + window)
    for k in range(start, end):
        for n in needles:
            if n in lines[k]:
                return True
    return False


def scan_file(path, lines):
    """Line-level findings for one file (excludes the package-scoped unbounded_map pass)."""
    out = []
    # bracket-depth tracker for "inside a for-loop body" (loop_alloc_scaling)
    for_depth_stack = []
    depth = 0
    for idx, raw in enumerate(lines):
        line = strip_comment(raw)
        stripped = line.strip()

        # ticker/timer without a nearby Stop()
        m = RE_TICKER.search(line)
        if m:
            var = m.group(1)
            if not lookahead_has(lines, idx, 8, f"{var}.Stop()"):
                out.append((idx + 1, "ticker_timer_no_stop", "high", stripped,
                            f"{var} has no .Stop() within 8 lines — defer {var}.Stop() right after"))
        elif RE_TICKER_BARE.search(line):
            # an inline NewTimer(...) with no captured var (e.g. <-time.NewTimer(d).C) —
            # the timer can't be Stopped; usually fine for one-shot but worth a glance.
            if "time.After(" not in line:
                out.append((idx + 1, "ticker_timer_no_stop", "med", stripped,
                            "timer created without a captured var to Stop()"))

        # context.With* whose cancel is not released nearby
        m = RE_CTX.search(line)
        if m:
            cancel = m.group(2)
            if cancel and cancel != "_" and not lookahead_has(lines, idx, 10, f"{cancel}()"):
                out.append((idx + 1, "ctx_cancel_unreleased", "high", stripped,
                            f"cancel func {cancel} not deferred/called within 10 lines"))

        # goroutine spawns
        if RE_GO_FUNC.search(line) or RE_GO_CALL.search(line):
            out.append((idx + 1, "goroutine_spawn", "med", stripped,
                        "trace this goroutine's exit: does it join (WaitGroup) or exit on ctx/closed chan?"))

        # body read without any close in the file (file-scoped heuristic; refined later)
        if RE_BODY_READ.search(line):
            out.append((idx + 1, "body_no_close", "med", stripped, "_filescope_"))

        # loop-scoped make([]T, var)
        depth_before = depth
        depth += line.count("{") - line.count("}")
        if re.search(r"\bfor\b", line) and "{" in line:
            for_depth_stack.append(depth_before)
        while for_depth_stack and depth <= for_depth_stack[-1]:
            for_depth_stack.pop()
        if for_depth_stack:
            m = RE_MAKE_SLICE_LOOP.search(line)
            if m:
                size = m.group(1).strip()
                if not re.fullmatch(r"\d+", size):  # non-constant size inside a loop
                    out.append((idx + 1, "loop_alloc_scaling", "low", stripped,
                                f"make([], {size}) inside a loop — reuse a scratch if {size} scales"))
    return out


# A write to a growing map matters most when it happens on the REQUEST/RUNTIME path. A
# map populated once at construction/load/registration is fixed-size, not a leak — so the
# enclosing function's name is the strongest cheap signal for separating the ctxmmu/normgate
# "held ledger grows per tool result" shape from the "tokenizer vocab loaded once" shape.
CONSTRUCTION_FUNC = re.compile(
    r"(?i)^(init|new\w*|load\w*|parse\w*|decode\w*|read\w*|build\w*|compile\w*|"
    r"register\w*|setup\w*|must\w*|open\w*|from\w*|with\w*|default\w*|make\w*)$")
RUNTIME_FUNC = re.compile(
    r"(?i)(admit|handle|serve|record|quarantine|add|put|insert|track|observe|"
    r"emit|append|push|store|remember|note|log|count|incr|update|set|fire)")
RE_FUNC_DECL = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\(\[]")


def enclosing_funcs(lines):
    """Return a list mapping line-index -> enclosing function name ('' at file scope)."""
    out = [""] * len(lines)
    cur = ""
    depth = 0
    for i, raw in enumerate(lines):
        line = strip_comment(raw)
        if depth == 0:
            m = RE_FUNC_DECL.match(line)
            if m:
                cur = m.group(1)
        out[i] = cur
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            depth = 0
            cur = ""
    return out


def package_unbounded_maps(pkg_files, file_lines):
    """Package-scoped pass: map names written but never delete()'d and with no cap tell.

    Signal is graded by the enclosing function of the FIRST write: a write on a
    construction/load function → low (fixed-size, e.g. a parsed vocab); a write on a
    request/runtime function (Admit/Record/...) with no delete and no cap tell → high
    (the unbounded-ledger shape); anything else → med.
    """
    deletes = set()
    map_decl_names = {}      # name -> (file, line)
    text_all = []
    func_at = {}
    for path in pkg_files:
        func_at[path] = enclosing_funcs(file_lines[path])
        for idx, raw in enumerate(file_lines[path]):
            line = strip_comment(raw)
            text_all.append(line)
            for dm in RE_DELETE.finditer(line):
                deletes.add(dm.group(1))
            dm = RE_MAP_DECL.search(line)
            if dm:
                map_decl_names.setdefault(dm.group(1), (path, idx + 1))
    cap_seen = bool(CAP_TELLS.search("\n".join(text_all)))

    write_sites = {}         # name -> (file, line, enclosing_func)
    for path in pkg_files:
        for idx, raw in enumerate(file_lines[path]):
            line = strip_comment(raw)
            for wm in RE_MAP_WRITE.finditer(line):
                nm = wm.group(1)
                if nm in map_decl_names and nm not in write_sites:
                    write_sites[nm] = (path, idx + 1, func_at[path][idx])

    findings = []
    for name, (decl_path, decl_line) in map_decl_names.items():
        if name not in write_sites:
            continue                     # declared but never written → not a growth risk
        if name in deletes:
            continue                     # has a removal path → bounded-ish
        wpath, wline, fn = write_sites[name]
        if RUNTIME_FUNC.search(fn or ""):
            sig = "med" if cap_seen else "high"
            why = f"written on runtime path {fn}()"
        elif CONSTRUCTION_FUNC.match(fn or ""):
            sig = "low"
            why = f"written only on construction path {fn}() — likely fixed-size"
        else:
            sig = "low" if cap_seen else "med"
            why = f"written on {fn or 'file-scope'}()"
        findings.append((wpath, wline, "unbounded_map", sig,
                         f"map {name} (decl {os.path.relpath(decl_path)}:{decl_line})",
                         f"map {name}: {why}; no delete({name},...) "
                         f"and no cap/LRU tell — confirm it cannot grow without bound"))
    return findings


def refine_body_findings(findings, file_lines):
    """Drop body_no_close findings in files that DO close a body somewhere (file-scoped)."""
    closes_in_file = {}
    refined = []
    for f in findings:
        path = f["file"]
        if f["klass"] != "body_no_close":
            refined.append(f)
            continue
        if path not in closes_in_file:
            closes_in_file[path] = any(RE_BODY_CLOSE.search(l) for l in file_lines[path])
        if not closes_in_file[path]:
            f = dict(f)
            f["hint"] = "file reads .Body but never .Body.Close() — confirm the body is closed on every path"
            refined.append(f)
    return refined


def partition(packages_by_count, lanes):
    """Greedy bin-pack packages into `lanes` lanes balanced by finding count."""
    bins = [{"lane": i + 1, "packages": [], "weight": 0} for i in range(max(1, lanes))]
    for pkg, cnt in sorted(packages_by_count.items(), key=lambda kv: (-kv[1], kv[0])):
        b = min(bins, key=lambda b: b["weight"])
        b["packages"].append(pkg)
        b["weight"] += max(cnt, 1)
    return [b for b in bins if b["packages"]]


def run(roots, include_tests, lanes, min_signal):
    # Normalize every path to forward slashes up front so finding keys, file_lines keys,
    # and package grouping all agree (Windows backslash paths otherwise mismatch).
    files = [p.replace("\\", "/") for p in iter_go_files(roots, include_tests)]
    file_lines = {}
    pkgs = collections.defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            file_lines[path] = fh.read().splitlines()
        pkgs[os.path.dirname(path)].append(path)

    findings = []
    for path in files:
        for (line, klass, sig, snippet, hint) in scan_file(path, file_lines[path]):
            findings.append({"file": path, "line": line, "klass": klass,
                             "signal": sig, "snippet": snippet[:200], "hint": hint})
    for pkg, pkg_files in pkgs.items():
        for (wpath, wline, klass, sig, snippet, hint) in package_unbounded_maps(pkg_files, file_lines):
            findings.append({"file": wpath, "line": wline, "klass": klass,
                             "signal": sig, "snippet": snippet[:200], "hint": hint})

    findings = refine_body_findings(findings, file_lines)
    floor = SIGNAL_RANK[min_signal]
    findings = [f for f in findings if SIGNAL_RANK[f["signal"]] >= floor]
    findings.sort(key=lambda f: (-SIGNAL_RANK[f["signal"]], f["file"], f["line"]))

    by_klass = collections.Counter(f["klass"] for f in findings)
    by_signal = collections.Counter(f["signal"] for f in findings)
    by_pkg = collections.Counter(
        os.path.dirname(f["file"]).replace("\\", "/") for f in findings)
    high = by_signal.get("high", 0)
    lanes_out = partition(by_pkg, lanes)

    return {
        "ok": high == 0,
        "roots": roots,
        "files_scanned": len(files),
        "findings_total": len(findings),
        "by_signal": dict(by_signal),
        "by_klass": dict(by_klass),
        "high_signal_count": high,
        "audit_partition": lanes_out,
        "findings": findings,
    }


def to_markdown(rep):
    L = []
    L.append(f"# Leak-candidate scan — {rep['files_scanned']} Go files, "
             f"{rep['findings_total']} candidates\n")
    L.append(f"- roots: `{'`, `'.join(rep['roots'])}`")
    L.append(f"- ok (no high-signal candidates): **{rep['ok']}**  "
             f"(high={rep['high_signal_count']})")
    L.append(f"- by signal: {rep['by_signal']}")
    L.append(f"- by klass: {rep['by_klass']}\n")
    L.append("## Suggested audit partition (one agent per lane)\n")
    for b in rep["audit_partition"]:
        pkgs = ", ".join(f"`{p}`" for p in b["packages"])
        L.append(f"- **lane {b['lane']}** (weight {b['weight']}): {pkgs}")
    L.append("\n## Candidates (ranked by signal)\n")
    L.append("| signal | klass | site | hint |")
    L.append("|--------|-------|------|------|")
    for f in rep["findings"][:400]:
        site = f"{f['file']}:{f['line']}"
        L.append(f"| {f['signal']} | {f['klass']} | `{site}` | {f['hint']} |")
    if rep["findings_total"] > 400:
        L.append(f"\n_({rep['findings_total'] - 400} more — see --json)_")
    L.append("\n> Heuristic & read-only: a clean scan does NOT prove leak-freedom; a flag "
             "is a candidate, not a verdict. Feed the partition to the AUDIT phase "
             "(leak-sweep workflow/skill), then PROVE each real finding with internal/leakcheck.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="static leak-candidate scanner for Go hot paths")
    ap.add_argument("roots", nargs="*", default=None,
                    help="dirs to scan (default: fak/internal fak/cmd)")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--min-signal", choices=["low", "med", "high"], default="low")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--md", dest="md_out")
    args = ap.parse_args(argv)
    roots = args.roots if args.roots else DEFAULT_ROOTS

    rep = run(roots, args.include_tests, args.lanes, args.min_signal)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
    md = to_markdown(rep)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if not args.md_out or not args.json_out:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
