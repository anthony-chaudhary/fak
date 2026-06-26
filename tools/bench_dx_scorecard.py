#!/usr/bin/env python3
"""Bench-DX scorecard  -  the measuring stick for the benchmarking DEVELOPER EXPERIENCE.

The sibling scorecards each grade a surface a reviewer cares about:
``code_quality_scorecard`` grades the Go module, ``agent_readiness_scorecard``
grades how attractive the repo is to an AI agent, ``industry_scorecard`` grades
the competitive claims. None of them grade the thing a *developer who wants to run
a benchmark* actually hits: can they **(1) discover** what benchmarks exist,
**(2) cold-start** at least one with no weights/GPU/dataset/key, **(3) learn one
flag vocabulary that transfers**, **(4) read a number and know what it means**, and
**(5) record + compare** a result? fak ships 27 benchmark surfaces (20 ``cmd/*bench*``
mains + 7 ``fak`` verbs)  -  its headline value-proof  -  and until now had no number
for how hard they are to use.

This scores the git-tracked tree on mechanical KPIs in five groups  -  the five steps
a developer walks  -  folds them into a weighted score and an A-F grade, and counts
**bench-DX-debt**: the total of concrete, re-derivable defects that make a
benchmark harder to find, start, learn, read, or compare. Each is a defect you fix
by *adding the missing affordance* (a catalog verb, an offline fallback, a shared
flag, an inline provenance line, a record command), not by writing prose.

  DISCOVER      -  a developer can enumerate the runnable benchmarks from one place
    catalog_verb     a `fak benchmarks` catalog verb exists that LISTS benchmarks
                     (not the `fak bench` A/B-ablation verb, which runs a suite)
    catalog_registry an in-binary registry (internal/benchcatalog) backs the verb,
                     so the list cannot drift from the cmd/ tree
    registry_covers_tree every cmd/*bench* main AND every `fak` bench verb appears
                     in the registry exactly once (no orphan main, no dead entry)

  COLD-START    -  a developer can run at least one meaningful benchmark with nothing
    offline_set      the registry marks which benchmarks need no weights/GPU/dataset/key
    webbench_runs_clean `fak webbench describe` works with NO --dataset (the doc's
                     "RUNNABLE NOW" claim is honest  -  it has an offline fallback)
    swebench_runs_clean `fak swebench describe` works with NO --difficulty/--dataset
                     (its RUNNABLE-NOW claim is honest  -  it has an offline fallback)
    no_spurious_paths no cmd/*bench* default flag points at a non-existent `fak/`-
                     prefixed path (the doubled-root bug that makes a bench fail cold)

  LEARN         -  one flag vocabulary transfers from one benchmark to the next
    out_flag_uniform a shared output convention (the registry documents -out per bench)
    quant_polarity_consistent the -quant flag's default does not flip polarity
                     between benches (same flag, same default meaning)

  READ          -  a number is self-describing: units, provenance, comparison
    provenance_doc   BENCHMARK-AUTHORITY.md exists (every number traces to it)
    summary_self_describes each registry row carries a one-line "what this number
                     means", so a developer reads intent without opening source

  COMPARE       -  run -> record -> compare is a documented, working loop
    bench_cli_path_ok the result-query CLI (tools/bench_cli.py) points at a path that
                     resolves from the repo root (no doubled `fak/` prefix)
    compare_documented a doc names the run->record->compare sequence (bench_cli.py
                     list/compare, or a Makefile target)

The headline metric is **bench-DX-debt**: the count of concrete HARD defects above.
Driving it to zero means a developer who wants to benchmark fak can find what to
run, run something in under a minute, learn one flag set, trust the number, and
compare it  -  without reading 20 mains. The companion process re-runs this, retires
the worst-first defect by adding the missing affordance, and re-runs to prove the
drop, folding into the unified scorecard_control_pane alongside the other sticks.

Deterministic + read-only by construction: it reads the git-tracked tree (so two
clones of the same commit score identically) and edits nothing. Run from the repo
ROOT::

    python tools/bench_dx_scorecard.py                 # human scorecard
    python tools/bench_dx_scorecard.py --json          # machine payload
    python tools/bench_dx_scorecard.py --markdown      # the committed snapshot body
    python tools/bench_dx_scorecard.py --compare base.json   # prove the debt moved
"""
from __future__ import annotations

import argparse
import json
import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-bench-dx-scorecard/1"
GENERATED_SNAPSHOT = "docs/BENCH-DX-SCORECARD.md"

# The five steps a developer walks, in order. Weights sum to 100; DISCOVER and
# COLD-START carry the most because a benchmark you can't find or can't start has
# zero value no matter how clean its flags are.
GROUPS = ["discover", "coldstart", "learn", "read", "compare"]
GROUP_WEIGHT = {"discover": 28, "coldstart": 28, "learn": 16, "read": 14, "compare": 14}

REGISTRY_GO = "internal/benchcatalog/catalog.go"
CATALOG_VERB_GO = "cmd/fak/benchmarks.go"
AUTHORITY_DOC = "BENCHMARK-AUTHORITY.md"
BENCH_CLI = "tools/bench_cli.py"
WEBBENCH_GO = "cmd/fak/webbench.go"
SWEBENCH_GO = "cmd/fak/swebench.go"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent.parent, *here.parents]:
        if (p / "go.mod").exists() and (p / "cmd").is_dir():
            return p
    return here.parent.parent


def _git_lines(args: list[str], root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _tracked(root: Path) -> list[str]:
    return _git_lines(["ls-files"], root)


def _read(root: Path, rel: str) -> str | None:
    p = root / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Tree facts  -  what benchmark surfaces actually exist, derived from git, never a
# hand-maintained list (so the score can't be gamed by editing the scorecard).
# ---------------------------------------------------------------------------

def cmd_bench_mains(tracked: list[str]) -> set[str]:
    """The standalone cmd/<name> benchmark binaries: a cmd dir whose name contains
    'bench' and that has a tracked .go file. Excludes cmd/fak itself."""
    names: set[str] = set()
    for f in tracked:
        m = re.match(r"^cmd/([^/]+)/[^/]+\.go$", f)
        if not m:
            continue
        name = m.group(1)
        if name == "fak":
            continue
        if "bench" in name:
            names.add(name)
    return names


def fak_bench_verbs(root: Path, tracked: list[str]) -> set[str]:
    """The `fak <verb>` subcommands that ARE benchmarks. Detected structurally:
    a `case "<verb>":` in cmd/fak/main.go whose verb name is benchmark-shaped
    (contains 'bench', or is one of the known non-'bench'-named bench verbs)."""
    main_go = _read(root, "cmd/fak/main.go") or ""
    cases = set(re.findall(r'case\s+"([a-z0-9-]+)":', main_go))
    # Bench verbs whose name lacks the substring 'bench' but which expose a
    # benchmark: `turntax` (the turn-tax A/B), `vcache` (its `bench`/`score`
    # subcommand), and `ablate` (the N-arm self-ablation sweep, the generalization
    # of `fak bench`). Kept as a small named allowlist so the detector stays structural.
    known_non_bench_named = {"turntax", "vcache", "ablate"}
    verbs = {c for c in cases if "bench" in c} | (cases & known_non_bench_named)
    # The catalog verb itself is not a benchmark; drop it.
    verbs.discard("benchmarks")
    return verbs


def registry_names(root: Path) -> set[str]:
    """The names declared in the in-binary registry (internal/benchcatalog). Parsed
    from the Go literal `Name: "<x>"` fields  -  cheap, deterministic, no Go build."""
    src = _read(root, REGISTRY_GO)
    if not src:
        return set()
    return set(re.findall(r'Name:\s*"([a-z0-9-]+)"', src))


def registry_summaries(root: Path) -> int:
    src = _read(root, REGISTRY_GO)
    if not src:
        return 0
    return len(re.findall(r'Summary:\s*"', src))


# ---------------------------------------------------------------------------
# KPIs. Each returns (passed, debt, detail). debt is the count of concrete defects
# this KPI surfaces (0 when passed). detail is a one-line human reason.
# ---------------------------------------------------------------------------

def kpi_catalog_verb(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    src = _read(root, "cmd/fak/main.go") or ""
    has_dispatch = 'case "benchmarks":' in src
    has_file = CATALOG_VERB_GO in tracked
    ok = has_dispatch and has_file
    if ok:
        return True, 0, "fak benchmarks catalog verb is wired and tracked"
    return False, 1, "no `fak benchmarks` catalog verb that LISTS benchmarks (must add a list/describe/run dispatcher)"


def kpi_catalog_registry(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    if REGISTRY_GO in tracked and registry_names(root):
        return True, 0, f"{REGISTRY_GO} registry present ({len(registry_names(root))} entries)"
    return False, 1, f"no in-binary registry at {REGISTRY_GO} backing the catalog (the list can drift from the tree)"


def kpi_registry_covers_tree(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    want = cmd_bench_mains(tracked) | fak_bench_verbs(root, tracked)
    have = registry_names(root)
    if not have:
        return False, max(1, len(want)), "registry empty  -  every benchmark surface is unlisted"
    missing = sorted(want - have)
    orphan = sorted(have - want)
    debt = len(missing) + len(orphan)
    if debt == 0:
        return True, 0, f"registry covers all {len(want)} surfaces one-to-one"
    bits = []
    if missing:
        bits.append(f"{len(missing)} unlisted: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}")
    if orphan:
        bits.append(f"{len(orphan)} dead entries: {', '.join(orphan[:6])}")
    return False, debt, "; ".join(bits)


def kpi_offline_set(root: Path) -> tuple[bool, int, str]:
    src = _read(root, REGISTRY_GO) or ""
    if "NeedNone" in src and "Offline" in src:
        n = src.count("Need: NeedNone")
        return True, 0, f"registry marks the offline set ({n} zero-asset benchmarks)"
    return False, 1, "registry does not classify cold-start need (no offline set a newcomer can filter to)"


def kpi_webbench_runs_clean(root: Path) -> tuple[bool, int, str]:
    src = _read(root, WEBBENCH_GO) or ""
    # The defect: describe exits with "--dataset is required". The fix: a fallback
    # to a committed sample when --dataset is empty.
    desc = ""
    m = re.search(r"func cmdWebbenchDescribe\(.*?\n\}", src, re.S)
    if m:
        desc = m.group(0)
    requires = "--dataset is required" in desc
    has_fallback = ("webbenchSampleDataset" in desc) or ("committed sample" in desc.lower())
    if has_fallback and not requires:
        return True, 0, "fak webbench describe falls back to a committed sample (runnable with zero args)"
    return False, 1, "`fak webbench describe` requires --dataset despite its RUNNABLE-NOW claim (add an offline fallback)"


def kpi_swebench_runs_clean(root: Path) -> tuple[bool, int, str]:
    # Same cold-start contract as webbench, the other RUNNABLE-NOW verb: the
    # registry marks `fak swebench describe` Need=offline / Run="fak swebench
    # describe", so it MUST produce a number with zero args. The defect: describe
    # routes through loadSwebenchSource, which errors ("pass --difficulty FILE or
    # --dataset FILE") when neither flag nor the FAK_SWEBENCH_* env is set. The
    # fix: a fallback to the committed difficulty sample when describe gets no
    # source. Score both pieces: describe's body must route to the fallback marker
    # and the named sample must actually exist in the tree.
    src = _read(root, SWEBENCH_GO) or ""
    desc = ""
    m = re.search(r"func cmdSwebenchDescribe\(.*?\n\}", src, re.S)
    if m:
        desc = m.group(0)
    sample = ""
    sm = re.search(r'swebenchSampleDifficulty\s*=\s*"([^"]+)"', src)
    if sm:
        sample = sm.group(1)
    has_fallback = "swebenchSampleDifficulty" in desc
    if desc and has_fallback and sample and (root / sample).exists():
        return True, 0, "fak swebench describe falls back to a committed sample (runnable with zero args)"
    if desc and has_fallback and sample:
        return False, 1, f"`fak swebench describe` fallback sample {sample} is missing"
    return False, 1, "`fak swebench describe` requires --difficulty/--dataset despite its RUNNABLE-NOW claim (add an offline fallback)"


def kpi_no_spurious_paths(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    """A cmd/*bench* default flag pointing at a `fak/experiments` or `fak/testdata`
    path is the doubled-module-root bug: the real path has no `fak/` prefix, so the
    bench fails to find committed assets and errors cold."""
    bad: list[str] = []
    for f in tracked:
        if not re.match(r"^cmd/[^/]*bench[^/]*/.*\.go$", f):
            continue
        src = _read(root, f) or ""
        for mm in re.finditer(r'flag\.String\([^,]+,\s*"(fak/(?:experiments|testdata)[^"]*)"', src):
            bad.append(f"{f}: {mm.group(1)}")
    if not bad:
        return True, 0, "no spurious fak/-prefixed default paths in cmd/*bench*"
    return False, len(bad), f"{len(bad)} spurious fak/ default path(s): {bad[0]}"


def kpi_out_flag_uniform(root: Path) -> tuple[bool, int, str]:
    """The registry documents a -out/--out for the benches that write a report, so a
    developer learns one 'where did my result go' convention. We score the registry's
    own consistency: every entry with a Flags list mentioning output uses '-out'/'--out'."""
    src = _read(root, REGISTRY_GO) or ""
    # crude but deterministic: count flag entries that look like an output flag
    out_dash = len(re.findall(r'"-out ', src)) + len(re.findall(r'"--out ', src))
    odd = len(re.findall(r'"-o ', src)) + len(re.findall(r'"-output ', src))
    if out_dash >= 1 and odd == 0:
        return True, 0, "output flag is documented uniformly as -out/--out in the registry"
    if odd > 0:
        return False, 1, f"registry documents {odd} non-uniform output flag spelling(s) (-o/-output) alongside -out"
    return False, 1, "registry documents no shared output convention"


def kpi_quant_polarity(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    """The -quant bool must not flip default polarity between benches: false in some,
    true in others, means the same flag does the opposite thing by default. This is a
    real source defect we surface (it is a LEARN-axis trap), counted as ONE debt since
    the fix is a single convention decision."""
    defaults: dict[str, list[str]] = {"true": [], "false": []}
    for f in tracked:
        if not re.match(r"^cmd/[^/]*bench[^/]*/.*\.go$", f):
            continue
        src = _read(root, f) or ""
        for mm in re.finditer(r'flag\.Bool\("quant",\s*(true|false)', src):
            defaults[mm.group(1)].append(f.split("/")[1])
    if defaults["true"] and defaults["false"]:
        return False, 1, (f"-quant default polarity is split: true in {defaults['true']}, "
                          f"false in {defaults['false']}")
    return True, 0, "-quant default polarity is consistent across benches"


def kpi_provenance_doc(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    if AUTHORITY_DOC in tracked:
        return True, 0, f"{AUTHORITY_DOC} present (every number traces to it)"
    return False, 1, f"no {AUTHORITY_DOC}  -  numbers have no provenance authority"


def kpi_summary_self_describes(root: Path) -> tuple[bool, int, str]:
    n = registry_summaries(root)
    want = len(registry_names(root))
    if want and n >= want:
        return True, 0, f"all {want} registry rows carry a one-line 'what this number means'"
    if want == 0:
        return False, 1, "no registry to self-describe"
    return False, want - n, f"{want - n} registry rows have no Summary (a number with no inline meaning)"


def kpi_bench_cli_path_ok(root: Path) -> tuple[bool, int, str]:
    src = _read(root, BENCH_CLI)
    if src is None:
        return False, 1, f"{BENCH_CLI} (the result-query CLI) is missing"
    # The doubled-root bug: BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark".
    m = re.search(r'BENCHMARK_DIR\s*=\s*ROOT\s*/\s*"([^"]+)"', src)
    if m and m.group(1) == "fak":
        return False, 1, f'{BENCH_CLI}: BENCHMARK_DIR has a spurious "fak/" prefix (path will not resolve from repo root)'
    # Confirm the path it points at exists.
    paths = re.findall(r'ROOT\s*((?:/\s*"[^"]+"\s*)+)', src)
    if paths:
        segs = re.findall(r'"([^"]+)"', paths[0])
        target = root.joinpath(*segs) if segs else root
        if segs and segs[0] == "fak":
            return False, 1, f'{BENCH_CLI}: BENCHMARK_DIR starts with "fak/" (doubled module root)'
        if not target.exists():
            return False, 1, f"{BENCH_CLI}: BENCHMARK_DIR {'/'.join(segs)} does not exist"
    return True, 0, f"{BENCH_CLI} BENCHMARK_DIR resolves from the repo root"


def kpi_compare_documented(root: Path, tracked: list[str]) -> tuple[bool, int, str]:
    # bench_cli.py documents list/compare; or a Makefile target chains run+record.
    cli = _read(root, BENCH_CLI) or ""
    has_compare = "compare" in cli and "list" in cli
    if has_compare:
        return True, 0, f"{BENCH_CLI} documents the list/compare loop"
    return False, 1, "no documented run->record->compare loop (bench_cli list/compare or a Makefile target)"


KPI_GROUP = {
    "catalog_verb": "discover",
    "catalog_registry": "discover",
    "registry_covers_tree": "discover",
    "offline_set": "coldstart",
    "webbench_runs_clean": "coldstart",
    "swebench_runs_clean": "coldstart",
    "no_spurious_paths": "coldstart",
    "out_flag_uniform": "learn",
    "quant_polarity_consistent": "learn",
    "provenance_doc": "read",
    "summary_self_describes": "read",
    "bench_cli_path_ok": "compare",
    "compare_documented": "compare",
}


def gather(root: Path) -> list[dict[str, Any]]:
    tracked = _tracked(root)
    results = [
        ("catalog_verb", *kpi_catalog_verb(root, tracked)),
        ("catalog_registry", *kpi_catalog_registry(root, tracked)),
        ("registry_covers_tree", *kpi_registry_covers_tree(root, tracked)),
        ("offline_set", *kpi_offline_set(root)),
        ("webbench_runs_clean", *kpi_webbench_runs_clean(root)),
        ("swebench_runs_clean", *kpi_swebench_runs_clean(root)),
        ("no_spurious_paths", *kpi_no_spurious_paths(root, tracked)),
        ("out_flag_uniform", *kpi_out_flag_uniform(root)),
        ("quant_polarity_consistent", *kpi_quant_polarity(root, tracked)),
        ("provenance_doc", *kpi_provenance_doc(root, tracked)),
        ("summary_self_describes", *kpi_summary_self_describes(root)),
        ("bench_cli_path_ok", *kpi_bench_cli_path_ok(root)),
        ("compare_documented", *kpi_compare_documented(root, tracked)),
    ]
    kpis = []
    for name, passed, debt, detail in results:
        kpis.append({
            "name": name,
            "group": KPI_GROUP[name],
            "passed": bool(passed),
            "debt": int(debt),
            "detail": detail,
        })
    return kpis


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def build_payload(workspace: str, kpis: list[dict[str, Any]], error: str = "") -> dict[str, Any]:
    debt = sum(k["debt"] for k in kpis)
    debt_by_group = {g: 0 for g in GROUPS}
    passed_by_group = {g: [0, 0] for g in GROUPS}  # [passed, total]
    for k in kpis:
        debt_by_group[k["group"]] += k["debt"]
        passed_by_group[k["group"]][1] += 1
        if k["passed"]:
            passed_by_group[k["group"]][0] += 1
    # Score: each group contributes its weight scaled by its pass fraction.
    score = 0.0
    for g in GROUPS:
        p, t = passed_by_group[g]
        if t:
            score += GROUP_WEIGHT[g] * (p / t)
    score = round(score, 1)
    return {
        "schema": SCHEMA,
        "workspace": workspace,
        "error": error,
        "ok": (not error) and debt == 0,
        "corpus": {
            "bench_dx_debt": debt,
            "score": score,
            "grade": grade_letter(score),
            "debt_by_group": debt_by_group,
        },
        "kpis": kpis,
    }


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / "go.mod").exists():
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not the fak repo root at {root} (no go.mod)")
    return build_payload(workspace=str(root), kpis=gather(root))


def render(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"error: {payload['error']}"
    c = payload["corpus"]
    lines = [
        "Bench-DX scorecard  -  the benchmarking developer experience",
        f"  score {c['score']}/100  grade {c['grade']}   bench-DX-debt {c['bench_dx_debt']}",
        "",
        "  by step:",
    ]
    for g in GROUPS:
        lines.append(f"    {g:<10} debt {c['debt_by_group'][g]}   (weight {GROUP_WEIGHT[g]})")
    lines.append("")
    lines.append("  KPIs (X = a defect to retire by ADDING the affordance):")
    for k in payload["kpis"]:
        mark = "ok" if k["passed"] else "X"
        extra = "" if k["passed"] else f"  [+{k['debt']}]"
        lines.append(f"    {mark} {k['name']:<26} {k['detail']}{extra}")
    if c["bench_dx_debt"] == 0:
        lines.append("\n  No bench-DX defects: a developer can find, start, learn, read, and compare a benchmark.")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("bench_dx_debt", 0), cur.get("bench_dx_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "inf (zero)" if cd == 0 else f"{bd / cd:.1f}x"
    lines = [
        f"bench-DX-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:         {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<10} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x bench-DX-debt reduction achieved ({bd} -> {cd}; target <= {target}).")
    else:
        lines.append(f"VERDICT: not yet 2x  -  need bench-DX-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], stamp: str | None) -> str:
    c = payload["corpus"]
    head = stamp or ""
    lines = [
        "---",
        'title: "Bench-DX scorecard  -  the benchmarking developer experience"',
        'description: "Deterministic score of how hard it is for a developer to discover, cold-start, learn, read, and compare a fak benchmark. Auto-generated; re-run tools/bench_dx_scorecard.py."',
        "---",
        "",
        f"# Bench-DX scorecard {head}".rstrip(),
        "",
        "_Auto-generated by `tools/bench_dx_scorecard.py`. Do not hand-edit; re-run the tool._",
        "",
        f"**score {c['score']}/100  -  grade {c['grade']}  -  bench-DX-debt {c['bench_dx_debt']}**",
        "",
        "The benchmarking developer experience, scored as the five steps a developer walks:",
        "discover what exists, cold-start one, learn one flag set, read a number, compare results.",
        "",
        "| step | debt | weight |",
        "|---|---:|---:|",
    ]
    for g in GROUPS:
        lines.append(f"| {g} | {c['debt_by_group'][g]} | {GROUP_WEIGHT[g]} |")
    lines += ["", "| KPI | step | status | detail |", "|---|---|:---:|---|"]
    for k in payload["kpis"]:
        mark = "[x]" if k["passed"] else f"[ ] +{k['debt']}"
        lines.append(f"| `{k['name']}` | {k['group']} | {mark} | {k['detail']} |")
    return "\n".join(lines) + "\n"



# The snapshot stamp lives in the H1 ("# Bench-DX scorecard <stamp>"); --check-doc re-renders
# with the SAME stamp so a date never causes spurious drift — only content drift fails.
_SNAPSHOT_STAMP_RE = re.compile(r"^# Bench-DX scorecard\s+(?P<stamp>\S+)\s*$", re.MULTILINE)


def snapshot_stamp(text: str) -> str:
    m = _SNAPSHOT_STAMP_RE.search(text)
    return m.group("stamp").strip() if m else ""


def check_markdown_doc(workspace: Path, payload: dict) -> dict:
    """Re-derive the snapshot markdown using the committed doc's own stamp and compare. A
    mismatch means the committed docs/BENCH-DX-SCORECARD.md drifted from the live score."""
    path = workspace / GENERATED_SNAPSHOT
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "doc": GENERATED_SNAPSHOT, "stamp": "", "reason": f"read {GENERATED_SNAPSHOT}: {exc}", "diff": []}
    stamp = snapshot_stamp(actual)
    if not stamp:
        return {"ok": False, "doc": GENERATED_SNAPSHOT, "stamp": "", "reason": "snapshot H1 stamp missing", "diff": []}
    expected = render_markdown(payload, stamp=stamp)
    if actual.rstrip() == expected.rstrip():
        return {"ok": True, "doc": GENERATED_SNAPSHOT, "stamp": stamp,
                "reason": f"{GENERATED_SNAPSHOT} matches the live score (stamp {stamp})", "diff": []}
    diff = list(difflib.unified_diff(actual.splitlines(), expected.splitlines(),
                                     fromfile=GENERATED_SNAPSHOT, tofile=f"{GENERATED_SNAPSHOT} (generated)", lineterm=""))
    return {"ok": False, "doc": GENERATED_SNAPSHOT, "stamp": stamp,
            "reason": f"{GENERATED_SNAPSHOT} is stale; regenerate with --markdown --stamp {stamp}", "diff": diff}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bench-DX scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the bench-DX-debt delta vs a prior baseline JSON")
    ap.add_argument("--check-doc", action="store_true",
                    help=f"fail if {GENERATED_SNAPSHOT} drifted from the live score (CI gate)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.check_doc:
        chk = check_markdown_doc(workspace, payload)
        print(chk["reason"])
        if not chk["ok"]:
            for line in chk["diff"][:40]:
                print(line)
            return 1
        return 0

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
