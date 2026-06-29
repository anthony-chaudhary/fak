#!/usr/bin/env python3
"""Release-readiness scorecard — the measuring stick for "can fak release at agentic speed?"

The sibling scorecards grade attractiveness (agent_readiness), code shape
(code_quality), prose (doc_appeal), and the competitive story (industry). None of
them grade the capability a fast-moving kernel lives or dies on: when the trunk
moves hundreds of commits a day, can the project **cut, validate, publish, and roll
back a release at the same speed** — or does `@latest` rot 1900 commits behind HEAD
because the cut is a hand-driven 7-step ritual? That used to be a vibe ("we have a
release skill, we're fine"). This is the number.

It scores the git-tracked tree + live release signals on mechanical KPIs in four
bands — the lifecycle a release walks — folds them into a weighted score and an A-F
grade, and counts **release-debt**: the total of concrete, re-derivable defects that
keep fak from releasing at agentic speed. Each is a defect you fix by *adding the
missing release affordance* (an automated cut, a discoverable verb, a post-publish
gate, a signed artifact), not by writing more prose.

  DISCOVER   — an agent can FIND and INVOKE the release path
    fak_release_verb     `fak release` is a dispatched subcommand (parsed live from
                         cmd/fak/main.go) — the binary, the agent's front door, knows
                         how to release; not Python-tools-only
    agents_md_release    AGENTS.md (the agent contract) documents the release path
    llms_release         llms.txt (the agent doc-map) points at the release path
    staleness_verb       `fak release-staleness` exists (the @latest lag signal)

  AUTOMATE   — the MACHINE cuts on green, not a human
    cadence_auto_cut     release-cadence.yml can cut on a scheduled tick (not
                         dry-run-only) — the root-cause lever for staleness
    staleness_wired      the staleness signal is wired into a make target / CI / loop,
                         not a verb nobody runs
    not_very_stale       @latest is NOT VERY_STALE vs HEAD (commits+days behind)
    fresh                @latest is FRESH vs HEAD (the agentic-speed end state)

  VALIDATE   — the cut is gated and the publish is verified
    release_tools_tested the release helpers carry unit tests (decide/cut/tag/publish)
    post_publish_verify  something verifies a published release (go install @vX.Y.Z,
                         binary-runs, sha256 match, container oci-tag match)
    lock_present         a single-writer release lock serializes concurrent cuts
    gotchas_bounded      the documented chicken-egg gotcha count is small (<=1)

  TRUST      — stable anchors, signed artifacts, rollback
    stable_exercised     at least one stable/* rollback anchor tag exists
    artifacts_signed     release artifacts carry signing / provenance (cosign/SLSA),
                         not just sha256 sidecars
    arm64_shipped        the linux/arm64 leg the workflow declares first-class has
                         actually shipped as a release asset
    artifacts_present    recent releases carry the cross-platform archives + checksums

The headline metric is **release-debt**: the count of concrete HARD defects above.
Driving release-debt to zero means fak can cut a release as fast as it writes code,
validate it before anyone trusts it, and roll it back if it's bad — release at
agentic speed. The companion epic (#1354) retires the worst-first defect by adding
the missing release affordance and re-runs this to prove the drop. It folds into the
unified scorecard_control_pane alongside the other inward sticks.

Read-only: it shells `git`, reads tracked files, and (best-effort) `gh`. Network/gh
failures degrade gracefully — a KPI that can't be witnessed is scored conservatively
(absent), never silently passed.

Usage:
  python tools/release_readiness_scorecard.py                 # human render
  python tools/release_readiness_scorecard.py --json          # control-pane payload
  python tools/release_readiness_scorecard.py --markdown --stamp 2026-06-29 > docs/RELEASE-READINESS-SCORECARD.md
  python tools/release_readiness_scorecard.py --check         # exit non-zero on HARD debt
  python tools/release_readiness_scorecard.py --compare baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "fak-release-readiness/1"

# VERY_STALE thresholds mirror internal/releasestale.DefaultThresholds so this
# scorecard's verdict agrees with `fak release-staleness`.
STALE_COMMITS = 20
STALE_DAYS = 14.0
VERY_STALE_COMMITS = 100
VERY_STALE_DAYS = 45.0


def _run(args: list[str], cwd: Path, timeout: float = 30.0) -> tuple[int, str]:
    """Run a command; return (returncode, stdout). Never raises on failure."""
    try:
        p = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout or "")
    except Exception:
        return 1, ""


def repo_root(start: Path) -> Path:
    rc, out = _run(["git", "rev-parse", "--show-toplevel"], start)
    if rc == 0 and out.strip():
        return Path(out.strip())
    return start


def _read(root: Path, rel: str) -> str:
    p = root / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Fact gathering — every fact re-derived from git/tree/gh, not hand-entered.
# ---------------------------------------------------------------------------


def gather(root: Path) -> dict:
    f: dict = {}

    # --- DISCOVER ---
    main_go = _read(root, "cmd/fak/main.go")
    # A dispatched verb appears as `case "release":` (the dispatch switch). We only
    # count a real `release` case, not `release-staleness`.
    f["fak_release_verb"] = bool(re.search(r'case\s+"release"\s*:', main_go))
    f["staleness_verb"] = bool(re.search(r'case\s+"release-staleness"\s*:', main_go))

    agents = _read(root, "AGENTS.md").lower()
    # Require a genuine release SECTION, not an incidental "released binary" mention.
    f["agents_md_release"] = bool(
        re.search(r"#+\s*releas", agents) or "cut a release" in agents or "/release" in agents
    )

    llms = _read(root, "llms.txt").lower()
    f["llms_release"] = (
        "skills/release" in llms
        or "release skill" in llms
        or "docs/releases" in llms
        or "how to release" in llms
        or "cut a release" in llms
    )

    # --- AUTOMATE ---
    cadence = _read(root, ".github/workflows/release-cadence.yml")
    cl = cadence.lower()
    # Auto-cut is WIRED when the cadence references the FAK_AUTO_RELEASE switch / a
    # named auto-cut path AND it is not a pure dry-run-only cadence. We do NOT infer
    # it from a cross-line regex (a "stop after plan" guard and a dispatch-only execute
    # are different steps; a DOTALL match across them falsely credits the capability).
    f["cadence_auto_cut"] = ("fak_auto_release" in cl or "auto-cut" in cl) and not (
        "stop after the plan" in cl and "fak_auto_release" not in cl and "auto-cut" not in cl
    )
    # Auto-cut is DEFAULT-ON (the strong posture) when a scheduled tick executes
    # UNLESS the kill switch is set — i.e. the cadence says auto-cut is default-on and
    # the arm logic tests `!= "0"` rather than `== "1"`. This is what "releases go
    # green automatically when the gates pass" means, with a kill switch not an arm.
    f["cadence_auto_cut_default_on"] = f["cadence_auto_cut"] and (
        "default-on" in cl and '!= "0"' in cadence
    )

    # staleness wired: a make target or any workflow references release-staleness.
    makefile = _read(root, "Makefile")
    wired = "release-staleness" in makefile or "release_staleness" in makefile
    if not wired:
        wf_dir = root / ".github" / "workflows"
        if wf_dir.is_dir():
            for wf in wf_dir.glob("*.yml"):
                try:
                    if "release-staleness" in wf.read_text(encoding="utf-8", errors="replace"):
                        wired = True
                        break
                except Exception:
                    pass
    f["staleness_wired"] = wired

    # staleness verdict, computed exactly like internal/releasestale.Gather.
    f.update(_staleness_facts(root))

    # --- VALIDATE ---
    tools_dir = root / "tools"
    tested = 0
    for stem in ("release_decide", "release_cut", "release_tag", "release_publish"):
        if (tools_dir / f"{stem}_test.py").is_file():
            tested += 1
    f["release_tools_tested_count"] = tested
    f["release_tools_tested"] = tested >= 3

    # post-publish verify: a workflow/tool that re-checks a published release.
    artifacts_wf = _read(root, ".github/workflows/release-artifacts.yml")
    f["post_publish_verify"] = (
        "release-verify" in artifacts_wf.lower()
        or "go install" in artifacts_wf
        or (tools_dir / "release_verify.py").is_file()
        or "verify-release" in artifacts_wf.lower()
    )

    f["lock_present"] = (tools_dir / "release_lock.py").is_file()

    # gotcha count: the skill's "Notes on this repo's release machinery" enumerates
    # ⚠ corrections. Count the ⚠ markers in the skill as the friction proxy.
    skill = _read(root, ".claude/skills/release/SKILL.md")
    f["gotcha_count"] = skill.count("⚠")
    f["gotchas_bounded"] = f["gotcha_count"] <= 1

    # --- TRUST ---
    rc, out = _run(["git", "tag", "--list", "stable/*"], root)
    f["stable_tags"] = [t for t in out.splitlines() if t.strip()]
    f["stable_exercised"] = len(f["stable_tags"]) > 0

    f["artifacts_signed"] = (
        "cosign" in artifacts_wf.lower()
        or "attest-build-provenance" in artifacts_wf.lower()
        or "slsa" in artifacts_wf.lower()
        or "sigstore" in artifacts_wf.lower()
    )

    # arm64 shipped + artifacts present: best-effort via gh on the latest release.
    f.update(_artifact_facts(root))

    return f


def _staleness_facts(root: Path) -> dict:
    out: dict = {
        "latest_tag": "",
        "commits_behind": 0,
        "days_behind": 0.0,
        "staleness_verdict": "UNKNOWN",
    }
    rc, tags = _run(
        ["git", "tag", "--merged", "HEAD", "--list", "v*", "--sort=-v:refname"], root
    )
    merged = [t for t in tags.splitlines() if t.strip()]
    if not merged:
        out["staleness_verdict"] = "NO_TAG"
        return out
    latest = merged[0]
    out["latest_tag"] = latest
    rc, cnt = _run(["git", "rev-list", "--count", f"{latest}..HEAD"], root)
    try:
        out["commits_behind"] = int(cnt.strip() or "0")
    except ValueError:
        out["commits_behind"] = 0
    rc, tt = _run(["git", "log", "-1", "--format=%ct", latest], root)
    rc2, ht = _run(["git", "log", "-1", "--format=%ct", "HEAD"], root)
    try:
        out["days_behind"] = round((int(ht.strip()) - int(tt.strip())) / 86400.0, 1)
    except (ValueError, ZeroDivisionError):
        out["days_behind"] = 0.0

    c, d = out["commits_behind"], out["days_behind"]
    if c >= VERY_STALE_COMMITS or d >= VERY_STALE_DAYS:
        out["staleness_verdict"] = "VERY_STALE"
    elif c >= STALE_COMMITS or d >= STALE_DAYS:
        out["staleness_verdict"] = "STALE"
    else:
        out["staleness_verdict"] = "FRESH"
    return out


def _artifact_facts(root: Path) -> dict:
    """Best-effort: read the latest release's assets via gh. Offline -> unknown."""
    out = {"latest_release_assets": [], "arm64_shipped": False, "artifacts_present": False, "gh_reachable": False}
    rc, latest = _run(["gh", "release", "list", "--limit", "1", "--json", "tagName", "--jq", ".[0].tagName"], root, timeout=20)
    tag = latest.strip()
    if rc != 0 or not tag:
        return out
    out["gh_reachable"] = True
    rc, names = _run(
        ["gh", "release", "view", tag, "--json", "assets", "--jq", ".assets[].name"], root, timeout=20
    )
    assets = [n for n in names.splitlines() if n.strip()]
    out["latest_release_assets"] = assets
    out["arm64_shipped"] = any("arm64" in a and "linux" in a for a in assets)
    out["artifacts_present"] = (
        any(a.endswith(".tar.gz") or a.endswith(".zip") for a in assets)
        and any(a.endswith(".sha256") or "SHA256" in a for a in assets)
    )
    return out


# ---------------------------------------------------------------------------
# KPI scoring
# ---------------------------------------------------------------------------

# Each KPI: (key, band, weight, lambda(facts)->bool|None, label, fix)
# A None result means "could not witness" -> scored as a miss (conservative), but
# NOT counted toward HARD release-debt (we don't punish an offline gh probe).
KPIS = [
    ("fak_release_verb", "discover", 3, lambda f: f["fak_release_verb"],
     "`fak release` is a dispatched verb", "Add a `fak release` subcommand (#1356)"),
    ("agents_md_release", "discover", 2, lambda f: f["agents_md_release"],
     "AGENTS.md documents the release path", "Add a Releasing section to AGENTS.md (#1357)"),
    ("llms_release", "discover", 2, lambda f: f["llms_release"],
     "llms.txt points at the release path", "Add a release-path entry to llms.txt (#1357)"),
    ("staleness_verb", "discover", 1, lambda f: f["staleness_verb"],
     "`fak release-staleness` exists", "Keep the staleness verb dispatched"),

    ("cadence_auto_cut", "automate", 3, lambda f: f["cadence_auto_cut"],
     "cadence can cut on a scheduled tick", "Add guarded auto-cut to release-cadence.yml (#1355)"),
    ("cadence_auto_cut_default_on", "automate", 2, lambda f: f["cadence_auto_cut_default_on"],
     "scheduled auto-cut is DEFAULT-ON (kill switch, not arm)", "Make auto-cut default-on with a FAK_AUTO_RELEASE=0 kill switch (#1355)"),
    ("staleness_wired", "automate", 2, lambda f: f["staleness_wired"],
     "staleness signal wired into make/CI", "Wire `fak release-staleness --check` into a target/CI (#1367)"),
    ("not_very_stale", "automate", 2, lambda f: f["staleness_verdict"] not in ("VERY_STALE",) and f["staleness_verdict"] != "UNKNOWN",
     "@latest is not VERY_STALE", "Cut a release; automate the cadence (#1355)"),
    ("fresh", "automate", 2, lambda f: f["staleness_verdict"] == "FRESH",
     "@latest is FRESH vs HEAD", "Cut on green at agentic cadence (#1355)"),

    ("release_tools_tested", "validate", 2, lambda f: f["release_tools_tested"],
     "release helpers carry unit tests", "Add tests for decide/cut/tag/publish"),
    ("post_publish_verify", "validate", 3, lambda f: f["post_publish_verify"],
     "a published release is verified", "Add post-publish verification (#1369)"),
    ("lock_present", "validate", 1, lambda f: f["lock_present"],
     "single-writer release lock present", "Keep release_lock.py"),
    ("gotchas_bounded", "validate", 2, lambda f: f["gotchas_bounded"],
     "documented gotcha count <= 1", "Eliminate the chicken-egg gotchas (#1368)"),

    ("stable_exercised", "trust", 2, lambda f: f["stable_exercised"],
     "a stable/* rollback anchor exists", "Cut the first stable/* tag (#1370)"),
    ("artifacts_signed", "trust", 2, lambda f: f["artifacts_signed"],
     "artifacts carry signing/provenance", "Sign artifacts with cosign/SLSA (#1372)"),
    ("arm64_shipped", "trust", 1, lambda f: (f["arm64_shipped"] if f["gh_reachable"] else None),
     "linux/arm64 leg actually shipped", "Ship the arm64 asset (#1371)"),
    ("artifacts_present", "trust", 1, lambda f: (f["artifacts_present"] if f["gh_reachable"] else None),
     "recent release carries archives+checksums", "Fix release-artifacts.yml"),
]

BANDS = ("discover", "automate", "validate", "trust")
BAND_LABEL = {
    "discover": "Discover — an agent can find & invoke the release path",
    "automate": "Automate — the machine cuts on green, not a human",
    "validate": "Validate — the cut is gated and the publish verified",
    "trust": "Trust — stable anchors, signed artifacts, rollback",
}


def score(facts: dict) -> dict:
    rows = []
    band_got = {b: 0.0 for b in BANDS}
    band_max = {b: 0.0 for b in BANDS}
    hard_debt = 0
    soft = 0
    for key, band, weight, fn, label, fix in KPIS:
        try:
            val = fn(facts)
        except Exception:
            val = None
        band_max[band] += weight
        if val is True:
            band_got[band] += weight
            ok = True
            counts = False
        elif val is None:
            # unwitnessed (e.g. offline gh) — neither credit nor HARD debt
            ok = False
            counts = False
            soft += 1
        else:
            ok = False
            counts = True
            hard_debt += 1
        rows.append({
            "key": key, "band": band, "weight": weight, "ok": ok,
            "unwitnessed": val is None, "label": label, "fix": fix,
        })

    total_got = sum(band_got.values())
    total_max = sum(band_max.values()) or 1.0
    composite = round(100.0 * total_got / total_max, 1)
    grade = _grade(composite)
    return {
        "rows": rows,
        "band_got": band_got,
        "band_max": band_max,
        "composite": composite,
        "grade": grade,
        "release_debt": hard_debt,
        "soft": soft,
    }


def _grade(s: float) -> str:
    if s >= 90:
        return "A"
    if s >= 80:
        return "B"
    if s >= 70:
        return "C"
    if s >= 60:
        return "D"
    return "F"


def build_payload(root: Path) -> dict:
    facts = gather(root)
    sc = score(facts)
    debt = sc["release_debt"]
    if debt == 0:
        verdict, finding = "OK", "release at agentic speed: no HARD debt"
    elif debt <= 4:
        verdict, finding = "RELEASE_DEBT", f"{debt} HARD release defect(s) — worst-first via #1354"
    else:
        verdict, finding = "RELEASE_DEBT", f"{debt} HARD release defects — release is hand-driven, not agentic"
    band_scores = {
        b: round(100.0 * sc["band_got"][b] / (sc["band_max"][b] or 1.0), 1) for b in BANDS
    }
    debt_by_band = {b: sum(1 for r in sc["rows"] if r["band"] == b and not r["ok"] and not r["unwitnessed"]) for b in BANDS}
    next_action = "fix the worst-first HARD defect under epic #1354"
    open_rows = [r for r in sc["rows"] if not r["ok"] and not r["unwitnessed"]]
    if open_rows:
        worst = max(open_rows, key=lambda r: r["weight"])
        next_action = worst["fix"]
    return {
        "schema": SCHEMA,
        "ok": debt == 0,
        "verdict": verdict,
        "finding": finding,
        "next_action": next_action,
        "workspace": str(root),
        "score": sc["composite"],
        "grade": sc["grade"],
        "release_debt": debt,
        "soft_signals": sc["soft"],
        "band_scores": band_scores,
        "debt_by_band": debt_by_band,
        "staleness": {
            "latest_tag": facts["latest_tag"],
            "commits_behind": facts["commits_behind"],
            "days_behind": facts["days_behind"],
            "verdict": facts["staleness_verdict"],
        },
        "gotcha_count": facts["gotcha_count"],
        "stable_tags": facts["stable_tags"],
        "rows": sc["rows"],
        "epic": 1354,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render(p: dict) -> str:
    lines = []
    lines.append(f"release-readiness: {p['score']}/100 (grade {p['grade']}) · release-debt {p['release_debt']}")
    st = p["staleness"]
    lines.append(
        f"@latest {st['latest_tag'] or '(none)'}: {st['commits_behind']} commits / {st['days_behind']}d behind HEAD -> {st['verdict']}"
    )
    for b in BANDS:
        lines.append(f"  {b:<9} {p['band_scores'][b]:>5}/100  (debt {p['debt_by_band'][b]})")
    for r in p["rows"]:
        mark = "ok " if r["ok"] else ("·· " if r["unwitnessed"] else "DEBT")
        lines.append(f"    [{mark}] {r['label']}")
    lines.append(f"next: {p['next_action']}")
    return "\n".join(lines)


def render_markdown(p: dict, stamp: str | None) -> str:
    st = p["staleness"]
    head = stamp or ""
    out = []
    out.append("---")
    out.append('title: "fak release-readiness scorecard — release-debt measuring stick"')
    out.append('description: "fak\'s deterministic release-readiness scorecard: KPIs across the release lifecycle — discover, automate, validate, trust — folded into a composite score and the headline release-debt metric, re-derived from git + the tracked tree + live release signals."')
    out.append("---")
    out.append("")
    out.append("# Release-readiness scorecard — can fak release at agentic speed")
    out.append("")
    out.append(f"<!-- release-readiness-scorecard: {head} · process: tools/release_readiness_scorecard.py -->")
    out.append("")
    out.append("The measuring stick for fak's **release velocity under truth**: a kernel that")
    out.append("writes hundreds of commits a day must be able to **cut, validate, publish, and")
    out.append("roll back** a release at the same speed — or `@latest` rots far behind HEAD. Every")
    out.append("number below is re-derived from git + the git-tracked tree + live release signals")
    out.append("by `tools/release_readiness_scorecard.py` — no hand-entry. The headline metric is")
    out.append("**release-debt**: the count of concrete, mechanical defects that keep fak from")
    out.append("releasing at agentic speed. The companion epic (#1354) retires the worst-first")
    out.append("defect by adding the missing release affordance.")
    out.append("")
    out.append("> Regenerate: `python tools/release_readiness_scorecard.py --markdown --stamp DATE > docs/RELEASE-READINESS-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Release-debt (total HARD defects)** | **{p['release_debt']}** |")
    out.append(f"| Composite score | {p['score']}/100 (grade {p['grade']}) |")
    out.append(f"| @latest staleness | {st['latest_tag'] or '(none)'} — {st['commits_behind']} commits / {st['days_behind']}d behind HEAD → **{st['verdict']}** |")
    out.append(f"| Release lifecycle | " + " · ".join(f"{b} {p['band_scores'][b]:.0f}" for b in BANDS) + " |")
    out.append(f"| Documented gotcha count | {p['gotcha_count']} |")
    out.append(f"| Stable rollback anchors | {len(p['stable_tags'])} |")
    out.append(f"| Unwitnessed (offline) signals | {p['soft_signals']} |")
    out.append("")
    out.append("## The release lifecycle, band by band")
    out.append("")
    for b in BANDS:
        out.append(f"### {BAND_LABEL[b]} — {p['band_scores'][b]:.0f}/100 (debt {p['debt_by_band'][b]})")
        out.append("")
        out.append("| KPI | State | Fix if open |")
        out.append("|---|---|---|")
        for r in p["rows"]:
            if r["band"] != b:
                continue
            state = "✅ met" if r["ok"] else ("➖ unwitnessed" if r["unwitnessed"] else "❌ **debt**")
            fix = "" if r["ok"] else r["fix"]
            out.append(f"| {r['label']} | {state} | {fix} |")
        out.append("")
    out.append(f"**Next action:** {p['next_action']}")
    out.append("")
    return "\n".join(out)


def render_compare(baseline: dict, cur: dict) -> str:
    bd, cd = baseline.get("release_debt", "?"), cur["release_debt"]
    bs, cs = baseline.get("score", "?"), cur["score"]
    arrow = "→"
    delta = ""
    try:
        d = cd - bd
        delta = f"  ({'+' if d > 0 else ''}{d} debt)"
    except TypeError:
        pass
    return (
        f"release-readiness: score {bs} {arrow} {cs} · release-debt {bd} {arrow} {cd}{delta}\n"
        f"staleness: {baseline.get('staleness',{}).get('verdict','?')} {arrow} {cur['staleness']['verdict']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Release-readiness scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--check", action="store_true", help="exit non-zero when release-debt > 0")
    ap.add_argument("--compare", default="", metavar="BASELINE.json", help="diff against a saved JSON payload")
    args = ap.parse_args(argv)

    # The markdown body carries arrows/emoji; a Windows console defaults to cp1252 and
    # would crash on redirect. Force UTF-8 so the snapshot regenerates on any host.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    start = Path(args.workspace) if args.workspace else Path.cwd()
    root = repo_root(start)
    payload = build_payload(root)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    if args.check and payload["release_debt"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
