#!/usr/bin/env python3
"""Skill-effectiveness scorecard — is each Claude Code skill BUILT to be effective?

The sibling scorecards grade product surfaces: ``code_quality_scorecard`` grades
the Go module, ``agent_readiness_scorecard`` grades how an AI agent adopts fak,
``doc_appeal_scorecard`` grades a doc's prose. None of them grade the thing that
decides whether the *skill pack itself* pulls its weight: when the model reaches
for a `.claude/skills/<name>/SKILL.md`, does that skill carry the affordances that
make a skill effective — can the model **discover** it (a sharp trigger), **operate**
it safely (a scoped tool surface, references that actually resolve), and **trust**
its output (a proof step, the commit-by-path discipline this shared trunk demands)?
That used to be a vibe ("we have lots of skills"). This is the number.

It reads every git-tracked ``.claude/skills/*/SKILL.md`` and scores nine mechanical
KPIs in four groups, folds them into a weighted score + an A-F grade, and counts
**skill-debt**: the total of concrete, re-derivable affordances a skill is missing.
Each is a defect you fix by *adding the real affordance to the skill* — a sharper
trigger, a resolving reference, a witness step — never by spraying a keyword.

  DISCOVER   — the model can find the skill and tell WHEN to fire it
    description_present  front-matter carries a non-trivial `description` (the
                         trigger surface the model matches the request against)
    trigger_clause       the description states WHEN to use it ("Use when / Use to
                         / Use after …") — without it the model can't tell a near-miss
                         request from a real one
    name_resolves        front-matter `name:` equals the directory name, so `/name`
                         actually invokes this skill (a mismatch is a dead command)

  OPERATE    — the skill can be run safely and its references are real
    refs_resolve         every repo-relative path the skill cites (a `tools/*.py`
                         helper, a `docs/*` link, a sibling `../x/SKILL.md`) exists
                         on disk — a skill pointing at a deleted tool is broken
                         (the ungameable anchor: cross-checked against the tree)
    tools_scoped         a skill that COMMITS declares `allowed-tools` — a high-
                         privilege skill states its tool surface (least privilege)

  TRUST      — the model/operator can trust what the skill ships
    commit_discipline    a skill that commits names the explicit-path discipline
                         (commit-by-path / `-- <paths>` / never `git add -A`) — the
                         shared-trunk safety rule that stops a skill sweeping a peer's
                         staged files
    proof_step           a skill that commits carries a verify / witness / re-measure
                         step, so "it shipped" is grounded in evidence, not narration

  ECONOMY    — advisory cost signals (SOFT — they score but emit no debt)
    anti_gaming   (SOFT) a metric-driving skill carries an explicit anti-gaming /
                         honesty clause (the cheap move to fix is gaming, so it's SOFT)
    context_budget (SOFT) SKILL.md stays under ~300 lines (the /clean-skill smell —
                         a long skill burns per-invocation context)

The headline metric is **skill-debt**: the count of concrete HARD defects above.
Driving it to zero means every skill in the pack is discoverable, safe to operate,
and trustworthy by construction. The companion process — the ``/skill-score`` skill,
an instance of the ``/score-2x`` loop — runs this, retires the worst-first defect by
adding the missing affordance, and re-runs to prove the drop. It folds into the
unified ``scorecard_control_pane`` alongside the other inward sticks.

Deterministic + read-only by construction: it reads the git-tracked skill tree (so
two clones of one commit score identically) and edits nothing. Run from the repo
ROOT::

    python tools/skill_effectiveness_scorecard.py                 # human scorecard
    python tools/skill_effectiveness_scorecard.py --json          # machine payload
    python tools/skill_effectiveness_scorecard.py --markdown      # the committed snapshot body
    python tools/skill_effectiveness_scorecard.py --compare base.json   # prove skill-debt moved
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-skill-effectiveness-scorecard/1"
SKILLS_REL = ".claude/skills"
GENERATED_SNAPSHOT = "docs/SKILL-EFFECTIVENESS-SCORECARD.md"

# ---------------------------------------------------------------------------
# The contract a skill is expected to carry. Each constant is a deliberate,
# named affordance the model (or the operator) reaches for — never a hand-picked
# allow-list where a rule would do. A skill that drops one scores lower, by
# construction.
# ---------------------------------------------------------------------------

GROUPS = ("discover", "operate", "trust", "economy")
KPI_GROUP: dict[str, str] = {
    "description_present": "discover",
    "trigger_clause": "discover",
    "name_resolves": "discover",
    "refs_resolve": "operate",
    "tools_scoped": "operate",
    "commit_discipline": "trust",
    "proof_step": "trust",
    "anti_gaming": "economy",
    "context_budget": "economy",
}
# Nine KPIs. `refs_resolve` carries the most weight — it is the one cross-checked
# against the real tree (a cited helper either exists or it doesn't), so it can't be
# gamed by editing the skill text. The commit-discipline cluster (commit_discipline,
# proof_step, tools_scoped) gates the highest-privilege skills — the ones that touch
# this shared trunk. The two SOFT KPIs still lower the score (a long or anti-gaming-
# free skill grades lower) but emit no debt — the cheap move to fix either is a
# keyword, which is the gaming this refuses. Sum is exactly 1.0; a test asserts the
# sum and that the weight set == the KPI set.
KPI_WEIGHTS: dict[str, float] = {
    # discover (0.34)
    "description_present": 0.12,
    "trigger_clause": 0.12,
    "name_resolves": 0.10,
    # operate (0.30)
    "refs_resolve": 0.20,
    "tools_scoped": 0.10,
    # trust (0.26)
    "commit_discipline": 0.14,
    "proof_step": 0.12,
    # economy (0.10, both SOFT)
    "anti_gaming": 0.05,
    "context_budget": 0.05,
}

CONTEXT_SOFT_MAX = 300  # /clean-skill's smell threshold — a SKILL.md longer than this
                        # tends to read multiple files per invocation and burn context.
DESC_MIN_CHARS = 40     # a description shorter than this can't carry a real trigger.

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
_CODE_RE = re.compile(r"`([^`]+)`")
# A "Use when / Use to / Use after …" clause — the Claude Code trigger convention.
TRIGGER_RE = re.compile(r"\bUse (?:when|to|after|for|during|on|whenever|if|this)\b", re.I)
# A skill that commits to the trunk is unambiguously high-privilege. Match a real
# commit INVOCATION (`git commit -s/-m/-F …`), not a prose mention ("does not
# authorize git commits", "operator handles git commit") — every committing skill
# writes the command with a flag, so the `-` is what separates run from mention.
COMMITS_RE = re.compile(r"git commit\s+-", re.I)
# The explicit-path discipline phrases this shared trunk demands.
PATHDISC_RE = re.compile(
    r"explicit path|by path|commit-by-path|--\s*<|git add -A|pathspec|<paths>|<explicit", re.I)
# A verify / witness / re-measure step — "it shipped" grounded in evidence.
PROOF_RE = re.compile(
    r"\b(prove|re-?measure|verif|witness|re-?run|validate|confirm|assert|audit)\b"
    r"|dos commit-audit", re.I)
# A skill is metric-driving (so it owes an anti-gaming clause) if it talks debt/score.
METRIC_RE = re.compile(r"\b(debt|scorecard|metric|score)\b", re.I)
ANTIGAME_RE = re.compile(
    r"anti-?gam|never (?:weaken|game|fake|silence)|don'?t game|gaming|honest|not yet", re.I)

# Repo-relative path prefixes a skill cites that we resolve against disk. Tokens
# without one of these prefixes are prose, not a reference.
REPO_TOP_DIRS = ("tools/", "docs/", "cmd/", "internal/", "examples/",
                 "experiments/", ".claude/")
# A concrete reference has a real file suffix (or is a sibling SKILL.md) — this keeps
# the resolver precise and skips prose like "the tools/ directory".
_CONCRETE_RE = re.compile(r"\.(py|md|go|json|toml|txt|ya?ml|sh|ps1|cfg|ini)$|/SKILL\.md$", re.I)


# ---------------------------------------------------------------------------
# Small pure helpers (the testable core).
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


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


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (flat frontmatter keys, body). The leading ``---`` YAML block is parsed
    as flat ``key: value`` lines (descriptions here are single-line); the body is
    everything after the closing ``---``. No YAML dependency — deterministic."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n")
    # find the closing fence (the second '---' line)
    close = None
    for i in range(1, len(parts)):
        if parts[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    fm: dict[str, str] = {}
    for ln in parts[1:close]:
        if ln.startswith((" ", "\t")) or ":" not in ln:
            continue  # nested (metadata children) or non-key line
        key, _, val = ln.partition(":")
        fm[key.strip()] = val.strip().strip('"').strip("'")
    body = "\n".join(parts[close + 1:])
    return fm, body


def is_template_slot(tok: str) -> bool:
    """A token that is an illustrative slot, not a concrete reference: an
    <angle-bracket> / {brace} fill-in, a glob, a $var, a version or date placeholder
    (X.Y.Z, YYYY-MM-DD) the doc uses to show the SHAPE of an output filename."""
    return ("<" in tok or ">" in tok or "*" in tok or "{" in tok
            or "X.Y.Z" in tok or "YYYY" in tok or "MM-DD" in tok
            or tok.startswith("$"))


def cited_paths(body: str) -> list[str]:
    """Every concrete repo-relative path a skill body cites — from markdown link
    targets AND inline-code spans. Returns the raw token strings (resolution is the
    impure shell's job). Skips externals, anchors, and template slots."""
    out: list[str] = []
    seen: set[str] = set()

    def consider(tok: str) -> None:
        tok = tok.strip()
        if not tok or tok in seen or is_template_slot(tok):
            return
        # strip a leading ./ and any trailing punctuation a sentence left attached
        cand = tok[2:] if tok.startswith("./") else tok
        cand = cand.rstrip(".,);:")
        if not _CONCRETE_RE.search(cand):
            return
        seen.add(tok)
        out.append(cand)

    for m in _LINK_RE.finditer(body):
        t = m.group("target").strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        consider(t.split("#", 1)[0].split("?", 1)[0])
    for m in _CODE_RE.finditer(body):
        span = m.group(1)
        for tok in re.split(r"\s+", span):
            if tok.lstrip("./").startswith(REPO_TOP_DIRS):
                consider(tok)
    return out


def is_flaggable(ref: str) -> bool:
    """Only resolve a reference we can attribute to the repo: one that carries a
    repo-top-dir prefix (`tools/…`, `docs/…`) or is an explicit relative path
    (`./x`, `../x/SKILL.md`). A BARE filename with no directory (`MEMORY_archive.md`,
    `MEMORY.md`) is ambiguous — an external runtime artifact or an illustrative name —
    so we never flag it as dead; that is the zero-false-positive boundary."""
    return ref.lstrip("./").startswith(REPO_TOP_DIRS) or ref.startswith(("./", "../"))


# ---------------------------------------------------------------------------
# Pure KPI functions over the gathered per-skill facts.
# Each fact dict: {name, fm_name, description, has_allowed_tools, commits,
#                  metric_driving, n_lines, dead_refs:[str], body_has_trigger,
#                  body_has_pathdisc, body_has_proof, body_has_antigame}
# Each KPI returns {kpi, group, score, detail, defects:[str], soft:[str]}.
# ---------------------------------------------------------------------------

def _kpi(name: str, applicable: int, failed: list[str], *, soft: bool = False) -> dict[str, Any]:
    n_app = max(applicable, 0)
    n_fail = len(failed)
    score = 100 if n_app == 0 else _clamp(100 * (n_app - n_fail) / n_app)
    detail = (f"{n_app - n_fail}/{n_app} skill(s) pass" if n_app
              else "n/a (no applicable skill)")
    return {"kpi": name, "group": KPI_GROUP[name], "score": score, "detail": detail,
            "defects": [] if soft else failed, "soft": failed if soft else []}


def kpi_description_present(facts: list[dict]) -> dict[str, Any]:
    failed = [f"{s['name']}: missing or too-short `description` front-matter "
              f"(< {DESC_MIN_CHARS} chars) — the model has no trigger surface to match"
              for s in facts if len(s["description"]) < DESC_MIN_CHARS]
    return _kpi("description_present", len(facts), failed)


def kpi_trigger_clause(facts: list[dict]) -> dict[str, Any]:
    app = [s for s in facts if len(s["description"]) >= DESC_MIN_CHARS]
    failed = [f"{s['name']}: description states WHAT it does but not WHEN — add a "
              "\"Use when / Use to / Use after …\" trigger clause"
              for s in app if not TRIGGER_RE.search(s["description"])]
    return _kpi("trigger_clause", len(app), failed)


def kpi_name_resolves(facts: list[dict]) -> dict[str, Any]:
    failed = [f"{s['name']}: front-matter name `{s['fm_name']}` != directory "
              f"`{s['name']}` — `/{s['name']}` won't resolve to this skill"
              for s in facts if s["fm_name"] and s["fm_name"] != s["name"]]
    failed += [f"{s['name']}: no `name:` in front-matter — the skill has no invocation id"
               for s in facts if not s["fm_name"]]
    return _kpi("name_resolves", len(facts), failed)


def kpi_refs_resolve(facts: list[dict]) -> dict[str, Any]:
    failed: list[str] = []
    for s in facts:
        for ref in s["dead_refs"]:
            failed.append(f"{s['name']}: cites `{ref}` which does not exist on disk "
                          "— a dead reference (deleted/renamed/typo)")
    return _kpi("refs_resolve", len(facts), failed)


def kpi_tools_scoped(facts: list[dict]) -> dict[str, Any]:
    app = [s for s in facts if s["commits"]]
    failed = [f"{s['name']}: commits to the trunk but declares no `allowed-tools` — "
              "a high-privilege skill should state its tool surface (least privilege)"
              for s in app if not s["has_allowed_tools"]]
    return _kpi("tools_scoped", len(app), failed)


def kpi_commit_discipline(facts: list[dict]) -> dict[str, Any]:
    app = [s for s in facts if s["commits"]]
    failed = [f"{s['name']}: commits but never names the explicit-path discipline "
              "(commit-by-path / `-- <paths>` / never `git add -A`) — it can sweep a "
              "peer's staged files on this shared trunk"
              for s in app if not s["body_has_pathdisc"]]
    return _kpi("commit_discipline", len(app), failed)


def kpi_proof_step(facts: list[dict]) -> dict[str, Any]:
    app = [s for s in facts if s["commits"]]
    failed = [f"{s['name']}: commits but carries no verify / witness / re-measure step "
              "— \"it shipped\" would rest on narration, not evidence"
              for s in app if not s["body_has_proof"]]
    return _kpi("proof_step", len(app), failed)


def kpi_anti_gaming(facts: list[dict]) -> dict[str, Any]:
    app = [s for s in facts if s["metric_driving"]]
    soft = [f"{s['name']}: drives a metric but states no anti-gaming / honesty rule "
            "— add the \"never weaken the check to score\" clause"
            for s in app if not s["body_has_antigame"]]
    return _kpi("anti_gaming", len(app), soft, soft=True)


def kpi_context_budget(facts: list[dict]) -> dict[str, Any]:
    soft = [f"{s['name']}: SKILL.md is {s['n_lines']} lines (> {CONTEXT_SOFT_MAX}) "
            "— a long skill burns per-invocation context (see /clean-skill)"
            for s in facts if s["n_lines"] > CONTEXT_SOFT_MAX]
    return _kpi("context_budget", len(facts), soft, soft=True)


KPI_FUNCS = (
    kpi_description_present, kpi_trigger_clause, kpi_name_resolves,
    kpi_refs_resolve, kpi_tools_scoped, kpi_commit_discipline, kpi_proof_step,
    kpi_anti_gaming, kpi_context_budget,
)


# ---------------------------------------------------------------------------
# The fold (pure over the KPI list).
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  n_skills: int, error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    skill_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    score_by_group = {g: 0.0 for g in GROUPS}
    wsum_by_group = {g: 0.0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
        w = KPI_WEIGHTS.get(k["kpi"], 0.0)
        score_by_group[k["group"]] += w * k["score"]
        wsum_by_group[k["group"]] += w
    group_scores = {g: (round(score_by_group[g] / wsum_by_group[g], 1)
                        if wsum_by_group[g] else 0.0) for g in GROUPS}

    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score, "grade": grade, "skill_debt": skill_debt,
        "soft_signals": n_soft, "skills": n_skills,
        "group_scores": group_scores, "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }
    gs = group_scores
    standing = (f"discover {gs['discover']:.0f} · operate {gs['operate']:.0f} "
                f"· trust {gs['trust']:.0f}")
    if skill_debt == 0:
        ok, verdict, finding = True, "OK", "skills_effective"
        reason = (f"skills effective: score {score}/100 (grade {grade}), zero skill-debt "
                  f"across {len(kpis)} KPIs over {n_skills} skills ({standing}; "
                  f"{n_soft} advisory). Every skill is discoverable, safe to operate, "
                  "and trustworthy by construction")
        next_action = ("hold the line; re-run after adding/editing a skill, or harden "
                       "the bar (tighten DESC_MIN_CHARS / promote a SOFT KPI) once saturated")
    else:
        ok, verdict, finding = False, "ACTION", "skill_debt"
        worst = breakdown[0]
        reason = (f"{skill_debt} unit(s) of skill-debt over {n_skills} skills; "
                  f"score {score}/100 (grade {grade}); heaviest: {worst['kpi']} "
                  f"({worst['debt']} defect(s)); standing {standing}")
        next_action = ("retire skill-debt worst-first (see corpus.breakdown + per-KPI "
                       "defects): add the missing trigger clause, resolve the dead "
                       "reference, name the commit-by-path discipline, add the witness "
                       "step, scope the tools; re-run to prove the drop")
    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


# ---------------------------------------------------------------------------
# Disk gathering (the impure shell around the pure KPIs).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _dead_refs(body: str, skill_dir: Path, root: Path) -> list[str]:
    """Resolve every cited concrete path against disk; return the ones that miss.
    A markdown link target resolves relative to the skill dir (so `../x/SKILL.md`
    works); an inline-code path resolves relative to the repo root (so `tools/x.py`
    works). We try BOTH bases and only flag a ref dead when neither resolves."""
    dead: list[str] = []
    seen: set[str] = set()
    for ref in cited_paths(body):
        if ref in seen or not is_flaggable(ref):
            continue
        seen.add(ref)
        rel = ref.lstrip("/")
        from_root = (root / rel)
        from_dir = (skill_dir / ref)
        if not from_root.exists() and not from_dir.exists():
            dead.append(ref)
    return dead


def gather(root: Path) -> tuple[list[dict[str, Any]], int]:
    skills_dir = root / SKILLS_REL
    facts: list[dict[str, Any]] = []
    if skills_dir.is_dir():
        for sk in sorted(skills_dir.iterdir()):
            md = sk / "SKILL.md"
            if not md.is_file():
                continue
            text = _safe_read(md)
            fm, body = split_frontmatter(text)
            facts.append({
                "name": sk.name,
                "fm_name": fm.get("name", ""),
                "description": fm.get("description", ""),
                "has_allowed_tools": "allowed-tools" in fm,
                "commits": bool(COMMITS_RE.search(body)),
                "metric_driving": bool(METRIC_RE.search(body)),
                "n_lines": text.count("\n") + 1 if text else 0,
                "dead_refs": _dead_refs(body, sk, root),
                "body_has_pathdisc": bool(PATHDISC_RE.search(body)),
                "body_has_proof": bool(PROOF_RE.search(body)),
                "body_has_antigame": bool(ANTIGAME_RE.search(body)),
            })
    kpis = [f(facts) for f in KPI_FUNCS]
    return kpis, len(facts)


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    skills_dir = root / SKILLS_REL
    if not skills_dir.is_dir():
        return build_payload(workspace=str(root), kpis=[], n_skills=0,
                             error=f"no skill pack at {skills_dir} — run from the repo ROOT")
    kpis, n = gather(root)
    return build_payload(workspace=str(root), kpis=kpis, n_skills=n)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    lines = [
        f"skill-effectiveness-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· SKILL-DEBT {c.get('skill_debt', 0)} · {c.get('soft_signals', 0)} advisory "
         f"· {c.get('skills', 0)} skills"),
        (f"by group:  discover {gs.get('discover', 0):.0f}  ·  operate {gs.get('operate', 0):.0f}"
         f"  ·  trust {gs.get('trust', 0):.0f}  ·  economy {gs.get('economy', 0):.0f}"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'group':<9} {'kpi':<20} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<9} "
                     f"{b['kpi']:<20} {b['detail']}")
    lines.append("")
    lines.append("skill-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:14]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > 14:
            lines.append(f"      ... and {len(k['defects']) - 14} more")
    if not any_defect:
        lines.append("  (none — zero skill-debt)")
    soft_all = [(k["kpi"], s) for k in payload.get("kpis", []) for s in k["soft"]]
    if soft_all:
        lines.append("")
        lines.append(f"advisory signals ({len(soft_all)}):")
        for kpi, s in soft_all[:14]:
            lines.append(f"      · [{kpi}] {s}")
        if len(soft_all) > 14:
            lines.append(f"      ... and {len(soft_all) - 14} more")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak skill-effectiveness scorecard — is each skill built to be effective"')
    out.append('description: "fak\'s deterministic skill-effectiveness scorecard: nine '
               'mechanical KPIs across discover, operate, trust, and economy, folded into a '
               'composite score and the headline skill-debt metric, re-derived from the '
               'git-tracked .claude/skills tree. It grades whether each Claude Code skill '
               'carries the affordances that make it discoverable, safe to operate, and '
               'trustworthy — a sharp trigger, references that resolve, the commit-by-path '
               'discipline."')
    out.append("---")
    out.append("")
    out.append("# Skill-effectiveness scorecard — is each skill built to be effective")
    out.append("")
    if stamp:
        out.append(f"<!-- skill-effectiveness-scorecard: {stamp} · process: tools/skill_effectiveness_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for the **skill pack itself**. The other "
               "scorecards grade product surfaces; this one asks whether each "
               "`.claude/skills/<name>/SKILL.md` is *built to be effective* — when the model "
               "reaches for it, can it **discover** the skill (a sharp \"Use when …\" "
               "trigger), **operate** it safely (a scoped tool surface, references that "
               "resolve on disk), and **trust** what it ships (a witness step, the "
               "commit-by-path discipline this shared trunk demands)? Every number below is "
               "re-derived from the git-tracked skill tree by "
               "`tools/skill_effectiveness_scorecard.py` — no hand-entry. The headline "
               "metric is **skill-debt**: the count of concrete affordances a skill is "
               "missing. Driving it to zero makes every skill in the pack pull its weight.")
    out.append("")
    out.append("> Regenerate: `python tools/skill_effectiveness_scorecard.py --markdown --stamp DATE > docs/SKILL-EFFECTIVENESS-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Skill-debt (total HARD defects)** | **{c.get('skill_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Skills graded | {c.get('skills', 0)} |")
    out.append(f"| By group | discover {gs.get('discover', 0):.0f} · operate {gs.get('operate', 0):.0f} "
               f"· trust {gs.get('trust', 0):.0f} · economy {gs.get('economy', 0):.0f} |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## The four things that make a skill effective")
    out.append("")
    out.append(f"{len(payload.get('kpis', []))} KPIs, each 0–100, grouped by the job they "
               "gate. `debt` = units of HARD skill-debt. `refs_resolve` is the ungameable "
               "anchor — a cited helper either exists on disk or it doesn't. The "
               "commit-discipline cluster (`commit_discipline`, `proof_step`, "
               "`tools_scoped`) gates only the skills that commit to the trunk. The two "
               "ECONOMY KPIs are SOFT — they lower the score but emit no debt, because the "
               "cheap way to fix either is a keyword, which is gaming.")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Skill-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No skill-debt: every skill is discoverable, safe to operate, and "
                   "trustworthy by construction. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("skill_debt", 0), cur.get("skill_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"skill-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:      {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<9} {gb} -> {gc}")
    target2 = max(0, bd // 2)
    target3 = max(0, (bd + 2) // 3)
    if cd <= target3:
        lines.append(f"VERDICT: ≥3× reduction achieved (skill-debt {bd}->{cd}, target ≤{target3}).")
    elif cd <= target2:
        lines.append(f"VERDICT: ≥2× (not yet 3×) — skill-debt {bd}->{cd}; 3× needs ≤{target3}.")
    else:
        lines.append(f"VERDICT: not yet 2× — need skill-debt ≤{target2} (now {cd}); 3× target ≤{target3}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Skill-effectiveness scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the skill-debt delta vs a prior baseline JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

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
