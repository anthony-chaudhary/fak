#!/usr/bin/env python3
"""Product scorecard — is each fak concept a DURABLE, REAL, USEFUL-TODAY product?

The sibling scorecards grade the tree's *internals*: ``code_quality`` grades the
Go module, ``docs`` grades the doc corpus, ``industry_scorecard`` grades fak's
competitive standing vs the field, ``agent_readiness`` grades whether an agent can
adopt fak. None of them asks the question a *person* asks of a project:

  **Of the concepts fak ships, which are a durable, real, useful-TODAY product —
  something a human can pick up and use this afternoon — and which are still a
  named gap, a research seam, or an overclaim?**

That is this scorecard. The "various items" are fak's product concepts (the ``##``
sections of ``CLAIMS.md`` — the honesty ledger). Each is positioned on three axes a
buyer/operator/developer actually feels:

  REAL         — is it actually built + SHIPPED (not a STUB / SIMULATED seam), and
                 does its CLAIMS tag back the maturity the row claims?
  USEFUL-TODAY — is there a copy-pasteable first command a person runs to USE it,
                 ideally OFFLINE (no GPU, no API key) on a laptop this afternoon?
  DURABLE      — is it PROVEN (a witness test / results doc that exists) and
                 DISCOVERABLE (an entry doc that exists), so it keeps working and a
                 newcomer can learn it — not a one-off demo that rots?

The product-quality **verdict** ladder folds those axes into one honest label:

  durable-product  shipped + OFFLINE first command + witness exists + entry doc exists
  usable-today     shipped + a first command, but it needs a GPU / key / network
  real-not-easy    shipped/real, but NO copy-pasteable command (a subsystem, not a surface)
  honest-stub      a STUB / SIMULATED seam, labeled honestly
  concept-only     a roadmap idea, not built

Every check CROSS-CHECKS the row against the real tree — the CLAIMS.md tag the
concept actually carries, whether the first-command's ``cmd`` dir exists and its
``fak`` verb is documented, whether the witness/entry paths exist on disk. So the
score CANNOT be gamed by editing the data alone; to drop debt you fix the real
thing (correct an overclaim, add a pointer that resolves, write the missing test).

Two numbers are driven, mirroring ``industry_scorecard``:

  PRODUCT-DEBT  honesty/quality defects of the rows that EXIST + coverage gaps
                (concept sections with no row). Drive it toward 0 — the repo-3x
                idiom is "cut it to a third." Folds into ``scorecard_control_pane``
                via ``corpus.product_debt``.
  COVERAGE      of the CLAIMS.md concept sections, how many are positioned at all.

Deterministic + read-only over the data (two clones at one commit score
identically); the only disk writes are the generated doc folder under
``--markdown-dir``. The source of truth is a DIRECTORY of small JSON files so the
category vocabulary and each category's rows evolve independently::

    tools/product_scorecard.data/
      _meta.json        meta + the declared category vocabulary
      rows-*.json       fak's product-concept rows, grouped by category

Run from the repo ROOT::

    python tools/product_scorecard.py                 # human scorecard
    python tools/product_scorecard.py --chart         # at-a-glance ASCII chart of the standing
    python tools/product_scorecard.py --json          # machine payload (control-pane / loop)
    python tools/product_scorecard.py --critical      # the most-critical-areas backlog (what to progress)
    python tools/product_scorecard.py --gaps          # the coverage backlog: concept sections with no row
    python tools/product_scorecard.py --compare base.json    # prove product-debt dropped (3x target)
    python tools/product_scorecard.py --markdown-dir docs/product-scorecard   # regenerate the doc folder
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-product-scorecard/1"
DATA_DIR_REL = "tools/product_scorecard.data"
GENERATED_DOC_DIR = "docs/product-scorecard"
CLAIMS_REL = "CLAIMS.md"
CLI_REF_REL = "docs/cli-reference.md"

# ---------------------------------------------------------------------------
# Closed vocabularies. `category` is DATA-defined (declared in _meta.json) so the
# product map can grow; the honesty vocabularies below ARE the doctrine and fixed.
# ---------------------------------------------------------------------------

MATURITIES = {
    "shipped": "real code on the critical path, reproducible now",
    "simulated": "modeled with labeled stand-in data (no live engine / no meter on the box)",
    "stub": "plumbing present, behavior deferred — labeled",
    "concept": "a roadmap idea, not built",
}
CLAIMS_TAGS = {"SHIPPED", "SIMULATED", "STUB"}
# A maturity must agree with the CLAIMS tag the concept's lead line carries.
MATURITY_TAG = {"shipped": "SHIPPED", "simulated": "SIMULATED", "stub": "STUB"}
AUDIENCES = {"buyer", "platform", "developer", "researcher"}

# The product-quality verdict ladder, best -> worst. The rank doubles as the
# "distance from a durable product" used to order the critical backlog.
VERDICTS = ["durable-product", "usable-today", "real-not-easy", "honest-stub", "concept-only"]
VERDICT_RANK = {v: i for i, v in enumerate(VERDICTS)}

# What KIND of surface a concept is — this is what stops a benchmark from posing
# as a "durable product." Only a `product` surface (something a person runs on
# THEIR own work) can reach durable-product; a `benchmark` you run to see/measure
# tops out at usable-today; a `subsystem`/`seam` is real-not-easy (no surface).
SURFACES = {
    "product": "a surface a person runs on their own work (a CLI verb, a server)",
    "benchmark": "a runnable demo/measurement a person runs to SEE or reproduce a result",
    "subsystem": "an internal mechanism with no direct product surface (proven by tests)",
    "seam": "a frozen ABI seam awaiting a backend/transport before it is usable",
}

# CLAIMS.md `##` sections that are NOT product concepts (so coverage doesn't demand
# a row for them). Everything else is a concept in the catalog.
NON_CONCEPT_SECTIONS = {"what fak is not", "prior-art posture"}

GROUPS = ("well-formed", "honesty", "usefulness", "durability")
KPI_GROUP: dict[str, str] = {
    "well_formed": "well-formed",
    "claim_honest": "honesty",
    "verdict_consistency": "honesty",
    "command_resolves": "usefulness",
    "witnessed": "durability",
    "discoverable": "durability",
}
KPI_WEIGHTS: dict[str, float] = {
    "well_formed": 0.12,
    "claim_honest": 0.22,
    "verdict_consistency": 0.22,
    "command_resolves": 0.14,
    "witnessed": 0.16,
    "discoverable": 0.14,
}
KPI_PENALTY: dict[str, int] = {
    "well_formed": 12,
    "claim_honest": 20,
    "verdict_consistency": 25,
    "command_resolves": 15,
    "witnessed": 15,
    "discoverable": 15,
}
# The composite blends the honesty/quality of the rows that EXIST with how much of
# the concept catalog is even positioned. An incomplete map costs grade.
HONESTY_WEIGHT = 0.60
COVERAGE_WEIGHT = 0.40

REQUIRED_FIELDS = (
    "id", "concept", "category", "surface", "what_you_get", "audience", "maturity",
    "claims_section", "claims_tag", "first_command", "first_command_verb",
    "needs_gpu", "needs_key", "witness_path", "witness", "entry_doc",
    "verdict", "gaps", "durability_note",
)


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


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def norm_section(s: Any) -> str:
    """Normalize a `## header` / claims_section to a comparable key: drop the
    leading hashes, cut at the first parenthesis / em-dash / colon, lowercase."""
    s = (s or "")
    if not isinstance(s, str):
        return ""
    s = s.strip()
    while s.startswith("#"):
        s = s[1:]
    s = s.strip()
    for sep in ("(", "—", ":"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    return " ".join(s.lower().split())


def section_match(row_section: str, catalog_norm: str) -> bool:
    """Tolerant match between a row's claims_section and a catalog section: equal,
    or one normalized form contains the other (guarded against trivial overlaps)."""
    a, b = norm_section(row_section), catalog_norm
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and (a in b or b in a):
        return True
    return False


def parse_command(cmd: Any) -> tuple[str | None, str | None]:
    """Pull (cmd_dir, verb) out of a `go run ./cmd/<dir> <verb> ...` command.

    cmd_dir is the directory under cmd/ (e.g. 'fak', 'fanbench'); verb is the next
    token (the fak subcommand) for the fak binary, else None. (None, None) when no
    `./cmd/<dir>` token is present."""
    if not _nonempty(cmd):
        return None, None
    m = re.search(r"\./cmd/([\w-]+)", cmd)
    if not m:
        return None, None
    cmd_dir = m.group(1)
    rest = cmd[m.end():].strip()
    verb = None
    for tok in rest.split():
        if tok.startswith("-"):
            break
        verb = tok
        break
    return cmd_dir, verb


# ---------------------------------------------------------------------------
# Per-KPI pure checks (honesty/quality of the rows that EXIST). Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of product-debt; soft = score-only judgment nudges.
# Every defect string is prefixed `<id>: ` so per-row debt is recoverable.
# ---------------------------------------------------------------------------

def _kpi(name: str, defects: list[str], ok_detail: str, *, soft: list[str] | None = None,
         bad_detail: str | None = None) -> dict[str, Any]:
    soft = soft or []
    pen = KPI_PENALTY[name]
    detail = (bad_detail or f"{len(defects)} defect(s)") if defects else ok_detail
    return {"kpi": name, "group": KPI_GROUP[name],
            "score": _clamp(100 - pen * len(defects) - min(10, 2 * len(soft))),
            "detail": detail, "defects": defects, "soft": soft}


def kpi_well_formed(rows: list[dict[str, Any]], categories: set[str]) -> dict[str, Any]:
    """A row must be shaped like a product position: required fields present, every
    enum inside its closed vocabulary, category declared, id unique. A malformed row
    can't be honestly graded, so it is hard debt."""
    defects: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        rid = r.get("id") if _nonempty(r.get("id")) else f"row[{i}]"
        for f in REQUIRED_FIELDS:
            if f not in r:
                defects.append(f"{rid}: missing field '{f}'")
        if not _nonempty(r.get("id")):
            defects.append(f"{rid}: missing id")
        elif r["id"] in seen:
            defects.append(f"{rid}: duplicate id")
        else:
            seen.add(r["id"])
        if categories and r.get("category") not in categories:
            defects.append(f"{rid}: category {r.get('category')!r} not declared in _meta.json")
        if r.get("surface") not in SURFACES:
            defects.append(f"{rid}: surface {r.get('surface')!r} not in {sorted(SURFACES)}")
        if r.get("maturity") not in MATURITIES:
            defects.append(f"{rid}: maturity {r.get('maturity')!r} not in {sorted(MATURITIES)}")
        if r.get("claims_tag") not in CLAIMS_TAGS:
            defects.append(f"{rid}: claims_tag {r.get('claims_tag')!r} not in {sorted(CLAIMS_TAGS)}")
        if r.get("audience") not in AUDIENCES:
            defects.append(f"{rid}: audience {r.get('audience')!r} not in {sorted(AUDIENCES)}")
        if r.get("verdict") not in VERDICT_RANK:
            defects.append(f"{rid}: verdict {r.get('verdict')!r} not in {VERDICTS}")
        for b in ("needs_gpu", "needs_key"):
            if not isinstance(r.get(b), bool):
                defects.append(f"{rid}: {b} must be a bool")
        if not _nonempty(r.get("what_you_get")):
            defects.append(f"{rid}: missing what_you_get (one plain sentence)")
        if not isinstance(r.get("gaps"), list):
            defects.append(f"{rid}: gaps must be a list")
    return _kpi("well_formed", defects, f"all {len(rows)} rows well-formed",
                bad_detail=f"{len(defects)} malformed field(s)")


def kpi_claim_honest(rows: list[dict[str, Any]],
                     section_tags: dict[str, set[str]]) -> dict[str, Any]:
    """The row's claimed maturity must match reality: (1) its claims_tag must be a
    tag the matched CLAIMS.md concept section actually CARRIES (a section mixes
    SHIPPED + STUB lines, so membership — not the lead line — is the honest test),
    and (2) the maturity must agree with that tag (shipped<->SHIPPED, …). This is
    the overclaim catch against the honesty ledger — you cannot call a STUB a
    shipped product, nor claim a tag the ledger never gave that concept."""
    defects: list[str] = []
    soft: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        # maturity <-> claims_tag agreement (skip pure concepts: no CLAIMS line).
        mat, tag = r.get("maturity"), r.get("claims_tag")
        if mat in MATURITY_TAG and tag in CLAIMS_TAGS and MATURITY_TAG[mat] != tag:
            defects.append(f"{rid}: maturity '{mat}' disagrees with claims_tag '{tag}' "
                           f"(expected {MATURITY_TAG[mat]})")
        # claims_tag vs the tag SET the matched CLAIMS.md section carries.
        tags = None
        for cat_norm, tagset in section_tags.items():
            if section_match(r.get("claims_section", ""), cat_norm):
                tags = tagset
                break
        if tags is None:
            if mat != "concept":
                soft.append(f"{rid}: claims_section {r.get('claims_section')!r} did not match a "
                            f"CLAIMS.md section — cannot cross-check the tag")
        elif tag in CLAIMS_TAGS and tag not in tags:
            defects.append(f"{rid}: claims_tag '{tag}' but CLAIMS.md section carries only "
                           f"{sorted(tags)} — overclaim vs the honesty ledger")
    return _kpi("claim_honest", defects,
                f"every claimed maturity matches CLAIMS.md ({len(soft)} unmatched section)",
                soft=soft, bad_detail=f"{len(defects)} maturity overclaim(s) vs CLAIMS.md")


def kpi_command_resolves(rows: list[dict[str, Any]], cmd_dirs: set[str],
                         doc_verbs: set[str]) -> dict[str, Any]:
    """If a row carries a first_command it must REALLY resolve: its `./cmd/<dir>`
    exists, and for the fak binary the verb is documented in the CLI reference. A
    command nobody can run is worse than an honest 'no command'."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        cmd = r.get("first_command", "")
        verb_field = r.get("first_command_verb", "")
        if not _nonempty(cmd):
            if _nonempty(verb_field):
                defects.append(f"{rid}: first_command_verb set but first_command is empty")
            continue
        cmd_dir, verb = parse_command(cmd)
        if cmd_dir is None:
            defects.append(f"{rid}: first_command has no `./cmd/<dir>` invocation — "
                           f"cannot verify it runs")
            continue
        if cmd_dir not in cmd_dirs:
            defects.append(f"{rid}: first_command runs ./cmd/{cmd_dir} which does not exist")
            continue
        if cmd_dir == "fak" and verb and verb.lower() not in doc_verbs:
            defects.append(f"{rid}: fak verb '{verb}' is not documented in docs/cli-reference.md")
    return _kpi("command_resolves", defects,
                "every first command resolves to a real cmd dir + documented verb",
                bad_detail=f"{len(defects)} unrunnable first command(s)")


def kpi_witnessed(rows: list[dict[str, Any]], exists: Callable[[str], bool]) -> dict[str, Any]:
    """A REAL product claim (shipped / simulated) must be PROVEN: a witness_path (a
    test dir / results doc) that exists on disk. An unproven 'shipped' is exactly the
    self-report this repo refuses; a stub/concept is excused (nothing to prove yet)."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        if r.get("maturity") in ("shipped", "simulated"):
            wp = r.get("witness_path", "")
            if not _nonempty(wp):
                defects.append(f"{rid}: {r.get('maturity')} but no witness_path — "
                               f"name the test dir / results doc that proves it")
            elif not exists(wp):
                defects.append(f"{rid}: witness_path '{wp}' does not exist in the tree")
    return _kpi("witnessed", defects, "every shipped/simulated concept is witnessed by a real path",
                bad_detail=f"{len(defects)} unproven product claim(s)")


def kpi_discoverable(rows: list[dict[str, Any]], exists: Callable[[str], bool]) -> dict[str, Any]:
    """A product a person can use must be DISCOVERABLE: an entry_doc (where a human
    learns it) that exists. Required for everything but a pure roadmap concept."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        if r.get("maturity") == "concept":
            continue
        ed = r.get("entry_doc", "")
        if not _nonempty(ed):
            defects.append(f"{rid}: no entry_doc — name where a person learns it")
        elif not exists(ed):
            defects.append(f"{rid}: entry_doc '{ed}' does not exist in the tree")
    return _kpi("discoverable", defects, "every usable concept has a real entry doc",
                bad_detail=f"{len(defects)} undiscoverable concept(s)")


def expected_verdict(row: dict[str, Any]) -> tuple[str, str]:
    """The product-quality verdict the evidence implies, from maturity + surface +
    whether a first command exists + whether it needs a GPU/key. Witness/entry
    existence is graded SEPARATELY (witnessed/discoverable) so it is not double-charged.

    The surface gate is what keeps the map honest: only a `product` surface can be a
    durable-product; a `benchmark` tops out at usable-today; a `subsystem`/`seam`
    (or anything with no runnable command) is real-not-easy."""
    mat = row.get("maturity")
    if mat == "concept":
        return "concept-only", "a roadmap idea (maturity=concept)"
    if mat in ("stub", "simulated"):
        return "honest-stub", f"a labeled seam (maturity={mat})"
    # shipped: the surface gate decides the ceiling.
    surface = row.get("surface")
    if surface in ("subsystem", "seam") or not _nonempty(row.get("first_command")):
        return "real-not-easy", f"shipped {surface or 'subsystem'} with no product surface a person runs directly"
    if surface == "benchmark":
        return "usable-today", "shipped benchmark/demo a person runs today to see or reproduce a result"
    # surface == product
    if row.get("needs_gpu") or row.get("needs_key"):
        return "usable-today", "shipped product surface, but its first command needs a GPU / key / network"
    return "durable-product", "shipped product surface + an OFFLINE first command a person runs today"


def kpi_verdict_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The stated verdict must match what the evidence implies. A 'durable-product'
    on a stub, or on a shipped concept with no runnable command, is an overclaim —
    the single most important thing this scorecard catches."""
    defects: list[str] = []
    for i, r in enumerate(rows):
        rid = r.get("id", i)
        declared = r.get("verdict")
        exp, why = expected_verdict(r)
        if declared != exp:
            defects.append(f"{rid}: claims '{declared}' but evidence implies '{exp}' — {why}")
    return _kpi("verdict_consistency", defects, f"every verdict matches its evidence",
                bad_detail=f"{len(defects)} verdict overclaim(s)")


# ---------------------------------------------------------------------------
# Coverage (how much of the concept catalog is positioned at all).
# ---------------------------------------------------------------------------

def coverage_report(catalog: list[dict[str, str]],
                    rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A concept section is 'covered' when at least one row's claims_section matches
    it. An uncovered section is coverage_debt: a product concept the scorecard
    silently omits — exactly the blind spot this exists to kill."""
    uncovered: list[dict[str, str]] = []
    covered = 0
    for sec in catalog:
        hit = any(section_match(r.get("claims_section", ""), sec["norm"]) for r in rows)
        if hit:
            covered += 1
        else:
            uncovered.append(sec)
    total = len(catalog)
    pct = round(100.0 * covered / total, 1) if total else 100.0
    return {
        "catalog_total": total,
        "covered": covered,
        "coverage_pct": pct,
        "coverage_debt": total - covered,
        "uncovered": uncovered,
    }


# ---------------------------------------------------------------------------
# Fold: KPIs + coverage -> composite score, grade, product-debt, payload.
# ---------------------------------------------------------------------------

def standing(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for r in rows:
        v = r.get("verdict")
        if v in counts:
            counts[v] += 1
    return counts


def per_row_debt(rows: list[dict[str, Any]], kpis: list[dict[str, Any]]) -> dict[str, int]:
    """How many HARD defects each row id accrued across every KPI."""
    out: dict[str, int] = {r.get("id", f"row[{i}]"): 0 for i, r in enumerate(rows)}
    for k in kpis:
        for d in k["defects"]:
            rid = d.split(":", 1)[0]
            if rid in out:
                out[rid] += 1
    return out


def leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        exp, _ = expected_verdict(r)
        out.append({
            "id": r.get("id"),
            "concept": r.get("concept"),
            "category": r.get("category"),
            "surface": r.get("surface"),
            "maturity": r.get("maturity"),
            "verdict": r.get("verdict"),
            "expected_verdict": exp,
            "what_you_get": r.get("what_you_get"),
            "first_command": r.get("first_command", ""),
            "offline": (_nonempty(r.get("first_command"))
                        and not r.get("needs_gpu") and not r.get("needs_key")),
            "witness_path": r.get("witness_path", ""),
            "entry_doc": r.get("entry_doc", ""),
        })
    return out


def critical_backlog(rows: list[dict[str, Any]], row_debt: dict[str, int]) -> list[dict[str, Any]]:
    """The most-critical-areas backlog: rows worst-first by (defect count, distance
    from a durable product). These are what 'progress the most critical areas' means."""
    out = []
    for r in rows:
        rid = r.get("id")
        out.append({
            "id": rid,
            "concept": r.get("concept"),
            "category": r.get("category"),
            "verdict": r.get("verdict"),
            "debt": row_debt.get(rid, 0),
            "distance": VERDICT_RANK.get(r.get("verdict"), 9),
            "gaps": r.get("gaps") or [],
        })
    out.sort(key=lambda x: (-x["debt"], -x["distance"], x["id"] or ""))
    return out


def run_kpis(rows: list[dict[str, Any]], categories: set[str],
             section_tags: dict[str, set[str]], cmd_dirs: set[str], doc_verbs: set[str],
             exists: Callable[[str], bool]) -> list[dict[str, Any]]:
    return [
        kpi_well_formed(rows, categories),
        kpi_claim_honest(rows, section_tags),
        kpi_verdict_consistency(rows),
        kpi_command_resolves(rows, cmd_dirs, doc_verbs),
        kpi_witnessed(rows, exists),
        kpi_discoverable(rows, exists),
    ]


def build_payload(*, workspace: str, data: dict[str, Any] | None, tree: dict[str, Any],
                  error: str | None = None) -> dict[str, Any]:
    if error or not isinstance(data, dict):
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error or "no data",
            "next_action": f"fix the read (run from repo ROOT; check {DATA_DIR_REL}/), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    meta = data.get("meta") or {}
    rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
    cat_defs = [c for c in (data.get("categories") or []) if isinstance(c, dict)]
    categories = {c.get("id") for c in cat_defs if _nonempty(c.get("id"))}

    section_tags = tree.get("section_tags") or {}
    catalog = tree.get("catalog") or []
    cmd_dirs = tree.get("cmd_dirs") or set()
    doc_verbs = tree.get("doc_verbs") or set()
    exists = tree.get("exists") or (lambda p: False)

    kpis = run_kpis(rows, categories, section_tags, cmd_dirs, doc_verbs, exists)
    by_name = {k["kpi"]: k for k in kpis}
    honesty_score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                              for n in KPI_WEIGHTS if n in by_name), 1)
    honesty_defects = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)

    cov = coverage_report(catalog, rows)
    product_debt = honesty_defects + cov["coverage_debt"]

    cov_pct = cov["coverage_pct"] if cov["catalog_total"] else 100.0
    score = round(HONESTY_WEIGHT * honesty_score + COVERAGE_WEIGHT * cov_pct, 1)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    row_debt = per_row_debt(rows, kpis)
    pos = standing(rows)
    crit = critical_backlog(rows, row_debt)
    n_durable = pos.get("durable-product", 0)

    corpus = {
        "score": score, "grade": grade,
        "honesty_score": honesty_score,
        "product_debt": product_debt,
        "honesty_defects": honesty_defects,
        "coverage_debt": cov["coverage_debt"],
        "coverage": cov,
        "soft_signals": n_soft,
        "rows": len(rows),
        "durable_products": n_durable,
        "as_of": meta.get("as_of", ""),
        "fak_version": meta.get("fak_version", ""),
        "standing": pos,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "leaderboard": leaderboard(rows),
        "critical": crit,
    }

    standing_line = (f"{pos['durable-product']} durable · {pos['usable-today']} usable-today "
                     f"· {pos['real-not-easy']} real-not-easy · {pos['honest-stub']} honest-stub "
                     f"· {pos['concept-only']} concept")
    cov_line = (f"coverage {cov['coverage_pct']}% ({cov['covered']}/{cov['catalog_total']} "
                f"concept sections positioned)")
    if product_debt == 0:
        ok, verdict, finding = True, "OK", "product_map_complete_and_honest"
        reason = (f"product map complete + honest: score {score}/100 (grade {grade}); {cov_line}; "
                  f"zero product-debt across {len(kpis)} KPIs over {len(rows)} concepts "
                  f"({standing_line}; {n_soft} advisory)")
        next_action = ("hold the line; when CLAIMS.md adds a concept section, coverage drops — "
                       "position it; when a stub ships, raise its verdict; re-run to keep debt at 0")
    elif honesty_defects == 0 and cov["coverage_debt"] > 0:
        ok, verdict, finding = False, "ACTION", "coverage_debt"
        reason = (f"{cov['coverage_debt']} concept section(s) not yet positioned; {cov_line}; "
                  f"score {score}/100 (grade {grade}); rows are honest (0 honesty-debt); "
                  f"standing {standing_line}")
        next_action = ("close coverage (see --gaps): add an honest product row for each uncovered "
                       "CLAIMS.md concept section (a real-not-easy / honest-stub row is valid); re-run")
    else:
        ok, verdict, finding = False, "ACTION", "product_debt"
        worst = breakdown[0]
        reason = (f"{honesty_defects} honesty/quality defect(s) + {cov['coverage_debt']} coverage "
                  f"gap(s) = product-debt {product_debt}; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']}); {cov_line}; standing {standing_line}")
        next_action = ("retire product-debt worst-first (--critical + per-KPI defects): fix every "
                       "overclaimed verdict, align maturity to the CLAIMS.md tag, point first_command "
                       "at a real verb, add the missing witness/entry doc; then close coverage "
                       "(--gaps); re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
        "_data": {"rows": rows, "categories": cat_defs, "catalog": catalog},
    }


# ---------------------------------------------------------------------------
# Disk shell — read the modular data DIRECTORY + the tree-facts the KPIs verify.
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _read_json(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as exc:
        return None, f"cannot parse {path.name}: {exc}"


def load_data_dir(d: Path) -> tuple[dict[str, Any] | None, str]:
    """Merge the modular data directory into one document: _meta.json contributes
    meta + categories; every other rows-*.json contributes its `rows`."""
    meta_doc, err = _read_json(d / "_meta.json")
    if err:
        return None, err
    if not isinstance(meta_doc, dict):
        return None, "_meta.json is not a JSON object"
    out: dict[str, Any] = {
        "meta": meta_doc.get("meta") or {},
        "categories": meta_doc.get("categories") or [],
        "rows": [],
    }
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        doc, err = _read_json(f)
        if err:
            return None, err
        for r in (doc or {}).get("rows") or []:
            if isinstance(r, dict):
                r.setdefault("_source_file", f.name)
                out["rows"].append(r)
    return out, ""


def load_data(path: Path) -> tuple[dict[str, Any] | None, str]:
    if path.is_dir():
        return load_data_dir(path)
    return None, f"missing data directory: {path}"


def parse_claims_catalog(text: str) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    """From CLAIMS.md, return (catalog, section_tags).

    catalog = the product-concept `##` sections (minus the NON_CONCEPT denylist),
    each {section, norm}. section_tags = {norm -> SET of every `- [TAG]` tag found
    under that section} (a section mixes SHIPPED + STUB lines, so the honest test is
    membership, not the lead line)."""
    catalog: list[dict[str, str]] = []
    tags: dict[str, set[str]] = {}
    cur_norm: str | None = None
    tag_re = re.compile(r"^\s*-\s*\[(SHIPPED|SIMULATED|STUB)\]")
    for line in text.splitlines():
        if line.startswith("## "):
            header = line[3:].strip()
            n = norm_section(header)
            cur_norm = n
            if n and n not in NON_CONCEPT_SECTIONS:
                if all(c["norm"] != n for c in catalog):
                    catalog.append({"section": header, "norm": n})
        elif cur_norm is not None:
            m = tag_re.match(line)
            if m:
                tags.setdefault(cur_norm, set()).add(m.group(1))
    return catalog, tags


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


def load_tree(root: Path) -> dict[str, Any]:
    """The real-tree facts the KPIs cross-check rows against."""
    claims_text = ""
    cp = root / CLAIMS_REL
    if cp.exists():
        try:
            claims_text = cp.read_text(encoding="utf-8")
        except OSError:
            claims_text = ""
    catalog, section_tags = parse_claims_catalog(claims_text)

    cmd_dirs: set[str] = set()
    cmd_root = root / "cmd"
    if cmd_root.is_dir():
        cmd_dirs = {p.name for p in cmd_root.iterdir() if p.is_dir()}

    doc_verbs: set[str] = set()
    cr = root / CLI_REF_REL
    if cr.exists():
        try:
            doc_verbs = _word_set(cr.read_text(encoding="utf-8"))
        except OSError:
            doc_verbs = set()

    return {
        "catalog": catalog,
        "section_tags": section_tags,
        "cmd_dirs": cmd_dirs,
        "doc_verbs": doc_verbs,
        "exists": lambda p: bool(p) and (root / p).exists(),
    }


def collect(workspace: Path, *, data_path: Path | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    path = data_path or (root / DATA_DIR_REL)
    data, err = load_data(path)
    tree = load_tree(root)
    return build_payload(workspace=str(root), data=data, tree=tree, error=err or None)


# ---------------------------------------------------------------------------
# Renderers — terminal, critical backlog, coverage gaps, compare, doc folder.
# ---------------------------------------------------------------------------

_MARK = {"durable-product": "★", "usable-today": "●", "real-not-easy": "◐",
         "honest-stub": "○", "concept-only": "·"}


def _bar(n: int, scale: int, width: int = 28, *, fill: str = "█", empty: str = "·") -> str:
    """A horizontal bar `width` cells wide, length proportional to n/scale. Any
    nonzero value shows at least a one-cell sliver so a real-but-small count is never
    rendered as an empty bar."""
    if scale <= 0:
        return empty * width
    cells = int(round(width * max(0, n) / scale))
    cells = max(0, min(width, cells))
    if n > 0 and cells == 0:
        cells = 1
    return fill * cells + empty * (width - cells)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    pos = c.get("standing") or {}
    lines = [
        f"product-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· PRODUCT-DEBT {c.get('product_debt', 0)} "
         f"(honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) "
         f"· {c.get('soft_signals', 0)} advisory"),
        (f"coverage: {cov.get('coverage_pct', 0)}% "
         f"({cov.get('covered', 0)}/{cov.get('catalog_total', 0)} concept sections positioned) "
         f"· {c.get('rows', 0)} concepts scored · {c.get('durable_products', 0)} durable products"),
        (f"standing: {pos.get('durable-product', 0)} durable-product · "
         f"{pos.get('usable-today', 0)} usable-today · {pos.get('real-not-easy', 0)} real-not-easy · "
         f"{pos.get('honest-stub', 0)} honest-stub · {pos.get('concept-only', 0)} concept-only"),
        ("debt by group: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "product concepts (best verdict first):",
        f"  {'verdict':<16} {'maturity':<10} {'cat':<11} {'today?':<7} concept",
    ]
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("category") or "")):
        mark = _MARK.get(row["verdict"], " ")
        today = "laptop" if row.get("offline") else ("cmd" if _nonempty(row.get("first_command")) else "—")
        flag = "" if row["verdict"] == row["expected_verdict"] else f"  ⚠ expected {row['expected_verdict']}"
        lines.append(f"  {mark} {row['verdict']:<14} {str(row.get('maturity')):<10} "
                     f"{str(row.get('category')):<11} {today:<7} {row.get('concept')}{flag}")
    lines += ["", "per-KPI (worst first):",
              f"  {'score':>5} {'debt':>4}  {'group':<13} {'kpi':<20} detail"]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<13} "
                     f"{b['kpi']:<20} {b['detail']}")
    lines.append("")
    if cov.get("uncovered"):
        lines.append(f"coverage gaps ({len(cov['uncovered'])} concept sections with no row):")
        for sec in cov["uncovered"][:15]:
            lines.append(f"      - {sec['section']}")
        lines.append("")
    lines.append("product-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > 12:
            lines.append(f"      ... and {len(k['defects']) - 12} more")
    if not any_defect:
        lines.append("  (none — zero honesty-debt; every product claim is honest)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_critical(payload: dict[str, Any]) -> str:
    """The most-critical-areas backlog: what 'progress the most critical areas' means."""
    c = payload.get("corpus") or {}
    lines = ["product critical backlog (progress worst-first):", ""]
    crit = c.get("critical", [])
    if not crit:
        lines.append("  (no concepts scored)")
        return "\n".join(lines)
    for it in crit:
        if it["debt"] == 0 and it["distance"] <= VERDICT_RANK["real-not-easy"]:
            continue
        lines.append(f"  [{it['debt']} debt · {it['verdict']}] {it['id']} ({it['category']})")
        for g in (it.get("gaps") or [])[:4]:
            lines.append(f"      - {g}")
    lines.append("")
    lines.append("(rows with 0 debt and a durable/usable/real verdict are omitted — they are not "
                 "critical; a stub/concept with gaps is shown because raising it is the work.)")
    return "\n".join(lines)


def render_gaps(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    lines = ["product coverage backlog (position every concept section):", ""]
    unc = cov.get("uncovered", [])
    lines.append(f"UNPOSITIONED — {len(unc)} CLAIMS.md concept section(s) with no product row:")
    if not unc:
        lines.append("  (none — every concept section is positioned)")
    for sec in unc:
        lines.append(f"  - {sec['section']}")
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("product_debt", 0), cur.get("product_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"product-debt: {bd} -> {cd}   ({ratio} fewer defects+gaps)",
        f"  honesty:    {b.get('honesty_defects', 0)} -> {cur.get('honesty_defects', 0)}",
        f"  coverage:   {b.get('coverage_debt', 0)} -> {cur.get('coverage_debt', 0)}",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"durable:      {b.get('durable_products', 0)} -> {cur.get('durable_products', 0)} durable products",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<13} {gb} -> {gc}")
    target3 = max(0, (bd + 2) // 3)
    target2 = max(0, bd // 2)
    if cd <= target3:
        lines.append(f"VERDICT: ≥3× reduction achieved (product-debt {bd}->{cd}, target ≤{target3}).")
    elif cd <= target2:
        lines.append(f"VERDICT: ≥2× (not yet 3×) — product-debt {bd}->{cd}; 3× needs ≤{target3}.")
    else:
        lines.append(f"VERDICT: not yet 2× — need product-debt ≤{target2} (now {cd}); 3× target ≤{target3}.")
    return "\n".join(lines)


def render_chart(payload: dict[str, Any]) -> str:
    """An at-a-glance ASCII chart of the product standing — what a person sees first.

    Three views over the same data the rest of the scorecard derives: (1) the
    verdict-ladder distribution as count-scaled bars (best -> roadmap); (2) the
    per-category verdict mix, one cell per concept using the same mark vocabulary as
    the ladder, so a category's product-vs-subsystem balance reads at a glance; (3)
    the use-today split (laptop-offline / needs-gpu-key / no-command) and the coverage
    gauge. Pure text + deterministic — two clones at one commit chart identically."""
    c = payload.get("corpus") or {}
    pos = c.get("standing") or {}
    cov = c.get("coverage") or {}
    lb = c.get("leaderboard") or []

    lines: list[str] = [
        (f"product standing chart — {c.get('rows', 0)} concepts · "
         f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) · "
         f"product-debt {c.get('product_debt', 0)}"),
        "",
        "verdict ladder (count of concepts, best -> roadmap):",
    ]
    maxn = max((pos.get(v, 0) for v in VERDICTS), default=0)
    for v in VERDICTS:
        n = pos.get(v, 0)
        lines.append(f"  {_MARK.get(v, ' ')} {v:<15} {_bar(n, maxn)} {n}")
    lines.append("")

    # per-category verdict mix: one mark per concept, sorted best-verdict-first.
    by_cat: dict[str, list[str]] = {}
    for r in lb:
        by_cat.setdefault(r.get("category") or "?", []).append(r.get("verdict"))
    lines.append("verdict mix by category (each cell = one concept):")
    for cat in sorted(by_cat):
        verds = sorted(by_cat[cat], key=lambda v: VERDICT_RANK.get(v, 9))
        spark = "".join(_MARK.get(v, " ") for v in verds)
        durable = sum(1 for v in verds if v == "durable-product")
        usable = sum(1 for v in verds if v == "usable-today")
        lines.append(f"  {cat:<12} {spark:<16} ({len(verds)} concept(s); "
                     f"{durable} durable, {usable} usable-today)")
    lines.append("")

    # use-today split: can a person actually run it, and offline?
    laptop = sum(1 for r in lb if r.get("offline"))
    needs = sum(1 for r in lb if (not r.get("offline")) and _nonempty(r.get("first_command")))
    nocmd = sum(1 for r in lb if not _nonempty(r.get("first_command")))
    total = len(lb)
    lines.append("can a person run it today?")
    lines.append(f"  laptop (offline)   {_bar(laptop, total)} {laptop}")
    lines.append(f"  needs gpu/key/net  {_bar(needs, total)} {needs}")
    lines.append(f"  no direct command  {_bar(nocmd, total)} {nocmd}")
    lines.append("")

    pct = cov.get("coverage_pct", 0.0)
    gauge = _bar(int(round(pct)), 100, width=32)
    lines.append(f"coverage  [{gauge}] {pct}%  "
                 f"({cov.get('covered', 0)}/{cov.get('catalog_total', 0)} concept sections positioned)")
    lines.append("")
    lines.append("legend: " + "   ".join(f"{_MARK[v]} {v}" for v in VERDICTS))
    return "\n".join(lines)


def _front_matter(title: str, desc: str) -> list[str]:
    return ["---", f'title: "{title}"', f'description: "{desc}"', "---", ""]


def render_doc_index(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    pos = c.get("standing") or {}
    out = _front_matter(
        "fak product scorecard — which concepts are durable, real, useful-today products",
        "Inward product-concept scorecard: each fak concept positioned on the durable / real / "
        "useful-today axes a person feels, with one honest verdict per concept and the most-critical "
        "areas to progress. Two driven numbers: coverage (of the concept catalog) and product-debt.")
    out.append("# Product scorecard — durable, real, useful-today")
    out.append("")
    if stamp:
        out.append(f"<!-- product-scorecard: {stamp} · process: tools/product_scorecard.py · "
                   f"data: tools/product_scorecard.data/ -->")
        out.append("")
    out.append("The sibling scorecards grade fak's internals (code, docs) and its competitive "
               "standing (industry). This one asks the question a *person* asks: **of the concepts "
               "fak ships, which can I actually pick up and use this afternoon — and which are still "
               "a named gap, a research seam, or an overclaim?** The source of truth is the concept "
               "catalog in [`CLAIMS.md`](../../CLAIMS.md); every number below is re-derived from "
               "`tools/product_scorecard.data/` by `tools/product_scorecard.py` and cross-checked "
               "against the real tree (the CLAIMS tag a concept carries, whether its first command "
               "resolves, whether its witness/entry paths exist). No verdict is hand-typed.")
    out.append("")
    out.append("> Regenerate: `python tools/product_scorecard.py --markdown-dir docs/product-scorecard`.")
    out.append("")
    out.append("> Person-facing snapshot (what you can run today + what's next): "
               "[`docs/PRODUCT-STATUS.md`](../PRODUCT-STATUS.md).")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Coverage** | **{cov.get('coverage_pct', 0)}%** "
               f"({cov.get('covered', 0)}/{cov.get('catalog_total', 0)} concept sections positioned) |")
    out.append(f"| **Product-debt** | **{c.get('product_debt', 0)}** "
               f"(honesty {c.get('honesty_defects', 0)} + coverage {c.get('coverage_debt', 0)}) |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Durable products | {c.get('durable_products', 0)} of {c.get('rows', 0)} concepts |")
    out.append(f"| As of | {c.get('as_of', '?')} (fak {c.get('fak_version', '?')}) |")
    out.append("")
    out.append("> **Read this right.** The score grades how *complete and honest the product map is* "
               "— not how much fak wins. A concept that is an honest `real-not-easy` subsystem or a "
               "labeled `honest-stub` is not a defect; an *overclaimed* verdict is.")
    out.append("")
    out.append("## Standing at a glance")
    out.append("")
    out.append("> Regenerate this chart in the terminal with `python tools/product_scorecard.py --chart`.")
    out.append("")
    out.append("```text")
    out.append(render_chart(payload))
    out.append("```")
    out.append("")
    out.append("## The verdict ladder")
    out.append("")
    out.append("| Verdict | Means |")
    out.append("|---|---|")
    out.append("| ★ durable-product | shipped + an OFFLINE first command (no GPU/key) + a witness that exists + an entry doc that exists — use it today on a laptop |")
    out.append("| ● usable-today | shipped + a first command, but it needs a GPU / key / network |")
    out.append("| ◐ real-not-easy | shipped/real, but no copy-pasteable command (a subsystem, not a surface) |")
    out.append("| ○ honest-stub | a STUB / SIMULATED seam, labeled honestly |")
    out.append("| · concept-only | a roadmap idea, not built |")
    out.append("")
    out.append("## The product concepts (best verdict first)")
    out.append("")
    out.append("| | Verdict | Maturity | Category | Use today? | Concept — what you get |")
    out.append("|---|---|---|---|---|---|")
    for row in sorted(c.get("leaderboard", []),
                      key=lambda x: (VERDICT_RANK.get(x["verdict"], 9), x.get("category") or "")):
        mark = _MARK.get(row["verdict"], " ")
        today = "laptop" if row.get("offline") else ("needs gpu/key" if _nonempty(row.get("first_command")) else "—")
        out.append(f"| {mark} | {row['verdict']} | {row.get('maturity')} | {row.get('category')} | "
                   f"{today} | **{row.get('concept')}** — {row.get('what_you_get')} |")
    out.append("")
    out.append("## Per-KPI (product-debt = honesty/quality of the rows that exist)")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    if cov.get("uncovered"):
        out.append("## Coverage gaps (concept sections not yet positioned)")
        out.append("")
        for sec in cov["uncovered"]:
            out.append(f"- {sec['section']}")
        out.append("")
    crit = [x for x in c.get("critical", []) if x["debt"] > 0 or x["distance"] > VERDICT_RANK["real-not-easy"]]
    if crit:
        out.append("## Most-critical areas to progress")
        out.append("")
        for it in crit[:12]:
            gaps = "; ".join((it.get("gaps") or [])[:3])
            out.append(f"- **{it['id']}** ({it['verdict']}, {it['debt']} debt) — {gaps}")
        out.append("")
    return "\n".join(out)


def render_doc_folder(payload: dict[str, Any], *, stamp: str | None = None) -> dict[str, str]:
    return {"README.md": render_doc_index(payload, stamp=stamp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Product-concept scorecard (read-only unless --markdown-dir).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--chart", action="store_true", help="an at-a-glance ASCII chart of the product standing")
    ap.add_argument("--critical", action="store_true", help="the most-critical-areas backlog")
    ap.add_argument("--gaps", action="store_true", help="the coverage backlog (unpositioned concept sections)")
    ap.add_argument("--compare", default="", help="baseline JSON to prove product-debt dropped")
    ap.add_argument("--markdown-dir", default="", help=f"regenerate the doc folder (e.g. {GENERATED_DOC_DIR})")
    ap.add_argument("--data", default="", help=f"data directory (default: {DATA_DIR_REL})")
    ap.add_argument("--stamp", default="", help="optional stamp embedded in the generated doc")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    data_path = Path(args.data).resolve() if args.data else None
    payload = collect(root, data_path=data_path)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
        return 0 if payload.get("ok") else 1

    if args.markdown_dir:
        out_dir = Path(args.markdown_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in render_doc_folder(payload, stamp=args.stamp or None).items():
            (out_dir / rel).write_text(content + "\n", encoding="utf-8")
        if not args.json:
            print(f"wrote product-scorecard doc folder -> {out_dir}")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.chart:
        print(render_chart(payload))
    elif args.critical:
        print(render_critical(payload))
    elif args.gaps:
        print(render_gaps(payload))
    elif not args.markdown_dir:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
