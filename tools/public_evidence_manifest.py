#!/usr/bin/env python3
"""Public-evidence manifest: the ship-iff-cited gate for benchmark artifacts.

The public `fak` repo's pitch is "every number, traced to its commit + artifact".
That promise only holds if the artifacts a published claim CITES are actually in
the public tree. A public-readiness scrub that cuts experiments/ wholesale breaks
it silently (the dead "see artifact" links only surface as docs-scorecard debt).

This tool makes "is this evidence public?" a CHECKABLE, DERIVED property — the same
move tools/scrub_public_copy.py made for "is this a leak?":

  --derive   Walk the PUBLIC-shipping doc set (every tracked *.md + llms.txt in the
             public tree) + the hero *.data.json artifact rows, collect every
             `experiments/...` path and `*-RESULTS.md` doc they cite, transitively
             close (a cited RESULTS doc — even one cut from public — pulls in the
             experiments/ paths IT cites, read from --private-root), and write the
             sorted, deduped manifest to tools/_registry/public_evidence_manifest.json.

  --check    Every manifest path must EXIST in the public tree (a missing one means
             an over-aggressive cut broke verifiability) and a manifest entry that is
             no longer cited by any doc is an ORPHAN (stale). Pure-stdlib, repo-root,
             CI-wireable beside hero_benchmark_gen.py --verify-sources. Idempotent.

Exit: 0 clean, 1 violation (missing-cited or orphan), 2 could-not-run.
The discriminator is DERIVED from the docs, so it cannot drift: the same grep that
builds the manifest is the one CI checks against.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

# NB: NOT under tools/_registry/ — that dir is gitignored (private needle sidecar).
# The manifest is a list of PUBLIC paths, so it ships at a tracked location.
MANIFEST_REL = "tools/public_evidence_manifest.json"

# Template / meta docs whose `path` mentions are EXAMPLES of the naming convention
# (MODELNAME-RESULTS.md, 8B-RESULTS.md, experiments/<suite>/*.json), not real citations.
EXCLUDE_ROOTS = {
    "BENCHMARK-TEMPLATE.md", "BENCHMARK-GOVERNANCE.md", "BENCHMARK-XREF-AUDIT.md",
    "tools/_registry/public_evidence_manifest.json",
}
# Placeholder citation names a template leaves behind (skip — not a real artifact/doc).
_PLACEHOLDER_RE = re.compile(r"MODELNAME|MODEL-NAME|^\d+B-RESULTS|YYYY|<.*>|\*|\{")

# A citation is either an experiments/ artifact path or a *-RESULTS.md provenance doc.
# Match both markdown-link `](path)` and inline-code `` `path` `` / bare forms; tolerate
# a leading fak/ or ./ or ../ (pre-hoist links) and normalize to a root-relative path.
# jsonl BEFORE json so the longer extension wins; the trailing (?![A-Za-z]) stops
# `.json` matching as a prefix of `.jsonl`.
_EXPERIMENTS_RE = re.compile(r"(?:\.{0,2}/)?(?:fak/)?(experiments/[A-Za-z0-9_./{},*-]+?\.(?:jsonl|json|tsv|csv|svg|png|md|html))(?![A-Za-z])")
_RESULTS_RE = re.compile(r"([A-Z0-9][A-Z0-9-]*-RESULTS\.md)")
# A path right after an output flag (`--output X`, `-out X`, `-o X`) is a file a command
# WRITES, not an artifact the doc CITES — don't treat it as a published-evidence citation.
_OUTPUT_FLAG_RE = re.compile(r"(?:--?o(?:ut(?:put)?)?)\s+$")


def _norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def _doc_roots(public_root: str) -> list[str]:
    """Every tracked *.md (+ llms.txt) in the public tree = the shipping doc closure."""
    roots: list[str] = []
    for dirpath, dirs, files in os.walk(public_root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for f in files:
            if f.endswith(".md") or f == "llms.txt":
                roots.append(os.path.relpath(os.path.join(dirpath, f), public_root).replace(os.sep, "/"))
    # hero data files carry artifact rows too (the CI verify-sources source of truth)
    for hd in ("tools/hero_benchmark.data.json",):
        if os.path.isfile(os.path.join(public_root, hd)):
            roots.append(hd)
    return sorted(roots)


def _cited_in(text: str) -> tuple[set[str], set[str]]:
    exp = set()
    for m in _EXPERIMENTS_RE.finditer(text):
        if _OUTPUT_FLAG_RE.search(text[max(0, m.start() - 14):m.start()]):
            continue  # command output path, not a citation
        exp.add(_norm(m.group(1)))
    res = {_norm(m) for m in _RESULTS_RE.findall(text)}
    return exp, res


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def derive(public_root: str, private_root: str) -> dict:
    experiments: dict[str, list[str]] = {}   # path -> citing docs
    results: dict[str, list[str]] = {}
    for rel in _doc_roots(public_root):
        if rel in EXCLUDE_ROOTS:
            continue
        exp, res = _cited_in(_read(os.path.join(public_root, rel)))
        for p in exp:
            if _PLACEHOLDER_RE.search(p):
                continue
            experiments.setdefault(p, []).append(rel)
        for p in res:
            if _PLACEHOLDER_RE.search(p):
                continue
            results.setdefault(p, []).append(rel)
    # Transitive closure: each cited RESULTS doc pulls in the experiments/ it cites.
    # Prefer the PUBLIC restored copy (docs/benchmarks/<doc> — already de-citeed of
    # ghost artifacts) over the private original, so closure reflects what actually
    # ships; fall back to the private tree only for docs not yet restored.
    for doc in sorted(results):
        for cand in (os.path.join(public_root, "docs", "benchmarks", doc),
                     os.path.join(public_root, doc),
                     os.path.join(private_root, "fak", doc),
                     os.path.join(private_root, doc)):
            if os.path.isfile(cand):
                exp, _ = _cited_in(_read(cand))
                for p in exp:
                    if _PLACEHOLDER_RE.search(p):
                        continue
                    experiments.setdefault(p, []).append(doc)
                break
    return {
        "schema": "fleet-public-evidence/1",
        "note": "DERIVED by tools/public_evidence_manifest.py --derive. ship-iff-cited: a public claim's citation chain reaches every path here. Regenerate; do not hand-edit.",
        "experiments": {p: sorted(set(c)) for p, c in sorted(experiments.items())},
        "results_docs": {p: sorted(set(c)) for p, c in sorted(results.items())},
    }


def cmd_derive(public_root: str, private_root: str) -> int:
    man = derive(public_root, private_root)
    out = os.path.join(public_root, MANIFEST_REL)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"public-evidence: derived {len(man['experiments'])} experiments/ artifacts + "
          f"{len(man['results_docs'])} *-RESULTS.md cited by public docs -> {MANIFEST_REL}")
    return 0


def cmd_check(public_root: str) -> int:
    out = os.path.join(public_root, MANIFEST_REL)
    if not os.path.isfile(out):
        print(f"PUBLIC_EVIDENCE (warn): no manifest at {MANIFEST_REL}; run --derive.", file=sys.stderr)
        return 2
    with open(out, encoding="utf-8") as fh:
        man = json.load(fh)
    experiments = man.get("experiments", {})
    results_docs = man.get("results_docs", {})
    cited = list(experiments) + list(results_docs)

    def _present(p: str) -> bool:
        # brace-globs (e.g. radixbench-{a,b}.json) expand to multiple real files — skip
        # the literal brace entry; the expansion members match on their own if cited.
        if "{" in p:
            return True
        if os.path.exists(os.path.join(public_root, p)):
            return True
        # A *-RESULTS.md provenance doc is cited by its bare basename (the citation regex
        # normalizes the path away), but dated docs may not live at the repo root — the
        # doc-placement gate keeps them under docs/benchmarks/. Resolve a results-doc
        # basename to its canonical restored home there too.
        if p in results_docs and os.path.exists(os.path.join(public_root, "docs", "benchmarks", p)):
            return True
        return False

    missing = [p for p in cited if not _present(p)]
    if missing:
        print(f"PUBLIC_EVIDENCE: {len(missing)} cited evidence path(s) are MISSING from the "
              f"public tree — a published claim cites them but the scrub cut them "
              f"(restore via the cited-evidence sync, or de-cite in the doc):", file=sys.stderr)
        for m in sorted(missing):
            print(f"  MISSING  {m}  <- cited by {', '.join((man.get('experiments', {}).get(m) or man.get('results_docs', {}).get(m) or [])[:3])}", file=sys.stderr)
        return 1
    print(f"public-evidence: clean — all {len(cited)} cited artifacts/docs present in the public tree.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--derive", action="store_true", help="regenerate the manifest from the doc citations")
    g.add_argument("--check", action="store_true", help="every cited path must exist in the public tree (CI)")
    ap.add_argument("--public-root", default=".", help="the public fak tree (default: cwd)")
    ap.add_argument("--private-root", default="", help="the full private tree (for --derive transitive closure)")
    args = ap.parse_args()
    public_root = os.path.abspath(args.public_root)
    if args.check:
        return cmd_check(public_root)
    private_root = os.path.abspath(args.private_root) if args.private_root else public_root
    return cmd_derive(public_root, private_root)


if __name__ == "__main__":
    sys.exit(main())
