#!/usr/bin/env python3
"""skill_overlap.py — flag content-redundant SKILL.md pairs as merge candidates (#3930).

The Comet/Perplexity "cutting agent context cost" writeup's one technique fak did
NOT already cover: detect skills with high pairwise CONTENT overlap and merge the
redundant ones, so a 55-file skill pack stays lean and discoverable. `skill-lifecycle`
archives *dead* skills and `skill-score` audits over-budget ones; neither measures
skill-to-skill redundancy. This is that missing measure.

It runs a pairwise token-similarity primitive over the `.claude/skills/*/SKILL.md`
BODIES (front-matter stripped, so a shared YAML shape never inflates the score) and
emits the pairs above a threshold, sorted by similarity — the same posture as
`fak dup` / `fak traj` (which do block- and query-level near-duplicate detection)
but self-contained so the witness runs hermetically, with no built `fak` binary.

Hard posture — PROPOSAL ONLY. This file never merges, edits, moves, or deletes a
skill. Its max action is to PRINT candidates for a human to judge, exactly like
`trajectory-garden` (near-dup queries) and `skill-lifecycle` (archive verdicts are
dry-run until `--apply`). There is no `--apply` here: merging two skills is a
semantic call the detector deliberately does not make.

Similarity is cosine over token frequency with a small function-word stoplist —
"skills A and B are N% token-similar" in the issue's own words. Views/frontmatter/
punctuation don't count; only body vocabulary does.

Usage:
  python skill_overlap.py [--skills-root PATH] [--threshold T] [--top N] [--json]

Flags: --skills-root PATH (default: this file's parent's parent, i.e. .claude/skills),
--threshold T (default 0.5; a pair must score >= T to surface), --top N (cap the
list), --json (machine-readable). Exit code is always 0 — this is advisory, and
"no candidates" is a clean, valid answer, not a failure.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_THRESHOLD = 0.5
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Function words only — deliberately NOT domain verbs like "run"/"use"/"score",
# which carry real signal in a skill doc. Dropping these keeps two skills from
# looking similar merely because English prose shares its connective tissue.
STOPWORDS = frozenset("""
a an and are as at be been being but by can could do does done for from had has have
he her his if in into is it its may might must no nor not of off on onto or our out
over per shall she should so than that the their them then there these they this those
to too up upon was we were what when where which while who whom whose why will with
would you your
""".split())


def body_of(text: str) -> str:
    """The SKILL.md content with a leading YAML front-matter block removed.

    Stripping front-matter is load-bearing: every skill's YAML shares the same
    `name:`/`description:` skeleton, and counting it would floor every pair's
    similarity above zero for a structural, non-redundant reason.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            nl = text.find("\n", end + 1)
            return text[nl + 1:] if nl >= 0 else ""
    return text


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def tf_vector(tokens: list[str]) -> Counter:
    return Counter(tokens)


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller vector's keys for the dot product.
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(count * large.get(tok, 0) for tok, count in small.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(c * c for c in a.values()))
    nb = math.sqrt(sum(c * c for c in b.values()))
    return dot / (na * nb)


def discover_skills(root: Path) -> list[tuple[str, Path]]:
    """(name, SKILL.md) for every live skill under root, sorted by name.

    Skips dot-directories (`.archive`, `.snapshots`, …) so an archived skill is
    never proposed for a merge.
    """
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            md = p / "SKILL.md"
            if md.exists():
                out.append((p.name, md))
    return out


def load_vectors(root: Path) -> dict[str, Counter]:
    vectors: dict[str, Counter] = {}
    for name, md in discover_skills(root):
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        vectors[name] = tf_vector(tokenize(body_of(text)))
    return vectors


def shared_terms(a: Counter, b: Counter, limit: int = 6) -> list[str]:
    """The highest-weight tokens both bodies carry — the 'why' behind a score."""
    common = {t: min(a[t], b[t]) for t in a.keys() & b.keys()}
    return [t for t, _ in sorted(common.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def overlap_pairs(root: Path, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Every skill pair whose body cosine similarity is >= threshold.

    Sorted by similarity descending, then by the pair names — deterministic, so
    the same skill pack always yields the same candidate ordering.
    """
    vectors = load_vectors(root)
    names = sorted(vectors)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sim = cosine(vectors[a], vectors[b])
            if sim >= threshold:
                pairs.append({
                    "a": a,
                    "b": b,
                    "similarity": round(sim, 4),
                    "shared_top": shared_terms(vectors[a], vectors[b]),
                })
    pairs.sort(key=lambda p: (-p["similarity"], p["a"], p["b"]))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skills-root", default=None,
                    help="skills directory (default: this file's parent's parent, .claude/skills)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"min body cosine similarity to surface a pair (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--top", type=int, default=0, help="cap the candidate list (0 = all)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    root = Path(args.skills_root) if args.skills_root else Path(__file__).resolve().parent.parent
    pairs = overlap_pairs(root, args.threshold)
    if args.top > 0:
        pairs = pairs[:args.top]

    if args.json:
        print(json.dumps({
            "skills_root": root.as_posix(),
            "threshold": args.threshold,
            "candidates": pairs,
        }, indent=2))
        return 0

    if not pairs:
        print(f"skill-overlap: no pair >= {args.threshold:g} similarity under {root.as_posix()} "
              f"— the pack is non-redundant at this threshold.")
        return 0

    print(f"skill-overlap: {len(pairs)} merge candidate(s) >= {args.threshold:g} "
          f"(proposal only — nothing is merged):\n")
    for p in pairs:
        print(f"  {p['similarity']:.2f}  {p['a']} <> {p['b']}")
        if p["shared_top"]:
            print(f"        shared: {', '.join(p['shared_top'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
