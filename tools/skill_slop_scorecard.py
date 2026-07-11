#!/usr/bin/env python3
"""Skill-slop scorecard — the structural admission gate for agent-authored skills (#2911).

The library's skills are increasingly *agent-authored*: an autonomous loop that
"produces at least one skill update per session" cannot be gated by human review
(the Hermes model — a HARDLINE authoring standard enforced by eyeballs — does not
scale, and its ungated agent path stored a 74KB transcript verbatim as one skill,
bug #3006's shape). This is the machine gate for the same defects a human reviewer
rejects by hand, run BEFORE a candidate SKILL.md is admitted to the library.

Six signals, all HARD (each one is true slop, never a style nit):

  verbatim_dump         one fenced block (or the whole body) so large it can only be a
                        pasted transcript/file, not authored instruction  [the #3006 reject]
  vacuous_body          headings and boilerplate with (almost) no substantive lines —
                        a skill that instructs nothing
  marketing             sales adjectives ("seamless", "revolutionary", …) — a skill sells
                        nothing; marketing voice marks generated filler
  one_off_narrative     first-person session narration ("I then ran…", timestamps) — a
                        transcript remnant, not a reusable procedure
  missing_verification  no verification/witness section AND no checkable command or
                        exit-code anywhere — an unfalsifiable skill
  duplicate_body        the substantive body near-duplicates an existing skill (opt-in,
                        needs --corpus) — a redundant skill paying context cost for no new
                        instruction; the "duplicated body vs an existing skill" reject (#2839)

The gate is calibrated to reject SLOP, not brevity: a terse skill whose few lines
name a real check ("run X; exit 0 means clean") ADMITS — thresholds only trip on
near-empty, dump-sized, or narration-shaped bodies (the "terse-but-legitimate"
confusion risk the issue names). This is a skill-authoring STRUCTURAL check; it is
not a code test-coverage gate (the other named confusion risk).

Read-only and hermetic by construction: pure text analysis of the given SKILL.md
paths — no network, no git, no toolchain. Run from anywhere::

    python tools/skill_slop_scorecard.py .claude/skills/foo/SKILL.md       # one skill
    python tools/skill_slop_scorecard.py .claude/skills --json             # scan a tree
    python tools/skill_slop_scorecard.py new/SKILL.md --corpus .claude/skills  # + dedup
    # exit 0 = every candidate ADMITted; 1 = at least one REJECT (the HARD gate);
    # exit 2 = a path was unreadable / no SKILL.md found (contract error, not a verdict)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-skill-slop-scorecard/1"

# The #3006 reject: one fenced block this large is a pasted artifact, not instruction.
DUMP_BLOCK_BYTES = 8_192
# A whole-body ceiling: the 74KB verbatim skill trips this even without fences.
DUMP_TOTAL_BYTES = 32_768
# Fewer substantive lines than this instructs nothing (headings/boilerplate excluded).
MIN_SUBSTANTIVE_LINES = 3
# Marketing hits at/above this are voice, not accident.
MARKETING_MAX = 2
# Narration hits at/above this mark a transcript remnant, not one figure of speech.
NARRATIVE_MAX = 3
# Duplicate-body reject (#2839): a candidate this fraction of whose substantive lines
# appear verbatim in ONE existing skill is a copy, not fresh instruction. Containment
# (not symmetric similarity) so a copy with extra padding is still caught; a title or
# frontmatter tweak that defeats a whole-file hash does not defeat it.
DUP_CONTAINMENT = 0.80
# Only skills with at least this many substantive lines are dedup-checked, so a terse
# legitimate skill cannot trip merely because its 2-3 lines echo a larger one.
DUP_MIN_LINES = 4

MARKETING_RE = re.compile(
    r"\b(seamless(?:ly)?|revolutionary|cutting[- ]edge|best[- ]in[- ]class|"
    r"world[- ]class|state[- ]of[- ]the[- ]art|game[- ]chang\w*|effortless(?:ly)?|"
    r"blazing(?:ly)?[- ]fast|supercharge\w*|magical(?:ly)?|enterprise[- ]grade|"
    r"next[- ]generation|turnkey|frictionless)\b", re.IGNORECASE)

NARRATIVE_RES = [
    re.compile(r"\b[Ii] (?:then |just |first |also )?(?:ran|tried|fixed|noticed|"
               r"realized|checked|edited|committed|discovered)\b"),
    re.compile(r"\bwe (?:then |just |first )?(?:ran|tried|fixed|noticed|checked)\b",
               re.IGNORECASE),
    re.compile(r"\bin this session\b", re.IGNORECASE),
    re.compile(r"\bthe user (?:asked|wanted|said)\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
]

VERIFICATION_HEADING_RE = re.compile(
    r"^#{1,6}\s+.*\b(verif\w*|witness\w*|validat\w*|checks?)\b", re.IGNORECASE)
VERIFICATION_LINE_RE = re.compile(
    r"\b(verify|witness|pytest|go test|make ci|exit\s+(?:codes?|\d+)|assert\w*|--check)\b",
    re.IGNORECASE)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---`` YAML frontmatter block; the body is what gets graded."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    return text[end + len("\n---"):]


def _fenced_blocks(body: str) -> list[str]:
    """The fenced code blocks, verbatim, so a pasted transcript is measurable."""
    blocks: list[str] = []
    cur: list[str] | None = None
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            if cur is None:
                cur = []
            else:
                blocks.append("\n".join(cur))
                cur = None
            continue
        if cur is not None:
            cur.append(line)
    if cur:  # an unclosed fence is still a block (dumps are rarely well-formed)
        blocks.append("\n".join(cur))
    return blocks


def _substantive_lines(body: str) -> list[str]:
    """Prose lines that could instruct: non-blank, non-heading, outside code fences."""
    out: list[str] = []
    in_fence = False
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s or s.startswith("#") or s.startswith("<!--"):
            continue
        if len(s) <= 2:
            continue
        out.append(s)
    return out


def _norm_line(line: str) -> str:
    """Whitespace- and case-folded form of a line, so a reflow or re-case is not a diff."""
    return re.sub(r"\s+", " ", line).strip().lower()


def body_shingles(text: str) -> frozenset[str]:
    """The set of normalized substantive lines of a skill body — the unit of the
    duplicate-vs-corpus comparison. Frontmatter (name/description) is excluded, so a
    copy that only renames the skill still collides on its instruction lines."""
    return frozenset(_norm_line(ln) for ln in _substantive_lines(_strip_frontmatter(text)))


def grade_text(
    text: str,
    corpus: list[tuple[str, frozenset[str]]] | None = None,
) -> dict[str, Any]:
    """Grade one candidate SKILL.md body. Pure; the CLI wraps file IO around this.

    ``corpus`` is an optional list of ``(existing-skill-label, body_shingles)`` to check
    the candidate against for duplication. Omit it (the default) and the duplicate_body
    signal is simply not evaluated — the gate stays hermetic and single-file, so every
    prior caller is unchanged."""
    body = _strip_frontmatter(text)
    findings: list[dict[str, Any]] = []

    blocks = _fenced_blocks(body)
    big = max((len(b.encode("utf-8")) for b in blocks), default=0)
    total = len(body.encode("utf-8"))
    if big > DUMP_BLOCK_BYTES or total > DUMP_TOTAL_BYTES:
        findings.append({
            "signal": "verbatim_dump",
            "detail": (f"largest fenced block {big}B (max {DUMP_BLOCK_BYTES}B), "
                       f"body {total}B (max {DUMP_TOTAL_BYTES}B) — pasted artifact, "
                       f"not authored instruction (#3006 shape)")})

    lines = _substantive_lines(body)
    has_witness_line = any(VERIFICATION_LINE_RE.search(ln) for ln in lines)
    # Terse-but-legitimate calibration: a short body whose few lines name a real
    # check is instruction, not slop — vacuous means short AND uncheckable.
    if len(lines) < MIN_SUBSTANTIVE_LINES and not has_witness_line:
        findings.append({
            "signal": "vacuous_body",
            "detail": (f"{len(lines)} substantive line(s) "
                       f"(min {MIN_SUBSTANTIVE_LINES}) — instructs nothing")})

    marketing = MARKETING_RE.findall(body)
    if len(marketing) >= MARKETING_MAX:
        findings.append({
            "signal": "marketing",
            "detail": f"marketing voice x{len(marketing)}: "
                      + ", ".join(sorted({m.lower() for m in marketing}))})

    narrative = [m.group(0) for rx in NARRATIVE_RES for m in rx.finditer(body)]
    if len(narrative) >= NARRATIVE_MAX:
        findings.append({
            "signal": "one_off_narrative",
            "detail": f"session narration x{len(narrative)}: "
                      + "; ".join(narrative[:4])})

    has_verification = any(VERIFICATION_HEADING_RE.match(ln)
                           for ln in body.splitlines()) or has_witness_line
    if not has_verification:
        findings.append({
            "signal": "missing_verification",
            "detail": "no verification/witness section and no checkable command "
                      "or exit-code named anywhere — unfalsifiable skill"})

    # Duplicated body vs an existing skill (#2839) — opt-in, needs a corpus to compare
    # against. A near-verbatim copy of a library skill is pure slop: it is loaded on
    # every session for zero new instruction. Containment (candidate lines found in ONE
    # existing skill) catches a copy even when a renamed title/frontmatter would defeat a
    # whole-file hash; the MIN_LINES floor keeps a terse legitimate skill from tripping.
    if corpus:
        cand = frozenset(_norm_line(ln) for ln in lines)
        if len(cand) >= DUP_MIN_LINES:
            best_label, best_ratio = "", 0.0
            for label, other in corpus:
                if not other:
                    continue
                ratio = len(cand & other) / len(cand)
                if ratio > best_ratio:
                    best_label, best_ratio = label, ratio
            if best_ratio >= DUP_CONTAINMENT:
                findings.append({
                    "signal": "duplicate_body",
                    "detail": (f"{best_ratio:.0%} of substantive lines duplicate an "
                               f"existing skill ('{best_label}') — a redundant skill pays "
                               f"context cost every session for no new instruction")})

    ok = not findings
    return {"ok": ok, "verdict": "ADMIT" if ok else "REJECT",
            "defects": len(findings), "score": 100 - 25 * len(findings),
            "findings": findings}


def grade_file(
    path: Path,
    corpus: list[tuple[str, frozenset[str]]] | None = None,
) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"path": str(path), "ok": False, "verdict": "UNREADABLE",
                "error": str(e), "defects": 0, "findings": []}
    return {"path": str(path), **grade_text(text, corpus=corpus)}


def load_corpus(dirs: list[str]) -> list[tuple[Path, frozenset[str]]]:
    """Read every SKILL.md under each --corpus dir into (resolved-path, shingles) rows.
    The resolved path lets the caller drop a candidate that is its own corpus entry so a
    skill never scores as a duplicate of itself. Unreadable corpus files are skipped —
    they are reference, not the graded artifact."""
    rows: list[tuple[Path, frozenset[str]]] = []
    for d in dirs:
        for sk in sorted(Path(d).rglob("SKILL.md")):
            try:
                rows.append((sk.resolve(),
                             body_shingles(sk.read_text(encoding="utf-8", errors="replace"))))
            except OSError:
                continue
    return rows


def discover(paths: list[Path]) -> list[Path]:
    """Explicit files verbatim; a directory contributes every SKILL.md under it."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("SKILL.md")))
        else:
            out.append(p)
    return out


def render(payload: dict[str, Any]) -> str:
    lines = [f"skill-slop: {payload['verdict']}  "
             f"({payload['admitted']}/{payload['graded']} admitted)"]
    for card in payload["cards"]:
        lines.append(f"  {card.get('verdict'):10s} {card.get('path')}"
                     + (f"  score={card.get('score')}" if "score" in card else ""))
        for f in card.get("findings", []):
            lines.append(f"      [{f['signal']}] {f['detail']}")
        if card.get("error"):
            lines.append(f"      [error] {card['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Skill-slop scorecard — HARD admission gate for SKILL.md (read-only).")
    ap.add_argument("paths", nargs="+", help="SKILL.md files or directories to scan")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--corpus", action="append", default=[], metavar="DIR",
                    help="existing-skill dir(s) to flag duplicate bodies against "
                         "(repeatable; a candidate that is its own corpus entry is skipped)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    files = discover([Path(p) for p in args.paths])
    if not files:
        print("skill-slop: contract error — no SKILL.md found under the given paths",
              file=sys.stderr)
        return 2
    # Build the dedup corpus once (empty unless --corpus was given). Each candidate is
    # graded against every corpus body EXCEPT its own entry, matched by resolved path, so
    # a skill that already lives under a --corpus dir never scores as a duplicate of
    # itself. Without --corpus the corpus is empty and the duplicate_body signal is
    # simply not evaluated — the gate stays hermetic and every prior caller is unchanged.
    corpus_rows = load_corpus(args.corpus)
    cards = []
    for p in files:
        corpus: list[tuple[str, frozenset[str]]] | None = None
        if corpus_rows:
            pr = p.resolve()
            corpus = [(cp.parent.name, sh) for cp, sh in corpus_rows if cp != pr]
        cards.append(grade_file(p, corpus=corpus))
    unreadable = any(c["verdict"] == "UNREADABLE" for c in cards)
    admitted = sum(1 for c in cards if c["ok"])
    payload = {"schema": SCHEMA, "graded": len(cards), "admitted": admitted,
               "verdict": "ADMIT" if admitted == len(cards) else "REJECT",
               "cards": cards}
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    if unreadable:
        return 2
    return 0 if admitted == len(cards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
