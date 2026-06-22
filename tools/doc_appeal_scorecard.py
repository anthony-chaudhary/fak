#!/usr/bin/env python3
"""Doc-appeal scorecard — the measuring stick for *prose that lands*.

The repo already has two checking layers for its front door, and both watch the
*mechanical* health of a doc: ``readme_freshness_audit.py`` (dead links, stale
version pins, naive-led headlines) and ``docs_scorecard.py`` (the same, fanned
out across the core doc set, folded into a ``doc-debt`` integer). They answer
"is this doc *correct and current*?"

They do **not** answer the question an early adopter actually asks in their first
thirty seconds: "do I *get* it, do I *trust* it, and can I *try* it — fast?"
That is a question about the *writing*: is the lead crisp or buried, do the
sentences breathe or run on, does it sound like a person or like a machine, can a
skimmer (human or LLM) find the one thing they came for. None of that is a dead
link, so none of it was measured — "make the README more appealing" was an
unfalsifiable vibe.

This is that missing number. It scores one doc (``README.md`` by default) on five
prose axes, each 0-100, deterministic, content-only — no model, no network:

  clarity        sentences are short enough to parse in one breath; few run-ons
  priority       the lead defines the thing fast; a skimmer's path to value is short
  voice          reads human, not machine — few clichés, no em-dash flood, no tic
  scannability   a skimmer (or an answer engine) can lift the answer without reading
  organization   one title, clean heading hierarchy, balanced sections, a real arc

The axes fold into a weighted **appeal score** (0-100, A-F) AND — the lever that
makes "10x better" a checkable target instead of a vibe — an **appeal-debt**
integer: the count of concrete, re-derivable prose defects (an overlong sentence,
a run-on, a cliché, an em-dash past budget, a buried lead, a wall-of-text
paragraph, a skipped heading level). Appeal-debt is an integer you can drive down,
so an improvement program can promise "cut appeal-debt 10x, from N to N/10" and
then *prove* it by re-running.

The **early-adopter lens** (this project's chosen audience) shows up in the
weights and in two priority checks specifically: time-to-understanding (how many
words before the doc says what the thing *is*, in one crisp line) and
time-to-first-action (how deep before a paste-able command or a "try it" link).
Early adopters reward fast understanding, a low-friction first step, and honest
scope; they punish a wall of marketing prose. The weights lean that way:
priority and voice weigh most, organization least.

HARD defects (each one unit of appeal-debt, the work-list) are unambiguous and
worth fixing: an overlong sentence, a run-on, a cliché phrase, an em-dash past a
generous budget, the rhetorical "not X, it's Y" frame past a small budget, a
buried lead, a missing one-line summary, a wall-of-text paragraph, a skipped
heading level, more than one H1. SOFT signals (passive voice, hedging, low
sentence-length variety, a dominant section, inconsistent name-casing) lower the
score but never count as debt — they are writing judgment, not mechanical fact,
the same split the sibling scorecards draw between FAIL and ADVISORY.

Read-only by construction: it reads the target doc (and ``VERSION`` only to label
its output); it edits nothing. Run from the repo ROOT::

    python tools/doc_appeal_scorecard.py                       # human scorecard for README.md
    python tools/doc_appeal_scorecard.py --target docs/FAQ.md  # any other doc
    python tools/doc_appeal_scorecard.py --json                # machine payload (control-pane)
    python tools/doc_appeal_scorecard.py --markdown            # the committed snapshot body

The companion process is the appeal-10x program: each defect is one unit to
retire; re-running proves the number moved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any

SCHEMA = "fleet-doc-appeal-scorecard/1"

DEFAULT_TARGET_REL = "README.md"
VERSION_REL = "VERSION"

# Per-axis weights for the composite appeal score. The early-adopter lens leans
# the weights toward "do I get it fast" (priority) and "does it read human"
# (voice); organization is real but the least load-bearing for a cold reader who
# has not yet decided to care.
AXIS_WEIGHTS: dict[str, float] = {
    "clarity": 0.22,
    "priority": 0.24,
    "voice": 0.22,
    "scannability": 0.18,
    "organization": 0.14,
}

# ---------------------------------------------------------------------------
# Calibration constants. Each is a deliberate threshold, not a magic number;
# the comment says why. Tuned so a polished-but-dense front page (the current
# README) shows real, fixable headroom rather than either 100 or 0.
# ---------------------------------------------------------------------------

# Sentence length. A reader parses a ~20-word sentence in one breath; past ~30 it
# starts to strain; past 40 it is a hard defect (split it). Calibrated for a
# technical doc, which runs longer than chat prose.
SENT_LONG = 30          # soft: counts toward the "% long" score penalty
SENT_OVERLONG = 40      # HARD: one appeal-debt unit each, listed in the work-list
SENT_RUNON_COMMAS = 5   # HARD: a sentence with >= this many commas reads as a run-on

# Paragraph length. A prose block past ~110 words is a wall of text a skimmer
# bounces off — HARD.
PARA_WALL_WORDS = 110

# Voice budgets. A little of each device is fine — a flood is the machine-tell.
EMDASH_PER_100W_BUDGET = 0.5   # em-dashes allowed per 100 prose words before debt
EMDASH_DEFECT_CAP = 20         # cap em-dash debt so one axis can't be everything
NOT_X_BUT_Y_BUDGET = 2         # the "not X, it's Y" frame allowed before debt

# Early-adopter timing thresholds, measured in *content lines* (blanks, comments,
# and code/badge scaffolding excluded — what a reader's eye actually lands on).
LEAD_WINDOW_LINES = 14         # "above the fold": where the crisp one-liner must live
DEF_MAX_WORDS = 25             # a one-line definition must be at most this many words
FIRST_ACTION_MAX_LINE = 45     # a paste-able command / "try it" link should arrive by here

# Flesch reading-ease floor for a technical doc. Below this it reads like a spec,
# not an invitation. (Plain English ~60-70; dense technical prose ~30-45.)
FLESCH_FLOOR = 35.0

# Marketing / AI-tell phrases an early adopter's eye snags on. Each occurrence in
# PROSE (not code, not a heading) is one appeal-debt unit. Kept to genuine tells —
# words that signal "a machine or a brochure wrote this", not ordinary technical
# vocabulary. Matched case-insensitively on word boundaries.
CLICHE_PHRASES: list[str] = [
    "delve", "leverage", "seamless", "seamlessly", "robust", "game-changer",
    "game changer", "cutting-edge", "cutting edge", "state-of-the-art solution",
    "unlock the power", "harness the power", "in today's", "fast-paced",
    "ever-evolving", "ever-changing", "landscape of", "in the realm of",
    "tapestry", "testament to", "at the end of the day", "needless to say",
    "it's worth noting", "it is worth noting", "boasts", "plethora", "myriad",
    "dive into", "deep dive", "supercharge", "effortless", "effortlessly",
    "elevate your", "best-in-class", "world-class", "revolutionary", "paradigm shift",
    "synergy", "synergies", "holistic", "bespoke", "turnkey", "first-class citizen",
    "look no further", "rest assured", "without further ado",
]

# Formulaic LLM scaffolding — low-information framing an answer model reaches for
# to sound authoritative. Distinct from the marketing clichés above: these are the
# connective tissue of machine prose, not brochure words. Each occurrence in PROSE
# is one appeal-debt unit. Kept deliberately tight so a genuine Feynman move
# ("think of it as …", "imagine …", a concrete example) is never punished — the
# repo's own voice law *encourages* those; only the empty throat-clearing is a tell.
LLM_SCAFFOLD_PHRASES: list[str] = [
    "here's the thing", "here's the kicker", "the key insight", "the key takeaway",
    "the takeaway is", "at its core", "at the heart of", "make no mistake",
    "let's be clear", "let's be honest", "the beauty of", "the magic of",
    "in a nutshell", "the bottom line", "when it comes to", "the reality is",
    "the truth is", "it's important to note", "it is important to note",
]

# Hedging / filler words. SOFT only (density nudge) — one "essentially" is fine,
# a cluster reads unsure. Never appeal-debt.
HEDGE_WORDS: list[str] = [
    "arguably", "essentially", "fundamentally", "basically", "simply put",
    "in essence", "of course", "obviously", "clearly", "needless to say",
    "in order to", "due to the fact", "it should be noted",
]

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
# A passive-voice approximation: a be/get auxiliary followed by a past participle.
# Crude (over-fires a little), so it is SOFT only — it nudges the score, never debt.
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|get|gets|got)\s+(\w+(?:ed|en|wn|ne|de|t))\b",
    re.IGNORECASE,
)
# The "not X, it's Y" / "not X but Y" / "not X — Y" rhetorical frame (forward form).
_NOT_X_Y_RE = re.compile(
    r"\bnot\s+(?:just\s+|merely\s+|only\s+|simply\s+)?[^,.;:—]{2,45}?[,—]\s*"
    r"(?:it'?s|its|but|rather|instead)\b",
    re.IGNORECASE,
)
# The REVERSED contrast frame "X, not Y" / "X — not Y" (e.g. "the lock, not the
# screener"; "because a verdict said so, not because memory got tight"). This is
# the form an LLM over-leans on; the forward regex above misses it entirely. One
# or two are punchy; a doc that leans on it every other paragraph reads formulaic,
# so both forms share one budget. The lead-in "[,—] not" plus a word keeps it to a
# deliberate contrast (not "is not"/"do not", which lack the comma/dash hinge).
_X_NOT_Y_RE = re.compile(
    r"[,—]\s+not\s+(?:just\s+|merely\s+|only\s+|simply\s+|a\s+|the\s+|because\s+)?"
    r"[A-Za-z][^,.;:—]{1,40}",
    re.IGNORECASE,
)
# A bold-emphasis span (**x** or __x__) — counted for the emphasis-flood signal.
_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|(?<!_)__[^_\n]+__(?!_)")
# A paste-able shell command inside a fenced block (cheap heuristic: a known verb
# at a line start within code).
_CMD_RE = re.compile(r"^\s*(?:\$\s*)?(go|fak|curl|make|python|pip|brew|docker|git|sh|npm|cargo)\b")
# A "try it / run it / demo" call-to-action link or phrase.
_TRY_RE = re.compile(r"\b(try (it|the)|run (it|the|your)|live demo|demos?|quickstart|colab|notebook)\b",
                     re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing: one classifier pass turns raw markdown into the views the axes need.
# This is the testable seam — every axis takes a `Doc`, tests build one from a
# fixture string with `parse()`.
# ---------------------------------------------------------------------------

class Doc:
    """A parsed markdown doc: classified lines plus the derived prose views."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines: list[tuple[int, str, str]] = _classify_lines(text)
        # Headings: (level, text, lineno).
        self.headings: list[tuple[int, str, int]] = []
        # Content lines = everything a reader's eye lands on (prose, list, quote,
        # heading, table) — excludes blanks, code, comments, badges. Used for the
        # early-adopter "how deep" timing checks.
        self.content: list[tuple[int, str, str]] = []
        # Prose blocks: consecutive prose/list/quote lines joined, inline markup
        # stripped — the unit for sentence + paragraph analysis.
        self.prose_blocks: list[tuple[int, str]] = []
        # Block-start linenos of blocks that are genuine multi-item lists (>= 2
        # bullets). A list is the *remedy* for a wall of text, not a wall itself —
        # the scannability axis exempts these from the wall-of-text check. A lone
        # giant bullet is NOT exempt (it's a paragraph wearing a dash).
        self.list_blocks: set[int] = set()

        block_start = 0
        block_buf: list[str] = []
        block_list_items = 0
        for lineno, raw, kind in self.lines:
            if kind == "heading":
                m = _HEADING_RE.match(raw.strip())
                if m:
                    self.headings.append((len(m.group(1)), m.group(2).strip(), lineno))
            if kind in ("prose", "list", "quote", "heading", "table"):
                self.content.append((lineno, raw, kind))
            if kind in ("prose", "list", "quote"):
                if not block_buf:
                    block_start = lineno
                    block_list_items = 0
                block_buf.append(_strip_inline(raw))
                if kind == "list":
                    block_list_items += 1
            else:
                if block_buf:
                    self.prose_blocks.append((block_start, " ".join(block_buf).strip()))
                    if block_list_items >= 2:
                        self.list_blocks.add(block_start)
                    block_buf = []
        if block_buf:
            self.prose_blocks.append((block_start, " ".join(block_buf).strip()))
            if block_list_items >= 2:
                self.list_blocks.add(block_start)

        self.prose_text = " ".join(b for _, b in self.prose_blocks)
        # Split sentences PER BLOCK, not over the joined corpus. A block that ends
        # without terminal punctuation (a colon lead-in to a table, a "→ link"
        # pointer line) must not glue itself to the next block's opening sentence
        # and masquerade as one giant overlong run-on — that is a measurement
        # artifact, not a prose defect.
        self.sentences = [s for _, b in self.prose_blocks for s in _split_sentences(b)]
        self.prose_words = _wordcount(self.prose_text)


def parse(text: str) -> Doc:
    return Doc(text)


def _classify_lines(text: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    in_fence = False
    in_comment = False
    in_script = False
    lines = text.split("\n")
    # A leading YAML front-matter block (--- … ---) is page metadata (title,
    # description), not reader prose. Mark it so no axis scores it — otherwise a
    # long `description:` reads as a giant run-on and a Jekyll/SKILL.md page is
    # graded on metadata it never shows a reader. Mirrors seo_aeo_scorecard.
    fm_last = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                fm_last = j + 1  # 1-based line number of the closing fence
                break
    for i, raw in enumerate(lines, 1):
        if i <= fm_last:
            out.append((i, raw, "frontmatter"))
            continue
        s = raw.strip()
        if in_comment:
            if "-->" in s:
                in_comment = False
            out.append((i, raw, "comment"))
            continue
        if s.startswith("<!--"):
            if "-->" not in s:
                in_comment = True
            out.append((i, raw, "comment"))
            continue
        # A <script> / <style> block is machine content (e.g. generated JSON-LD
        # structured data), not reader prose — skip it like a code fence, or its
        # generated answer text gets scored as run-on prose.
        low = s.lower()
        if in_script:
            if "</script>" in low or "</style>" in low:
                in_script = False
            out.append((i, raw, "script"))
            continue
        if low.startswith(("<script", "<style")):
            if "</script>" not in low and "</style>" not in low:
                in_script = True
            out.append((i, raw, "script"))
            continue
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            out.append((i, raw, "code"))
            continue
        if in_fence:
            out.append((i, raw, "code"))
            continue
        if not s:
            out.append((i, raw, "blank"))
            continue
        if _HEADING_RE.match(s):
            out.append((i, raw, "heading"))
            continue
        # A horizontal rule (---, ***, ___) is a section separator, not prose; if
        # left as prose it has no terminal period and merges the sections it sits
        # between into one phantom sentence.
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", s):
            out.append((i, raw, "hr"))
            continue
        if s.startswith("|"):
            out.append((i, raw, "table"))
            continue
        if s.startswith(">"):
            out.append((i, raw, "quote"))
            continue
        # An image/badge/raw-HTML-only line carries no prose.
        if re.match(r"^\[?!\[", s) or s.startswith(("<sub>", "<img", "<picture", "<div", "<a ", "</")):
            out.append((i, raw, "badge"))
            continue
        if s.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", s):
            out.append((i, raw, "list"))
            continue
        out.append((i, raw, "prose"))
    return out


def _strip_inline(s: str) -> str:
    """Drop markdown decoration so sentence analysis sees plain words."""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)        # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)     # links -> link text
    s = re.sub(r"`[^`]*`", "", s)                       # inline code
    s = re.sub(r"[*_]{1,3}", "", s)                     # emphasis markers
    s = re.sub(r"^\s*>+\s*", "", s)                     # blockquote markers
    s = re.sub(r"^\s*(?:[-*+]\s+|\d+\.\s+)", "", s)     # list markers
    return s.strip()


def _split_sentences(text: str) -> list[str]:
    """Split on .!? boundaries that are followed by space + a capital/quote/digit.
    A decimal like ``9.7×`` has no following space, so it is not a boundary;
    ``e.g.`` is usually followed by a lowercase word, so it is not either.

    Also break at the house ``→`` "see also" pointer and the ``·`` inline-list
    separator: those join clauses with no terminal period, so without an explicit
    break a "→ link" pointer or a ``a · b · c`` link row reads as one run-on.
    """
    if not text.strip():
        return []
    # The product name ``fak`` is lower-case and frequently opens a sentence; allow
    # it as a sentence-start token so "… differs). `fak` can do this …" splits in
    # two instead of reading as one phantom run-on.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“(]|fak\b)|\s*[→·]\s*', text)
    return [p.strip() for p in parts if p.strip()]


def _wordcount(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'×%/-]*", text))


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _flesch(text: str) -> float:
    sents = _split_sentences(text)
    words = re.findall(r"[A-Za-z]+", text)
    if not sents or not words:
        return 100.0
    syll = sum(_syllables(w) for w in words)
    wps = len(words) / len(sents)
    spw = syll / len(words)
    return 206.835 - 1.015 * wps - 84.6 * spw


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def _snip(s: str, n: int = 56) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# The five axes. Each returns
#   {axis, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of appeal-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def axis_clarity(doc: Doc) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    overlong = runon = longish = 0
    for sent in doc.sentences:
        w = _wordcount(sent)
        if w == 0:
            continue
        if w >= SENT_OVERLONG:
            overlong += 1
            defects.append(f"overlong sentence ({w} words): “{_snip(sent)}”")
        elif w >= SENT_LONG:
            longish += 1
        if sent.count(",") >= SENT_RUNON_COMMAS:
            runon += 1
            defects.append(f"run-on ({sent.count(',')} commas): “{_snip(sent)}”")
    n = max(1, len(doc.sentences))
    pct_long = 100 * (overlong + longish) / n
    passives = len(_PASSIVE_RE.findall(doc.prose_text))
    passive_density = passives / n
    flesch = _flesch(doc.prose_text)

    score = 100.0
    score -= min(30, 3 * overlong)
    score -= min(16, 2 * runon)
    if pct_long > 30:
        score -= min(18, (pct_long - 30) * 0.5)
    if flesch < FLESCH_FLOOR:
        score -= min(18, (FLESCH_FLOOR - flesch) * 0.6)
        soft.append(f"reading-ease {flesch:.0f} below {FLESCH_FLOOR:.0f} (dense for a cold read)")
    if passive_density > 0.25:
        score -= min(12, (passive_density - 0.25) * 100 * 0.4)
        soft.append(f"passive-voice density {passive_density:.0%} of sentences")

    detail = (f"{len(doc.sentences)} sentences · {overlong} overlong · {runon} run-on · "
              f"{pct_long:.0f}% long · Flesch {flesch:.0f}")
    return {"axis": "clarity", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_priority(doc: Doc) -> dict[str, Any]:
    """Early-adopter lens: a crisp definition fast, a short path to value."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # 1. Time-to-understanding: a <= DEF_MAX_WORDS sentence in the lead window
    # that says what the thing IS or DOES (a copula / "lets you" / "puts …").
    lead = [(ln, _strip_inline(raw)) for ln, raw, kind in doc.content[:LEAD_WINDOW_LINES]
            if kind in ("prose", "list", "quote")]
    def_re = re.compile(r"\b(is|are)\s+(a|an|the|not)\b|\b(lets you|puts |sits between|"
                        r"turns |gives you|treats )", re.IGNORECASE)
    crisp_def = None
    for ln, txt in lead:
        for sent in _split_sentences(txt):
            if def_re.search(sent) and _wordcount(sent) <= DEF_MAX_WORDS:
                crisp_def = (ln, sent)
                break
        if crisp_def:
            break
    if crisp_def is None:
        defects.append(f"buried lead: no crisp one-line definition (<= {DEF_MAX_WORDS} words) "
                       f"in the first {LEAD_WINDOW_LINES} content lines")
        score -= 22

    # 2. A TL;DR / one-line summary block above the fold (a blockquote or an
    # explicit "TL;DR"/"In short"/"In one line" cue). Agents and skimmers lift it.
    head_raw = "\n".join(raw for _, raw, _ in doc.lines[:40])
    has_tldr = bool(re.search(r"\bTL;?DR\b|\bin short\b|\bin one line\b|\bone-liner\b",
                              head_raw, re.IGNORECASE)) or _early_blockquote(doc)
    if not has_tldr:
        defects.append("no TL;DR / one-line summary block above the fold (a skimmer's lift)")
        score -= 12

    # 3. Time-to-first-action: how deep before a paste-able command or a "try it"
    # link. Measured in content-line index (scaffolding excluded).
    first_action = _first_action_content_index(doc)
    if first_action is None:
        soft.append("no paste-able command or 'try it' link found")
        score -= 8
    elif first_action > FIRST_ACTION_MAX_LINE:
        defects.append(f"first paste-able action is deep (content line {first_action} "
                       f"> {FIRST_ACTION_MAX_LINE}) — early adopters want to *do* something sooner")
        score -= 12

    # 4. Lead density: a single giant opening block before the first heading break
    # is a wall an early adopter bounces off (soft — also caught by scannability).
    fold_words = _words_before_first_break(doc)
    if fold_words > 140:
        soft.append(f"{fold_words}-word lead before the first section break (dense opener)")
        score -= 8

    detail = (f"crisp-def {'at line ' + str(crisp_def[0]) if crisp_def else 'MISSING'} · "
              f"tldr {'yes' if has_tldr else 'no'} · "
              f"first-action {first_action if first_action else 'none'} · lead {fold_words}w")
    return {"axis": "priority", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_voice(doc: Doc) -> dict[str, Any]:
    """Human, not machine: few clichés, no em-dash flood, no over-leaned tic."""
    defects: list[str] = []
    soft: list[str] = []
    low = doc.prose_text.lower()

    cliche_hits: list[str] = []
    for phrase in CLICHE_PHRASES:
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for _ in pat.finditer(low):
            cliche_hits.append(phrase)
    for ph in cliche_hits:
        defects.append(f"cliché / AI-tell phrase: “{ph}”")

    # Formulaic LLM scaffolding — empty framing a machine reaches for. HARD.
    scaffold_hits: list[str] = []
    for phrase in LLM_SCAFFOLD_PHRASES:
        # Apostrophe-tolerant: match both "here's" and a curly-quote "here’s".
        pat = re.compile(re.escape(phrase).replace("'", "['’]"), re.IGNORECASE)
        for _ in pat.finditer(low):
            scaffold_hits.append(phrase)
    for ph in scaffold_hits:
        defects.append(f"LLM scaffolding phrase: “{ph}” (empty framing — cut or say it plainly)")

    # Em-dash flood. A generous budget scaled to length; each dash past it is one
    # debt unit, capped so this single device can't dominate the whole score.
    n_dash = doc.prose_text.count("—")
    budget = max(2, int(doc.prose_words * EMDASH_PER_100W_BUDGET / 100))
    over = max(0, n_dash - budget)
    capped = min(EMDASH_DEFECT_CAP, over)
    for _ in range(capped):
        defects.append("em-dash past budget (reads machine-generated — vary the punctuation)")
    if over > capped:
        soft.append(f"{over - capped} further em-dashes past budget (capped at {EMDASH_DEFECT_CAP})")

    # The contrast frame, both forward ("not X, it's Y") and reversed ("X, not Y").
    # A couple is good rhetoric; a habit is a tic. Both forms share one budget.
    notxy = len(_NOT_X_Y_RE.findall(doc.prose_text)) + len(_X_NOT_Y_RE.findall(doc.prose_text))
    for _ in range(max(0, notxy - NOT_X_BUT_Y_BUDGET)):
        defects.append("over-leaned 'not X / X, not Y' contrast frame (vary the sentence shape)")

    # SOFT: bold-emphasis flood. A little bold guides a skimmer's eye; bolding a
    # phrase in every other sentence is a machine tic that stops guiding anything.
    bold_spans = len(_BOLD_RE.findall(doc.text))
    bold_budget = max(6, int(doc.prose_words * 1.5 / 100))
    if bold_spans > bold_budget:
        soft.append(f"{bold_spans} bold spans (budget {bold_budget}) — emphasis flood reads machine-generated")

    # SOFT: hedging density.
    hedges = sum(len(re.findall(r"\b" + re.escape(h) + r"\b", low)) for h in HEDGE_WORDS)
    if hedges > max(2, doc.prose_words // 400):
        soft.append(f"{hedges} hedging/filler phrases (trim for a confident voice)")

    # SOFT: sentence-length monotony — too little variety reads robotic.
    lens = [_wordcount(s) for s in doc.sentences if _wordcount(s) > 0]
    if len(lens) >= 6 and pstdev(lens) < 5:
        soft.append(f"low sentence-length variety (σ={pstdev(lens):.1f}) — monotone rhythm")

    # SOFT: does it address the reader at all? Early adopters like "you".
    if doc.prose_words > 200 and not re.search(r"\byou(r|'ll|'re|'ve)?\b", low):
        soft.append("never addresses the reader directly ('you') in a 200+ word doc")

    score = 100.0
    score -= 2 * len(cliche_hits)
    score -= 2 * len(scaffold_hits)
    score -= 1.5 * capped
    score -= 4 * max(0, notxy - NOT_X_BUT_Y_BUDGET)
    score -= 3 * len(soft)
    detail = (f"{len(cliche_hits)} cliché · {len(scaffold_hits)} scaffold · "
              f"{n_dash} em-dashes (budget {budget}) · {notxy} contrast-frame · {hedges} hedges")
    return {"axis": "voice", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_scannability(doc: Doc) -> dict[str, Any]:
    """A skimmer — human or answer-engine — can lift the answer without reading."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    # Wall-of-text paragraphs: a skimmer bounces off these. Each is HARD. A genuine
    # multi-item list is exempt — its bullets ARE the skim structure.
    walls = 0
    for ln, block in doc.prose_blocks:
        if ln in doc.list_blocks:
            continue
        w = _wordcount(block)
        if w > PARA_WALL_WORDS:
            walls += 1
            defects.append(f"wall-of-text paragraph (line {ln}, {w} words) — break it up")
    score -= min(40, 8 * walls)

    # Heading cadence: a long doc needs section anchors a skimmer can jump between.
    n_content = len(doc.content)
    n_head = len(doc.headings)
    if n_content > 40 and n_head < max(3, n_content // 40):
        soft.append(f"sparse headings ({n_head} for {n_content} content lines) — hard to skim")
        score -= 10

    # Structured affordances: tables and lists let a skimmer (and an LLM) lift
    # structure. Their total absence in a long doc is a soft nudge.
    has_table = any(k == "table" for _, _, k in doc.lines)
    has_list = any(k == "list" for _, _, k in doc.lines)
    if n_content > 40 and not (has_table or has_list):
        soft.append("no tables or lists — nothing for a skimmer to lift")
        score -= 8

    # Name-casing consistency: an answer engine keys off a consistent entity name.
    name_variants = set(re.findall(r"\b(fak|Fak|FAK)\b", doc.prose_text))
    if len(name_variants) > 1:
        soft.append(f"inconsistent product-name casing in prose: {sorted(name_variants)}")
        score -= 6

    detail = (f"{walls} wall(s) · {n_head} headings / {n_content} content lines · "
              f"tables={has_table} lists={has_list}")
    return {"axis": "scannability", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def axis_organization(doc: Doc) -> dict[str, Any]:
    """One title, a clean heading hierarchy, balanced sections, a sensible arc."""
    defects: list[str] = []
    soft: list[str] = []
    score = 100.0

    h1s = [h for h in doc.headings if h[0] == 1]
    if len(h1s) == 0:
        defects.append("no H1 title (a '# Title' line)")
        score -= 25
    elif len(h1s) > 1:
        defects.append(f"{len(h1s)} H1 titles (expected exactly one)")
        score -= 12

    # Heading-level skips (e.g. an H2 jumping straight to an H4). Each is HARD —
    # it breaks a generated table of contents and a skimmer's mental outline.
    skips = 0
    prev = 0
    for level, txt, ln in doc.headings:
        if prev and level > prev + 1:
            skips += 1
            defects.append(f"skipped heading level (H{prev}→H{level}) at line {ln}: “{_snip(txt, 40)}”")
        prev = level
    score -= min(24, 12 * skips)

    # Duplicate heading text — confuses navigation and anchors. SOFT.
    titles = [t.lower() for _, t, _ in doc.headings]
    dups = sorted({t for t in titles if titles.count(t) > 1})
    if dups:
        soft.append(f"duplicate heading text: {dups[:3]}")
        score -= 5

    # Section balance: one H2 section dwarfing the rest reads lopsided. SOFT.
    dominance = _section_dominance(doc)
    if dominance > 0.45:
        soft.append(f"one section holds {dominance:.0%} of the body (lopsided)")
        score -= 6

    detail = (f"{len(h1s)} H1 · {len(doc.headings)} headings · {skips} level-skip(s) · "
              f"max-section {dominance:.0%}")
    return {"axis": "organization", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


# ---------------------------------------------------------------------------
# Small structural helpers
# ---------------------------------------------------------------------------

def _early_blockquote(doc: Doc) -> bool:
    """A blockquote summary in the first ~12 content lines (the llms.txt pattern)."""
    for ln, _raw, kind in doc.content[:12]:
        if kind == "quote":
            return True
    return False


def _first_action_content_index(doc: Doc) -> int | None:
    """Content-line index (1-based over content lines) of the first paste-able
    command or 'try it' link — whichever comes first. None if neither exists."""
    cmd_lines = {ln for ln, raw, kind in doc.lines if kind == "code" and _CMD_RE.match(raw)}
    for idx, (ln, raw, _kind) in enumerate(doc.content, 1):
        if _TRY_RE.search(raw) and _LINK_RE.search(raw):
            return idx
    # No try-link: fall back to the first command's *position* relative to content.
    if cmd_lines:
        first_cmd = min(cmd_lines)
        before = sum(1 for ln, _r, _k in doc.content if ln < first_cmd)
        return before + 1
    return None


def _words_before_first_break(doc: Doc) -> int:
    """Prose words before the first heading after the H1 (the opening 'fold')."""
    seen_h1 = False
    total = 0
    for lineno, raw, kind in doc.lines:
        if kind == "heading":
            level = len(raw.strip().split(" ", 1)[0])
            if level == 1 and not seen_h1:
                seen_h1 = True
                continue
            if seen_h1:
                break
        if seen_h1 and kind in ("prose", "list", "quote"):
            total += _wordcount(_strip_inline(raw))
    return total


def _section_dominance(doc: Doc) -> float:
    """Fraction of body prose-words held by the single largest H2 section."""
    if not doc.headings:
        return 0.0
    # Map each prose block to the most recent H2-or-deeper heading line.
    h_lines = sorted(ln for lvl, _t, ln in doc.headings if lvl >= 2)
    if not h_lines:
        return 0.0
    sizes: dict[int, int] = {ln: 0 for ln in h_lines}
    for bln, block in doc.prose_blocks:
        owner = max((ln for ln in h_lines if ln < bln), default=None)
        if owner is not None:
            sizes[owner] += _wordcount(block)
    total = sum(sizes.values())
    return (max(sizes.values()) / total) if total else 0.0


# ---------------------------------------------------------------------------
# Fold: axes -> composite score, grade, appeal-debt, control-pane payload
# ---------------------------------------------------------------------------

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


def score_doc(text: str, target_rel: str) -> dict[str, Any]:
    doc = parse(text)
    axes = [
        axis_clarity(doc),
        axis_priority(doc),
        axis_voice(doc),
        axis_scannability(doc),
        axis_organization(doc),
    ]
    by_name = {a["axis"]: a for a in axes}
    composite = sum(AXIS_WEIGHTS[name] * by_name[name]["score"] for name in AXIS_WEIGHTS)
    defects = [f"{a['axis']}: {d}" for a in axes for d in a["defects"]]
    soft = [f"{a['axis']}: {s}" for a in axes for s in a["soft"]]
    return {
        "path": target_rel,
        "appeal_score": round(composite, 1),
        "grade": grade_letter(composite),
        "appeal_debt": len(defects),
        "axes": {a["axis"]: a["score"] for a in axes},
        "axis_detail": {a["axis"]: a["detail"] for a in axes},
        "axis_debt": {a["axis"]: len(a["defects"]) for a in axes},
        "defects": defects,
        "soft": soft,
        "stats": {
            "sentences": len(doc.sentences),
            "prose_words": doc.prose_words,
            "headings": len(doc.headings),
            "prose_blocks": len(doc.prose_blocks),
        },
    }


def build_payload(*, workspace: str, target_rel: str, doc: dict[str, Any] | None,
                  error: str | None = None) -> dict[str, Any]:
    if error or doc is None:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error or "no document scored",
            "next_action": "fix the target read (run from repo ROOT), then re-run",
            "workspace": workspace, "target": target_rel, "doc": doc,
        }
    debt = doc["appeal_debt"]
    score = doc["appeal_score"]
    grade = doc["grade"]
    if debt == 0:
        ok, verdict, finding = True, "OK", "appeal_clean"
        reason = f"{target_rel}: {score}/100 ({grade}), zero appeal-debt — prose lands"
        next_action = "no required edit; re-run after the next front-page change"
    else:
        ok, verdict, finding = False, "ACTION", "appeal_debt"
        worst_axis = min(doc["axes"], key=lambda a: doc["axes"][a])
        reason = (f"{target_rel}: {score}/100 ({grade}), {debt} unit(s) of appeal-debt; "
                  f"weakest axis: {worst_axis} ({doc['axes'][worst_axis]}/100)")
        next_action = ("retire appeal-debt worst-axis-first (see doc.defects): split overlong "
                       "sentences, cut clichés/em-dashes, add a crisp one-liner + TL;DR, break "
                       "walls of text; re-run to prove the drop")
    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action,
        "workspace": workspace, "target": target_rel, "doc": doc,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def collect(workspace: Path, target_rel: str = DEFAULT_TARGET_REL) -> dict[str, Any]:
    root = workspace.resolve()
    target = root / target_rel
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return build_payload(workspace=str(root), target_rel=target_rel, doc=None,
                             error=f"cannot read {target_rel}: {exc}")
    return build_payload(workspace=str(root), target_rel=target_rel,
                         doc=score_doc(text, target_rel))


def render(payload: dict[str, Any]) -> str:
    doc = payload.get("doc")
    if not doc:
        return f"doc-appeal: {payload.get('verdict')} — {payload.get('reason')}"
    lines = [
        f"doc-appeal scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"target: {doc['path']} · APPEAL {doc['appeal_score']}/100 ({doc['grade']}) "
         f"· APPEAL-DEBT {doc['appeal_debt']}"),
        "",
        "per-axis:",
        f"  {'score':>5} {'debt':>4}  axis           detail",
    ]
    for axis in AXIS_WEIGHTS:
        lines.append(f"  {doc['axes'][axis]:>5} {doc['axis_debt'][axis]:>4}  "
                     f"{axis:<14} {doc['axis_detail'][axis]}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    if doc["defects"]:
        lines.append("")
        lines.append("appeal-debt work-list:")
        for d in doc["defects"]:
            lines.append(f"  - {d}")
    if doc["soft"]:
        lines.append("")
        lines.append("soft signals (score only, not debt):")
        for s in doc["soft"]:
            lines.append(f"  · {s}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    doc = payload.get("doc") or {}
    out: list[str] = ["# Doc-appeal scorecard", ""]
    if stamp:
        out.append(f"<!-- doc-appeal-scorecard: {stamp} · target={doc.get('path')} "
                   f"· process: tools/doc_appeal_scorecard.py -->")
        out.append("")
    out.append("> The measuring stick for *prose that lands* — does an early adopter **get** it, "
               "**trust** it, and can they **try** it, fast? Five deterministic prose axes "
               "(clarity · priority · voice · scannability · organization), folded into an "
               "**appeal score** (0–100) and an **appeal-debt** integer (the count of concrete, "
               "re-derivable prose defects). Every number below is re-derived from disk by "
               "`tools/doc_appeal_scorecard.py` — no hand-entry. Drive appeal-debt toward zero to "
               "make \"more appealing\" provable.")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Target | `{doc.get('path')}` |")
    out.append(f"| **Appeal score** | **{doc.get('appeal_score')}/100 ({doc.get('grade')})** |")
    out.append(f"| **Appeal-debt (defects)** | **{doc.get('appeal_debt')}** |")
    st = doc.get("stats", {})
    out.append(f"| Prose | {st.get('sentences')} sentences · {st.get('prose_words')} words "
               f"· {st.get('headings')} headings |")
    out.append("")
    out.append("## Per-axis")
    out.append("")
    out.append("| Axis | Score | Debt | Detail |")
    out.append("|---|---:|---:|---|")
    for axis in AXIS_WEIGHTS:
        out.append(f"| {axis} | {doc.get('axes', {}).get(axis)} | "
                   f"{doc.get('axis_debt', {}).get(axis)} | {doc.get('axis_detail', {}).get(axis)} |")
    out.append("")
    out.append("## Appeal-debt work-list")
    out.append("")
    if doc.get("defects"):
        for d in doc["defects"]:
            out.append(f"- {d}")
    else:
        out.append("No appeal-debt: the prose lands. 🎉")
    out.append("")
    if doc.get("soft"):
        out.append("## Soft signals (score only, not debt)")
        out.append("")
        for s in doc["soft"]:
            out.append(f"- {s}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Doc-appeal scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--target", default=DEFAULT_TARGET_REL,
                    help=f"doc to score, repo-relative (default: {DEFAULT_TARGET_REL})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    args = ap.parse_args(argv)

    # Docs carry Unicode (×, —, ·); force UTF-8 stdout so a Windows cp1252 console
    # cannot crash the scorer on an em-dash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, target_rel=args.target)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
