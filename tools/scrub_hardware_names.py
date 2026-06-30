#!/usr/bin/env python3
"""scrub_hardware_names.py -- rewrite the operator's private lab hardware names
(NVIDIA "A100" / "DGX" / "SXM4") out of human-readable DOCUMENTATION PROSE, while
leaving every CODE/DATA IDENTIFIER that happens to contain "dgx"/"a100" untouched.

Why this exists
---------------
The lab's GPU box is a private detail; the public docs should describe it
generically ("GPU server", "datacenter GPU") the same way the private control bridge
codename is already normalized in tools/scrub_public_copy.py. But "dgx"/"a100" is
also baked into code the docs legitimately reference -- the `dgxbridge` command,
`register_dgx_run()`, the `FAK_DGX_REQ_` response marker, the `"dgx"` machine_id in
benchmark JSON, the `sm_80` arch constant, artifact paths like
`experiments/qwen36/dgx-r4-20260622/`. Rewriting those would break the build and the
bench-data joins. So the rule is: rewrite PROSE, preserve IDENTIFIERS.

The prose/identifier boundary, made mechanical
----------------------------------------------
1. Fenced code blocks (``` ... ```) are skipped wholesale -- they are commands/output.
2. Inline code spans (`...`) are masked before rewriting and restored after -- they are
   identifiers (`cmd/dgxbridge`, `sm_80`, `"machine_id": "dgx"`, ...).
3. In the remaining prose, the token rules match only the UPPERCASE forms `DGX`/`A100`
   with word boundaries. Identifiers use lowercase `dgx`/`a100` or `_`-joined caps
   (`FAK_DGX_REQ_`, `DGX_RUN`) -- `\bDGX\b` does not match inside `FAK_DGX_REQ_`
   because `_` is a word character, and lowercase `dgx` is never matched. That single
   case+boundary invariant is what keeps prose and code on opposite sides of the line.

Modes
-----
  --check (default): lint. Exit 1 if any tracked doc still carries a prose hardware
          name. This is the CI/boundary-lint enforcement.
  --apply:  rewrite the files in place.
  --dry-run: print a unified diff of what --apply would change; touch nothing.
  --audit-message MSGFILE: the COMMIT-MSG gate. Exit 1 if the commit message carries a
          private hardware tell (a node label in the SUBJECT/BODY), so it is blocked before
          riding into immutable history. Escape: FLEET_ALLOW_HW=1; mode: FLEET_HW_GUARD.

File set: the doc args given on the command line, else the default doc set
(git-tracked *.md minus the generated artifacts, whose SOURCES are scrubbed instead).
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Generated docs: their bytes are emitted by a tool, so scrubbing the artifact is
# pointless (the next cron run clobbers it). Their SOURCES are scrubbed separately
# (tools/industry_scorecard.data/*.json, the generator .py comments) and the artifact
# is regenerated. Excluded from the default set + the --check lint.
GENERATED_DOCS = {
    "docs/bench-plan.md",
    "llms-full.txt",
    "llms.txt",
}
GENERATED_DIR_PREFIXES = (
    "docs/industry-scorecard/",  # generated from tools/industry_scorecard.data/*.json
)

# ONE declarative list of the bare private-hardware TOKENS — the single source of truth
# consumed by THREE surfaces: the doc-rewrite (PROSE_RULES tail, below), the doc-lint and
# the commit-message gate (RESIDUAL_TELLS / raw_hardware_hits, below). Adding a new private
# term (a host like da33, a SKU like H200) is ONE tuple here, not an edit scattered across
# the rewrite + the lint + the gate. Each entry is:
#   (pattern, replacement, competitor_guarded, is_tell)
#     pattern            — a word-bounded regex (so it never bites an identifier like
#                          cmd/dgxbridge or FAK_DGX_REQ_); may carry an inline (?i) flag.
#     replacement        — the generic public wording the rewrite substitutes.
#     competitor_guarded — True ⇒ skip the match on a line citing a competitor / third party
#                          (a published "1xA100" benchmark is a fact, not a leak). See
#                          COMPETITOR_MARKERS.
#     is_tell            — True ⇒ this token is a HARD leak signal the --check lint and the
#                          commit-message gate refuse. A competitor-guarded token (bare A100)
#                          is NOT a hard tell (a citation legitimately keeps it), so it
#                          rewrites in prose but does not block.
HARDWARE_TERMS: list[tuple[str, str, bool, bool]] = [
    # bare uppercase DGX (word-bounded; never matches `dgx`/`FAK_DGX_REQ_`).
    (r"\bDGX\b", "GPU server", False, True),
    # SXM4 — only ever the lab SKU; a hard tell. (Standalone rewrite is conservative; it
    # normally appears inside a phrase the phrase-rules already consumed.)
    (r"\bSXM4\b", "GPU", False, True),
    # digit-suffixed dgxN node label (dgx3/dgx2/dgx1 AND uppercase DGX3/DGX2) — the ONE
    # CASE-INSENSITIVE term. The `[0-9]+` requirement is itself the identifier guard
    # (FAK_DGX_REQ_ / cmd/dgxbridge have no digit after dgx); two negative lookaheads keep
    # the channel names (dgx3-control), artifact stems (dgx3-a100-*), schema ids
    # (dgx3-node-state), and the dgx1.example.lab FQDN intact while still scrubbing a
    # sentence-final "dgx1." (the char after the dot is space/EOL, not [A-Za-z0-9]).
    (r"(?i)\bdgx[0-9]+(?![-\w])(?!\.[A-Za-z0-9])", "GPU server", False, True),
    # da33 — the operator's AVX2-only CPU host (EPYC-7742, the #1127 GLM-5.2 CPU baseline
    # box). Same shape and guards as the dgxN rule: case-insensitive (prose writes both
    # "da33" and "DA33"), word-bounded, and the two negative lookaheads keep the channel
    # name (da33-control), the artifact/schema stems (da33-class, da33-node-state), and an
    # FQDN shortname (da33.example.lab) intact while still scrubbing a sentence-final
    # "da33." and a possessive "da33's". The literal "33" anchors it to the one real host —
    # "da" alone is a common English fragment, so unlike dgxN this is NOT a bare-prefix rule.
    # Replacement is "CPU server" (NOT the dgxN "GPU server"): da33 is the CPU-only box the
    # GLM-5.2 docs deliberately CONTRAST against the GPU server, so folding both to one label
    # would erase the distinction the prose is making (a CPU vs GPU throughput comparison).
    (r"(?i)\bda33(?![-\w])(?!\.[A-Za-z0-9])", "CPU server", False, True),
    # bare A100 — OVERLOADED: fak's box in fak's own runs, but a public fact when citing a
    # COMPETITOR's published hardware (Sarathi-Serve on 1xA100). competitor_guarded so the
    # rewrite skips a citation line; is_tell=False so it never BLOCKS a commit (a citation
    # legitimately keeps it). The guard is enforced via COMPETITOR_MARKERS.
    (r"\bA100\b", "datacenter GPU", True, False),
]

# UNCONDITIONAL rules: "DGX" and "SXM4" are ONLY ever the operator's lab box, and the
# multi-GPU "A100" PHRASES below ("A100 DGX", "8×A100-SXM4-40GB", "lab A100") only ever
# describe fak's box, so they are always safe to scrub. Phrases first (most specific) so
# the bare-token rules cannot re-rewrite a fragment a phrase already consumed. The bare
# unguarded token rules at the TAIL are DERIVED from HARDWARE_TERMS (see below the list)
# so the rewrite and the lint cannot drift.
PROSE_RULES: list[tuple[str, str]] = [
    # --- multi-GPU server phrasings (fak's box) -------------------------------------
    # The SKU size suffix is matched generically ([0-9]+GB), not pinned to -40GB, so a
    # -80GB / -141GB variant softens cleanly instead of leaving a "-80GB" tail behind a
    # half-rewritten phrase. The 8×DGX-A100-NNGB and 8×A100-SXM4-NNGB forms come FIRST
    # (most specific) so the bare-A100 and bare-DGX rules can't bite a fragment a phrase
    # already consumed (that adjacency is what produced "GPU server-datacenter GPU").
    # The size suffix tolerates optional markdown bold around the NNGB ("**80GB**") so a
    # table cell like "8×A100-SXM4-**80GB**" softens whole instead of falling through to
    # the bare "8×A100" rule and leaving a dangling "-SXM4-**80GB**".
    (r"8\s*[x×]\s*DGX-A100-\*{0,2}[0-9]+GB\*{0,2}", "8-GPU datacenter server"),
    (r"8\s*[x×]\s*A100-SXM4-\*{0,2}[0-9]+GB\*{0,2}", "8-GPU datacenter server"),
    (r"8\s*[x×]\s*A100-\*{0,2}[0-9]+GB\*{0,2}", "8-GPU datacenter server"),
    (r"8\s*[x×]\s*A100", "8-GPU datacenter server"),
    (r"DGX-A100-\*{0,2}[0-9]+GB\*{0,2}", "datacenter GPU"),
    (r"A100-SXM4-\*{0,2}[0-9]+GB\*{0,2}", "datacenter GPU"),
    (r"A100-SXM4", "datacenter GPU"),
    (r"A100-\*{0,2}[0-9]+GB\*{0,2}", "datacenter GPU"),
    # --- "A100 DGX" / "DGX A100" machine name (fak's box) ---------------------------
    (r"\blab A100 DGX\b", "lab GPU server"),
    (r"\bA100 DGX\b", "GPU server"),
    (r"\bDGX A100\b", "GPU server"),
    (r"\bDGX-A100\b", "GPU server"),
    # --- "DGX-<word>" hyphenated prose ("DGX-fleet readiness") — the hyphen is a word
    # boundary so the bare \bDGX\b rule below would leave "GPU server-fleet"; rewrite the
    # whole "DGX-<lowercaseword>" to "GPU server <word>". Uppercase-suffixed forms
    # (DGX-OVERNIGHT, GPU-SERVER-GLM52) are link/heading IDs already masked as path tokens.
    (r"\bDGX-([a-z]\w*)", r"GPU server \1"),
    # --- plan name (specific lowercase string, safe) --------------------------------
    (r"PLAN-model-ladder-dgx-a100", "PLAN-model-ladder-gpu-server"),
    # --- "the/a/lab DGX" ------------------------------------------------------------
    (r"\blab DGX\b", "lab GPU server"),
    (r"\bthe DGX\b", "the GPU server"),
    (r"\ba DGX\b", "a GPU server"),
    # --- bare unguarded tokens (DGX, SXM4, dgxN) appended from HARDWARE_TERMS below ---
    # (extended right after the list is closed, so the source of truth for these tells is
    # the single HARDWARE_TERMS list, not a second copy here).
]
# Append the bare UNGUARDED token rules (the hard tells) to the PROSE_RULES tail, in
# HARDWARE_TERMS order, so "phrases first, bare tokens last" holds and the rewrite tail
# stays in lockstep with the lint/gate tells (both derive from HARDWARE_TERMS).
PROSE_RULES += [(pat, repl) for (pat, repl, guarded, _tell) in HARDWARE_TERMS if not guarded]

# GUARDED rule: bare "A100" is OVERLOADED — it is fak's private box in fak's own runs
# ("(A100; cf. ...)", "on the A100"), but it is ALSO a public fact when citing a
# COMPETITOR's published benchmark hardware (Sarathi-Serve on 1xA100, "needs 8×80 GB
# H100/A100"). Scrubbing the latter would falsify a citation, so the bare-A100 rule is
# SKIPPED on any line carrying a competitor / third-party / generic-hardware marker.
# DERIVED: the single competitor_guarded entry in HARDWARE_TERMS (consumed by
# _rewrite_prose's COMPETITOR_MARKERS branch). A future guarded SKU is added there, once.
A100_BARE = next((pat, repl) for (pat, repl, guarded, _t) in HARDWARE_TERMS if guarded)

# CLEANUP: a multi-GPU phrase followed by a bare DGX ("8× A100-SXM4-40GB DGX",
# "8xA100 DGX") rewrites in two steps -- the phrase consumes the A100 part, then the
# trailing bare DGX becomes "GPU server" -- yielding a doubled "...datacenter server
# GPU server". Collapse the adjacent duplicates as a final idempotent pass.
CLEANUP_RULES: list[tuple[str, str]] = [
    (r"datacenter server GPU server", "datacenter server"),
    (r"GPU server GPU server", "GPU server"),
    (r"datacenter GPU GPU server", "GPU server"),
    # ARTICLE fix: "A100"/"DGX" take "an"/"a" by their own sound, but the generic
    # replacements ("datacenter GPU", "GPU server") both start with a consonant sound, so
    # an "an A100" / "an A100-SXM4-80GB" that softened to "an datacenter GPU" now reads
    # wrong. Demote the article. Word-bounded + case-preserving for a sentence-initial "An".
    (r"\ban (datacenter GPU|GPU server)\b", r"a \1"),
    (r"\bAn (datacenter GPU|GPU server)\b", r"A \1"),
    # LEFTOVER fix: an earlier, incomplete scrub baked "A100-80GB" → "datacenter GPU-80GB"
    # into ~6 committed docs (the bare-A100 rule fired but the "-NNGB" tail was left). The
    # size tail is now redundant noise on the generic label — drop it so a re-run heals the
    # half-scrubbed artifact instead of preserving it. (Markdown-bold tail too.)
    (r"datacenter GPU-\*{0,2}[0-9]+GB\*{0,2}", "datacenter GPU"),
]
COMPETITOR_MARKERS = re.compile(
    r"H100|Sarathi|vLLM|SGLang|TensorRT|DeepSpeed|Mooncake|DistServe|Falcon|Mistral|"
    r"Yi-|arxiv|OSDI|NSDI|MLSys|\b[1-4]\s*[x×]\s*A100\b",
    re.IGNORECASE,
)

# --check scans prose for these residual tells. Bare A100 is NOT a hard tell (competitor
# citations legitimately keep it; is_tell=False); only DGX/SXM4/dgxN are. DERIVED from the
# single HARDWARE_TERMS list, so the lint, the commit-message gate (raw_hardware_hits), and
# the rewrite (PROSE_RULES tail) all share ONE source of truth — a tell that flags a token
# the rewrite won't fix is impossible by construction.
RESIDUAL_TELLS = [pat for (pat, _repl, _guarded, tell) in HARDWARE_TERMS if tell]

FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Things a prose rule must NEVER rewrite, masked in this order before the rules run:
#   1. inline `code` spans (identifiers: cmd/dgxbridge, sm_80, "machine_id": "dgx")
#   2. markdown link/image DISPLAY TEXT that is itself a filename token (`[GPU-SERVER-OVERNIGHT-PLAN](…)`)
#   3. markdown link/image TARGETS `](...)` (paths like ...GPU-DGX-A100-...md)
#   4. bare URLs
#   5. bare filename/path tokens with a known extension (so `\bDGX-A100\b` can't mangle a
#      filename that appears outside a link). Renames are a SEPARATE deterministic pass.
# The URL/path classes EXCLUDE the \x00 sentinel ([^\s\x00]+ not \S+) so a later mask can
# never swallow an earlier mask's placeholder — e.g. on "[x](https://h)" the link-target
# mask leaves "[x\x000\x00" and the URL mask must NOT then eat "https://…\x000\x00"
# into a nested placeholder. Nested placeholders + forward _unmask was the latent bug that
# corrupted plain markdown links to "...\x000\x00 " on a first --apply.
#
# Mask 2 (the link-DISPLAY-TEXT filename) closes a false-positive class: a link whose
# VISIBLE text is a file reference — `[GPU-SERVER-OVERNIGHT-PLAN](../nightrun/GPU-SERVER-OVERNIGHT-PLAN-…md)`
# — is an identifier, not prose, exactly like the target the next mask already protects. The
# old masks shielded the `](target)` half but left the `[text]` half exposed, so `--check`
# flagged the visible filename and `--apply` corrupted the link to `[GPU server-OVERNIGHT-PLAN]`
# (a broken display/target mismatch the agent could never clear). The match is scoped to a
# FILENAME-SHAPED token only — no internal whitespace AND at least one `-`/`/`/`_`/`.` separator
# — so it masks `[GPU-SERVER-OVERNIGHT-PLAN]` and `[GPU-SERVER-CROSS-ENGINE-DATA]` but NOT real prose link
# text like `[the DGX runbook](…)` (has spaces) or a bare `[DGX]` (no separator), which stay
# prose and are still scrubbed. It excludes the \x00 sentinel for the same nesting reason.
MASK_RES = [
    re.compile(r"`[^`]*`"),
    re.compile(r"\[(?=[^\]\s\x00]*[-/_.])[^\]\s\x00]+\]"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"https?://[^\s\x00]+"),
    re.compile(r"[\w./\\-]+\.(?:md|json|go|py|sh|txt|png|svg|jpg|ya?ml|toml|csv|html)\b"),
]


def _mask_inline_code(line: str) -> tuple[str, list[str]]:
    """Replace code/link/path spans with placeholders so prose rules can't touch them."""
    spans: list[str] = []

    def grab(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    for rx in MASK_RES:
        line = rx.sub(grab, line)
    return line, spans


def _unmask(line: str, spans: list[str]) -> str:
    # Restore in REVERSE index order: if span j (j>i) ever contains span i's placeholder
    # (a nested mask), restoring j first reveals "\x00i\x00" so the i-pass still resolves
    # it. Forward order would restore i before j unhid it, leaving the inner placeholder as
    # literal "\x00i\x00" garbage in the output. (With the \x00-excluding masks above masks
    # no longer nest, but reverse restore is the belt-and-suspenders that makes a stray
    # nesting heal instead of leak.)
    for i in range(len(spans) - 1, -1, -1):
        line = line.replace(f"\x00{i}\x00", spans[i])
    return line


def _rewrite_prose(text: str) -> str:
    masked, spans = _mask_inline_code(text)
    for pat, repl in PROSE_RULES:
        masked = re.sub(pat, repl, masked)
    # Bare A100 only on lines that are NOT a competitor / third-party citation. The
    # guard checks the ORIGINAL line so a marker inside a `code` span still counts.
    if not COMPETITOR_MARKERS.search(text):
        masked = re.sub(A100_BARE[0], A100_BARE[1], masked)
    for pat, repl in CLEANUP_RULES:
        masked = re.sub(pat, repl, masked)
    return _unmask(masked, spans)


def transform(content: str) -> str:
    """Rewrite prose lines; pass fenced code blocks through verbatim."""
    out: list[str] = []
    in_fence = False
    for line in content.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        nl = ""
        body = line
        if body.endswith("\n"):
            body, nl = body[:-1], "\n"
        out.append(_rewrite_prose(body) + nl)
    return "".join(out)


def residual_hits(content: str) -> list[tuple[int, str]]:
    """Lines (outside code) that still carry a prose hardware tell."""
    hits: list[tuple[int, str]] = []
    in_fence = False
    tells = [re.compile(t) for t in RESIDUAL_TELLS]
    for n, line in enumerate(content.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        masked, _ = _mask_inline_code(line)
        if any(t.search(masked) for t in tells):
            hits.append((n, line.rstrip()))
    return hits


# Pre-compiled (pattern, competitor_guarded) for the hard tells — the commit-message /
# raw-text gate detector. Built once from HARDWARE_TERMS so the gate shares the doc-lint's
# single source of truth.
_TELL_RES = [(re.compile(pat), guarded) for (pat, _r, guarded, tell) in HARDWARE_TERMS if tell]


def raw_hardware_hits(content: str) -> list[tuple[int, str]]:
    """Lines carrying a hard hardware tell, scanned as RAW TEXT — the gate oracle.

    The sibling of residual_hits for surfaces with NO markdown contract (a commit MESSAGE,
    or any plain text): it does NOT skip fenced blocks and does NOT mask inline `code`
    spans, so a label hidden in a backtick span (``datacenter server (`dgx3`)``) or a
    fence-style comment (``# ON DGX3``) is still caught — exactly the forms residual_hits
    deliberately exempts as identifiers in docs. A competitor_guarded tell (bare A100) is
    skipped on a line carrying a COMPETITOR_MARKERS match, so a cited "1xA100" benchmark is
    not flagged. Returns [(1-based line number, line)].
    """
    hits: list[tuple[int, str]] = []
    for n, line in enumerate(content.splitlines(), 1):
        competitor = COMPETITOR_MARKERS.search(line) is not None
        for rx, guarded in _TELL_RES:
            if guarded and competitor:
                continue
            if rx.search(line):
                hits.append((n, line.rstrip()))
                break
    return hits


def default_doc_set() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files: list[Path] = []
    for rel in tracked:
        rel = rel.replace("\\", "/")
        if rel in GENERATED_DOCS:
            continue
        if any(rel.startswith(p) for p in GENERATED_DIR_PREFIXES):
            continue
        files.append(REPO / rel)
    return files


def audit_message(msg_path: str) -> int:
    """Scan a COMMIT MESSAGE file for a hardware tell — the commit-msg gate.

    The forward fix for the leak class where a private node label rides into immutable
    history via the commit SUBJECT/BODY (e.g. "docs: add the dgx3 decode") because the
    secret-needle commit-msg gate never checked hardware names. Mirrors
    scrub_public_copy.audit_message: it replicates git's own message processing — drop
    leading-`#` comment lines, stop at the scissors line git appends under `-v` — then runs
    raw_hardware_hits (RAW text, so a backtick/fence label is caught too) on the survivor.

    Mode env FLEET_HW_GUARD (block default; "off"/"warn" downgrade); escape FLEET_ALLOW_HW=1
    (a commit legitimately discussing the scrubber, or a competitor citation). Exit 1 on a
    tell, 0 if clean, 2 if the file is unreadable (the hook falls open on 2).
    """
    mode = os.environ.get("FLEET_HW_GUARD", "block")
    if mode == "off" or os.environ.get("FLEET_ALLOW_HW") == "1":
        print("hardware-tell (message): skipped (FLEET_HW_GUARD=off / FLEET_ALLOW_HW=1).")
        return 0
    try:
        with open(msg_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError as exc:
        print(f"hardware-tell (warn): could not read commit message {msg_path}: {exc}", file=sys.stderr)
        return 2
    # Keep only what git keeps in the final message: drop comment lines and everything from
    # the scissors marker down (the `-v` diff preview the content gate, not this one, owns).
    kept: list[str] = []
    for line in raw.splitlines():
        if line.startswith("# ------------------------ >8"):
            break
        if line.startswith("#"):
            continue
        kept.append(line)
    hits = raw_hardware_hits("\n".join(kept))
    if not hits:
        print("hardware-tell (message): clean.")
        return 0
    print(f"HARDWARE_TELL: the commit MESSAGE carries {len(hits)} private hardware name(s):", file=sys.stderr)
    for n, line in hits:
        print(f"  message:{n}: {line.strip()[:80]}", file=sys.stderr)
    print("  fix: describe the box generically (GPU server / datacenter GPU), per "
          "PUBLIC-SCRUB-POLICY.md.", file=sys.stderr)
    print("  override once: FLEET_ALLOW_HW=1 <git cmd>  (a competitor citation / a commit "
          "about the scrubber).", file=sys.stderr)
    return 0 if mode == "warn" else 1


def main() -> int:
    try:  # Windows consoles default to cp1252; doc prose carries ×, →, ✅, etc.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="lint; exit 1 on residual prose hardware names")
    mode.add_argument("--apply", action="store_true", help="rewrite files in place")
    mode.add_argument("--dry-run", action="store_true", help="print the diff; change nothing (default)")
    mode.add_argument("--audit-message", metavar="MSGFILE", default=None,
                      help="commit-msg gate: scan a commit message file for a hardware tell "
                           "(so a private node label in the SUBJECT/BODY is blocked before it "
                           "rides into history)")
    ap.add_argument("files", nargs="*", help="files to process (default: tracked *.md minus generated)")
    args = ap.parse_args()

    if args.audit_message:
        return audit_message(args.audit_message)

    files = [Path(f) for f in args.files] if args.files else default_doc_set()
    changed = 0
    residual_files = 0

    for f in files:
        if not f.exists():
            continue
        original = f.read_text(encoding="utf-8")
        if args.check:
            hits = residual_hits(original)
            if hits:
                residual_files += 1
                rel = f.relative_to(REPO) if f.is_absolute() else f
                for n, line in hits[:6]:
                    print(f"  {rel}:{n}: {line.strip()}")
            continue
        new = transform(original)
        if new == original:
            continue
        changed += 1
        rel = f.relative_to(REPO) if f.is_absolute() else f
        if args.apply:
            f.write_text(new, encoding="utf-8", newline="")
            print(f"  scrubbed {rel}")
        else:  # dry-run
            diff = difflib.unified_diff(
                original.splitlines(), new.splitlines(),
                fromfile=str(rel), tofile=str(rel), lineterm="",
            )
            print("\n".join(diff))

    if args.check:
        if residual_files:
            print(f"\nhardware-name lint: FAIL -- {residual_files} doc(s) carry a prose A100/DGX/SXM4 tell")
            print("fix: python tools/scrub_hardware_names.py --apply <file>  (or describe the box generically)")
            return 1
        print("hardware-name lint: clean (no prose A100/DGX/SXM4 tells in the doc set)")
        return 0
    print(f"\n{'applied' if args.apply else 'would change'} {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
