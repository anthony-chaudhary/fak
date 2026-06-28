#!/usr/bin/env python3
"""scrub_hardware_names.py -- rewrite the operator's private lab hardware names
(NVIDIA "A100" / "DGX" / "SXM4") out of human-readable DOCUMENTATION PROSE, while
leaving every CODE/DATA IDENTIFIER that happens to contain "dgx"/"a100" untouched.

Why this exists
---------------
The lab's GPU box is a private detail; the public docs should describe it
generically ("GPU server", "datacenter GPU") the same way the Slack control-bridge
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

File set: the doc args given on the command line, else the default doc set
(git-tracked *.md minus the generated artifacts, whose SOURCES are scrubbed instead).
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
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

# UNCONDITIONAL rules: "DGX" and "SXM4" are ONLY ever the operator's lab box, and the
# multi-GPU "A100" PHRASES below ("A100 DGX", "8×A100-SXM4-40GB", "lab A100") only ever
# describe fak's box, so they are always safe to scrub. Phrases first (most specific) so
# the bare-token rules cannot re-rewrite a fragment a phrase already consumed.
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
    # (DGX-OVERNIGHT, DGX-GLM52) are link/heading IDs already masked as path tokens.
    (r"\bDGX-([a-z]\w*)", r"GPU server \1"),
    # --- plan name (specific lowercase string, safe) --------------------------------
    (r"PLAN-model-ladder-dgx-a100", "PLAN-model-ladder-gpu-server"),
    # --- "the/a/lab DGX" ------------------------------------------------------------
    (r"\blab DGX\b", "lab GPU server"),
    (r"\bthe DGX\b", "the GPU server"),
    (r"\ba DGX\b", "a GPU server"),
    # --- bare uppercase DGX (word-bounded; never matches `dgx`/`FAK_DGX_REQ_`) ------
    (r"\bDGX\b", "GPU server"),
    # --- digit-suffixed dgxN machine label (dgx3/dgx2/dgx1, AND uppercase DGX3/DGX2) —
    # the ONE intentional CASE-INSENSITIVE prose rule. Unlike bare DGX (uppercase-only,
    # because lowercase `dgx` is an identifier), the dgxN node labels appear in prose in
    # BOTH cases ("ran on dgx3", "serve GLM-5.2 on DGX3", "DGX3's host-CPU tier"), so case
    # cannot separate prose from code here — the lab uses the same label either way. The
    # `[0-9]+` requirement is itself the identifier guard (FAK_DGX_REQ_ has no digit after
    # DGX; cmd/dgxbridge has no digit), and two negative lookaheads re-establish the rest
    # of the prose/identifier boundary for the digit-suffixed form:
    #   (?![-\w])         — refuse a hyphen/word-char continuation, so the raw-prose
    #                       channel names (dgx3-control), the dgx3-a100-* artifact stem,
    #                       the dgx3-node-state schema stem, and dgx3_glm_* survive even
    #                       when the `.sh`/`.json`/`.md` path mask does not fire.
    #   (?!\.[A-Za-z0-9]) — refuse a dot that STARTS another id segment, so the FQDN
    #                       dgx1.example.lab and the .v1/.sh suffixes survive — but NOT a
    #                       sentence-final period: "ran on dgx1." still scrubs, because the
    #                       char after the dot is a space/EOL/`)`, not [A-Za-z0-9].
    # No competitor guard (unlike bare A100): dgxN is only ever fak's lab box — no third
    # party cites "dgx3" — so the asymmetry is intentional, not an oversight.
    (r"(?i)\bdgx[0-9]+(?![-\w])(?!\.[A-Za-z0-9])", "GPU server"),
]

# GUARDED rule: bare "A100" is OVERLOADED — it is fak's private box in fak's own runs
# ("(A100; cf. ...)", "on the A100"), but it is ALSO a public fact when citing a
# COMPETITOR's published benchmark hardware (Sarathi-Serve on 1xA100, "needs 8×80 GB
# H100/A100"). Scrubbing the latter would falsify a citation, so the bare-A100 rule is
# SKIPPED on any line carrying a competitor / third-party / generic-hardware marker.
A100_BARE = (r"\bA100\b", "datacenter GPU")

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
# citations legitimately keep it); only DGX/SXM4 are. The lowercase dgxN tell carries the
# SAME pattern + lookaheads as the PROSE_RULES rewrite above, so the lint and the rewrite
# share one source of truth — a tell that flags a token the rule won't fix (a raw-prose
# dgx3-control / dgx1.example.lab / dgx3-node-state.v1) is impossible by construction.
RESIDUAL_TELLS = [r"\bDGX\b", r"\bSXM4\b", r"(?i)\bdgx[0-9]+(?![-\w])(?!\.[A-Za-z0-9])"]

FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Things a prose rule must NEVER rewrite, masked in this order before the rules run:
#   1. inline `code` spans (identifiers: cmd/dgxbridge, sm_80, "machine_id": "dgx")
#   2. markdown link/image TARGETS `](...)` (paths like ...GPU-DGX-A100-...md)
#   3. bare URLs
#   4. bare filename/path tokens with a known extension (so `\bDGX-A100\b` can't mangle a
#      filename that appears outside a link). Renames are a SEPARATE deterministic pass.
# The URL/path classes EXCLUDE the \x00 sentinel ([^\s\x00]+ not \S+) so a later mask can
# never swallow an earlier mask's placeholder — e.g. on "[x](https://h)" the link-target
# mask (2) leaves "[x\x000\x00" and the URL mask (3) must NOT then eat "https://…\x000\x00"
# into a nested placeholder. Nested placeholders + forward _unmask was the latent bug that
# corrupted plain markdown links to "...\x000\x00 " on a first --apply.
MASK_RES = [
    re.compile(r"`[^`]*`"),
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
    ap.add_argument("files", nargs="*", help="files to process (default: tracked *.md minus generated)")
    args = ap.parse_args()

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
