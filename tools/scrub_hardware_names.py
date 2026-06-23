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
    "docs/dispatch-status.md",
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
    (r"8\s*[x×]\s*A100-SXM4-40GB", "8-GPU datacenter server"),
    (r"8\s*[x×]\s*A100-40GB", "8-GPU datacenter server"),
    (r"8\s*[x×]\s*A100", "8-GPU datacenter server"),
    (r"A100-SXM4-40GB", "datacenter GPU"),
    (r"A100-SXM4", "datacenter GPU"),
    (r"A100-40GB", "datacenter GPU"),
    # --- "A100 DGX" / "DGX A100" machine name (fak's box) ---------------------------
    (r"\blab A100 DGX\b", "lab GPU server"),
    (r"\bA100 DGX\b", "GPU server"),
    (r"\bDGX A100\b", "GPU server"),
    (r"\bDGX-A100\b", "GPU server"),
    # --- plan name (specific lowercase string, safe) --------------------------------
    (r"PLAN-model-ladder-dgx-a100", "PLAN-model-ladder-gpu-server"),
    # --- "the/a/lab DGX" ------------------------------------------------------------
    (r"\blab DGX\b", "lab GPU server"),
    (r"\bthe DGX\b", "the GPU server"),
    (r"\ba DGX\b", "a GPU server"),
    # --- bare uppercase DGX (word-bounded; never matches `dgx`/`FAK_DGX_REQ_`) ------
    (r"\bDGX\b", "GPU server"),
]

# GUARDED rule: bare "A100" is OVERLOADED — it is fak's private box in fak's own runs
# ("(A100; cf. ...)", "on the A100"), but it is ALSO a public fact when citing a
# COMPETITOR's published benchmark hardware (Sarathi-Serve on 1xA100, "needs 8×80 GB
# H100/A100"). Scrubbing the latter would falsify a citation, so the bare-A100 rule is
# SKIPPED on any line carrying a competitor / third-party / generic-hardware marker.
A100_BARE = (r"\bA100\b", "datacenter GPU")
COMPETITOR_MARKERS = re.compile(
    r"H100|Sarathi|vLLM|SGLang|TensorRT|DeepSpeed|Mooncake|DistServe|Falcon|Mistral|"
    r"Yi-|arxiv|OSDI|NSDI|MLSys|\b[1-4]\s*[x×]\s*A100\b",
    re.IGNORECASE,
)

# --check scans prose for these residual tells (uppercase, word-bounded). Bare A100 is
# NOT a hard tell (competitor citations legitimately keep it); only DGX/SXM4 are.
RESIDUAL_TELLS = [r"\bDGX\b", r"\bSXM4\b"]

FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Things a prose rule must NEVER rewrite, masked in this order before the rules run:
#   1. inline `code` spans (identifiers: cmd/dgxbridge, sm_80, "machine_id": "dgx")
#   2. markdown link/image TARGETS `](...)` (paths like ...GPU-DGX-A100-...md)
#   3. bare URLs
#   4. bare filename/path tokens with a known extension (so `\bDGX-A100\b` can't mangle a
#      filename that appears outside a link). Renames are a SEPARATE deterministic pass.
MASK_RES = [
    re.compile(r"`[^`]*`"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"https?://\S+"),
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
    for i, s in enumerate(spans):
        line = line.replace(f"\x00{i}\x00", s)
    return line


def _rewrite_prose(text: str) -> str:
    masked, spans = _mask_inline_code(text)
    for pat, repl in PROSE_RULES:
        masked = re.sub(pat, repl, masked)
    # Bare A100 only on lines that are NOT a competitor / third-party citation. The
    # guard checks the ORIGINAL line so a marker inside a `code` span still counts.
    if not COMPETITOR_MARKERS.search(text):
        masked = re.sub(A100_BARE[0], A100_BARE[1], masked)
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
