#!/usr/bin/env bash
# commit-audit-in-60s: watch `dos commit-audit` catch a commit whose SUBJECT claims a
# code fix but whose DIFF only edits a README — then watch the SAME subject clear when
# the diff actually does the work. No key, no model, no GPU, no network.
set -euo pipefail

DOS_BIN="${DOS_BIN:-dos}"
if ! command -v "$DOS_BIN" >/dev/null 2>&1; then
  echo "commit-audit-in-60s: 'dos' not found on PATH." >&2
  echo "  install it once:  pip install dos-kernel==0.29.0   (the bare 'dos' package is an unrelated squatter)" >&2
  echo "  or set DOS_BIN=/path/to/dos" >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "commit-audit-in-60s: 'git' not found on PATH — install git and rerun." >&2
  exit 2
fi

# A throwaway git repo so the demo never touches your real history. `dos commit-audit`
# reads the commit FROM this workspace, so we pass --workspace explicitly — without it
# `dos` resolves the workspace upward and would audit the wrong tree.
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

git init -q "$WORK"
cd "$WORK"
git config user.email "demo@example.com"
git config user.name "demo"
git config commit.gpgsign false

# A tiny project to make commits against.
printf '# calc\n\nA tiny demo project.\n' > README.md
printf 'def add(a, b):\n    return a + b\n' > calc.py
git add README.md calc.py
git commit -q -m "chore: initial project"

SUBJECT="fix: handle nulls in add()"

echo "== 1/2 THE OVER-CLAIM — subject says '$SUBJECT', diff touches only README.md =="
printf '\nMore documentation.\n' >> README.md
git add README.md
git commit -q -m "$SUBJECT"
# `dos commit-audit`'s EXIT CODE is the verdict: 0 = witnessed, 1 = an unwitnessed
# claim, 2 = an unreadable ref. Case 1 is supposed to exit 1, so we capture the code
# instead of letting `set -e` abort the demo on the deliberate "failure".
set +e
"$DOS_BIN" commit-audit HEAD --workspace "$WORK" 2>&1
code=$?
set -e
echo "  -> exit $code  (non-zero: the claim is NOT witnessed by the diff)"
echo

echo "== 2/2 THE HONEST COMMIT — same subject, diff touches calc.py (a source file) =="
printf 'def add(a, b):\n    if a is None or b is None:\n        return 0\n    return a + b\n' > calc.py
git add calc.py
git commit -q -m "$SUBJECT"
set +e
"$DOS_BIN" commit-audit HEAD --workspace "$WORK" 2>&1
code=$?
set -e
echo "  -> exit $code  (zero: the diff witnesses the claim)"
echo
echo "Same subject, same auditor, no model in the loop. The only thing that changed"
echo "between case 1 and case 2 is the DIFF — and the verdict flips on the diff, not on"
echo "the subject text. That is the verify-don't-trust split: a commit subject is"
echo "forgeable (whoever wrote the message authored it); the files it touched are not."
