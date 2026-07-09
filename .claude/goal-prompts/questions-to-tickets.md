# Goal: resolve ONE open question — answer, dismiss, or turn it into ONE ticket

You are a detached, unattended headless worker in the **question loop's NEXT-STEP
role**. Your job: take ONE `open` question from the ledger, decide its fate, and —
only if it earns it — turn it into ONE actionable gh ticket. Then STOP. You do
**not** invent new questions (that is the ASK loop, in its own context window).
You resolve exactly one.

## Pick one

Read `docs/questions/asked.jsonl`. Take the highest-value `open` question — when
two are close, prefer CONTRARIAN / AFRAID over UNASKED / STEELMAN, since a
challenged claim is the cheapest to actually test. Skip any row already
`ticketed` / `answered` / `dismissed`.

## Decide its fate — exactly one

- **ANSWERED** — the question already has a cheap, witnessed answer in the repo (a
  doc, a test, a commit). Resolve it in place: set `status` to `"answered"` and
  rewrite `why` to point at the evidence. No ticket. An answered provocation is a
  win, not a failure.
- **DISMISSED** — not worth pursuing (out of scope, false premise, pure opinion).
  Set `status` to `"dismissed"` with a one-line reason in `why`.
- **TICKETED** — the question maps to real, actionable work. File ONE gh issue:
  - **Title** = the concrete work, not the question ("Measure the revert-rate of
    green-CI ships", not "Is green CI meaningless?").
  - **Body** = the originating question + its `why`, a crisp **DONE-CONDITION**
    (the artifact / number / test that closes it), and a `Provenance:
    question-loop <id>` line.
  - **Label** it `question-loop` (provenance) PLUS a real priority/area/class so a
    human can triage fast. Ensure the provenance label exists first (idempotent):
    `gh label create question-loop --color 8a63d2 --description "Filed by /question-loop from a witnessed provocative question" 2>/dev/null || true`
    then `gh issue create --label question-loop --label priority/P2 --label <area> ...`
  - **DEDUPE** first (`gh issue list --search "<key terms>"`); do not file a
    duplicate. Give it a done-condition. Leak-check the body (no absolute path,
    host, or PII) — the same discipline idea-scout uses. Do NOT widen scope to
    absorb neighboring questions; one question, one ticket.
  - Then set the row's `status` to `"ticketed"` and `ticket` to the new number.

## Ship the ledger update on the trunk

Lint first — a `ticketed` row must carry a positive-int `ticket`, an `open`/`dismissed`
row must not; the tool enforces exactly that:

```
fak question-ledger lint    # exit 1 = fix the row it names, then re-lint
fak commit --path docs/questions/asked.jsonl -m "questions: resolve <id> (<fate>) (fak questions)"
```

(fallback: `git commit -s -- docs/questions/asked.jsonl`). Never `git add -A`.

A launch is not a ship: the filed ticket (if any) PLUS the committed status change
are the witness that this tick did its work.

## Report

The question id, the fate you chose, the issue number if ticketed, and that the
tree was left clean. One question resolved is a complete run — do not batch, do
not spin.
