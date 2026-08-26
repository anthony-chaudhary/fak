# Mac guard dogfood readout — 2026-08-25

`fak guard` completed a real Codex shell turn on this Mac from the clean,
installed trunk build `6f65ee0de25b`. The shell call crossed the capability
floor, the guard sampled the live child tree eight times, the audit chain is
intact, and the default Darwin session journal needed no environment override.

## Final witness

The installed binary reported `fak 0.45.0`, build `6f65ee0de25b`, Go 1.26.6,
Darwin/arm64. `fak self-update --force` built and gated `origin/main`, replaced
the stale scheduled binary, and read it back at that revision.

The final run wrapped Codex 0.149.1 and required one real shell call:

```text
fak guard --provider openai --expose-profile headless \
  --audit .dispatch-runs/guard-audit/final-shell-probe-6f65ee0de.jsonl \
  --banner compact --split off -- \
  codex exec --ephemeral -C . --approve-for-me \
  'Use the shell exactly once to run pwd; sleep 4.'
```

Observed result:

- Codex called `exec_command`; guard allowed it and Codex returned the repo root.
- The 12-second session completed after eight resource samples. Guard peaked at
  60.3 MiB RSS; the owned child tree peaked at 185.6 MiB RSS.
- The audit contains two rows: one `ALLOW` for `exec_command` and one clean
  `CHILD_EXIT`. `fak audit verify` read back both hash-chained rows intact.
- `~/Library/Application Support/fak/session-journal/events.jsonl` recorded the
  guarded Codex registration through terminal state `completed` without
  `FAK_SESSION_JOURNAL`.
- The sweep exercised at least five guarded child launches: the first bounded
  crash-restart run alone emitted five child-exit rows, followed by the Darwin
  RSS failure, the post-fix model-catalog probe, and this successful shell run.
  The owned tree reached three processes during the exit-race reproduction; the
  one-second monitor interval stayed below the three-second intervention target.

The final module revisions are `internal/sessionjournal@r16+g759168c95`,
`internal/procguard@r17+gd527c9506`, and
`internal/gateway@r714+g6f65ee0de`.

## Defects and disposition

| Finding | Disposition |
|---|---|
| Darwin defaulted the session journal to root-owned `/var/lib` | Fixed and closed in #9046 |
| Darwin RSS census treated normal descendant exit races as monitor failure | Fixed and closed in #9056 |
| Gateway emitted internal service-tier fields that Codex 0.149.1 could not decode | Fixed and closed in #9058 |
| Guarded Codex still needs a paired direct-versus-guard MCP/custom-tool catalog witness | Reopened #8566; the final built-in shell witness is green, but the broader catalog claim remains open |
| Clean `CHILD_EXIT` is outside the guard RSI fold's closed verdict vocabulary | Filed as #9061 |
| Live RSI routing said it created an issue while returning `ok:false` and no issue | Filed as #9062 |
| Isolated worker landing omitted declared untracked files or left reverse-staged residue | Filed as #9063 |

## Verification

- Focused gateway/modelroute tests passed; the three package-wide rich-dashboard
  failures passed five times each in isolation and remain unrelated ordering
  flakes rather than evidence for this change.
- Committed-tip `fak-dev ci-preflight` and the pre-push trunk build gate passed at
  `6f65ee0de25b`.
- `fak guard-rsi-scorecard --json` reported grade A, zero guard-RSI debt, and
  169 real journal rows after this run.
- The final RSI replay kept a witnessed 50-to-100 quality improvement, then
  routed its single unknown-verdict honesty hole to #9061.

This closes the #8981 dogfood unit: the run used the repo's live guarded Codex
path, its readout is committed, and every surfaced defect has a durable tracker.
