<!-- fak-sessionaudit-key: sessionaudit-pi-jsonl-adapter-20261103 -->
# feat(sessionaudit): pi JSONL adapter so fak session-audit parses pi transcripts natively

```routing
lane: sessionaudit
paths: ["internal/sessionaudit/**", "cmd/fak/sessionaudit*.go"]
expected_steps: 5
```

## Working spine
Teach `fak session-audit` the pi session schema so trajectory audits over `~/.pi/agent/sessions` report real turns, tool calls, and errors instead of zero.

## Current state
Pi sessions store one JSON record per line: `type: message` envelopes with `message.role` in {user, assistant, toolResult} and content blocks of kind thinking/text/toolCall/toolResult (toolCall carries id/name/arguments; toolResult carries isError + content). `fak session-audit deep` on an active pi session reports `records=198 turns=0 tool_calls=0` while the file holds 159 thinking blocks and dozens of toolCall records — the auditor is blind to the format, so every pi trajectory campaign falls back to ad-hoc python extraction (witnessed 2026-11-03 by the GOAL-pi-traj-issue-fleet fleet: prefilter + extract scripts written by hand because no dedicated verb existed; operator quote from that window: "why are you greping? there should be dedicated verbs here").

## Why this is next
Meta-gap: it made a whole review campaign expensive and makes every future pi trajectory audit pay the same tax. Cheapest durable fix is the native adapter.

## Parent context
none (campaign: fak-private GOAL-pi-traj-issue-fleet)

## Core through-line
Pi JSONL reader in sessionaudit -> discover/audit/deep/budget/summary emit correct turns/tool_calls/errors for pi stores -> `go test ./internal/sessionaudit/... -count=1` green plus a fixture witness on a recorded pi session shape.

## Gold-plating boundary
No changes to claude/codex/opencode readers' behavior; no new verdict classes beyond what existing actions already express; no live-session tailing/watch mode.

## Done condition / witness
The defect is resolved and tests pass.
Witness: `go test ./internal/sessionaudit/... -count=1`

## Definition of done
- [ ] A pi-format detector identifies pi session files (message envelope + role set) without false positives on claude/codex/opencode shapes.
- [ ] `deep`/`audit`/`summary` report non-zero turns and tool_calls on a committed pi-session fixture (fixture mirrors the observed schema: user/assistant/toolResult roles, thinking/text/toolCall/toolResult blocks, isError flag).
- [ ] Tool-error counting includes toolResult records with isError=true and captures the error text for signature clustering.
- [ ] Compaction detection recognizes pi compaction markers if present in the observed store (else document ABSENT).
- [ ] Committed with `(fak sessionaudit)` trailer citing this ticket.

## Acceptance gate
`go test ./internal/sessionaudit/... -count=1` exit 0.

## Closure binding
Resolving commit cites this ticket and carries the `(fak sessionaudit)` ship-stamp.

## Lane
sessionaudit

## Likely files
- internal/sessionaudit/pi_reader.go
- internal/sessionaudit/pi_reader_test.go
- internal/sessionaudit/sessionaudit_test.go

## Expected steps
5

- Centrality: Core
- P1 Context: advanced - transcript parsing seam
- P2 Net value: preserved - other harness readers untouched
- P3 Adaptation: N/A - schema-driven parse
- P4 Operations: advanced - real path, witnessed blindness

## Work estimate
Estimate: 4 points

## Overall completion contribution
Contribution: 4/8 points

## Completion standard
landed
