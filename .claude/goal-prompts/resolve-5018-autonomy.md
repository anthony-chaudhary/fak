You are a detached, unattended headless worker. Resolve ONE specific GitHub issue —
**#5018** — ship the fix WITNESSED, then stop. Other workers run beside you in the
SAME tree on ADJACENT cmd/fak files, so lane discipline is load-bearing.

## Your issue: #5018 — anti-stall for the Stop-hook refuse-and-repeat loop
Read it first: `gh issue view 5018`. STAY IN SCOPE:
- `cmd/fak/guard_stophook.go` and `cmd/fak/guard_stops.go` (+ their `_test.go`).
Primary deliverable (the part that Fixes #5018): the Stop hook must DETECT a
refuse-and-repeat stall — when it has re-nudged a session ≥2 times and the model's
stop message is substantially identical to the prior one AND no new witnessed
progress (no new commit) occurred between them — and STOP re-nudging, emitting a
single structured terminal blocked-state (a clear reason token, e.g. a
`NEEDS_HUMAN`/blocked verdict) instead of looping. A principled refusal is a
first-class terminal outcome, not something to nudge in a circle. Make the
restatement cap a named constant (default 2). Suppressing AskUserQuestion under an
active goal is a SEPARATE concern — if it doesn't fit cleanly here, file it as a
follow-up issue rather than widening this diff.

## The one loop
1. **Take a lane first.** `dos arbitrate --workspace . --lane <guess>` covering
   `cmd/fak/guard_stop*.go`. Honor a REFUSE. NEVER --force. Do NOT edit files
   outside your lane — siblings are on accounts/dispatch launcher files.
2. **Reproduce first, then fix.** Add a `guard_stophook_test.go` case: feed two
   near-identical refusals with no commit between; assert the hook emits the
   terminal blocked-state and stops re-nudging (does not emit a 3rd continuation).
   Failing before, passing after, SAME commit as the fix.
3. **Ship on the trunk, by explicit path.** Stay on `main`. Green first
   (`go test ./cmd/fak/...`, or `./test.ps1`). Then `fak commit --preview` then
   `fak commit --path cmd/fak/guard_stophook.go --path cmd/fak/guard_stops.go
   --path cmd/fak/guard_stophook_test.go -m "feat(autonomy): halt the Stop-hook refuse-and-repeat loop (fak guard)"`.
   NEVER `git add -A`; `-m` before any `--`.
4. **Close by ancestry.** `Fixes #5018` in the commit BODY. Do NOT `gh issue close`.
   Verify: `dos commit-audit --json`.
5. **Leave the tree clean, then stop.** `git status --porcelain -- cmd/fak/guard_stop*.go`
   empty, release the lane, end.

## Hard boundaries (enforced below you)
- A launch is not a ship; only a witnessed trunk commit resolves the issue.
- Out-of-scope findings → file a gh issue (dedupe → done-condition → leak-check).
- Never publish an absolute path, hostname, or personal id in issue/commit text.
- Guard refusal (OFF_TRUNK / COLLISION_RISK / …): recover per AGENTS.md or STOP.
- Do NOT "fix" early-stop by pressuring the model to abandon a correct refusal —
  the deliverable is to END the loop cleanly, not override judgment.

Do NOT end by narrating leftover work — file each follow-up as an OPEN gh issue
first (the AskUserQuestion-suppression piece almost certainly becomes one).
Self-check: `fak headless-lint --leftovers --issues-filed <N>`. Report: the issue #,
the witnessing commit SHA (or `not yet` + missing witness), follow-up issue
numbers, and whether the tree was left clean.
