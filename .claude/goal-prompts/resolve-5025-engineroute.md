You are a detached, unattended headless worker. Resolve ONE specific GitHub issue —
**#5025** — ship the fix WITNESSED, then stop. Other workers run beside you in the
SAME tree on ADJACENT cmd/fak files, so lane discipline is load-bearing.

## Your issue: #5025 — route engine by task-class (codex for grind, ultracode Opus for rigor)
Read it first: `gh issue view 5025`. As of 2026-08-14, #5025 is OPEN. Related launch work is split: #5016 and #5019 are CLOSED; #5017 and #5018 remain OPEN. Re-read the current dispatch seams before editing; recovery and worker-worktree isolation have moved since this prompt was first written. STAY IN SCOPE:
- `cmd/fak/dispatch_model_policy.go` and the work-kind → launch seam
  (`cmd/fak/dispatch_tick_route.go` / `cmd/fak/dispatch_workkind_launch_test.go`) + `cmd/fak/dispatch_model_policy_test.go`.
Primary deliverable (the part that Fixes #5025): add a work-class → ENGINE
resolution (Claude vs codex) defaulted by a small documented table — grind/gardening
classes → codex, rigor/engineering → Claude/Opus, UNKNOWN/untagged → the CURRENT
engine so a default tick is byte-identical to today. An explicit operator engine/
command pin ALWAYS wins over the table (match the resolver's "human intent wins"
precedence). This is a DEFAULT-selection seam + table + tests; do not rewire the
codex launcher itself.

## The one loop
1. **Take a lane first.** `dos arbitrate --workspace . --lane <guess>` covering
   `cmd/fak/dispatch_*.go`. Honor a REFUSE. NEVER --force. Do NOT edit files outside
   your lane — siblings are on accounts/guard launcher files.
2. **Reproduce first, then fix.** Add a route-policy test that fails before the fix in `dispatch_model_policy_test.go` (or route
   test) case: assert a gardening/grind class resolves to the codex engine, an
   engineering/rigor class resolves to Claude, an untagged class is unchanged
   (byte-identical to prior), and an explicit pin overrides the table. Failing
   before, passing after, SAME commit as the fix.
3. **Ship on the trunk, by explicit path.** Stay on `main`. Green first
   (`go test ./cmd/fak/...`, or `./test.ps1`). Then `fak commit --preview` then
   `fak commit --path cmd/fak/dispatch_model_policy.go --path <route/seam file>
   --path <test> -m "feat(dispatch): route engine by task-class (fak dispatch)"`.
   NEVER `git add -A`; `-m` before any `--`.
4. **Close by ancestry.** `Fixes #5025` in the commit BODY. Do NOT `gh issue close`.
   Verify: `dos commit-audit --json`.
5. **Leave the tree clean, then stop.** `git status --porcelain -- cmd/fak/dispatch_*.go`
   empty, release the lane, end.

## Hard boundaries (enforced below you)
- A launch is not a ship; only a witnessed trunk commit resolves the issue.
- Out-of-scope findings → file a gh issue (dedupe → done-condition → leak-check).
- Never publish an absolute path, hostname, or personal id in issue/commit text.
- Guard refusal (OFF_TRUNK / COLLISION_RISK / …): recover per AGENTS.md or STOP.
- Untagged/unknown class MUST stay byte-identical to today — the seam is additive.

Do NOT end by narrating leftover work — file each follow-up as an OPEN gh issue
first. Self-check: `fak headless-lint --leftovers --issues-filed <N>`. Report: the
issue #, the witnessing commit SHA (or `not yet` + missing witness), follow-up
issue numbers, and whether the tree was left clean.
