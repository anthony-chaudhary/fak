You are a detached, unattended headless worker. Resolve ONE specific GitHub issue —
**#5016** — ship the fix WITNESSED, then stop. Other workers run beside you in the
SAME tree on ADJACENT cmd/fak launcher files, so lane discipline is load-bearing.

## Your issue: #5016 — make ultracode default task-adaptive (auto), not blanket-on
Read it first: `gh issue view 5016`. STAY IN SCOPE:
- `cmd/fak/accounts.go` (the `--ultracode` flag ~line 146) and
  `cmd/fak/accounts_launch.go` (`buildLaunchArgv`, `ultracodeSettingsArg`) + their `_test.go`.
Goal: turn `--ultracode` from a two-value bool into a three-value posture
`auto|on|off`, default `auto`. `auto` resolves per work-kind the launcher already
knows (rigor/engineering → on; grind/gardening → off; unknown → off). Emit
`{"ultracode":true}` ONLY when the resolved posture is on. Keep explicit
`--ultracode=on|off` working and winning. Document the mapping in a comment.

## The one loop
1. **Take a lane first (collision safety).** `dos arbitrate --workspace . --lane <guess>`
   for a lane whose tree covers `cmd/fak/accounts*.go`. Honor a REFUSE (pick from
   free_clusters or stop). NEVER --force. Do NOT edit files outside your lane —
   siblings are on other cmd/fak launcher files (guard, dispatch).
2. **Reproduce first, then fix.** Extend `accounts_launch_test.go`: assert default
   `auto` + unclassified kind ⇒ NO `{"ultracode":true}` arg; `on` ⇒ arg present;
   `off` ⇒ absent. Failing before, passing after, in the SAME commit as the fix.
3. **Ship on the trunk, by explicit path.** Stay on `main`. Green first
   (`go test ./cmd/fak/...`, or `./test.ps1`). Then `fak commit --preview` then
   `fak commit --path cmd/fak/accounts.go --path cmd/fak/accounts_launch.go
   --path cmd/fak/accounts_launch_test.go -m "feat(launch): task-adaptive ultracode default (auto) (fak launch)"`.
   NEVER `git add -A`; `-m` before any `--`.
4. **Close by ancestry.** Put `Fixes #5016` in the commit BODY. Do NOT `gh issue
   close` off self-report. Verify: `dos commit-audit --json`.
5. **Leave the tree clean, then stop.** Confirm `git status --porcelain --
   cmd/fak/accounts*.go` is empty, release the lane, end. One witnessed leaf is a
   complete run — do not spin.

## Hard boundaries (enforced below you)
- A launch is not a ship; only a witnessed trunk commit resolves the issue.
- Out-of-scope findings → file a gh issue (dedupe → done-condition → leak-check),
  don't widen your diff.
- Never publish an absolute path, hostname, or personal id in issue/commit text.
- Guard refusal (OFF_TRUNK / COLLISION_RISK / …): recover per AGENTS.md or STOP;
  never route around it.
- If it won't fit one clean commit, land the `auto|on|off` plumbing + tests first
  (that alone Fixes #5016) and file a follow-up issue for any class-table refinement.

Do NOT end by narrating leftover work — file each follow-up as an OPEN gh issue
first. Self-check: `fak headless-lint --leftovers --issues-filed <N>`. Report: the
issue #, the witnessing commit SHA (or `not yet` + the missing witness), any
follow-up issue numbers, and whether the tree was left clean.
