# fak issue-backlog dispatch loop — handoff

_Generated 2026-06-29 from a session that shipped 26 issues (20 example demos + 6 tooling/doc fixes) via headless lane-disjoint workers. Grounded in that session's laws + a live backlog probe. The adversarial red-team pass was cut by a session limit — treat the tooling sketches as proposals, not vetted specs._

## Goal prompt (paste-ready)

```
GOAL: Keep closing the fak issue backlog via headless lane-disjoint workers — ship verified-on-origin, decline honestly, never pad to a number.

CADENCE: Work the dispatchable leaves below in WAVES OF 5 (never 10 — 10-wide + per-worker builds saturated this Windows box to zero throughput for 25min). One worker per disjoint file-tree; prove disjointness with dos_arbitrate before launch. Never use Workflow isolation:'worktree' (every branch hits OFF_TRUNK and the whole run fails at 0 tokens).

EACH WORKER: pre-check the lane isn't already peer/cron-shipped (quiet ls / dos_verify) → build the thing using the ON-DISK fak.exe (v0.34.0), NO fresh `go build` per worker → run ONE live witness → `git commit -s -F msg -- <explicit paths>` (never `git add -A`), subject led with add/fix/implement + a `(fak <leaf>)` trailer → `git show HEAD --stat` (any deletion on an additive change = stale-base peer-wipe, stop) → re-verify the SHA is actually on origin (`git log origin/main..` / `git branch -r --contains`), not just dos_commit_audit (local-only).

DON'T probe git (log/fetch/status) from the orchestrator during a commit burst — it collides with worker index writes and hangs; audit each reported SHA with dos_commit_audit instead. If a fleet-wide GH013 push wedge hits, land your CLEAN SHA via detached-worktree cherry-pick; never amend/force-push a peer's poison commit (operator-gated).

HONESTY BAR: report on-origin ships vs done-pending-push vs not-applicable separately. A phantom/aspirational deliverable → documented walkthrough + honest not-yet, NOT a failure and NOT a fabricated file. Stop at the verified count when clean leaves thin out; most of the rest is epics/research.
```

## Operating laws (the must-follows)

Ordered by how badly skipping it bites:

1. **No `worktree` isolation for dispatch** — `Workflow isolation:'worktree'` puts each worker on a new branch; the trunk guard refuses every one (`OFF_TRUNK`) and the whole parallel run dies at 0 tokens. Use shared-tree lane-disjoint workers.
2. **Commit by explicit pathspec, then `git show HEAD --stat`** — `git commit -- <file>` commits your STALE blob and silently deletes any peer block landed after your base; `git add -A` sweeps peer hunks under your message. A non-zero deletion count on an additive change is a peer-wipe until proven otherwise.
3. **Prove lane-disjointness with dos_arbitrate before launching** — one worker per disjoint tree (new `examples/<dir>/` or a clean `internal/<pkg>/`); never two workers in one package (they race `.git/index`).
4. **No fresh `go build ./cmd/fak` per worker** — the on-disk `fak.exe` v0.34.0 works for witnesses; 10 concurrent compiles saturated the box (67min/worker, `ls` timed out). Build only if a needed verb is missing. Single biggest throughput lever.
5. **Commit the green lane FIRST, then test** — a peer `reset --hard`/`checkout` wipes uncommitted tracked edits AND untracked new files mid-lane; a committed git object survives. (`go test` is OS-blocked on this Windows box anyway — use the example-demo witness via on-disk fak.exe, or WSL/CI.)
6. **Re-verify on origin — a worker "PASS" is unverified** — `dos_commit_audit` proves LOCAL well-formedness only; a push wedge leaves a well-formed commit local-only. Check `git log origin/main..` / `git branch -r --contains <sha>` for actually-on-origin.
7. **Don't probe git from the orchestrator during a commit burst** — `git log`/`fetch`/`status` collide with workers' index writes and hang; trust the completion notification's SHA + `dos_commit_audit` (instant local object read).
8. **Lead the commit subject with add/fix/implement** — never surface/render/expose/print/pin/post; the latter ABSTAIN the commit-audit witness so `dos verify` can't plan-bind the ship. End with the `(fak <leaf>)` trailer matching the lane.
9. **Run waves of 5, cap each worker to ONE live witness** — 10-wide saturates; many witnesses → 65min/142k-tokens per worker.
10. **Pre-check each lane before dispatch** (`ls examples/<dir>/` / `dos_verify`) — peers and the night cron clear the SAME backlog; tell workers to self-dedup and refuse a phantom lane rather than fabricate a deliverable.
11. **Escape a push wedge via detached-worktree cherry-pick** — `git worktree add --detach <tmp> origin/main; cherry-pick <your-clean-sha>; push HEAD:main; remove`. Never amend/rebase/force-push/allowlist a peer's poison commit — that's an operator security decision; report not-yet with evidence.
12. **A phantom/aspirational issue → walkthrough + honest not-yet** — don't convert an open follow-on into a failure or invent a substitute file; don't pad the tally with blocked/research work.
13. **Never foreground-`sleep` to wait** — harness-blocked; use Bash `run_in_background` + until-loop, Monitor, or ScheduleWakeup.

## What to work on next

Four genuinely dispatchable leaves (dos_arbitrate-confirmed disjoint against zero live leases). **Strongest single pick: #1216** — cleanest greenfield verb with an existing spec and a natural golden-table test. _(Re-run the pre-check before dispatching — these were OPEN as of 2026-06-29 and a peer/cron may have since taken one.)_

**internal-pkg + cmd verb (greenfield, strongest):**
- **#1216** — `grammar(C8): fak claim-check` net-true-value claim-check as a single verb. Lane: `cmd/fak/claim_check.go` + `internal/claimcheck/`. No existing `*claim*` file — genuine greenfield, driven by the existing spec `docs/standards/net-true-value.md` (six-question rubric), golden-table testable. Child of epic #1208 but needs no prior rung.

**internal-pkg (self-contained pairs):**
- **#1149** — `perf(metrics): per-span cost (ns + token-delta), folded into fak rungstats`. Lane: `internal/rungobs/` + `cmd/fak/rungstats.go` (both already exist as a pair with their own tests; golden rungstats table verifies it).
- **#1180** — `feat(dormancy): fak dormancy view + fak_dormancy_* metrics`. Lane: `internal/dormancy/` + `cmd/fak/dormancy.go`. Prereq #1179 (LastActiveAt/Horizon clock primitives) already SHIPPED. **CAVEAT:** keep the worker inside `internal/dormancy` + the cmd verb + a self-registered metric source; if the `fak_dormancy_*` gauges must hang off the shared `internal/gateway` `/metrics` handler, SPLIT that wiring into a follow-on so the leaf stays disjoint.

**docs (safest doc-only ship):**
- **#1153** — `devex(tooling): the loop-stage→tool map`. Lane: one new doc, e.g. `docs/loop-stage-tool-map.md`, mapping Orient/Plan/Act/Verify/Ship/Learn → the concrete fak/dos/skill verb. Source material already in AGENTS.md + the dos answer corpus. Must add an `INDEX.md`/`llms.txt` line (repo doc-placement rule) or `commit -- path` refuses.

**What's NOT a clean ship (don't dispatch):** the backlog is overwhelmingly epics + their children. Skip: epics #1146/#1148/#1167/#1173/#1193/#1208/#1217 (umbrellas, not leaves); #1199/#1200/#1151 (blocked-on-rung — need a keystone shipped first); #1210/#1227 (pure research/design rungs); #1241 (canon secret-fixture trips push-protection — a fleet-wide security concern, not independently verifiable); #1194/#1145 (CI-surface / shared architest-tier maintenance — cross-cutting, rots RED from peer churn).

## Per-worker brief template

Copy, fill `{LANE}` / `{ISSUE}` / `{WHAT}`, spawn one headless worker per disjoint lane:

```
You are a headless worker shipping ONE fak issue end-to-end on the shared trunk (main). Work ONLY in lane {LANE}. Do not touch any file outside it.

ISSUE: #{ISSUE} — {WHAT}

1. PRE-CHECK (self-dedup): A peer or the night cron may have already shipped this. Quietly `ls {LANE}` and run dos_verify for the issue's plan/phase. If a fresh resolving commit already exists, report ALREADY-SHIPPED with the SHA and do NOT create a redundant commit.

2. BUILD THE THING in {LANE}. If the issue body's literal CLI/file deliverable does not exist (wrong module path, aspirational one-liner, phantom target file), produce a documented walkthrough citing real green tests and report an honest not-yet — do NOT fabricate a substitute file or call it a failure.

3. NO FRESH BINARY: use the on-disk fak.exe (v0.34.0) for witnesses. Do NOT run `go build ./cmd/fak`. (go test is OS-blocked here; the demo/CLI run IS the witness.) Run exactly ONE live witness.

4. COMMIT BY PATH, before any further testing:
   git commit -s -F <msgfile> -- <explicit paths in {LANE} only>
   Subject MUST lead with add/fix/implement (never surface/render/expose/print/pin) and end with the trailer (fak <leaf>). Link #{ISSUE}.
   Then `git show HEAD --stat` — if an additive change shows ANY deletions, STOP (stale-base peer-wipe) and report.

5. VERIFY ON ORIGIN: run dos_commit_audit <SHA> (must be OK / diff-witnessed), THEN confirm it actually reached origin: `git log origin/main..<SHA>` empty / `git branch -r --contains <SHA>` shows origin/main. If a GH013 push wedge blocks the stack, land your CLEAN SHA via detached-worktree cherry-pick; never touch a peer's poison commit.

6. RETURN structured: {issue, lane, sha, on_origin: true|false, witness: <one line of what you ran and saw>, status: shipped-on-origin | done-pending-push | not-applicable-walkthrough}.
```

## Tooling to build (durable, ranked)

Ranked by leverage against this session's actual pain. All five are additive over existing leaves — none rewrite the loop. Prefer Go in `cmd/fak` (verbs register via the switch in `cmd/fak/main.go` + a `cmd<Name>.go` file, NOT cobra).

**★ THE KEYSTONE — `fak ship-state` (lock-free, batched git probe).** Kills the single worst pain: every git probe hung during commit bursts (`.git/index.lock` contention + slow origin), and `GIT_OPTIONAL_LOCKS=0` / `--no-optional-locks` appears NOWHERE in the dispatch/ship tooling. Sketch: a read-only Go verb `fak ship-state --shas a,b,c --json` running every git read with `GIT_OPTIONAL_LOCKS=0` + a short timeout + offline default, answering `{sha: {in_head, on_origin, subject}}` from `git rev-list`/`for-each-ref` reads that never write `.git/index.lock`. Factor a lock-free reader package that `internal/safecommit` (already shells git) and this verb share. This is the keystone because it makes burst-time verification possible at all — laws 6 and 7 both exist only because this tool doesn't.

2. **`fak on-origin` (batch ancestry checker).** Today "is this commit on origin" needs `git branch -r --contains <sha>` PER SHA (N slow forks); dos_review/dos_commit_audit witness the local DIFF but neither answers "reached origin/main yet?". Sketch: `fak on-origin --shas <list> --remote origin --branch main --json` doing ONE `git rev-list origin/main` membership set (or a single `merge-base --is-ancestor` sweep) for O(1) lookups. Pairs with ship-state's `GIT_OPTIONAL_LOCKS=0` so the burst probe is both batched and lock-free. (Build alongside the keystone — same lock-free reader package.)

3. **`fak dispatch next` (emit next N lane-disjoint leaves).** No single command emits the next N dispatchable, mutually-disjoint leaves with lanes pre-computed — this session triaged 30 issues into batches by hand across 4 tools. Sketch: `fak dispatch next --n 10 --json` chaining `issue_lane_router` (route backlog → lanes) → `dispatch_order` (collapse stale dups + cooldown) → greedy lane-disjoint selection via dos_arbitrate (no two emitted leaves share a tree) → emit `[{issue, lane, tree, mode, confidence}]` capped at N and at preflight headroom. One call replaces the manual 4-tool assembly.

4. **`fak dispatch claim` (peer/cron already-shipped preflight).** `dispatch_preflight.py` gates HOST/ACCOUNT/CAP but not "this exact lane already has a live worker or a fresh resolving commit" — two operators (or operator + the FleetIssueDispatch cron) can dispatch the same lane. Sketch: fold THREE existing signals before SPAWN_OK — (a) dos_arbitrate over live_leases, (b) a live `resolve-*.pid` sidecar on this issue, (c) dos_verify / fresh OPEN_WITNESSED commit (already shipped, just needs closing) — refuse with a structured `LANE_LIVE`/`ALREADY_SHIPPED` reason from the dos closed vocabulary.

5. **issue-body cache (shared batched fetch).** Each worker re-ran `gh issue view` (~2min) on spawn; `issue_lane_router` and `issue_closure_audit` each hit gh independently too. Sketch (this one stays in `tools/*.py` — the family lives there): a `.dispatch-runs/issue-cache/<N>.json` write-through cache populated by one `gh issue list --json number,title,body,labels,updatedAt --limit K`; `fetch_issue`/`lane_router`/`closure_audit` read cache-first (TTL keyed on `updatedAt`), single-issue gh fetch only on miss.

Build order: ship-state + on-origin together (shared lock-free reader, kills the burst-time hang), then `fak dispatch next`/`claim` (collapses the manual triage), then the issue-body cache (removes the 2min/worker spawn stall).
```
