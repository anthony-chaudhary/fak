---
name: clear-out-wip
description: Clear backed-up local repository work without sweeping up peers - inventory the dirty tree, remove proven generated junk, prioritize coherent slices, ship green partial/enabling work honestly, park or issue the rest, and use stale-work/fleet workflows only after ownership and path contracts are explicit. Use when the operator says "clear out local WIP", "clean up the dirty tree", "get this backlog of changes shipped", "sort out uncommitted work", or asks what can be committed even though the larger feature is not.
allowed-tools: Read, Bash
metadata:
  opencode: claude-only   # the commit-clean / no-bulk-stage boundary is load-bearing and Claude-only — opencode drops it
---

# /clear-out-wip

Reduce backed-up work to the smallest truthful residual set. "Clear out" means every observed
path ends in one witnessed disposition: **ship, keep active, park, remove generated junk, or file
and dispatch**. A smaller `git status` is an outcome, not the objective; never get there by hiding,
resetting, stashing, or absorbing another worker's changes.

This is the operator workflow over existing primitives. Use `/commit-clean` for each ship unit,
`/stale-work-loop` for stale or ownerless candidates, and `/fleet-wave` only for already-filed,
collision-priced follow-on units. Do not recreate those workflows here.

## Non-negotiable fences

- Work on `main`; do not create a branch, hand-rolled worktree, stash, reset, clean unignored
  files, or bulk-stage the checkout.
- Treat the shared checkout as peer-dirty. A path's presence in `git status` does not make it
  yours. Mutate or commit only paths whose ownership is evidenced by the operator, current task,
  a lane/lease record, or a coherent diff you can attribute independently.
- If `MERGE_HEAD` exists and the merge is not yours, do not finish or abort it. Leave your edits
  unstaged and wait.
- Never add a `.gitignore` rule merely to make an unknown path disappear. Ignore only a
  reproducible generated class with a named producer; verify no tracked or durable source path
  matches. Prefer allocating output through `fak tree-doctor --scratch-dir/--scratch-path`.
- Never ship broken default builds, secrets, generated binaries, private run fuel, misleading
  claims, or half of a cross-file symbol dependency. A build tag may protect genuine in-progress
  source, but is not a way to launder abandoned code into trunk.
- Do not close a parent issue merely because an independently useful enabling slice shipped.
  Describe and witness exactly the slice that landed; keep or file the remaining done-condition.

## 0. Open the lifecycle receipt

Before any mutation, run:

```bash
fak wip lifecycle begin --kind clear-out --root .
```

Keep the returned `operation_id`. This persists the `fak-wip-inventory/1` before snapshot outside the working tree. If capture fails, report the typed `WIP_LIFECYCLE_CAPTURE_FAILED` condition but continue any safety-critical recovery; never reinterpret capture failure as a zero WIP count.

## 1. Freeze a read-only census

Run the inventory before editing anything. Save bulky machine output under an allocated scratch
path, not a new repository-root filename.

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse -q --verify MERGE_HEAD
fak sweep --json
fak tree-doctor --json
```

Also inspect lane/lease and WIP records when available:

```bash
fak wip list --json
fak worktree worker list --json
```

If a verb differs in the installed build, inspect `fak <verb> -h` and use its current read-only
form. Do not replace a missing read-only verb with an ad hoc mutation.

Build a path ledger with these columns:

| Unit | Paths | Evidence of owner | Intent | Dependency seam | Test/proof | Age/origin | Disposition |
|---|---|---|---|---|---|---|---|

Group paths by one coherent outcome, not merely by directory. Keep unknown ownership as
`UNKNOWN`; do not infer it from timestamps, author names in nearby commits, or apparent quality.
Record the before count and grouped unit count so the final reduction is measurable.

## 2. Retire hygiene debt first, but only proven hygiene

Classify maintenance residue before feature work because it obscures the real queue:

1. **Generated junk:** known outputs, temp binaries, logs, caches, or allocated scratch. Preview
   ignored cleanup with `fak tree-doctor --sweep-scratch --dry-run`; apply only after every listed
   path is known disposable. Use `fak sweep --clean-junk` only for paths that the sweep itself
   freshly classifies as junk.
2. **Durable generated output:** promote a successful reusable artifact up the tooling ladder
   instead of discarding it. In this repo that means a Go leaf/verb plus witness, not a new Python
   helper.
3. **Ignore gap:** identify the producer, choose the narrowest pattern, prove it does not match
   tracked/durable files (`git ls-files` plus a pattern-specific check), edit `.gitignore`, rerun
   the producer or equivalent witness, and commit the maintenance change as its own coherent unit.
4. **Unknown or possibly human-authored file:** retain it unchanged and route it to ownership
   adjudication. It is not hygiene.

Never use broad unignored cleanup, broad wildcard ignores, or removal as a triage shortcut.

## 3. Score units for the next action

Apply this deterministic order; do not prioritize by easiest diff alone:

1. **Safety and trunk health:** secret exposure, build breakage, merge residue, generated binary
   collisions, or an active lease conflict.
2. **Ready-to-ship user value:** a coherent outcome with an appropriate green witness.
3. **Enabling value:** an independently useful seam that unblocks other work even if the parent
   feature remains disabled.
4. **Decay risk:** old, ownerless, duplicated, or conflict-prone work.
5. **Cosmetic cleanup:** formatting or organization with no independent outcome.

Within a tier, prefer highest user/unblock value, then strongest witness, then smallest collision
surface. Use the FAK value frame for every proposed ship unit: **For / Problem / Today / Better
because / Witness**. Reject a slice that cannot explain its own value without promising the
unfinished parent.

Assign exactly one next action:

- `SHIP_NOW` - coherent, owned, truthful, and witnessable now.
- `FINISH_SMALL_GAP` - one bounded repair/test makes the coherent unit shippable in this session.
- `SHIP_ENABLER` - useful and safe independently, while the larger feature remains openly incomplete.
- `KEEP_ACTIVE` - current owner/lease and near-term continuation are evidenced.
- `STALE_ADJUDICATE` - ownerless/old/ambiguous; use `/stale-work-loop`, whose discovery phase does
  not edit the candidate.
- `PARK` - valuable bytes must be preserved but are not active or safely shippable; use the repo's
  witnessed parking/WIP mechanism, never a stash or mystery archive.
- `DELETE_JUNK` - only the generated-junk proof above permits this.
- `FILE_FOLLOWUP` - concrete residual work with a done-condition and explicit path tree.

## 4. Decide whether incomplete work can ship

A parent feature need not be fully enabled for a slice to land. The slice may ship as
`SHIP_ENABLER` only when **all** rows pass:

| Gate | Required evidence |
|---|---|
| Independent value | A caller, operator, or next implementation can use the slice without pretending the parent is done. |
| Coherent boundary | Named paths form one leaf and contain both sides of any changed contract. |
| Safe default | Default build/runtime behavior stays green and fail-closed; disabled means honestly unreachable, not silently broken. |
| Durable witness | Focused test or captured artifact proves the slice's actual claim. |
| Honest surface | Help, docs, claims, issue state, and flags do not advertise unfinished behavior. |
| Residual tracked | The remaining parent done-condition is already an open issue or is filed before the run ends. |

Examples that can qualify: a tested parser used by an existing path, a fail-closed interface plus
fake, a disabled-but-selfchecking demo spine, a migration reader landed before its writer, or a
narrow `.gitignore` producer fix. Orphan structs, speculative frameworks, tests for unreachable
behavior, and a flag wired to a stub do not qualify.

If only a small gap prevents qualification, finish that gap before committing. Otherwise keep,
park, or issue it; do not lower the gates to reduce the dirty count.

## 5. Ship serially and re-census after every unit

For each `SHIP_NOW` or `SHIP_ENABLER` unit:

1. Confirm ownership and inspect the complete diff for the explicit paths.
2. Run the narrow witness, then `fak validate --mine <path> ...` for the committed tip plus only
   that unit. On this Windows host, use the repository's WSL test path where tests are required.
3. Follow `/commit-clean`: preview the subject, commit exactly the paths under the lock, verify the
   committed path set, and push through the safe sync path. One coherent unit, one commit, one leaf.
4. Cite the changed `module@rev` and witness. State `parent remains open` for an enabling slice.
5. Rerun `fak sweep --json`; peers may have changed the tree while the unit landed. Never apply a
   stale census to the next commit.

Do not batch unrelated maintenance and feature work just because both are ready. Do not repeatedly
patch the same issue across several commits; finish its coherent acceptance boundary first.

## 6. Distribute and start contract-ready residual work

Local dirty paths are not worker fuel. First convert residuals into dedicated issues with:

- one outcome and done-condition;
- explicit owned path tree;
- current candidate/evidence digest;
- acceptance witness;
- dependency and collision notes; and
- retain/update/remove authority where stale work is involved.

Use `/stale-work-loop` for candidate adjudication. A one-unit or same-path queue is serial work,
not a fleet wave. For multiple disjoint issues, use `/issue-orchestrator` (`fak issue-orchestrator --plan-waves`)
and split overlaps into later waves before generating or starting sessions.

### Execution intent

Classify the invocation before launch:

- **Plan-only:** "inspect", "triage", "what can ship?", or "plan a clear-out" produces the census,
  dispositions, issue contracts, and priced wave; it starts no session.
- **Live clear-out:** "clear out", "work through", "distribute", "start sessions", "run the
  backlog", or an unattended/deadline request authorizes the native dry-run -> live launch path for
  contract-ready residual issues. Local `SHIP_NOW` units still land serially first.
- **Explicit no-fan-out:** "locally", "do not launch", or an equivalent constraint keeps all work
  serial even if several issues are ready.

When intent is unclear, complete the safe local spine and render the dry-run receipt. Never claim
that work was distributed merely because issues or a plan were created.

### Generate the guarded wave

Use the installed native dispatcher as the single workflow generator and launcher. Do not handwrite
child prompts, spawn raw Codex processes, wrap the whole wave in one parent guard, or mix in the
legacy launcher.

```powershell
fak dispatch wave --help
fak fleet-accounts wave --count <N> --work-kind codex --product codex --json
fak fleet-accounts status --provider codex --json
fak dispatch wave --count <N> --backend codex --work-kind codex --max-workers <C> `
  --goal high-priority --workspace . --json
```

Choose `N` from the number of contract-ready issues, not the number of dirty paths. Choose `C` no
higher than the collision-safe concurrency and independently observed host/account headroom. Read
the dry-run receipt and require all of the following before launch:

- selected issue IDs and bounded issue-derived worker fuel;
- pairwise-safe lane/path leases, with overlaps moved to later waves;
- distinct admitted account/session slots and a truthful shortfall;
- guarded worker commands using the Codex backend;
- implement -> test -> witness -> commit -> push ownership; and
- typed refusal/downsize reasons for every candidate not started.

A refusal is a recovery input, not permission to retry blindly. Apply the receipt's recovery action,
re-price once state changes, and preserve the refusal in the run report.

### Start sessions once

For live-clear-out intent and a clean dry run, launch the exact plan once:

```powershell
fak dispatch wave --count <N> --backend codex --work-kind codex --max-workers <C> `
  --goal high-priority --workspace . --live --json
```

Record the wave ID plus each issue, lane, account/session slot, PID, and run/log path. `--count` is
the total requested session slots over refill time; `--max-workers` is the concurrent process
ceiling. Do not rerun the launcher to top up. Let the native refill controller use newly available
capacity; use `fak dispatch auto` only when the installed `/fleet-wave` contract directs it.

### Monitor and harvest

Monitor from outside worker-owned trees and judge progress by independently readable state:

```powershell
fak dispatch status --runs-dir <RUNS_DIR> --json
fak dispatch progress --target <N> --json
fak dispatch closure-audit --workspace . --json
```

Follow `/fleet-wave` for liveness and account recovery and `/wave-harvest` for reconciliation.
`SHIPPED` requires git/read-back, focused green acceptance evidence, issue state, and DOS/commit
witnesses; a process exit, worker log, or final message is insufficient. Release leases and account
slots after reconciliation, then run another census. Start a later collision wave only from the
new state and only while live-clear-out intent remains in force.
## 6.5. Close the lifecycle receipt

After the last mutation and verification, run:

```bash
fak wip lifecycle end --root . --id <operation_id>
```

The resulting `fak-wip-lifecycle/1` receipt links ordered before/after inventory artifacts under one operation identity. Include its `receipt_path` in the final report.

## 7. Exit with a residual proof

Repeat the read-only census and reconcile every original unit plus any newly observed path. Report:

- before/after dirty path and grouped-unit counts;
- shipped commits with `module@rev` and focused witness;
- generated junk removed and the producer/classification that proved disposal;
- active or parked units with owner and recovery command/location;
- filed residual issue numbers and any launched/harvested wave;
- unknown/peer-owned paths explicitly untouched; and
- the next checkable action.

The run is complete only when every observed unit has a disposition and every named follow-up is an
open issue. A still-dirty tree can be a successful clear-out when the residue is owned and typed; a
clean-looking tree with hidden, stashed, removed, or untracked obligations is a failure.
