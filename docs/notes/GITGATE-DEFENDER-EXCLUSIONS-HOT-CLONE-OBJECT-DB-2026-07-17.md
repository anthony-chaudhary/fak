# Windows Defender exclusions for the hot-clone git object DB — 2026-07-17

Operator runbook for the environment half of #4602 (tracked as #5080, Phase 5 of
#4602's plan). The code half — object-DB maintenance that actually folds loose
objects — is being fixed in phases (Phase 1 landed in `e2a19b789`,
`internal/gitgate`); this note covers the piece fak deliberately does **not**
automate: the host's antivirus posture. fak must never silently reconfigure a
host's AV; the fix here is to state the required exclusions, the exact elevated
commands, and the honest trade-off, and let the operator apply them.

Status: **advisory, operator-applied**. Nothing in fak reads or enforces this.

## The problem this addresses

#4602 (measured 2026-07-13, HEAD `b6af74709`, git 2.51.0.windows.1) diagnosed
intermittent ~2-minute cold git stalls on the always-hot Windows shared clone
`C:\work\fak`:

- 67,885 loose objects (74.2 MiB) in `.git/objects`, ~58,382 of them
  unreachable (86% of loose).
- Warm git is fine (`git status --porcelain` 136 ms, `git log -20` 115 ms) —
  the stall regime is cold-start object-store walks.
- Windows Defender real-time protection is ON, and the object DB was not known
  to be excluded, so every cold walk drags tens of thousands of loose-object
  files through the AV scanner.

The Defender scan cost **compounds with, and is independent of,** the
loose-object backlog itself: even after `fak git-maint`'s grace tier
(`cmd/fak/git_maint.go`, `internal/gitgate/gitmaint.go`) folds loose objects
into packs, an unexcluded object DB still pays scan cost on every cold walk of
whatever files remain.

**Provenance of the perf claim:** the Defender contribution to the ~2-minute
stall is **modeled, not measured**. No before/after with the exclusions applied
has been captured on this host, so no speedup number is claimed here — per
[`docs/standards/net-true-value.md`](../standards/net-true-value.md), the
measured before/after is `not yet`. What *is* witnessed is the input side:
real-time protection on, a large unexcluded loose-object store, and stalls
confined to the cold-walk regime where AV scanning sits on the path (#4602).

## The exclusion set

Three items, ordered narrow-to-broad. The host may already carry the broader
fleet baseline from
[`docs/host-defender-exclusions.md`](../host-defender-exclusions.md) (which
excludes the whole `C:\work\fak` tree); if it does, item 1 is already covered
and only items 2–3 add anything. If the operator prefers a **narrower** posture
than excluding the whole repo tree, item 1 is the minimal path that addresses
the #4602 stall.

1. **The object DB (narrow) or the clone root (broad).**
   `C:\work\fak\.git\objects` is where the cold walk happens; excluding just it
   keeps Defender scanning the working tree (where fetched/authored content
   actually lands) while removing the scanner from the loose-object hot path.
   Excluding the clone root `C:\work\fak` instead is the broader fleet-baseline
   choice — simpler, also covers working-tree compile churn, but gives up
   real-time coverage of everything under the repo.

2. **The per-worker worktree roots.** The sanctioned detached-worktree flow
   (`fak worktree worker prepare|land|reap`, #1334 / epic #3165) creates
   worktrees under `%LOCALAPPDATA%\Fleet\worker-worktrees` by default
   (override: `FLEET_WORKER_WORKTREE_ROOT`; dir names carry the
   `fak-worker-wt-` marker — `internal/workerworktree/workerworktree.go`,
   `DefaultRoot`). Linked worktrees share the main clone's `.git/objects`, so
   item 1 covers their object reads/writes — but each worktree's own checkout
   plus its in-worktree `GOCACHE`/`GOTMPDIR` build churn is fresh scan load per
   prepare. Add any scratch worktree roots the host uses to the same list.

3. **The `git.exe` process.** The fleet baseline excludes the Git *install
   path* (`C:\Program Files\Git`) and the shells, but not the `git.exe`
   *process image*. A process exclusion stops Defender scanning the files
   `git.exe` touches wherever they live — the cold object walk itself — at the
   cost of being broader than any path exclusion (see trade-off below).

## Apply it (elevated PowerShell)

`Add-MpPreference` needs administrator rights and **appends** (re-running is
safe and additive). It survives reboot.

Narrow variant (object DB only, plus worktrees and git.exe):

```powershell
Add-MpPreference -ExclusionPath 'C:\work\fak\.git\objects', "$env:LOCALAPPDATA\Fleet\worker-worktrees"
Add-MpPreference -ExclusionProcess 'git.exe'
```

Broad variant (fleet baseline of `docs/host-defender-exclusions.md` + this
note's additions):

```powershell
Add-MpPreference -ExclusionPath 'C:\work\fak', "$env:LOCALAPPDATA\Fleet\worker-worktrees"
Add-MpPreference -ExclusionProcess 'git.exe'
```

Adjust `C:\work\fak` to the actual clone root, and add the
`FLEET_WORKER_WORKTREE_ROOT` override path instead if the host sets one.

## Verify (elevated)

Reading exclusions back **also** requires an elevated shell — from a normal
session `Get-MpPreference` answers *"Must be an administrator"*, which is why an
unexcluded host stays silently unexcluded until someone checks:

```powershell
Get-MpPreference | Select-Object ExclusionPath, ExclusionProcess
```

The host is covered for this note's scope when the object DB (or clone root),
the worker-worktree root, and `git.exe` appear in the output.

## The honest trade-off

Every exclusion trades security surface for throughput; state what is given up:

- **`.git/objects` (narrow):** Defender stops real-time-scanning git object
  files. Objects are zlib-deflated content-addressed blobs — not directly
  executable, and anything malicious in them only becomes dangerous when
  checked out into a working tree, which stays scanned under the narrow
  variant. This is the cheapest coverage give-up of the three.
- **`C:\work\fak` (broad):** Defender no longer scans anything written under
  the repo tree, including working-tree files an agent fetches or authors.
  Standard trade on a dev/build box; a real coverage reduction on a host that
  also handles untrusted input.
- **Worker-worktree root:** same character as the clone-root exclusion, scoped
  to short-lived build checkouts under `%LOCALAPPDATA%`. Do **not** widen it to
  all of `%LOCALAPPDATA%` or `%TEMP%` — droppers land in temp roots, and
  keeping those scanned is the point of leaving Defender on.
- **`git.exe` process:** broader than any path exclusion — it exempts files
  that image touches *wherever it runs*, not just under the repo. Skip it if
  the narrow-path variant proves sufficient on measurement; keep the process
  list short.

Reverse any of it with the matching elevated `Remove-MpPreference`:

```powershell
Remove-MpPreference -ExclusionPath 'C:\work\fak\.git\objects', "$env:LOCALAPPDATA\Fleet\worker-worktrees"
Remove-MpPreference -ExclusionProcess 'git.exe'
```

## Measured before/after — `not yet`

Per the net-true-value standard, a speedup may only be quoted against the tuned
baseline with a witness. None has been captured: applying these exclusions
needs an elevated operator action this note cannot take. When an operator
applies them, capture cold-start `git status` / `git log -20` timings before
and after (against a comparably-sized loose-object store) and record them here
with the date and host. Until then the Defender contribution to the #4602 stall
remains **modeled**.

## Related

- #4602 — parent: safe object-DB maintenance never folds loose objects
  (Phase 1: `e2a19b789`); #4605 — Phase 3 (trigger), tracked separately.
- #5080 — this note's issue (Phase 5, the ops/non-code half).
- [`docs/host-defender-exclusions.md`](../host-defender-exclusions.md) — the
  fleet-host Defender baseline (per-spawn scan tax; measured ~21%/core
  `MsMpEng`); this note narrows/extends it for the git object DB specifically.
- #3153 — kernel-path churn umbrella; `tools/host_stall_mitigations.ps1`
  already tames some non-fak daemon-floor sources reversibly.
