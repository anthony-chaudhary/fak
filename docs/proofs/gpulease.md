---
title: "fak proof: gpulease machine-wide mutual exclusion"
description: "Proof that fak's GPU lease grants at most one live holder machine-wide via flock, fails closed when busy, and reclaims a crashed holder's lease."
---

# D15 · gpulease

`gpulease` is a **machine-wide advisory lease** so that at most one GPU-heavy process
(e.g. a `-metal` modelbench that uploads gigabytes to unified memory) loads a model at a
time. A process `Acquire()`s the lease before loading and holds it until exit; concurrent
launches **queue** instead of stacking residency and overrunning physical RAM (the
2026-06-18 jetsam/watchdog cascade this module exists to prevent — see `lease.go:1-16`).
The lease is an OS-level `flock` on one lockfile, so it coordinates *unrelated* processes
and is released automatically if the holder dies (the kernel drops the flock when the fd
closes).

This is a **regime D (decision-procedure soundness)** module: "correct" means the
lock-grant decision is *sound* — it never grants two live holders machine-wide (mutual
exclusion), it **fails closed** for a busy lock (`ErrBusy` / queue, never a spurious
grant), and its lifecycle operations are well-defined: `Release` is idempotent and a
crashed holder's lease is reclaimable. The two theorems below discharge exactly those
properties against the real code, each with a deterministic Go test actually run on this
node.

---

## Theorem 1 — at most one holder machine-wide (mutual exclusion)

**THEOREM.** At any instant at most one process holds the lease. While one `Acquire`'s
`Lease` is live, a second `Acquire` on the same lockfile cannot obtain a `Lease`: with
`NoWait` it returns `ErrBusy`, otherwise it queues; the second succeeds only after the
first `Release`.

**REGIME.** D — decision-procedure soundness (the grant decision admits ≤ 1 holder and
fails closed when busy).

**PROOF.** The grant is an exclusive, non-blocking `flock`:
`syscall.Flock(int(f.Fd()), LOCK_EX|LOCK_NB)` at `lock_unix.go:12` (Windows mirror:
`LockFileEx(EXCLUSIVE|FAIL_IMMEDIATELY)` at `lock_windows.go:25-42`). `Acquire`'s loop
returns a `&Lease{...}` **only** when `tryLock` returns `nil` (`lease.go:95-104`); a busy
lock with `NoWait` set returns `ErrBusy` (`lease.go:110-112`) and otherwise polls until the
deadline (`lease.go:114-131`). Because an exclusive `flock` is granted by the kernel to at
most one open fd at a time, no two `Acquire` calls (in the same or different processes) can
simultaneously be holding a `Lease`. `flock` is per-fd, so two separate `Acquire` opens of
one file in a single process contend exactly as two processes would (witness comment,
`lease_test.go:13-17`).

**WITNESS.**
```
go test ./internal/gpulease/ -count=1 -timeout 120s -run 'TestNoWaitBusyThenFree' -v
```
`TestNoWaitBusyThenFree` (`lease_test.go:18-37`) holds lease `a`, asserts a second
`NoWait` `Acquire` returns `ErrBusy`, then after `a.Release()` asserts the second `Acquire`
succeeds. `TestWaitTimesOut` and `TestWaitThenSucceed` cover the blocking-queue branches
(timeout-honored / wait-then-win).

**VERDICT.** PROVEN — 2026-06-20, native darwin/arm64 (the macOS fleet node,
`go test` ran green: `PASS: TestNoWaitBusyThenFree`; `ok ... 0.273s`). Honest scope: this
is the in-process witness; the *active-hold* cross-process direction rests on the same
kernel `flock` primitive, and the cross-process reclaim direction is independently
witnessed by Theorem 2's `TestReleaseOnProcessExit`. A full data-race-freedom proof of the
advisory lock is the named **Gobra** upgrade path (00-METHOD.md §6) and is SCOPED-OUT, not
claimed.

**DOS.** bound at ship.

---

## Theorem 2 — release is idempotent; a crashed holder's lease is reclaimable

**THEOREM.** `Release` is idempotent — a second `Release`, and a `Release` on a `nil`
`*Lease`, are no-ops that do not panic — AND a lease held by a process that exits *without*
calling `Release` is reclaimable by a subsequent `Acquire` (the OS drops the `flock` at fd
close on process death).

**REGIME.** D — lifecycle soundness of the gate (well-defined release; crash-safe reclaim).

**PROOF.** *Idempotence.* `Release` (`lease.go:136-143`) returns early when `l == nil` or
`l.f == nil` (`lease.go:137-139`); on the live path it `unlock`s, `Close`s, and sets
`l.f = nil` (`lease.go:140-142`). A second call therefore hits the `nil`-`f` guard — no
double-unlock, no double-close, no panic — and a `nil`-receiver call is caught by the same
`l == nil` guard. *Crash-reclaim.* The lease holds the lock purely via the open fd's
`flock` (`lock_unix.go:12`); when the holding **process** exits, the kernel closes the fd
and drops the `flock` automatically (documented at `lease.go:13-15`), so the next `Acquire`
succeeds with no explicit `Release`. `unlock` (`lock_unix.go:19-21`, `LOCK_UN`) is only on
the explicit-Release path.

**WITNESS.**
```
go test ./internal/gpulease/ -count=1 -timeout 120s -run 'TestReleaseIdempotent|TestReleaseOnProcessExit' -v
```
`TestReleaseIdempotent` (`lease_test.go:132-142`) acquires, calls `Release()` twice, then
`Release()` on a `nil *Lease` — none panic. `TestReleaseOnProcessExit`
(`lease_test.go:100-129`) re-execs the test binary as a child (`GPULEASE_HELPER_PATH` set)
that `Acquire`s (prints `ACQUIRED`) and `os.Exit(0)` *without* `Release`; the parent
confirms the child held it, then asserts its own `NoWait` `Acquire` succeeds — proving the
`flock` did not leak past process death.

**VERDICT.** PROVEN — 2026-06-20, native darwin/arm64 (`PASS: TestReleaseIdempotent`,
`PASS: TestReleaseOnProcessExit`; `ok ... 0.273s`). Both halves are witnessed by a real
test; the reclaim half is a genuine *cross-process* re-exec, not an in-process proxy.

**DOS.** bound at ship.

---

### Reproduce

```bash
go test ./internal/gpulease/ -count=1 -timeout 120s
```
Native on this macOS arm64 node (`uname`: `Darwin ... arm64`); on the Windows host run
through WSL via `.\fak\test.ps1 ./internal/gpulease/` (root `CLAUDE.md`).
