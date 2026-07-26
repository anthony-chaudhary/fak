package main

import "sync/atomic"

// wip_spawns.go carries the WIP spine's git-subprocess counter. It lives beside
// wip.go (already at the god-file growth ceiling) rather than inside it.
//
// gitWipSpawns counts every git child process the WIP spine starts, so the scaling
// invariant of the checkpoint listing is WITNESSED rather than asserted by reading
// code: wipListRecords must issue O(1) git spawns in the number of refs (#5336).
// The regression it fences is the one that made the reconciliation spine
// unrunnable — a `git log -1` per ref, which at the ~4k local checkpoints a week
// of fleet work accumulates fanned out to thousands of children and timed out past
// 120s, so `fak wip status | reconcile | attribute | reap` never completed and
// orphaned WIP was never reconciled.
//
// It is pure observability: no code path reads it to make a decision, so a wrong
// count can never change behavior. Atomic because the spine runs passes
// concurrently (and the tests that read it run under -race).
var gitWipSpawns atomic.Int64
