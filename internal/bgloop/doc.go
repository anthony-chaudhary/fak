// Package bgloop is fak's IN-KERNEL BACKGROUND-LOOP RUNTIME — the supervisor that
// keeps recurring work progressing while the kernel (`fak serve`) is up, and makes
// each loop observable.
//
// Tier: foundation (1) — see internal/architest. It is stdlib-only and imports
// nothing internal, so it sits at the bottom of the layering DAG and any higher
// layer (the gateway, a demo command) may construct it. An upward import would fail
// the architest gate.
//
// # The gap it closes
//
// fak frames itself as "loops all the way down" (docs/explainers/
// engineering-is-building-loops.md): the tool-call, turn, session, fleet, and RSI
// loops. But every one of those is driven from OUTSIDE the running kernel — an OS
// scheduled task on a 10/15/30-minute cadence (the dispatch fleet), a one-shot
// `cmd/rsiloop` invocation, or an agent harness re-invoking itself. When `fak serve`
// is the process that is actually up, there was no first-class notion of a loop the
// KERNEL itself owns and keeps ticking. bgloop is that notion: a registered Loop runs
// in its own supervised goroutine on the serve lifecycle context, restarts on a
// panic or error with capped exponential backoff (a misbehaving loop never takes the
// kernel down), and exposes a point-in-time Snapshot of its progress.
//
// # How it complements loopmgr (the ledger) and looprecover (the worklist)
//
// internal/loopmgr is the durable, hash-chained JSONL LEDGER of loop events, and its
// own doc is blunt that it "does not schedule, spawn, notify, or authorize anything
// by itself; those stay in the callers that produce events." bgloop is exactly that
// missing caller for in-kernel loops: it RUNS them. The two compose without coupling
// — bgloop stays dependency-free and exposes two seams a host wires to loopmgr at a
// higher tier:
//
//   - WithObserver(func(Status)) is the PUSH seam: a host can fold each tick into the
//     loopmgr ledger (an armed/heartbeat/end event) so an in-kernel loop shows up in
//     `fak loop status` next to the externally-scheduled ones. Metrics, by contrast,
//     use the PULL model — the gateway reads Snapshot() at /metrics scrape time — so
//     no observer is needed just to expose Prometheus state.
//   - WithAdmit(func(name) (ok, reason)) is the BACKPRESSURE seam: a host can gate
//     each fire through loopmgr.Governor.Admit, giving an operator the existing
//     pause/disable/cadence-floor knobs over a loop the kernel runs.
//
// # Pure, supervised, observable
//
// The runtime is stdlib-only and deterministic to test: production code reads the
// real clock, and the time-dependent witnesses drive it under testing/synctest's
// virtual time. The three invariants it ships:
//
//   - PROGRESS: an interval Loop keeps ticking while the kernel is up (Status.Ticks
//     climbs, NextTickAt advances).
//   - CONTAINMENT: a Tick that panics or errors is recovered and counted (Panics /
//     Errors / Restarts), the loop backs off and resumes, and neither the supervisor
//     nor any sibling loop is affected.
//   - CLEAN SHUTDOWN: on context cancel (or Shutdown), every loop reaches Stopped and
//     its goroutine is joined within the deadline; a Tick that ignores cancellation
//     makes Shutdown return a timeout error rather than hang silently.
//
// See AGENTS.md and internal/architest for the layering contract; the worked example
// is the `fak bgloop` verb (offline demo + live-server status) and the gateway wiring
// that registers the kernel's built-in heartbeat loop.
package bgloop
