// Package fleetmon is the evidence-derived monitor, janitor, ledger, and
// replacement engine for a headless-worker fleet run (#1856–#1859).
//
// A fleet run launches N headless Claude workers, one per issue, each under its
// own account bucket. During the run an operator (or a supervising agent) needs
// four durable answers that a worker's own "busy" self-report cannot give:
//
//   - monitor (#1856): which workers are healthy, done, dead, stale, blocked, or
//     wedged on a stale child command — classified from REGISTRY + PROCESS +
//     TRANSCRIPT evidence, never from the worker's word.
//   - janitor (#1857): which stale child process trees (a 5-minute `ls`, a
//     wedged `go test`, a broad scan) can be terminated to unstick a worker,
//     WITHOUT killing the worker root or its DOS MCP server.
//   - fold (#1858): a witnessed run ledger recording each issue's real outcome
//     (patch-with-witness / blocked-scoped / read-only / crashed / incomplete /
//     superseded), extracted from the JSONL transcript, not the terminal.
//   - replace (#1859): an account-aware replacement launcher that refuses to
//     relaunch a healthy worker and carries the run's safe-prompt discipline
//     forward when a worker is genuinely unrecoverable.
//
// The package is PURE: every decision function takes an injected snapshot
// (registry rows, process relations, transcript bytes) and returns a typed
// verdict, so the whole surface is unit-tested without a live fleet. The thin
// cmd/fak/fleet.go shell does the actual OS collection (via internal/procguard
// and internal/sessionaudit discovery) and the actual process termination (via
// procguard.KillPID's tree kill). It reuses internal/procguard for the process
// forest + reuse-safe start-time keys and follows internal/nightrun's JSONL
// ledger shape (typed builder + validator), so it adds no new contract the rest
// of the tree does not already keep.
package fleetmon
